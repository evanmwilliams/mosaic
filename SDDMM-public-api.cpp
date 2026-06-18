// ============================================================================
// SDDMM-public-api.cpp
//
// A STANDALONE driver (its own main) that uses ONLY Mosaic's public API to,
// for the SDDMM expression, enumerate every rearranged schedule and print ALL
// possible library-function bindings for each rearrangement -- BEFORE the
// greedy "first match / first schedule" choice that autoAccelerate() makes.
//
// It does NOT modify the original mosaic/ sources, and it never calls
// Module::compile() (no kernel JIT, no MKL/CUDA/x86 artifact needed), so it
// runs natively on macOS arm64.
//
// SDDMM:   A(i,k) = B(i,k) * sum(j, C(i,j) * D(j,k))
//   - B is sparse (CSR) -> the "sampling" mask
//   - sum(j, C(i,j)*D(j,k)) is a dense-dense matrix multiply
//
// Mapping to Mosaic's pipeline (see src/index_notation/index_notation.cpp):
//   autoAccelerate():        possibleRewrites = generateEquivalentStmts(stmt,3)
//                            (each rewrite == one candidate SCHEDULE)
//   helperCheckForMatches(): per rewrite, greedily binds the FIRST matching
//                            function per sub-expression
//   concretizeAccelerated(): picks stmts[0] (first schedule)  <-- greedy
//
// Faithfulness to the Mosaic PAPER (Bansal, Hsu, Olukotun, Kjolstad,
// "Mosaic: An Interoperable Compiler for Tensor Algebra", PLDI 2023, PACMPL
// 7(122)). SDDMM (A(i,k)=B(i,k)*sum(j,C(i,j)*D(j,k))) is the paper's own
// running example (Sec 2, Fig 2-4, Fig 9-10). The paper's automated search
// (Sec 6, Fig 9) is a 5-step pipeline:
//   1. filter functions by datatype;
//   2. mathematical rewrites -- "Add Identity (Zero Matrix)" + Commutativity +
//      Distributivity (Sec 6.1)            == generateEquivalentStmts(stmt,3);
//   3. Tensor Index-Variable Matching, "resolving conditions on tensor order"
//                                          == allMatchedOpPatterns/hasPreciseMatch;
//   4. Tiling Validation -- order-reduce a sub-expr (e.g. fix outer index vars,
//      Fig 2 ".fix(iw,jw)") so a lower-order function fits, validated by SMT
//                                          == the hold-constant branch
//                                             (makeCombi/toMatchVars, 4856-4899);
//   5. return ALL valid mappings (the paper: "select one schedule out of all";
//      "We leave [the ranking] autoscheduler as future work", Sec 3/6).
//
// This driver reproduces step 2 (generateEquivalentStmts) and then, for each
// rearrangement, calls Mosaic's OWN matcher helperCheckForMatches() with a
// FRESH per-schedule catalog, which runs steps 3+4 -- so the catalog is the
// COMPLETE set of valid mappings, including the order-reduced (tiled) ones:
//   - DIRECT bind: order already matches (paper's `bind`, no tiling needed).
//   - via TILING/fix: function is lower-order and matches only after holding
//     outer index vars constant (paper's `map`, Sec 5 / Fig 3) -- e.g. the 1-D
//     dot product cblas_sdot mapping the j-contraction with i,k fixed.
// We also report the default pick (first directly-bindable mapping); per the
// paper there is no autoscheduler, so ordering decides.
//
// NOTE (the fix): an earlier version replicated ONLY step 3 (precise match) and
// so MISSED every cblas_sdot mapping (it found 2 functions where Mosaic finds 3,
// verified against stmt.autoAccelerate's catalog). Delegating to
// helperCheckForMatches restores step 4 without reimplementing the file-local
// (static) tiling helpers, which could silently diverge.
// ============================================================================

#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_notation/accelerator_notation.h"
#include "taco/accelerator_notation/accelerate_search.h"
#include "taco/accelerator_interface/cblas_interface.h"

using namespace taco;

// generateEquivalentStmts(IndexStmt, int depth) is the exact overload
// autoAccelerate() uses (with depth==3). It is compiled into libtaco but only
// the 1-arg form is declared in the public header, so we forward-declare the
// depth overload here to call the very same library symbol -- no source edit.
namespace taco {
std::vector<IndexStmt> generateEquivalentStmts(IndexStmt stmt, int depth);
}

// Pull the reduction-notation RHS out of a function interface's semantics --
// this mirrors helperCheckForMatches() lines 4813-4821: getStmt() ->
// AcceleratorAssignment -> makeReductionNotation -> getRhs().
static AcceleratorExpr referenceRhs(const FunctionInterface& fi) {
  AcceleratorStmt refStmt = fi.getNode()->getStmt();
  AcceleratorAssignment assign = to<AcceleratorAssignment>(refStmt);
  AcceleratorAssignment redux = makeReductionNotation(assign);
  return redux.getRhs();
}

template <typename T>
static std::string toStr(const T& x) {
  std::stringstream ss;
  ss << x;
  return ss.str();
}

int main() {
  std::cout << "============================================================\n";
  std::cout << " SDDMM via Mosaic public API: schedules + possible bindings\n";
  std::cout << " (enumerated BEFORE the greedy first-match/first-schedule pick)\n";
  std::cout << "============================================================\n\n";

  // ---- 1. Build the SDDMM statement -------------------------------------
  const int I = 10, J = 10, K = 10;
  Tensor<float> B("B", {I, K}, CSR);                 // sparse sampling mask
  Tensor<float> C("C", {I, J}, {Dense, Dense});
  Tensor<float> D("D", {J, K}, {Dense, Dense});
  Tensor<float> A("A", {I, K}, {Dense, Dense}, 0);

  IndexVar i("i"), j("j"), k("k");
  A(i, k) = B(i, k) * sum(j, C(i, j) * D(j, k));

  IndexStmt stmt = A.getAssignment();
  if (isEinsumNotation(stmt)) stmt = makeReductionNotation(stmt);

  std::cout << "SDDMM statement: " << stmt << "\n\n";

  // ---- 2. Make a few library functions "available" ----------------------
  // MatrixMultiply and BlasSymmLeft BOTH have semantics z(i,k)=x(i,j)*y(j,k),
  // so BOTH match SDDMM's inner sum(j, C(i,j)*D(j,k)) -- this is the
  // "one sub-expression, two functions" case the greedy path collapses.
  std::vector<FunctionInterface> functions = {
      new Saxpy(),          // x(i) = x(i) + y(i)            (vector add)
      new Sdot(),           // s    = x(i) * y(i)            (dot product)
      new MatrixMultiply(), // z(i,k) = x(i,j) * y(j,k)      (matmul)  -> cblas_sgemm
      new BlasSymmLeft(),   // z(i,k) = x(i,j) * y(j,k)      (matmul)  -> cblas_ssymm
      new Sgemm(),          // z(i,k) = x(i,j)*y(j,k)+z(i,k) (matmul + accumulate)
  };

  std::cout << "Available functions registered:\n";
  for (auto& f : functions)
    std::cout << "   - " << f.getNode()->getFunctionName()
              << "   semantics: " << toStr(f.getNode()->getStmt()) << "\n";
  std::cout << "\n";

  // Pre-compute each function's reduction-notation reference RHS once.
  std::vector<AcceleratorExpr> refRhs;
  for (auto& f : functions) refRhs.push_back(referenceRhs(f));

  // ---- 3. Enumerate rearrangements (== candidate schedules) -------------
  // Exactly the call autoAccelerate() makes. forms[0] is the original stmt.
  std::vector<IndexStmt> forms = generateEquivalentStmts(stmt, 3);

  // generateEquivalentStmts can emit duplicates; dedup by printed form so the
  // schedule list stays readable while still covering every distinct rewrite.
  std::set<std::string> seen;
  int scheduleNo = 0;

  for (auto& form : forms) {
    if (!isa<Assignment>(form)) continue;
    std::string key = toStr(form);
    if (!seen.insert(key).second) continue;
    scheduleNo++;

    IndexExpr rhs = to<Assignment>(form).getRhs();

    // COMPLETE candidate set: delegate to Mosaic's own matcher with a FRESH
    // catalog. This captures BOTH precise and hold-constant matches.
    std::set<std::pair<std::string, std::string>> catalog;
    form.helperCheckForMatches(form, functions, catalog);

    // Greedy winner = first PRECISE match (registration order), which is what
    // actually binds into the lowered statement (hold-constant ones don't).
    std::string greedyPick;
    std::set<std::string> preciseKeys;  // "sub||fname" that bind precisely
    for (size_t f = 0; f < functions.size(); f++) {
      for (auto& sub : allMatchedOpPatterns(rhs, refRhs[f])) {
        if (hasPreciseMatch(sub, refRhs[f]).possible) {
          preciseKeys.insert(toStr(sub) + "||" +
                             functions[f].getNode()->getFunctionName());
          if (greedyPick.empty())
            greedyPick = toStr(sub) + "  ->  " +
                         functions[f].getNode()->getFunctionName();
        }
      }
    }

    std::cout << "__Schedule " << scheduleNo << "__\n";
    std::cout << "   rearranged expression : " << form << "\n";
    std::cout << "   valid mappings (Mosaic 5-step search, steps 3+4; pre-select):\n";
    if (catalog.empty()) {
      std::cout << "       (none -- this schedule would run as pure TACO)\n";
    } else {
      for (auto& m : catalog) {
        bool direct = preciseKeys.count(m.first + "||" + m.second) > 0;
        std::cout << "       * " << m.first << "  ->  " << m.second
                  << "   [" << (direct ? "DIRECT bind: order matches"
                                       : "via TILING/fix: order-reduced map")
                  << "]\n";
      }
    }
    std::cout << "   >> default pick (no autoscheduler; first direct bind): "
              << (greedyPick.empty() ? "(nothing)" : greedyPick) << "\n\n";
  }

  std::cout << "------------------------------------------------------------\n";
  std::cout << "Total distinct schedules (rearrangements): " << scheduleNo << "\n";
  std::cout << "NOTE: a schedule may list several valid mappings for the SAME\n";
  std::cout << "sub-expression -- both DIRECT binds (e.g. cblas_sgemm, cblas_ssymm)\n";
  std::cout << "and TILING/fix maps (e.g. the 1-D cblas_sdot order-reduced onto the\n";
  std::cout << "j-contraction). Mosaic's search returns ALL of them; with no\n";
  std::cout << "autoscheduler (paper, future work) the default pick is the first\n";
  std::cout << "direct bind. This driver shows the full set, pre-selection.\n";
  return 0;
}
