#include <vector>
#include "bench.h"
#include "benchmark/benchmark.h"
#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/accelerator_interface/tensor_interface.h"

using namespace taco;


static void bench_gemv_tblis_all_schedules(benchmark::State& state) {

     // actual computation
   int dim = state.range(0);

   Tensor<float> b("b", {dim}, Format{Dense});
   Tensor<float> A("A", {dim, dim}, Format{Dense, Dense});

   for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
         A.insert({i, j}, (float) i + j);
      }
   }

   A.pack();
   for (int i = 0; i < dim; i++) {
      b.insert({i}, (float) i);
   }

   IndexVar i("i");
   IndexVar j("j");
   IndexExpr accelerateExpr = A(i, j) * b(j);

   std::vector<FunctionInterface> availableInterfaces = {
       FunctionInterface(new TblisGemv())
   };

   for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> scheduleProbe("scheduleProbe", {dim}, Format{Dense});
    scheduleProbe(i) = accelerateExpr;

    IndexStmt probeStmt = makeReductionNotation(scheduleProbe.getAssignment());
    std::vector<IndexStmt> probeSchedules =
        probeStmt.autoAccelerate(probeStmt, availableInterfaces);

    for (size_t scheduleIndex = 0;
         scheduleIndex < probeSchedules.size();
         scheduleIndex++) {
      // A Tensor can only proceed through compile/assemble/compute once. Build
      // each schedule from a fresh result tensor so those lifecycle flags and
      // generated modules are not shared between schedules.
      Tensor<float> res("res", {dim}, Format{Dense});
      res(i) = accelerateExpr;

      IndexStmt stmt = makeReductionNotation(res.getAssignment());
      std::vector<IndexStmt> boundSchedules =
          stmt.autoAccelerate(stmt, availableInterfaces);

      if (scheduleIndex >= boundSchedules.size()) {
        state.ResumeTiming();
        state.SkipWithError(
            "autoAccelerate returned an inconsistent schedule count");
        return;
      }

      IndexStmt boundSchedule = boundSchedules[scheduleIndex];

      std::cout << "\n========== FULL SCHEDULE "
                << (scheduleIndex + 1)
                << " ==========\n\n";
      std::cout << "----- Schedule -----\n";
      std::cout << stmt << "\n";

      std::cout << "----- Bound Schedule -----\n";
      std::cout << boundSchedule << "\n";

      res.compile(boundSchedule);
      std::cout << "\n\n----- Generated C -----\n";
      std::cout << res.getSource();
      std::cout << "\n========== END FULL SCHEDULE "
                << (scheduleIndex + 1)
                << " ==========\n\n";

      res.assemble();
      auto func = res.compute_split();
      auto pair = res.returnFuncPackedRaw(func);
      state.ResumeTiming();
      pair.first(func.data());
      state.PauseTiming();
    }

    state.ResumeTiming();
  }

}

TACO_BENCH(bench_gemv_tblis_all_schedules)->DenseRange(250, 5000, 250);
