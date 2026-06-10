#include <iostream>
#include <set>
#include <sstream>
#include <vector>
#include "test.h"
#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_interface/cblas_interface.h"
#include "taco/accelerator_interface/tblis_interface.h"
#include "taco/accelerator_interface/gsl_interface.h"
#include "taco/accelerator_interface/avx2_interface.h"
#include "taco/accelerator_interface/mkl_interface.h"

namespace taco {
std::vector<IndexStmt> generateEquivalentStmts(IndexStmt stmt, int depth);
}

using namespace taco;

TEST(mytest, getAllSchedulesSddmm) {
    int NUM_I = 10;
    int NUM_K = 10;
    int NUM_J = 10;

    Tensor<float> B("B", {NUM_I, NUM_K}, CSR);
    Tensor<float> C("C", {NUM_I, NUM_J}, {Dense, Dense});
    Tensor<float> D("D", {NUM_J, NUM_K}, {Dense, Dense});
    Tensor<float> A("A", {NUM_I, NUM_K}, {Dense, Dense}, 0);

    IndexVar i("i"), j("j"), k("k");
    A(i, k) = B(i, k) * sum(j, C(i, j) * D(j, k));

    // autoAccelerate needs reduction notation, not fully concrete notation.
    // concretize() does both steps; here we only do the first.
    IndexStmt stmt = makeReductionNotation(A.getAssignment());

    std::vector<FunctionInterface> functionInterfaces = {
        FunctionInterface(new Saxpy()),
        FunctionInterface(new MatrixMultiply()),
        FunctionInterface(new Sdot()),
        FunctionInterface(new MklDot()),
        FunctionInterface(new MklMM()),
        FunctionInterface(new AVXSaxpy()),
        FunctionInterface(new GSLDot()),
        FunctionInterface(new GSLMM()),
        FunctionInterface(new GSLVecAdd()),
        FunctionInterface(new TblisMultiply()),
        FunctionInterface(new TblisDot()),
        FunctionInterface(new TblisGemv()),
    };

    std::set<std::pair<std::string, std::string>> bindingOpportunities;
    bindingOpportunities.insert({"", ""});
    stmt.helperCheckForMatches(stmt, functionInterfaces, bindingOpportunities);
    for (const auto& rewrite : generateEquivalentStmts(stmt, 3)) {
        stmt.helperCheckForMatches(rewrite, functionInterfaces, bindingOpportunities);
    }

    std::cout << "\n=== Unique Binding Opportunities ===\n";
    std::cout << "Unique Binding Opportunity Count: " << bindingOpportunities.size() << "\n";
    std::cout << "Non-Empty Binding Opportunity Count: " << bindingOpportunities.size() - 1 << "\n";

    int tblisBindingCount = 0;
    for (const auto& binding : bindingOpportunities) {
        if (binding.second.find("tblis") != std::string::npos) {
            tblisBindingCount++;
        }
    }
    std::cout << "TBLIS Binding Opportunity Count: " << tblisBindingCount << "\n";

    int bindingIndex = 0;
    for (const auto& binding : bindingOpportunities) {
        std::cout << "\n--- Binding Opportunity " << bindingIndex++ << " ---\n";
        if (binding.first.empty() && binding.second.empty()) {
            std::cout << "offload: <none>\n";
            std::cout << "function: <none>\n";
            std::cout << "meaning: fully fused generated code\n";
        }
        else {
            std::cout << "offload: " << binding.first << "\n";
            std::cout << "function: " << binding.second << "\n";
        }
    }
    std::cout << "=== End Unique Binding Opportunities ===\n";

    std::vector<IndexStmt> schedules = stmt.autoAccelerate(stmt, functionInterfaces);

    std::set<std::string> uniqueScheduleStrings;
    for (const auto& schedule : schedules) {
        std::stringstream ss;
        ss << schedule;
        uniqueScheduleStrings.insert(ss.str());
    }

    std::cout << "\n=== AutoAccelerate Returned Schedule Statements ===\n";
    std::cout << "AutoAccelerate Returned Schedule Count: " << schedules.size() << "\n";
    std::cout << "AutoAccelerate Unique Printed Schedule Count: " << uniqueScheduleStrings.size() << "\n";

    std::cout << "\n=== Unique Schedule Statements ===\n";
    int uniqueIndex = 0;
    for (const auto& scheduleString : uniqueScheduleStrings) {
        std::cout << "\n--- Unique Schedule " << uniqueIndex++ << " ---\n"
                  << scheduleString << "\n";
    }
    std::cout << "=== End Unique Schedule Statements ===\n";

    std::cout << "\n=== Returned Schedule Statements ===\n";
    for (int s = 0; s < (int)schedules.size(); s++) {
        std::cout << "\n--- Returned Schedule " << s << " ---\n"
                  << schedules[s] << "\n";
    }
    std::cout << "=== End Returned Schedule Statements ===\n";
}
