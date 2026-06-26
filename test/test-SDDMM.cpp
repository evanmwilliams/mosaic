#include "taco/tensor.h"
#include "taco/index_notation/index_notation.h"
#include "taco/accelerator_notation/accel_interface.h"
#include "taco/accelerator_interface/cblas_interface.h"

using namespace taco;

int main() {
  const int I = 10, J = 10, K = 10;
  Tensor<float> B("B", {I, K}, CSR);
  Tensor<float> C("C", {I, J}, {Dense, Dense});
  Tensor<float> D("D", {J, K}, {Dense, Dense});
  Tensor<float> A("A", {I, K}, {Dense, Dense}, 0);

  IndexVar i("i"), j("j"), k("k");
  A(i, k) = B(i, k) * sum(j, C(i, j) * D(j, k));

  A.registerAccelerator(new Saxpy());          
  A.registerAccelerator(new Sdot());           
  A.registerAccelerator(new MatrixMultiply());  
  
  IndexStmt stmt = A.getAssignment();
  if (isEinsumNotation(stmt)) stmt = makeReductionNotation(stmt);
  stmt.autoAccelerate(stmt, A.getRegisteredAccelerators());

  return 0;
}
