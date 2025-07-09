// RUN: sdy_opt %s 2>&1 | FileCheck %s

sdy.mesh @mesh1 = <["x"=2, "y"=2]>

// CHECK-LABEL: func @basic_edge_sharding
func.func @basic_edge_sharding(%arg0 : tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK: %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[
  // CHECK:   {step-2 = [{"y" = operand-1 -> [operand-0, result-0]}]},
  // CHECK:   {step-12345 = [{"x" = result-0 -> [operand-0]}]}]>,
  // CHECK:   sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"y"}, {"x"}]>]>
  // CHECK: } : tensor<16x8xf32>
  %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-2 = [{"y" = operand-1 -> [operand-0, result-0]}]}, {step-12345 = [{"x" = result-0 -> [operand-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"y"}, {"x" }]>]>} : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
