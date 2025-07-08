// RUN: sdy_opt %s 2>&1 | FileCheck %s

sdy.mesh @mesh1 = <["x"=2, "y"=2]>

// CHECK-LABEL: func @basic_edge_sharding
func.func @basic_edge_sharding(%arg0 : tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK: %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[
  // CHECK:   {step_2 = [{"y" = OPERAND_1 -> [OPERAND_0, RESULT_0]}]},
  // CHECK:   {step_12345 = [{"x" = RESULT_0 -> [OPERAND_0]}]}]>,
  // CHECK:   sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"y"}, {"x"}]>]>
  // CHECK: } : tensor<16x8xf32>
  %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step_2 = [{"y" = OPERAND_1 -> [OPERAND_0, RESULT_0]}]}, {step_12345 = [{"x" = RESULT_0 -> [OPERAND_0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh1, [{"y"}, {"x" }]>]>} : tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

