// RUN: sdy_opt %s -split-input-file -sdy-add-data-flow-edges -sdy-basic-propagate -sdy-sink-data-flow-edges 2>&1 | FileCheck %s

sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>

// CHECK-LABEL: func @func_token_arg_skipped(
// CHECK-SAME:      %arg0: !stablehlo.token) -> !stablehlo.token
func.func @func_token_arg_skipped(%arg0: !stablehlo.token) -> !stablehlo.token {
  // CHECK-NEXT: return %arg0 : !stablehlo.token
  return %arg0 : !stablehlo.token
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>

func.func private @callee(%arg0: !stablehlo.token, %arg1: tensor<4xi64>) -> (!stablehlo.token, tensor<4xi64>) {
  return %arg0, %arg1 : !stablehlo.token, tensor<4xi64>
}

// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: !stablehlo.token,
// CHECK-SAME:      %arg1: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>})
func.func @main(%arg0: !stablehlo.token, %arg1: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>}) -> (!stablehlo.token, tensor<4xi64>) {
  %0:2 = call @callee(%arg0, %arg1) : (!stablehlo.token, tensor<4xi64>) -> (!stablehlo.token, tensor<4xi64>)
  // Add a use that expects sharding to verify propagation
  %1 = sdy.sharding_constraint %0#1 <@mesh_a_2_b_2, [{"a"}]> : tensor<4xi64>
  // CHECK: return %0#0, %1 : !stablehlo.token, tensor<4xi64>
  return %0#0, %1 : !stablehlo.token, tensor<4xi64>
}
