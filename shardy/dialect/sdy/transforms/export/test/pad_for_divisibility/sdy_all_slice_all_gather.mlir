// RUN: sdy_opt %s -sdy-pad-for-divisibility

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @expand_then_trim
func.func @expand_then_trim(%arg0: tensor<4x7xi32> ) -> tensor<4x7xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
  // CHECK-NEXT: %[[ALL_SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi32>
  // CHECK-NEXT: %[[AG:.*]] = sdy.all_gather [{}, {"y"}] %[[ALL_SLICE]] out_sharding=<@mesh_4_2, [{}, {}]> : tensor<4x8xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[AG]] [0:4, 0:7] : (tensor<4x8xi32>) -> tensor<4x7xi32>
  // CHECK-NEXT: return %[[SLICE]] : tensor<4x7xi32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>
  %1 = sdy.all_gather [{}, {"y"}] %0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<4x7xi32>
  return %1 : tensor<4x7xi32>
}
