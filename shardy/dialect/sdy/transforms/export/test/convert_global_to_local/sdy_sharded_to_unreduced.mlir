// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// CHECK-LABEL: func.func @sharded_to_unreduced_single_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
// CHECK-SAME: -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}], unreduced={"y"}>})
func.func @sharded_to_unreduced_single_dim(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}], unreduced={"y"}>}) {
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4x16xf32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[C4_UI32:.*]] = stablehlo.constant dense<4> : tensor<ui32>
  // CHECK-NEXT: %[[REM:.*]] = stablehlo.remainder %[[PID]], %[[C4_UI32]] : tensor<ui32>
  // CHECK-NEXT: %[[IDX:.*]] = stablehlo.convert %[[REM]] : (tensor<ui32>) -> tensor<i32>
  // CHECK-NEXT: %[[C4_I32:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.multiply %[[IDX]], %[[C4_I32]] : tensor<i32>
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.dynamic_update_slice %[[ZERO]], %[[ARG0]], %[[OFF0]], %[[OFF1]] : (tensor<4x16xf32>, tensor<4x4xf32>, tensor<i32>, tensor<i32>) -> tensor<4x16xf32>
  %0 = sdy.sharded_to_unreduced [{}, {"y"}] %arg0 out_sharding=<@mesh_2_4, [{"x"}, {}], unreduced={"y"}> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<4x16xf32>
  return %0 : tensor<8x16xf32>
}
