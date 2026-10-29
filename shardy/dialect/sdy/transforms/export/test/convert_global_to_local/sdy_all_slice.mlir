// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>
sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>


// CHECK-LABEL: func @one_dim_two_axes_xy
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
// CHECK-SAME: -> (tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y"}]>})
func.func @one_dim_two_axes_xy(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y"}]>}) {
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[C8:.*]] = stablehlo.constant dense<8> : tensor<ui32>
  // CHECK-NEXT: %[[DIV0:.*]] = stablehlo.divide %[[PID]], %[[C8]] : tensor<ui32>
  // CHECK-NEXT: %[[C2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // CHECK-NEXT: %[[DIV1:.*]] = stablehlo.divide %[[PID]], %[[C2]] : tensor<ui32>
  // CHECK-NEXT: %[[C4_UI32_1:.*]] = stablehlo.constant dense<4> : tensor<ui32>
  // CHECK-NEXT: %[[REM1:.*]] = stablehlo.remainder %[[DIV1]], %[[C4_UI32_1]] : tensor<ui32>
  // CHECK-NEXT: %[[C4_UI32:.*]] = stablehlo.constant dense<4> : tensor<ui32>
  // CHECK-NEXT: %[[MUL0:.*]] = stablehlo.multiply %[[DIV0]], %[[C4_UI32]] : tensor<ui32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[MUL0]], %[[REM1]] : tensor<ui32>
  // CHECK-NEXT: %[[IDX:.*]] = stablehlo.convert %[[ADD]] : (tensor<ui32>) -> tensor<i32>
  // CHECK-NEXT: %[[C4_I32:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.multiply %[[IDX]], %[[C4_I32]] : tensor<i32>
  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[OFF0]], %[[OFF1]], sizes = [16, 4] : (tensor<16x32xf32>, tensor<i32>, tensor<i32>) -> tensor<16x4xf32>
  %0 = sdy.all_slice [{}, {"x", "y"}] %arg0 out_sharding=<@mesh_2_4_2, [{}, {"x", "y"}]> : tensor<16x32xf32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<16x4xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @one_dim_two_axes_zx
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}]>})
// CHECK-SAME: -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"z", "x"}]>})
func.func @one_dim_two_axes_zx(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}]>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"z", "x"}]>}) {
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[C2_0:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // CHECK-NEXT: %[[REM0:.*]] = stablehlo.remainder %[[PID]], %[[C2_0]] : tensor<ui32>
  // CHECK-NEXT: %[[C8_UI32:.*]] = stablehlo.constant dense<8> : tensor<ui32>
  // CHECK-NEXT: %[[DIV1:.*]] = stablehlo.divide %[[PID]], %[[C8_UI32]] : tensor<ui32>
  // CHECK-NEXT: %[[C2_1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // CHECK-NEXT: %[[MUL0:.*]] = stablehlo.multiply %[[REM0]], %[[C2_1]] : tensor<ui32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[MUL0]], %[[DIV1]] : tensor<ui32>
  // CHECK-NEXT: %[[IDX:.*]] = stablehlo.convert %[[ADD]] : (tensor<ui32>) -> tensor<i32>
  // CHECK-NEXT: %[[C8_I32:.*]] = stablehlo.constant dense<8> : tensor<i32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.multiply %[[IDX]], %[[C8_I32]] : tensor<i32>
  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[OFF0]], %[[OFF1]], sizes = [16, 8] : (tensor<16x32xf32>, tensor<i32>, tensor<i32>) -> tensor<16x8xf32>
  %0 = sdy.all_slice [{}, {"z", "x"}] %arg0 out_sharding=<@mesh_2_4_2, [{}, {"z", "x"}]> : tensor<16x32xf32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<16x8xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @one_dim_two_axes_subaxis
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}]>})
// CHECK-SAME: -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y":(2)2}]>})
func.func @one_dim_two_axes_subaxis(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}]>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y":(2)2}]>}) {
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[C8_UI32:.*]] = stablehlo.constant dense<8> : tensor<ui32>
  // CHECK-NEXT: %[[DIV0:.*]] = stablehlo.divide %[[PID]], %[[C8_UI32]] : tensor<ui32>
  // CHECK-NEXT: %[[C2_1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // CHECK-NEXT: %[[DIV1:.*]] = stablehlo.divide %[[PID]], %[[C2_1]] : tensor<ui32>
  // CHECK-NEXT: %[[C2_2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // CHECK-NEXT: %[[REM1:.*]] = stablehlo.remainder %[[DIV1]], %[[C2_2]] : tensor<ui32>
  // CHECK-NEXT: %[[C2_0:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // CHECK-NEXT: %[[MUL0:.*]] = stablehlo.multiply %[[DIV0]], %[[C2_0]] : tensor<ui32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[MUL0]], %[[REM1]] : tensor<ui32>
  // CHECK-NEXT: %[[IDX:.*]] = stablehlo.convert %[[ADD]] : (tensor<ui32>) -> tensor<i32>
  // CHECK-NEXT: %[[C8_I32:.*]] = stablehlo.constant dense<8> : tensor<i32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.multiply %[[IDX]], %[[C8_I32]] : tensor<i32>
  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[OFF0]], %[[OFF1]], sizes = [16, 8] : (tensor<16x32xf32>, tensor<i32>, tensor<i32>) -> tensor<16x8xf32>
  %0 = sdy.all_slice [{}, {"x", "y":(2)2}] %arg0 out_sharding=<@mesh_2_4_2, [{}, {"x", "y":(2)2}]> : tensor<16x32xf32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<16x8xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func.func @two_dims_full_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
// CHECK-SAME: -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"y"}]>})
func.func @two_dims_full_axes(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[C8_UI32:.*]] = stablehlo.constant dense<8> : tensor<ui32>
  // CHECK-NEXT: %[[DIV0:.*]] = stablehlo.divide %[[PID0]], %[[C8_UI32]] : tensor<ui32>
  // CHECK-NEXT: %[[IDX0:.*]] = stablehlo.convert %[[DIV0]] : (tensor<ui32>) -> tensor<i32>
  // CHECK-NEXT: %[[C8_I32:.*]] = stablehlo.constant dense<8> : tensor<i32>
  // CHECK-NEXT: %[[OFF0:.*]] = stablehlo.multiply %[[IDX0]], %[[C8_I32]] : tensor<i32>
  // CHECK-NEXT: %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[C2_UI32:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // CHECK-NEXT: %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C2_UI32]] : tensor<ui32>
  // CHECK-NEXT: %[[C4_UI32:.*]] = stablehlo.constant dense<4> : tensor<ui32>
  // CHECK-NEXT: %[[REM1:.*]] = stablehlo.remainder %[[DIV1]], %[[C4_UI32]] : tensor<ui32>
  // CHECK-NEXT: %[[IDX1:.*]] = stablehlo.convert %[[REM1]] : (tensor<ui32>) -> tensor<i32>
  // CHECK-NEXT: %[[C8_I32_1:.*]] = stablehlo.constant dense<8> : tensor<i32>
  // CHECK-NEXT: %[[OFF1:.*]] = stablehlo.multiply %[[IDX1]], %[[C8_I32_1]] : tensor<i32>
  // CHECK-NEXT: %[[RESULT:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[OFF0]], %[[OFF1]], sizes = [8, 8] : (tensor<16x32xf32>, tensor<i32>, tensor<i32>) -> tensor<8x8xf32>
  %0 = sdy.all_slice [{"x"}, {"y"}] %arg0 out_sharding=<@mesh_2_4_2, [{"x"}, {"y"}]> : tensor<16x32xf32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<8x8xf32>
  return %0 : tensor<16x32xf32>
}
