// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>
sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>


// CHECK-LABEL: func @one_dim_two_axes_xy
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
// CHECK-SAME: -> (tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y"}]>})
func.func @one_dim_two_axes_xy(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y"}]>}) {
  // CHECK-DAG: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-DAG: %[[PIDI64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-DAG: %[[OFF0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28]> : tensor<16xi64>
  // CHECK: %[[DS:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PIDI64]], sizes = [1] : (tensor<16xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFF1:.*]] = stablehlo.reshape %[[DS]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[OFF0]], %[[OFF1]], sizes = [16, 4] : (tensor<16x32xf32>, tensor<i64>, tensor<i64>) -> tensor<16x4xf32>
  %0 = sdy.all_slice [{}, {"x", "y"}] %arg0 out_sharding=<@mesh_2_4_2, [{}, {"x", "y"}]> : tensor<16x32xf32>
  // CHECK: return %[[RESULT]] : tensor<16x4xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @one_dim_two_axes_zx
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}]>})
// CHECK-SAME: -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"z", "x"}]>})
func.func @one_dim_two_axes_zx(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}]>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"z", "x"}]>}) {
  // CHECK-DAG: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-DAG: %[[PIDI64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-DAG: %[[OFF0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 16, 0, 16, 0, 16, 0, 16, 8, 24, 8, 24, 8, 24, 8, 24]> : tensor<16xi64>
  // CHECK: %[[DS:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PIDI64]], sizes = [1] : (tensor<16xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFF1:.*]] = stablehlo.reshape %[[DS]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[OFF0]], %[[OFF1]], sizes = [16, 8] : (tensor<16x32xf32>, tensor<i64>, tensor<i64>) -> tensor<16x8xf32>
  %0 = sdy.all_slice [{}, {"z", "x"}] %arg0 out_sharding=<@mesh_2_4_2, [{}, {"z", "x"}]> : tensor<16x32xf32>
  // CHECK: return %[[RESULT]] : tensor<16x8xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @one_dim_two_axes_subaxis
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}]>})
// CHECK-SAME: -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y":(2)2}]>})
func.func @one_dim_two_axes_subaxis(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}]>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y":(2)2}]>}) {
  // CHECK-DAG: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-DAG: %[[PIDI64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-DAG: %[[OFF0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 8, 8, 8, 8, 16, 16, 16, 16, 24, 24, 24, 24]> : tensor<16xi64>
  // CHECK: %[[DS:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PIDI64]], sizes = [1] : (tensor<16xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFF1:.*]] = stablehlo.reshape %[[DS]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[OFF0]], %[[OFF1]], sizes = [16, 8] : (tensor<16x32xf32>, tensor<i64>, tensor<i64>) -> tensor<16x8xf32>
  %0 = sdy.all_slice [{}, {"x", "y":(2)2}] %arg0 out_sharding=<@mesh_2_4_2, [{}, {"x", "y":(2)2}]> : tensor<16x32xf32>
  // CHECK: return %[[RESULT]] : tensor<16x8xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func.func @two_dims_full_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
// CHECK-SAME: -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"y"}]>})
func.func @two_dims_full_axes(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"y"}]>}) {
  // CHECK-DAG:  %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-DAG:  %[[PIDI64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-DAG:  %[[TABLE0:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8]> : tensor<16xi64>
  // CHECK:  %[[DS0:.*]] = stablehlo.dynamic_slice %[[TABLE0]], %[[PIDI64]], sizes = [1] : (tensor<16xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK:  %[[OFF0:.*]] = stablehlo.reshape %[[DS0]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK-DAG:  %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-DAG:  %[[PIDI64_1:.*]] = stablehlo.convert %[[PID1]] : (tensor<ui32>) -> tensor<i64>
  // CHECK:  %[[TABLE1:.*]] = stablehlo.constant dense<[0, 0, 8, 8, 16, 16, 24, 24, 0, 0, 8, 8, 16, 16, 24, 24]> : tensor<16xi64>
  // CHECK:  %[[DS1:.*]] = stablehlo.dynamic_slice %[[TABLE1]], %[[PIDI64_1]], sizes = [1] : (tensor<16xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK:  %[[OFF1:.*]] = stablehlo.reshape %[[DS1]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK:  %[[RESULT:.*]] = stablehlo.dynamic_slice %arg0, %[[OFF0]], %[[OFF1]], sizes = [8, 8] : (tensor<16x32xf32>, tensor<i64>, tensor<i64>) -> tensor<8x8xf32>
  %0 = sdy.all_slice [{"x"}, {"y"}] %arg0 out_sharding=<@mesh_2_4_2, [{"x"}, {"y"}]> : tensor<16x32xf32>
  // CHECK: return %[[RESULT]] : tensor<8x8xf32>
  return %0 : tensor<16x32xf32>
}
