// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_2 = <["x"=2]>
// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
// CHECK: sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>
sdy.mesh @mesh_2_4_2 = <["x"=2, "y"=4, "z"=2]>

// CHECK-LABEL: func.func @func_returning_sharded_arg
// CHECK-SAME:    (%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>})
func.func @func_returning_sharded_arg(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}]>}) {
  // CHECK-NEXT:  return %arg0 : tensor<8xf32>
  return %arg0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @func_with_dot_then_add
// CHECK-SAME:    (%arg0: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
// CHECK-SAME:    %arg1: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>},
// CHECK-SAME:    %arg2: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
// CHECK-SAME:    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
func.func @func_with_dot_then_add(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>},
  %arg2: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
  -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT:  %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[DOT]], %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : tensor<4x8xf32>
  %1 = stablehlo.add %0, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : tensor<8x32xf32>
  // CHECK-NEXT:  return %[[ADD]] : tensor<4x8xf32>
  return %1 : tensor<8x32xf32>
}

// CHECK-LABEL: func.func @sdy_all_reduce
// CHECK-SAME:    (%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>})
// CHECK-SAME:    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>}) {
func.func @sdy_all_reduce(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>}) {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_2_4, [{}, {"y"}]> : tensor<16x8xf32>
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_2_4, [{}, {"y"}]> : tensor<16x32xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<16x8xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @sdy_all_slice_on_1dim_2axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
// CHECK-SAME: -> (tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y"}]>})
func.func @sdy_all_slice_on_1dim_2axes(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"x", "y"}]>}) {
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[PIDI64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[OFF0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28]> : tensor<16xi64>
  // CHECK: %[[DS:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PIDI64]], sizes = [1] : (tensor<16xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFF1:.*]] = stablehlo.reshape %[[DS]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RESULT:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[OFF0]], %[[OFF1]], sizes = [16, 4] : (tensor<16x32xf32>, tensor<i64>, tensor<i64>) -> tensor<16x4xf32>
  %0 = sdy.all_slice [{}, {"x", "y"}] %arg0 out_sharding=<@mesh_2_4_2, [{}, {"x", "y"}]> : tensor<16x32xf32>
  // CHECK: return %[[RESULT]] : tensor<16x4xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func.func @sdy_all_slice_on_2dims
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
// CHECK-SAME: -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"y"}]>})
func.func @sdy_all_slice_on_2dims(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {}], replicated={"x", "y"}>})
    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {"y"}]>}) {
  // CHECK:  %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK:  %[[PIDI64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK:  %[[TABLE0:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8]> : tensor<16xi64>
  // CHECK:  %[[DS0:.*]] = stablehlo.dynamic_slice %[[TABLE0]], %[[PIDI64]], sizes = [1] : (tensor<16xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK:  %[[OFF0:.*]] = stablehlo.reshape %2 : (tensor<1xi64>) -> tensor<i64>
  // CHECK:  %[[TABLE1:.*]] = stablehlo.constant dense<[0, 0, 8, 8, 16, 16, 24, 24, 0, 0, 8, 8, 16, 16, 24, 24]> : tensor<16xi64>
  // CHECK:  %[[DS1:.*]] = stablehlo.dynamic_slice %[[TABLE1]], %[[PIDI64]], sizes = [1] : (tensor<16xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK:  %[[OFF1:.*]] = stablehlo.reshape %4 : (tensor<1xi64>) -> tensor<i64>
  // CHECK:  %[[RESULT:.*]] = stablehlo.dynamic_slice %arg0, %[[OFF0]], %[[OFF1]], sizes = [8, 8] : (tensor<16x32xf32>, tensor<i64>, tensor<i64>) -> tensor<8x8xf32>
  %0 = sdy.all_slice [{"x"}, {"y"}] %arg0 out_sharding=<@mesh_2_4_2, [{"x"}, {"y"}]> : tensor<16x32xf32>
  // CHECK: return %6 : tensor<8x8xf32>
  return %0 : tensor<16x32xf32>
}
