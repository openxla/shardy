
// RUN: sdy_opt %s --sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// CHECK-LABEL: func @replicated_after_all_gather
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) -> tensor<10x18xf32>
func.func @replicated_after_all_gather(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) -> tensor<10x18xf32> {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  %pv = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{all_gather_dim = 0 : i64,
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]>
  // CHECK-SAME: (tensor<4x16xf32>) -> tensor<8x16xf32>
  %0 = sdy.all_gather [{"x"}, {}] %arg0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<8x16xf32>
  // CHECK: %[[PAD:.*]] = stablehlo.pad %[[GATHER]], %[[CST]], low = [1, 1], high = [1, 1], interior = [0, 0] : (tensor<8x16xf32>, tensor<f32>) -> tensor<10x18xf32>
  %1 = stablehlo.pad %0, %pv, low = [1, 1], high = [1, 1], interior = [0, 0] : (tensor<8x16xf32>, tensor<f32>) -> tensor<10x18xf32>
  // CHECK: return %[[PAD]]
  return %1 : tensor<10x18xf32>
}

// CHECK-LABEL: func @pad_non_sharded_dim
// CHECK-SAME:  %[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
// CHECK-SAME:  -> (tensor<8x24xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
func.func @pad_non_sharded_dim(%arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
 -> (tensor<16x24xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  // CHECK: %[[RES:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 4], high = [0, 4], interior = [0, 0] : (tensor<8x16xf32>, tensor<f32>) -> tensor<8x24xf32>
  %pv = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %pv, low=[0, 4], high=[0, 4], interior=[0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}]>]>} : (tensor<16x16xf32>, tensor<f32>) -> tensor<16x24xf32>
   // CHECK: return %[[RES]] : tensor<8x24xf32>
  return %0 : tensor<16x24xf32>
}

// CHECK-LABEL: func @pad_sharded_dim_uniform_on_partitions_1
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
// CHECK-SAME:  -> (tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
func.func @pad_sharded_dim_uniform_on_partitions_1(%arg0: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
 -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  %pv = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[RES:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 1], high = [0, 0], interior = [0, 1] : (tensor<2x4xf32>, tensor<f32>) -> tensor<2x8xf32>
  %0 = stablehlo.pad %arg0, %pv, low=[0, 1], high=[0, 0], interior=[0, 1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : (tensor<4x16xf32>, tensor<f32>) -> tensor<4x32xf32>
   // CHECK: return %[[RES]] : tensor<2x8xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @pad_sharded_dim_uniform_on_partitions_2
// CHECK-SAME:  %[[ARG0:.*]]: tensor<8x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
// CHECK-SAME:  -> (tensor<16x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
func.func @pad_sharded_dim_uniform_on_partitions_2(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
 -> (tensor<32x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  %pv = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[RES:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [1, 0], interior = [1, 0] : (tensor<8x1xf32>, tensor<f32>) -> tensor<16x1xf32>
  %0 = stablehlo.pad %arg0, %pv, low=[0, 0], high=[1, 0], interior=[1, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : (tensor<16x4xf32>, tensor<f32>) -> tensor<32x4xf32>
  // CHECK: return %[[RES]] : tensor<16x1xf32>
  return %0 : tensor<32x4xf32>
}

// CHECK-LABEL: func @pad_sharded_dim_non_uniform_on_partitions(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<8x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
// CHECK-SAME:  -> (tensor<25x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
func.func @pad_sharded_dim_non_uniform_on_partitions(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>})
 -> (tensor<50x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  %pv = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[SAFE_PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [2, 0], high = [2, 0], interior = [2, 0] : (tensor<8x1xf32>, tensor<f32>) -> tensor<26x1xf32>
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 0, 0, 0, 1, 1, 1, 1]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID_I64]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFF0:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFF1:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[RES:.*]] = stablehlo.dynamic_slice %[[SAFE_PAD]], %[[OFF0]], %[[OFF1]], sizes = [25, 1] : (tensor<26x1xf32>, tensor<i64>, tensor<i64>) -> tensor<25x1xf32>
  %0 = stablehlo.pad %arg0, %pv, low=[2, 0], high=[2, 0], interior=[2, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>} : (tensor<16x4xf32>, tensor<f32>) -> tensor<50x4xf32>
  // CHECK: return %[[RES]] : tensor<25x1xf32>
  return %0 : tensor<50x4xf32>
}

// CHECK-LABEL: func @pad_negative_edges_non_sharded_dim(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
// CHECK-SAME:  -> (tensor<8x12xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
func.func @pad_negative_edges_non_sharded_dim(%arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
 -> (tensor<16x12xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  %pv = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[RES:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, -2], high = [0, -2], interior = [0, 0] : (tensor<8x16xf32>, tensor<f32>) -> tensor<8x12xf32>
  %0 = stablehlo.pad %arg0, %pv, low=[0, -2], high=[0, -2], interior=[0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}]>]>} : (tensor<16x16xf32>, tensor<f32>) -> tensor<16x12xf32>
   // CHECK: return %[[RES]] : tensor<8x12xf32>
  return %0 : tensor<16x12xf32>
}

// CHECK-LABEL: func @pad_negative_edges_sharded_dim(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>})
// CHECK-SAME:  -> (tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>}) {
func.func @pad_negative_edges_sharded_dim(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>})
 -> (tensor<12xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  %pv = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[SAFE_PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0], high = [0], interior = [0] : (tensor<8xf32>, tensor<f32>) -> tensor<8xf32>
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[2, 2, 2, 2, 0, 0, 0, 0]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID_I64]], sizes = [1] : (tensor<8xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK: %[[OFF:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[RES:.*]] = stablehlo.dynamic_slice %[[SAFE_PAD]], %[[OFF]], sizes = [6] : (tensor<8xf32>, tensor<i64>) -> tensor<6xf32>
  %0 = stablehlo.pad %arg0, %pv, low=[-2], high=[-2], interior=[0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}]>]>} : (tensor<16xf32>, tensor<f32>) -> tensor<12xf32>
  // CHECK: return %[[RES]] : tensor<6xf32>
  return %0 : tensor<12xf32>
}

// CHECK-LABEL: func @pad_negative_edges_interior_sharded_dim(
// CHECK-SAME:  %[[ARG0:.*]]: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
// CHECK-SAME:  -> (tensor<13x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
func.func @pad_negative_edges_interior_sharded_dim(%arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
 -> (tensor<26x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
  // CHECK: %[[CST:.*]] = stablehlo.constant
  %pv = stablehlo.constant dense<0.0> : tensor<f32>
  // CHECK: %[[SAFE_PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [1, 0], high = [0, 0], interior = [1, 0] : (tensor<8x4xf32>, tensor<f32>) -> tensor<16x4xf32>
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[CVT:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[3, 3, 3, 3, 0, 0, 0, 0]> : tensor<8xi64>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[CVT]], sizes = [1]
  // CHECK: %[[OFF0:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK: %[[OFF1:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK: %[[RES:.*]] = stablehlo.dynamic_slice %[[SAFE_PAD]], %[[OFF0]], %[[OFF1]], sizes = [13, 4] : (tensor<16x4xf32>, tensor<i64>, tensor<i64>) -> tensor<13x4xf32>
  %0 = stablehlo.pad %arg0, %pv, low=[-2, 0], high=[-3, 0], interior=[1, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {}]>]>} : (tensor<16x4xf32>, tensor<f32>) -> tensor<26x4xf32>
  // CHECK: return %[[RES]] : tensor<13x4xf32>
  return %0 : tensor<26x4xf32>
}
