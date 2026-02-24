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

// CHECK-LABEL: func @all_gather_one_dim
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
// CHECK-SAME:    -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
func.func@all_gather_one_dim(%arg0 : tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x", "y"}, {}]>})
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [ {"x"}, {} ]>}){
  // CHECK: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{
  // CHECK-SAME:   all_gather_dim = 0 : i64,
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
  // CHECK-SAME{literal}:   replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>,
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> : (tensor<1x16xf32>) -> tensor<4x16xf32>
  %0 = sdy.all_gather[{"y"}, {}] %arg0 out_sharding = <@mesh_2_4, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK: return %[[GATHER]] : tensor<4x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @all_gather_two_axes
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "y"}, {"z"}]>})
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{}, {"z"}]>})
func.func@all_gather_two_axes(%arg0 : tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "y"}, {"z"}]>})
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [ {}, {"z"} ]>}){
  // CHECK: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{
  // CHECK-SAME:   all_gather_dim = 0 : i64,
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>,
  // CHECK-SAME{literal}:   replica_groups = dense<[[0, 2, 4, 6, 8, 10, 12, 14], [1, 3, 5, 7, 9, 11, 13, 15]]> : tensor<2x8xi64>,
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> : (tensor<1x8xf32>) -> tensor<8x8xf32>
  %0 = sdy.all_gather[{"x", "y"}, {}] %arg0 out_sharding = <@mesh_2_4_2, [{}, {"z"}]> : tensor<8x16xf32>
  // CHECK: return %[[GATHER]] : tensor<8x8xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @all_gather_two_dims
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<1x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "y"}, {"z"}]>})
// CHECK-SAME:    -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x"}, {}]>})
func.func@all_gather_two_dims(%arg0 : tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [{"x", "y"}, {"z"}]>})
  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4_2, [ {"x"}, {} ]>}){
  // CHECK: %[[GATHER1:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{
  // CHECK-SAME:   all_gather_dim = 0 : i64,
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>,
  // CHECK-SAME{literal}:   replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]]> : tensor<4x4xi64>,
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> : (tensor<1x8xf32>) -> tensor<4x8xf32>
  // CHECK: %[[GATHER2:.*]] = "stablehlo.all_gather"(%[[GATHER1]]) <{
  // CHECK-SAME:   all_gather_dim = 1 : i64,
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>,
  // CHECK-SAME{literal}:   replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]> : tensor<8x2xi64>,
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> : (tensor<4x8xf32>) -> tensor<4x16xf32>
  %0 = sdy.all_gather[{"y"}, {"z"}] %arg0 out_sharding = <@mesh_2_4_2, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK: return %[[GATHER2]] : tensor<4x16xf32>
  return %0 : tensor<8x16xf32>
}
