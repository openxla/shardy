// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_2 = <["x"=2]>
// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

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

// CHECK-LABEL: func.func @all_reduce_not_partitioned
// CHECK-SAME:    (%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}]>}) -> tensor<16xf32>
func.func @all_reduce_not_partitioned(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{}]>}) -> tensor<16xf32> {
  // CHECK-NEXT:          %[[ALL_REDUCE:.*]] = "stablehlo.all_reduce"(%arg0)
  // CHECK-SAME:          channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  // CHECK-SAME:           use_global_device_ids
  // CHECK-NEXT:           ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  // CHECK-NEXT:             %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<f32>
  // CHECK-NEXT:              stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT:            }) : (tensor<16xf32>) -> tensor<16xf32>
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh_2, [{}]> : tensor<16xf32>
  // CHECK-NEXT:  return %[[ALL_REDUCE]] : tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @all_reduce_partitioned
// CHECK-SAME:    (%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>})
func.func @all_reduce_partitioned(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) {
  // CHECK-NEXT:          %[[ALL_REDUCE:.*]] = "stablehlo.all_reduce"(%arg0)
  // CHECK-SAME:          channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>
  // CHECK-SAME:           use_global_device_ids
  // CHECK-NEXT:           ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
  // CHECK-NEXT:             %[[ADD:.*]] = stablehlo.add %arg1, %arg2 : tensor<f32>
  // CHECK-NEXT:              stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT:            }) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  %0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh_2_4, [{"x"}, {}]> : tensor<16x32xf32>
  // CHECK-NEXT:          return %[[ALL_REDUCE]] : tensor<8x32xf32>
  return %0 : tensor<16x32xf32>
}


