// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// CHECK-LABEL: func @shard_result
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y":(2)2}]>})
// CHECK-SAME: -> (tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y":(2)2}]>}) {
func.func @shard_result(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y":(2)2}]>})
  -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y":(2)2}]>}) {
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[ARG0]])
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 2], [1, 3], [4, 6], [5, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
  // CHECK: ^bb0(%[[ACC:.*]]: tensor<f32>, %[[UPD:.*]]: tensor<f32>):
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[ACC]], %[[UPD]] : tensor<f32>
  // CHECK: stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK: }) : (tensor<16x16xf32>) -> tensor<16x16xf32>
  %0 = sdy.all_reduce {"y":(1)2} %arg0 out_sharding=<@mesh_2_4, [{}, {"y":(2)2}]> : tensor<16x32xf32>
  // CHECK: return %[[RES]] : tensor<16x16xf32>
  return %0 : tensor<16x32xf32>
}

// CHECK-LABEL: func @not_shard_result
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x32xf32>) -> tensor<16x32xf32> {
func.func @not_shard_result(%arg0: tensor<16x32xf32>)
  -> tensor<16x32xf32> {
  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[ARG0]])
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
  // CHECK: ^bb0(%[[ACC:.*]]: tensor<f32>, %[[UPD:.*]]: tensor<f32>):
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[ACC]], %[[UPD]] : tensor<f32>
  // CHECK: stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK: }) : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %0 = sdy.all_reduce {"x", "y":(1)2} %arg0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<16x32xf32>
  // CHECK: return %[[RES]] : tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}
