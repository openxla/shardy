// RUN: sdy_opt %s --sdy-convert-global-to-local | FileCheck %s

// CHECK: sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>
sdy.mesh @mesh_2_4 = <["x"=2, "y"=4]>

// CHECK-LABEL: func @fully_replicated
func.func @fully_replicated(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {}]>}) -> tensor<8x32xf32> {
  // CHECK: %[[RES:.*]] = stablehlo.dot %arg0, %arg1 : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %0 = stablehlo.dot %arg0, %arg1 : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK: return %[[RES]] : tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @sharded_non_contracting_dims
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>})
// CHECK-SAME: -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
func.func @sharded_non_contracting_dims(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"y"}]>})
  -> (tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {"y"}]>}) {
  // CHECK: %[[RES:.*]] = stablehlo.dot %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>}
  // CHECK-SAME: (tensor<4x16xf32>, tensor<16x8xf32>) -> tensor<4x8xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{"x"}, {"y"}]>]>}
   : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  // CHECK: return %[[RES]] : tensor<4x8xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func @sharded_contracting_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) -> tensor<8x32xf32> {
func.func @sharded_contracting_dim(
  %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{}, {"x"}]>},
  %arg1: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_4, [{"x"}, {}]>}) -> tensor<8x32xf32> {
  // CHECK: %[[DOT:.*]] = stablehlo.dot %[[ARG0]], %[[ARG1]]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}], unreduced={"x"}>]>}
  // CHECK-SAME:(tensor<8x8xf32>, tensor<8x32xf32>) -> tensor<8x32xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_4, [{}, {}], unreduced={"x"}>]>}
   : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>

  // CHECK: %[[RES:.*]] = "stablehlo.all_reduce"(%[[DOT]])
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>
  %1 = sdy.all_reduce {"x"} %0 out_sharding=<@mesh_2_4, [{}, {}]> : tensor<8x32xf32>

  // CHECK: return %[[RES]] : tensor<8x32xf32>
  return %1 : tensor<8x32xf32>
}

