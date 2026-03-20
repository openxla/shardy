// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @replicated_after_all_gather
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"y"}, {}]>}) -> tensor<4x8xf32>
func.func @replicated_after_all_gather(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"y"}, {}]>}) -> tensor<4x8xf32> {
  // CHECK: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{all_gather_dim = 0 : i64,
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // CHECK-SAME{LITERAL}: replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]>
  // CHECK-SAME: (tensor<4x16xf32>) -> tensor<8x16xf32>
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x16xf32>
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[GATHER]] [0:4, 0:8] : (tensor<8x16xf32>) -> tensor<4x8xf32>
  %1 = stablehlo.slice %0 [0:4, 0:8] : (tensor<8x16xf32>) -> tensor<4x8xf32>
  // CHECK: return %[[SLICE]]
  return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @slicing_dim_not_sharded(
// CHECK-SAME:    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>})
// CHECK-SAME:    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>}) {
func.func @slicing_dim_not_sharded(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>})
    -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>}) {
  // CHECK:         %[[RES:.*]] = stablehlo.slice %arg0 [4:12:2, 0:8] : (tensor<16x8xf32>) -> tensor<4x8xf32>
  %0 = stablehlo.slice %arg0 [4:12:2, 0:32] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"x"}]>]>} : (tensor<16x32xf32>) -> tensor<4x32xf32>
  // CHECK:         return %[[RES]] : tensor<4x8xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @slicing_dim_sharded(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
// CHECK-SAME:  -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>}) {
func.func @slicing_dim_sharded(%arg0: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
  -> (tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT:    %[[RES:.*]] = stablehlo.slice %arg0 [0:8:2, 0:8] : (tensor<8x8xf32>) -> tensor<4x8xf32>
  %0 = stablehlo.slice %arg0 [0:32:2, 0:16] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>]>} : (tensor<32x16xf32>) -> tensor<16x16xf32>
  // CHECK-NEXT:    return %[[RES]] : tensor<4x8xf32>
  return %0 : tensor<16x16xf32>
}
