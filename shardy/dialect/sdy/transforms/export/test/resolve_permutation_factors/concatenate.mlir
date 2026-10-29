// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s --check-prefixes=CHECK,HALO

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>
// CHECK: @mesh_a_4 = <["a"=4]>
sdy.mesh @mesh_a_4 = <["a"=4]>

// CHECK-LABEL: func @concatenate_rotate_right_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func @concatenate_rotate_right_sharded(
    %arg0: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
    -> (tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  // REPL-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]> : tensor<10xf32>
  // REPL-NEXT: %[[SLICE0:.*]] = stablehlo.slice %[[RESHARD0]] [4:10] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<10xf32>) -> tensor<6xf32>
  // REPL-NEXT: %[[RESHARD_S0:.*]] = sdy.reshard %[[SLICE0]] <@mesh, [{"a"}]> : tensor<6xf32>
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]> : tensor<10xf32>
  // REPL-NEXT: %[[SLICE1:.*]] = stablehlo.slice %[[RESHARD1]] [0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<10xf32>) -> tensor<4xf32>
  // REPL-NEXT: %[[RESHARD_S1:.*]] = sdy.reshard %[[SLICE1]] <@mesh, [{"a"}]> : tensor<4xf32>
  // REPL-NEXT: %[[RESHARD_C0:.*]] = sdy.reshard %[[RESHARD_S0]] <@mesh, [{}]> : tensor<6xf32>
  // REPL-NEXT: %[[RESHARD_C1:.*]] = sdy.reshard %[[RESHARD_S1]] <@mesh, [{}]> : tensor<4xf32>
  // REPL-NEXT: %[[CONCAT_REPL:.*]] = stablehlo.concatenate %[[RESHARD_C0]], %[[RESHARD_C1]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<6xf32>, tensor<4xf32>) -> tensor<10xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONCAT_REPL]] <@mesh, [{"a"}]> : tensor<10xf32>

  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]]) in_shardings=[<@mesh, [{"a"}]>] out_shardings=[<@mesh, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<5xf32>) {
  // HALO-NEXT:   %[[S_LEFT:.*]] = stablehlo.slice %arg1 [4:5] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<5xf32>) -> tensor<1xf32>
  // HALO-NEXT:   %[[S_RIGHT:.*]] = stablehlo.slice %arg1 [0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<5xf32>) -> tensor<4xf32>
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[S_RIGHT]]) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 0], [3, 1]]> : tensor<4x2xi64>
  // HALO-SAME:   }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<4xf32>) -> tensor<4xf32>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[S_LEFT]], %[[CP]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<1xf32>, tensor<4xf32>) -> tensor<5xf32>
  // HALO-NEXT:   sdy.return %[[CONCAT]] : tensor<5xf32>
  // HALO-NEXT: } : (tensor<10xf32>) -> tensor<10xf32>
  %0 = stablehlo.slice %arg0 [4:10] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>
  } : (tensor<10xf32>) -> tensor<6xf32>
  %1 = stablehlo.slice %arg0 [0:4] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>
  } : (tensor<10xf32>) -> tensor<4xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>
  } : (tensor<6xf32>, tensor<4xf32>) -> tensor<10xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<10xf32>
  return %2 : tensor<10xf32>
}

// CHECK-LABEL: func @concatenate_rotate_right_with_aligned_padding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>})
func.func @concatenate_rotate_right_with_aligned_padding(
    %arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>})
    -> (tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>}) {
  // REPL-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh_a_4, [{}]> : tensor<6xf32>
  // REPL-NEXT: %[[SLICE0:.*]] = stablehlo.slice %[[RESHARD0]] [4:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<6xf32>) -> tensor<2xf32>
  // REPL-NEXT: %[[RESHARD_S0:.*]] = sdy.reshard %[[SLICE0]] <@mesh_a_4, [{"a"}]> : tensor<2xf32>
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a_4, [{}]> : tensor<6xf32>
  // REPL-NEXT: %[[SLICE1:.*]] = stablehlo.slice %[[RESHARD1]] [0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<6xf32>) -> tensor<4xf32>
  // REPL-NEXT: %[[RESHARD_S1:.*]] = sdy.reshard %[[SLICE1]] <@mesh_a_4, [{"a"}]> : tensor<4xf32>
  // REPL-NEXT: %[[RESHARD_C0:.*]] = sdy.reshard %[[RESHARD_S0]] <@mesh_a_4, [{}]> : tensor<2xf32>
  // REPL-NEXT: %[[RESHARD_C1:.*]] = sdy.reshard %[[RESHARD_S1]] <@mesh_a_4, [{}]> : tensor<4xf32>
  // REPL-NEXT: %[[CONCAT_REPL:.*]] = stablehlo.concatenate %[[RESHARD_C0]], %[[RESHARD_C1]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<2xf32>, tensor<4xf32>) -> tensor<6xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONCAT_REPL]] <@mesh_a_4, [{"a"}]> : tensor<6xf32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0], high = [2], interior = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>} : (tensor<6xf32>, tensor<f32>) -> tensor<8xf32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[PAD]]) in_shardings=[<@mesh_a_4, [{"a"}]>] out_shardings=[<@mesh_a_4, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<2xf32>) {
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%arg1) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 0]]> : tensor<3x2xi64>
  // HALO-SAME:   }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<2xf32>) -> tensor<2xf32>
  // HALO-NEXT:   sdy.return %[[CP]] : tensor<2xf32>
  // HALO-NEXT: } : (tensor<8xf32>) -> tensor<8xf32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>} : (tensor<8xf32>) -> tensor<6xf32>
  %0 = stablehlo.slice %arg0 [4:6] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>
  } : (tensor<6xf32>) -> tensor<2xf32>
  %1 = stablehlo.slice %arg0 [0:4] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>
  } : (tensor<6xf32>) -> tensor<4xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>
  } : (tensor<2xf32>, tensor<4xf32>) -> tensor<6xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<6xf32>
  return %2 : tensor<6xf32>
}

// CHECK-LABEL: func @concatenate_rotate_right_with_unaligned_padding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>})
func.func @concatenate_rotate_right_with_unaligned_padding(
    %arg0: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>})
    -> (tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>}) {
  // REPL-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh_a_4, [{}]> : tensor<10xf32>
  // REPL-NEXT: %[[SLICE0:.*]] = stablehlo.slice %[[RESHARD0]] [4:10] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<10xf32>) -> tensor<6xf32>
  // REPL-NEXT: %[[RESHARD_S0:.*]] = sdy.reshard %[[SLICE0]] <@mesh_a_4, [{"a"}]> : tensor<6xf32>
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a_4, [{}]> : tensor<10xf32>
  // REPL-NEXT: %[[SLICE1:.*]] = stablehlo.slice %[[RESHARD1]] [0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<10xf32>) -> tensor<4xf32>
  // REPL-NEXT: %[[RESHARD_S1:.*]] = sdy.reshard %[[SLICE1]] <@mesh_a_4, [{"a"}]> : tensor<4xf32>
  // REPL-NEXT: %[[RESHARD_C0:.*]] = sdy.reshard %[[RESHARD_S0]] <@mesh_a_4, [{}]> : tensor<6xf32>
  // REPL-NEXT: %[[RESHARD_C1:.*]] = sdy.reshard %[[RESHARD_S1]] <@mesh_a_4, [{}]> : tensor<4xf32>
  // REPL-NEXT: %[[CONCAT_REPL:.*]] = stablehlo.concatenate %[[RESHARD_C0]], %[[RESHARD_C1]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<6xf32>, tensor<4xf32>) -> tensor<10xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONCAT_REPL]] <@mesh_a_4, [{"a"}]> : tensor<10xf32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0], high = [2], interior = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>} : (tensor<10xf32>, tensor<f32>) -> tensor<12xf32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[PAD]]) in_shardings=[<@mesh_a_4, [{"a"}]>] out_shardings=[<@mesh_a_4, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<3xf32>) {
  // HALO-NEXT:   %[[ROT0:.*]] = "stablehlo.collective_permute"(%arg1) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 0], [3, 1]]> : tensor<4x2xi64>
  // HALO-SAME:   }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<3xf32>) -> tensor<3xf32>
  // HALO-NEXT:   %[[S0:.*]] = stablehlo.slice %arg1 [1:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<3xf32>) -> tensor<2xf32>
  // HALO-NEXT:   %[[CP0:.*]] = "stablehlo.collective_permute"(%[[S0]]) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 3], [1, 0], [2, 1], [3, 2]]> : tensor<4x2xi64>
  // HALO-SAME:   }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<2xf32>) -> tensor<2xf32>
  // HALO-NEXT:   %[[S1:.*]] = stablehlo.slice %arg1 [0:1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<3xf32>) -> tensor<1xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[S1]]) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 0], [3, 1]]> : tensor<4x2xi64>
  // HALO-SAME:   }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<1xf32>) -> tensor<1xf32>
  // HALO-NEXT:   %[[ROT1:.*]] = stablehlo.concatenate %[[CP0]], %[[CP1]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<2xf32>, tensor<1xf32>) -> tensor<3xf32>
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[PID_I32:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C3:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[OFFSET:.*]] = stablehlo.multiply %[[PID_I32]], %[[C3]] : tensor<i32>
  // HALO-NEXT:   %[[C6:.*]] = stablehlo.constant dense<6> : tensor<i32>
  // HALO-NEXT:   %[[BOUNDARY:.*]] = stablehlo.subtract %[[C6]], %[[OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[BOUNDARY]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<i32>) -> tensor<3xi32>
  // HALO-NEXT:   %[[IOTA:.*]] = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : tensor<3xi32>
  // HALO-NEXT:   %[[PRED:.*]] = stablehlo.compare LT, %[[IOTA]], %[[BCAST]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  // HALO-NEXT:   %[[SEL:.*]] = stablehlo.select %[[PRED]], %[[ROT1]], %[[ROT0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : tensor<3xi1>, tensor<3xf32>
  // HALO-NEXT:   sdy.return %[[SEL]] : tensor<3xf32>
  // HALO-NEXT: } : (tensor<12xf32>) -> tensor<12xf32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:10] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>} : (tensor<12xf32>) -> tensor<10xf32>
  %0 = stablehlo.slice %arg0 [4:10] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>
  } : (tensor<10xf32>) -> tensor<6xf32>
  %1 = stablehlo.slice %arg0 [0:4] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>
  } : (tensor<10xf32>) -> tensor<4xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>
  } : (tensor<6xf32>, tensor<4xf32>) -> tensor<10xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<10xf32>
  return %2 : tensor<10xf32>
}

// CHECK-LABEL: func @concatenate_rotate_right_2d_dim1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
func.func @concatenate_rotate_right_2d_dim1(
    %arg0: tensor<4x10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
    -> (tensor<4x10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // REPL-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{"a"}, {}]> : tensor<4x10xf32>
  // REPL-NEXT: %[[SLICE0:.*]] = stablehlo.slice %[[RESHARD0]] [0:4, 4:10] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<4x10xf32>) -> tensor<4x6xf32>
  // REPL-NEXT: %[[RESHARD_S0:.*]] = sdy.reshard %[[SLICE0]] <@mesh, [{"a"}, {"b"}]> : tensor<4x6xf32>
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{"a"}, {}]> : tensor<4x10xf32>
  // REPL-NEXT: %[[SLICE1:.*]] = stablehlo.slice %[[RESHARD1]] [0:4, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<4x10xf32>) -> tensor<4x4xf32>
  // REPL-NEXT: %[[RESHARD_S1:.*]] = sdy.reshard %[[SLICE1]] <@mesh, [{"a"}, {"b"}]> : tensor<4x4xf32>
  // REPL-NEXT: %[[RESHARD_C0:.*]] = sdy.reshard %[[RESHARD_S0]] <@mesh, [{"a"}, {}]> : tensor<4x6xf32>
  // REPL-NEXT: %[[RESHARD_C1:.*]] = sdy.reshard %[[RESHARD_S1]] <@mesh, [{"a"}, {}]> : tensor<4x4xf32>
  // REPL-NEXT: %[[CONCAT_REPL:.*]] = stablehlo.concatenate %[[RESHARD_C0]], %[[RESHARD_C1]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<4x6xf32>, tensor<4x4xf32>) -> tensor<4x10xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONCAT_REPL]] <@mesh, [{"a"}, {"b"}]> : tensor<4x10xf32>

  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<2x5xf32>) {
  // HALO-NEXT:   %[[S_LEFT:.*]] = stablehlo.slice %arg1 [0:2, 4:5] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x5xf32>) -> tensor<2x1xf32>
  // HALO-NEXT:   %[[S_RIGHT:.*]] = stablehlo.slice %arg1 [0:2, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x5xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[S_RIGHT]]) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 0], [2, 3], [3, 2]]> : tensor<4x2xi64>
  // HALO-SAME:   }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[S_LEFT]], %[[CP]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x1xf32>, tensor<2x4xf32>) -> tensor<2x5xf32>
  // HALO-NEXT:   sdy.return %[[CONCAT]] : tensor<2x5xf32>
  // HALO-NEXT: } : (tensor<4x10xf32>) -> tensor<4x10xf32>
  %0 = stablehlo.slice %arg0 [0:4, 4:10] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<4x10xf32>) -> tensor<4x6xf32>
  %1 = stablehlo.slice %arg0 [0:4, 0:4] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<4x10xf32>) -> tensor<4x4xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<4x6xf32>, tensor<4x4xf32>) -> tensor<4x10xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<4x10xf32>
  return %2 : tensor<4x10xf32>
}

// CHECK-LABEL: func @concatenate_rotate_right_multi_use_slice_not_matched
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func @concatenate_rotate_right_multi_use_slice_not_matched(
    %arg0: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
    -> (tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
        tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  // REPL-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]> : tensor<10xf32>
  // REPL-NEXT: %[[SLICE0:.*]] = stablehlo.slice %[[RESHARD0]] [4:10] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<10xf32>) -> tensor<6xf32>
  // REPL-NEXT: %[[S0:.*]] = sdy.reshard %[[SLICE0]] <@mesh, [{"a"}]> : tensor<6xf32>

  // HALO-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[S0:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST0]]) in_shardings=[<@mesh, [{"a"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<5xf32>, %arg2: tensor<f32>) {
  // HALO-NEXT:   %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[PID0_I32:.*]] = stablehlo.convert %[[PID0]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C2_0:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[DIV0:.*]] = stablehlo.divide %[[PID0_I32]], %[[C2_0]] : tensor<i32>
  // HALO-NEXT:   %[[C2_1:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[REM0:.*]] = stablehlo.remainder %[[DIV0]], %[[C2_1]] : tensor<i32>
  // HALO-NEXT:   %[[CP0:.*]] = "stablehlo.collective_permute"(%arg1) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>
  // HALO-SAME:   }> : (tensor<5xf32>) -> tensor<5xf32>
  // HALO-NEXT:   %[[C1_0:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP0:.*]] = stablehlo.compare LT, %[[REM0]], %[[C1_0]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PAD0:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<5xf32>
  // HALO-NEXT:   %[[BCAST0:.*]] = stablehlo.broadcast_in_dim %[[CMP0]], dims = [] : (tensor<i1>) -> tensor<5xi1>
  // HALO-NEXT:   %[[SEL0:.*]] = stablehlo.select %[[BCAST0]], %[[CP0]], %[[BCAST_PAD0]] : tensor<5xi1>, tensor<5xf32>
  // HALO-NEXT:   %[[CONCAT0:.*]] = stablehlo.concatenate %arg1, %[[SEL0]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<5xf32>, tensor<5xf32>) -> tensor<10xf32>
  // HALO-NEXT:   %[[PAD0:.*]] = stablehlo.pad %[[CONCAT0]], %arg2, low = [3], high = [3], interior = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<10xf32>, tensor<f32>) -> tensor<16xf32>
  // HALO-NEXT:   %[[CM2_0:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:   %[[C7_0:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // HALO-NEXT:   %[[MUL0:.*]] = stablehlo.multiply %[[REM0]], %[[CM2_0]] : tensor<i32>
  // HALO-NEXT:   %[[ADD0:.*]] = stablehlo.add %[[MUL0]], %[[C7_0]] : tensor<i32>
  // HALO-NEXT:   %[[CZ0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[MAX0:.*]] = stablehlo.maximum %[[ADD0]], %[[CZ0]] : tensor<i32>
  // HALO-NEXT:   %[[DS0:.*]] = stablehlo.dynamic_slice %[[PAD0]], %[[MAX0]], sizes = [3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<16xf32>, tensor<i32>) -> tensor<3xf32>
  // HALO-NEXT:   sdy.return %[[DS0]] : tensor<3xf32>
  // HALO-NEXT: } : (tensor<10xf32>, tensor<f32>) -> tensor<6xf32>
  %0 = stablehlo.slice %arg0 [4:10] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>
  } : (tensor<10xf32>) -> tensor<6xf32>

  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]> : tensor<10xf32>
  // REPL-NEXT: %[[SLICE1:.*]] = stablehlo.slice %[[RESHARD1]] [0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<10xf32>) -> tensor<4xf32>
  // REPL-NEXT: %[[S1:.*]] = sdy.reshard %[[SLICE1]] <@mesh, [{"a"}]> : tensor<4xf32>

  // HALO-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[S1:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST1]]) in_shardings=[<@mesh, [{"a"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<5xf32>, %arg2: tensor<f32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[PID1_I32:.*]] = stablehlo.convert %[[PID1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C2_2:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1_I32]], %[[C2_2]] : tensor<i32>
  // HALO-NEXT:   %[[C2_3:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[REM1:.*]] = stablehlo.remainder %[[DIV1]], %[[C2_3]] : tensor<i32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>
  // HALO-SAME:   }> : (tensor<5xf32>) -> tensor<5xf32>
  // HALO-NEXT:   %[[C1_1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare GE, %[[REM1]], %[[C1_1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PAD1:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<5xf32>
  // HALO-NEXT:   %[[BCAST1:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<5xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BCAST1]], %[[CP1]], %[[BCAST_PAD1]] : tensor<5xi1>, tensor<5xf32>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<5xf32>, tensor<5xf32>) -> tensor<10xf32>
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [2], high = [2], interior = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<10xf32>, tensor<f32>) -> tensor<14xf32>
  // HALO-NEXT:   %[[CM3_1:.*]] = stablehlo.constant dense<-3> : tensor<i32>
  // HALO-NEXT:   %[[C7_1:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // HALO-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[REM1]], %[[CM3_1]] : tensor<i32>
  // HALO-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[C7_1]] : tensor<i32>
  // HALO-NEXT:   %[[CZ1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[MAX1:.*]] = stablehlo.maximum %[[ADD1]], %[[CZ1]] : tensor<i32>
  // HALO-NEXT:   %[[DS1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[MAX1]], sizes = [2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<14xf32>, tensor<i32>) -> tensor<2xf32>
  // HALO-NEXT:   sdy.return %[[DS1]] : tensor<2xf32>
  // HALO-NEXT: } : (tensor<10xf32>, tensor<f32>) -> tensor<4xf32>
  %1 = stablehlo.slice %arg0 [0:4] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>
  } : (tensor<10xf32>) -> tensor<4xf32>

  // CHECK-NEXT: %[[RESHARD_C0:.*]] = sdy.reshard %[[S0]] <@mesh, [{}]> : tensor<6xf32>
  // CHECK-NEXT: %[[RESHARD_C1:.*]] = sdy.reshard %[[S1]] <@mesh, [{}]> : tensor<4xf32>
  // CHECK-NEXT: %[[CONCAT_REPL:.*]] = stablehlo.concatenate %[[RESHARD_C0]], %[[RESHARD_C1]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<6xf32>, tensor<4xf32>) -> tensor<10xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONCAT_REPL]] <@mesh, [{"a"}]> : tensor<10xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>
  } : (tensor<6xf32>, tensor<4xf32>) -> tensor<10xf32>
  // CHECK-NEXT: return %[[RES]], %[[S0]] : tensor<10xf32>, tensor<6xf32>
  return %2, %0 : tensor<10xf32>, tensor<6xf32>
}

// CHECK-LABEL: func @concatenate_identity_order_slices_not_matched
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func @concatenate_identity_order_slices_not_matched(
    %arg0: tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
    -> (tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  // REPL-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]> : tensor<10xf32>
  // REPL-NEXT: %[[SLICE0:.*]] = stablehlo.slice %[[RESHARD0]] [0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<10xf32>) -> tensor<4xf32>
  // REPL-NEXT: %[[S0:.*]] = sdy.reshard %[[SLICE0]] <@mesh, [{"a"}]> : tensor<4xf32>

  // HALO-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[S0:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST0]]) in_shardings=[<@mesh, [{"a"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<5xf32>, %arg2: tensor<f32>) {
  // HALO-NEXT:   %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[PID0_I32:.*]] = stablehlo.convert %[[PID0]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C2_0:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[DIV0:.*]] = stablehlo.divide %[[PID0_I32]], %[[C2_0]] : tensor<i32>
  // HALO-NEXT:   %[[C2_1:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[REM0:.*]] = stablehlo.remainder %[[DIV0]], %[[C2_1]] : tensor<i32>
  // HALO-NEXT:   %[[CP0:.*]] = "stablehlo.collective_permute"(%arg1) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>
  // HALO-SAME:   }> : (tensor<5xf32>) -> tensor<5xf32>
  // HALO-NEXT:   %[[C1_0:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP0:.*]] = stablehlo.compare GE, %[[REM0]], %[[C1_0]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PAD0:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<5xf32>
  // HALO-NEXT:   %[[BCAST0:.*]] = stablehlo.broadcast_in_dim %[[CMP0]], dims = [] : (tensor<i1>) -> tensor<5xi1>
  // HALO-NEXT:   %[[SEL0:.*]] = stablehlo.select %[[BCAST0]], %[[CP0]], %[[BCAST_PAD0]] : tensor<5xi1>, tensor<5xf32>
  // HALO-NEXT:   %[[CONCAT0:.*]] = stablehlo.concatenate %[[SEL0]], %arg1, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<5xf32>, tensor<5xf32>) -> tensor<10xf32>
  // HALO-NEXT:   %[[PAD0:.*]] = stablehlo.pad %[[CONCAT0]], %arg2, low = [2], high = [2], interior = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<10xf32>, tensor<f32>) -> tensor<14xf32>
  // HALO-NEXT:   %[[CM3_0:.*]] = stablehlo.constant dense<-3> : tensor<i32>
  // HALO-NEXT:   %[[C7_0:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // HALO-NEXT:   %[[MUL0:.*]] = stablehlo.multiply %[[REM0]], %[[CM3_0]] : tensor<i32>
  // HALO-NEXT:   %[[ADD0:.*]] = stablehlo.add %[[MUL0]], %[[C7_0]] : tensor<i32>
  // HALO-NEXT:   %[[CZ0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[MAX0:.*]] = stablehlo.maximum %[[ADD0]], %[[CZ0]] : tensor<i32>
  // HALO-NEXT:   %[[DS0:.*]] = stablehlo.dynamic_slice %[[PAD0]], %[[MAX0]], sizes = [2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<14xf32>, tensor<i32>) -> tensor<2xf32>
  // HALO-NEXT:   sdy.return %[[DS0]] : tensor<2xf32>
  // HALO-NEXT: } : (tensor<10xf32>, tensor<f32>) -> tensor<4xf32>
  %0 = stablehlo.slice %arg0 [0:4] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>
  } : (tensor<10xf32>) -> tensor<4xf32>

  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]> : tensor<10xf32>
  // REPL-NEXT: %[[SLICE1:.*]] = stablehlo.slice %[[RESHARD1]] [4:10] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<10xf32>) -> tensor<6xf32>
  // REPL-NEXT: %[[S1:.*]] = sdy.reshard %[[SLICE1]] <@mesh, [{"a"}]> : tensor<6xf32>

  // HALO-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[S1:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST1]]) in_shardings=[<@mesh, [{"a"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<5xf32>, %arg2: tensor<f32>) {
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[PID1_I32:.*]] = stablehlo.convert %[[PID1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C2_3:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1_I32]], %[[C2_3]] : tensor<i32>
  // HALO-NEXT:   %[[C2_4:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[REM1:.*]] = stablehlo.remainder %[[DIV1]], %[[C2_4]] : tensor<i32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>
  // HALO-SAME:   }> : (tensor<5xf32>) -> tensor<5xf32>
  // HALO-NEXT:   %[[C1_1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[CMP1:.*]] = stablehlo.compare LT, %[[REM1]], %[[C1_1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PAD1:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<f32>) -> tensor<5xf32>
  // HALO-NEXT:   %[[BCAST1:.*]] = stablehlo.broadcast_in_dim %[[CMP1]], dims = [] : (tensor<i1>) -> tensor<5xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BCAST1]], %[[CP1]], %[[BCAST_PAD1]] : tensor<5xi1>, tensor<5xf32>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %arg1, %[[SEL1]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<5xf32>, tensor<5xf32>) -> tensor<10xf32>
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [3], high = [3], interior = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<10xf32>, tensor<f32>) -> tensor<16xf32>
  // HALO-NEXT:   %[[CM2_1:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:   %[[C7_1:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // HALO-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[REM1]], %[[CM2_1]] : tensor<i32>
  // HALO-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[C7_1]] : tensor<i32>
  // HALO-NEXT:   %[[CZ1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[MAX1:.*]] = stablehlo.maximum %[[ADD1]], %[[CZ1]] : tensor<i32>
  // HALO-NEXT:   %[[DS1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[MAX1]], sizes = [3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<16xf32>, tensor<i32>) -> tensor<3xf32>
  // HALO-NEXT:   sdy.return %[[DS1]] : tensor<3xf32>
  // HALO-NEXT: } : (tensor<10xf32>, tensor<f32>) -> tensor<6xf32>
  %1 = stablehlo.slice %arg0 [4:10] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>
  } : (tensor<10xf32>) -> tensor<6xf32>

  // CHECK-NEXT: %[[RESHARD_C0:.*]] = sdy.reshard %[[S0]] <@mesh, [{}]> : tensor<4xf32>
  // CHECK-NEXT: %[[RESHARD_C1:.*]] = sdy.reshard %[[S1]] <@mesh, [{}]> : tensor<6xf32>
  // CHECK-NEXT: %[[CONCAT_REPL:.*]] = stablehlo.concatenate %[[RESHARD_C0]], %[[RESHARD_C1]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<4xf32>, tensor<6xf32>) -> tensor<10xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONCAT_REPL]] <@mesh, [{"a"}]> : tensor<10xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>
  } : (tensor<4xf32>, tensor<6xf32>) -> tensor<10xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<10xf32>
  return %2 : tensor<10xf32>
}

// CHECK-LABEL: func @plain_concatenate_replicated_dim_not_touched
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}, %[[ARG1:.*]]: tensor<6x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>})
func.func @plain_concatenate_replicated_dim_not_touched(
    %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>},
    %arg1: tensor<6x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>})
    -> (tensor<10x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}) {
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.concatenate %[[ARG0]], %[[ARG1]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>} : (tensor<4x8xf32>, tensor<6x8xf32>) -> tensor<10x8xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>
  } : (tensor<4x8xf32>, tensor<6x8xf32>) -> tensor<10x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<10x8xf32>
  return %0 : tensor<10x8xf32>
}

// CHECK-LABEL: func @plain_concatenate_sharded_dim_replicated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}, %[[ARG1:.*]]: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func @plain_concatenate_sharded_dim_replicated(
    %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
    %arg1: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
    -> (tensor<10xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  // CHECK-NEXT: %[[RESHARD_C0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]> : tensor<4xf32>
  // CHECK-NEXT: %[[RESHARD_C1:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}]> : tensor<6xf32>
  // CHECK-NEXT: %[[CONCAT_REPL:.*]] = stablehlo.concatenate %[[RESHARD_C0]], %[[RESHARD_C1]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<4xf32>, tensor<6xf32>) -> tensor<10xf32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[CONCAT_REPL]] <@mesh, [{"a"}]> : tensor<10xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>
  } : (tensor<4xf32>, tensor<6xf32>) -> tensor<10xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<10xf32>
  return %0 : tensor<10xf32>
}

// CHECK-LABEL: func @concatenate_rotate_right_with_unaligned_partial_shards
// CHECK-SAME: (%[[ARG0:.*]]: tensor<5xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>})
func.func @concatenate_rotate_right_with_unaligned_partial_shards(
    %arg0: tensor<5xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>})
    -> (tensor<5xf32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>}) {
  // REPL-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh_a_4, [{}]> : tensor<5xf32>
  // REPL-NEXT: %[[SLICE0:.*]] = stablehlo.slice %[[RESHARD0]] [3:5] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<5xf32>) -> tensor<2xf32>
  // REPL-NEXT: %[[RESHARD_S0:.*]] = sdy.reshard %[[SLICE0]] <@mesh_a_4, [{"a"}]> : tensor<2xf32>
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a_4, [{}]> : tensor<5xf32>
  // REPL-NEXT: %[[SLICE1:.*]] = stablehlo.slice %[[RESHARD1]] [0:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<5xf32>) -> tensor<3xf32>
  // REPL-NEXT: %[[RESHARD_S1:.*]] = sdy.reshard %[[SLICE1]] <@mesh_a_4, [{"a"}]> : tensor<3xf32>
  // REPL-NEXT: %[[RESHARD_C0:.*]] = sdy.reshard %[[RESHARD_S0]] <@mesh_a_4, [{}]> : tensor<2xf32>
  // REPL-NEXT: %[[RESHARD_C1:.*]] = sdy.reshard %[[RESHARD_S1]] <@mesh_a_4, [{}]> : tensor<3xf32>
  // REPL-NEXT: %[[CONCAT_REPL:.*]] = stablehlo.concatenate %[[RESHARD_C0]], %[[RESHARD_C1]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<2xf32>, tensor<3xf32>) -> tensor<5xf32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[CONCAT_REPL]] <@mesh_a_4, [{"a"}]> : tensor<5xf32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // HALO-NEXT: %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0], high = [3], interior = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>} : (tensor<5xf32>, tensor<f32>) -> tensor<8xf32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[PAD]]) in_shardings=[<@mesh_a_4, [{"a"}]>] out_shardings=[<@mesh_a_4, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<2xf32>) {
  // HALO-NEXT:   %[[ROT0:.*]] = "stablehlo.collective_permute"(%arg1) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 0]]> : tensor<3x2xi64>
  // HALO-SAME:   }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<2xf32>) -> tensor<2xf32>
  // HALO-NEXT:   %[[S0:.*]] = stablehlo.slice %arg1 [1:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<2xf32>) -> tensor<1xf32>
  // HALO-NEXT:   %[[CP0:.*]] = "stablehlo.collective_permute"(%[[S0]]) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 0], [2, 1]]> : tensor<3x2xi64>
  // HALO-SAME:   }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<1xf32>) -> tensor<1xf32>
  // HALO-NEXT:   %[[S1:.*]] = stablehlo.slice %arg1 [0:1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<2xf32>) -> tensor<1xf32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[S1]]) <{
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 0]]> : tensor<3x2xi64>
  // HALO-SAME:   }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<1xf32>) -> tensor<1xf32>
  // HALO-NEXT:   %[[ROT1:.*]] = stablehlo.concatenate %[[CP0]], %[[CP1]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[PID_I32:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[C2:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[OFFSET:.*]] = stablehlo.multiply %[[PID_I32]], %[[C2]] : tensor<i32>
  // HALO-NEXT:   %[[C2_0:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[BOUNDARY:.*]] = stablehlo.subtract %[[C2_0]], %[[OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[BOUNDARY]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<i32>) -> tensor<2xi32>
  // HALO-NEXT:   %[[IOTA:.*]] = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : tensor<2xi32>
  // HALO-NEXT:   %[[PRED:.*]] = stablehlo.compare LT, %[[IOTA]], %[[BCAST]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // HALO-NEXT:   %[[SEL:.*]] = stablehlo.select %[[PRED]], %[[ROT1]], %[[ROT0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}]>]>} : tensor<2xi1>, tensor<2xf32>
  // HALO-NEXT:   sdy.return %[[SEL]] : tensor<2xf32>
  // HALO-NEXT: } : (tensor<8xf32>) -> tensor<8xf32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:5] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>} : (tensor<8xf32>) -> tensor<5xf32>
  %0 = stablehlo.slice %arg0 [3:5] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>
  } : (tensor<5xf32>) -> tensor<2xf32>
  %1 = stablehlo.slice %arg0 [0:3] {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>
  } : (tensor<5xf32>) -> tensor<3xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>
  } : (tensor<2xf32>, tensor<3xf32>) -> tensor<5xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<5xf32>
  return %2 : tensor<5xf32>
}
