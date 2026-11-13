// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s --check-prefixes=CHECK,HALO

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>
// CHECK: @mesh_a4 = <["a"=4, "b"=2]>
sdy.mesh @mesh_a4 = <["a"=4, "b"=2]>
// CHECK: @mesh_a8 = <["a"=8]>
sdy.mesh @mesh_a8 = <["a"=8]>
// CHECK: @mesh_a3_b4 = <["a"=3, "b"=4]>
sdy.mesh @mesh_a3_b4 = <["a"=3, "b"=4]>

// CHECK-LABEL: func @strided_slice
// CHECK-SAME: (%[[ARG0:.*]]: tensor<12x8xi32>
func.func @strided_slice(
  %arg0: tensor<12x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>})
  -> tensor<5x8xi32> {
  // REPL: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}, {"b"}]> : tensor<12x8xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD0]] [0:10:2, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {"b"}]>]>} : (tensor<12x8xi32>) -> tensor<5x8xi32>
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[SLICE]] <@mesh_a4, [{"a"}, {"b"}]> : tensor<5x8xi32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[RESHARD1]] <@mesh_a4, [{}, {}]> : tensor<5x8xi32>

  // HALO: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a"}, {"b"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<3x4xi32>, %arg2: tensor<i32>) {
  // HALO:      %[[HOP_SLICE:.*]] = stablehlo.slice %arg1 [0:2, 0:4]
  // HALO-NEXT: %[[CP:.*]] = "stablehlo.collective_permute"(%[[HOP_SLICE]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [4, 2], [5, 3], [6, 4], [7, 5]]> : tensor<6x2xi64>
  // HALO:      %[[SEL:.*]] = stablehlo.select {{.*}}, %[[CP]], {{.*}}
  // HALO-NEXT: %[[CONCAT:.*]] = stablehlo.concatenate %arg1, %[[SEL]], dim = 0
  // HALO-NEXT: %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [0, 0], high = [3, 0], interior = [0, 0]
  // HALO:      %[[DYN_SLICE:.*]] = stablehlo.dynamic_slice %[[PAD]], {{.*}}, sizes = [3, 4]
  // HALO-NEXT: %[[STRIDED:.*]] = stablehlo.slice %[[DYN_SLICE]] [0:3:2, 0:4]
  // HALO-NEXT: sdy.return %[[STRIDED]] : tensor<2x4xi32>
  // HALO-NEXT: }
  // HALO-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[MC]] [0:5, 0:8]
  // HALO-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a4, [{}, {}]> : tensor<5x8xi32>
  %slice = stablehlo.slice %arg0 [0:10:2, 0:8]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>]>}
    : (tensor<12x8xi32>) -> tensor<5x8xi32>

  %0 = sdy.reshard %slice <@mesh_a4, [{}, {}]> : tensor<5x8xi32>

  // CHECK: return %[[RES]] : tensor<5x8xi32>
  return %0 : tensor<5x8xi32>
}

// CHECK-LABEL: func @slice_comm_free
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4xi32>
func.func @slice_comm_free(
  %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
  -> tensor<3xi32> {
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:3]
  %0 = stablehlo.slice %arg0 [0:3] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}]>]>
  } : (tensor<4xi32>) -> tensor<3xi32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.all_gather [{"a"}] %[[SLICE]]
  %1 = sdy.all_gather [{"a"}] %0 out_sharding=<@mesh, [{}]> : tensor<3xi32>
  // CHECK-NEXT: return %[[RES]]
  return %1 : tensor<3xi32>
}

// CHECK-LABEL: func @slice_comm_free_to_one
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<8xi32>
func.func @slice_comm_free_to_one(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
  -> tensor<1xi32> {
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:1]
  %0 = stablehlo.slice %arg0 [0:1] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}]>]>
  } : (tensor<8xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.all_gather [{"a"}] %[[SLICE]]
  %1 = sdy.all_gather [{"a"}] %0 out_sharding=<@mesh, [{}]> : tensor<1xi32>
  // CHECK-NEXT: return %[[RES]]
  return %1 : tensor<1xi32>
}

// CHECK-LABEL: func @slice_partition_partial_dim_with_communication
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<8xi32>
func.func @slice_partition_partial_dim_with_communication(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
  -> tensor<3xi32> {
  // REPL-NEXT:  %[[RESHARD_0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]>
  // REPL-NEXT:  %[[SLICE:.*]] = stablehlo.slice {{.*}} [0:3]
  // REPL-NEXT:  %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh, [{"a"}]>

  // HALO-NEXT:  %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:  %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<4xi32>, %arg2: tensor<i32>) {
  // HALO:         %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:    %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:    %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:    %[[CONV:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:    %[[S_HOP:.*]] = stablehlo.slice %arg1 [2:4]
  // HALO-NEXT:    %[[CP1:.*]] = "stablehlo.collective_permute"(%[[S_HOP]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>
  // HALO-NEXT:    %[[S_SELF:.*]] = stablehlo.slice %arg1 [0:2]
  // HALO-NEXT:    %[[CONCAT:.*]] = stablehlo.concatenate %[[CP1]], %[[S_SELF]], dim = 0
  // HALO-NEXT:    %[[C_STRIDE:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:    %[[C_OFFSET:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:    %[[MUL:.*]] = stablehlo.multiply %[[CONV]], %[[C_STRIDE]] : tensor<i32>
  // HALO-NEXT:    %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_OFFSET]] : tensor<i32>
  // HALO-NEXT:    %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:    %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT]], %[[ADD]], sizes = [2]
  // HALO-NEXT:    sdy.return %[[SLICE1]] : tensor<2xi32>
  // HALO-NEXT:  }
  // HALO-NEXT:  %[[RES:.*]] = stablehlo.slice %[[MC]] [0:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : (tensor<4xi32>) -> tensor<3xi32>
  %0 = stablehlo.slice %arg0 [0:3] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}]>]>
  } : (tensor<8xi32>) -> tensor<3xi32>

 // CHECK-NEXT:  %[[AG:.*]] = sdy.all_gather [{"a"}] %[[RES]] out_sharding=<@mesh, [{}]> : tensor<3xi32>
  %1 = sdy.all_gather [{"a"}] %0 out_sharding=<@mesh, [{}]> : tensor<3xi32>
  // CHECK-NEXT:  return %[[AG]] : tensor<3xi32>
  return %1 : tensor<3xi32>
}

// CHECK-LABEL: func @slice_multidim_mixed
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x4xi32>
func.func @slice_multidim_mixed(
  %arg0: tensor<4x4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}, {}]>})
  -> tensor<3x3x2xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{"a"}, {}, {}]> : tensor<4x4x4xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [0:3, 1:4, 1:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}, {}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh, [{"a"}, {"b"}, {}]> : tensor<3x3x2xi32>

  // HALO-NEXT: %[[CST_0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST_0]]) in_shardings=[<@mesh, [{"a"}, {"b"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}, {}]>] manual_axes={"a", "b"} (%arg1: tensor<2x2x4xi32>, %arg2: tensor<i32>) {
  // HALO:        %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_MOD_S:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[REM_S:.*]] = stablehlo.remainder %[[PID2]], %[[C_MOD_S]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV2:.*]] = stablehlo.convert %[[REM_S]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[S_SELF:.*]] = stablehlo.slice %arg1 [0:2, 1:2, 0:4]
  // HALO-NEXT:   %[[S_HOP:.*]] = stablehlo.slice %arg1 [0:2, 0:1, 0:4]
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[S_HOP]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>
  // HALO:        %[[SEL3:.*]] = stablehlo.select {{.*}}, %[[CP3]], {{.*}}
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[S_SELF]], %[[SEL3]], dim = 1
  // HALO-NEXT:   %[[C_OFFSET2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO2:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[CONCAT2]], %[[C_ZERO2]], %[[C_OFFSET2]], %[[C_ZERO2]], sizes = [2, 2, 4]
  // HALO-NEXT:   %[[REP_SLICE:.*]] = stablehlo.slice %[[SLICE2]] [0:2, 0:2, 1:3]
  // HALO-NEXT:   sdy.return %[[REP_SLICE]] : tensor<2x2x2xi32>
  // HALO-NEXT: }
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:3, 0:3, 0:2]

  %0 = stablehlo.slice %arg0 [0:3, 1:4, 1:3] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}, {}]>]>
  } : (tensor<4x4x4xi32>) -> tensor<3x3x2xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh, [{}, {}, {}]> : tensor<3x3x2xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}, {}]> : tensor<3x3x2xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<3x3x2xi32>
  return %1 : tensor<3x3x2xi32>
}

// CHECK-LABEL: func @slice_multiple_hops_shift
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @slice_multiple_hops_shift(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>})
  -> tensor<1x8xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}, {"b"}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [5:6, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {"b"}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a4, [{"a"}, {"b"}]> : tensor<1x8xi32>

  // HALO-NEXT: %[[CST_0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST_0]]) in_shardings=[<@mesh_a4, [{"a"}, {"b"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<2x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[S_SRC:.*]] = stablehlo.slice %arg1 [1:2, 0:4]
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV_MH:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV_MH:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV_MH]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV_MH]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[BCAST_PAD:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<1x4xi32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[S_SRC]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[4, 0], [5, 1]]> : tensor<2x2xi64>
  // HALO-NEXT:   %[[C_TARGET:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET:.*]] = stablehlo.compare EQ, %[[CONV1]], %[[C_TARGET]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET]], dims = [] : (tensor<i1>) -> tensor<1x4xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BCAST_PRED]], %[[CP1]], %[[BCAST_PAD]] : tensor<1x4xi1>, tensor<1x4xi32>
  // HALO-NEXT:   sdy.return %[[SEL1]] : tensor<1x4xi32>
  // HALO-NEXT: }
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:1, 0:8]

  %0 = stablehlo.slice %arg0 [5:6, 0:8] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>) -> tensor<1x8xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh_a4, [{}, {}]> : tensor<1x8xi32>
  %1 = sdy.reshard %0 <@mesh_a4, [{}, {}]> : tensor<1x8xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<1x8xi32>
  return %1 : tensor<1x8xi32>
}

// The data is on the last shard and only shard 0 holds output, so one
// collective-permute sends the sliced element from shard 7 to shard 0.
// CHECK-LABEL: func @slice_single_source_far_end
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xi32>
func.func @slice_single_source_far_end(
  %arg0: tensor<16xi32> {sdy.sharding = #sdy.sharding<@mesh_a8, [{"a"}]>})
  -> tensor<1xi32> {
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a8, [{}]> : tensor<16xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [15:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a8, [{}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a8, [{"a"}]> : tensor<1xi32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a8, [{"a"}]>, <@mesh_a8, []>] out_shardings=[<@mesh_a8, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<2xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[S_SRC:.*]] = stablehlo.slice %arg1 [1:2]
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[SHARD:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[BCAST_PAD:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<1xi32>
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[S_SRC]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[7, 0]]> : tensor<1x2xi64>
  // HALO-NEXT:   %[[C_TARGET:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET]], dims = [] : (tensor<i1>) -> tensor<1xi1>
  // HALO-NEXT:   %[[SEL:.*]] = stablehlo.select %[[BCAST_PRED]], %[[CP]], %[[BCAST_PAD]] : tensor<1xi1>, tensor<1xi32>
  // HALO-NEXT:   sdy.return %[[SEL]] : tensor<1xi32>
  // HALO-NEXT: } : (tensor<16xi32>, tensor<i32>) -> tensor<8xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:1]
  %0 = stablehlo.slice %arg0 [15:16] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a8, [{"a"}]>]>
  } : (tensor<16xi32>) -> tensor<1xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh_a8, [{}]> : tensor<1xi32>
  %1 = sdy.reshard %0 <@mesh_a8, [{}]> : tensor<1xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<1xi32>
  return %1 : tensor<1xi32>
}

// Same slice, but the operand is a reshard. Without the single-source lowering
// this slice replicates its operand (7 hops on 8 shards), and that is kept.
// CHECK-LABEL: func @slice_single_source_far_end_of_reshard
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xi32>
func.func @slice_single_source_far_end_of_reshard(
  %arg0: tensor<16xi32> {sdy.sharding = #sdy.sharding<@mesh_a8, [{}]>})
  -> tensor<1xi32> {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh_a8, [{"a"}]> : tensor<16xi32>
  %0 = sdy.reshard %arg0 <@mesh_a8, [{"a"}]> : tensor<16xi32>

  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[RESHARD0]] <@mesh_a8, [{}]> : tensor<16xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [15:16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a8, [{}]>]>} : (tensor<16xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a8, [{"a"}]> : tensor<1xi32>
  %1 = stablehlo.slice %0 [15:16] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a8, [{"a"}]>]>
  } : (tensor<16xi32>) -> tensor<1xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh_a8, [{}]> : tensor<1xi32>
  %2 = sdy.reshard %1 <@mesh_a8, [{}]> : tensor<1xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<1xi32>
  return %2 : tensor<1xi32>
}

// The operand is a reshard, but the data is on shard 6, only 6 hops from shard
// 0. Such a slice exchanges whole shards hop by hop rather than replicating
// its operand, so it takes the single-source lowering.
// CHECK-LABEL: func @slice_single_source_multi_hop_of_reshard
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xi32>
func.func @slice_single_source_multi_hop_of_reshard(
  %arg0: tensor<16xi32> {sdy.sharding = #sdy.sharding<@mesh_a8, [{}]>})
  -> tensor<1xi32> {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh_a8, [{"a"}]> : tensor<16xi32>
  %0 = sdy.reshard %arg0 <@mesh_a8, [{"a"}]> : tensor<16xi32>

  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[RESHARD0]] <@mesh_a8, [{}]> : tensor<16xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [13:14] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a8, [{}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a8, [{"a"}]> : tensor<1xi32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[RESHARD0]], %[[CST]]) in_shardings=[<@mesh_a8, [{"a"}]>, <@mesh_a8, []>] out_shardings=[<@mesh_a8, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<2xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[S_SRC:.*]] = stablehlo.slice %arg1 [1:2]
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[SHARD:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[BCAST_PAD:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<1xi32>
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[S_SRC]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[6, 0]]> : tensor<1x2xi64>
  // HALO-NEXT:   %[[C_TARGET:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET]], dims = [] : (tensor<i1>) -> tensor<1xi1>
  // HALO-NEXT:   %[[SEL:.*]] = stablehlo.select %[[BCAST_PRED]], %[[CP]], %[[BCAST_PAD]] : tensor<1xi1>, tensor<1xi32>
  // HALO-NEXT:   sdy.return %[[SEL]] : tensor<1xi32>
  // HALO-NEXT: } : (tensor<16xi32>, tensor<i32>) -> tensor<8xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:1]
  %1 = stablehlo.slice %0 [13:14] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a8, [{"a"}]>]>
  } : (tensor<16xi32>) -> tensor<1xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh_a8, [{}]> : tensor<1xi32>
  %2 = sdy.reshard %1 <@mesh_a8, [{}]> : tensor<1xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<1xi32>
  return %2 : tensor<1xi32>
}

// The 5 sliced elements are on shard 2. They are padded to 6 and split into
// pieces of 2 for shards 0, 1 and 2: shard 2 keeps its piece and sends the
// other two.
// CHECK-LABEL: func @slice_single_source_among_targets
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xi32>
func.func @slice_single_source_among_targets(
  %arg0: tensor<24xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}]>})
  -> tensor<5xi32> {
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}]> : tensor<24xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [12:17] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a4, [{"a"}]> : tensor<5xi32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<6xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[S_SRC:.*]] = stablehlo.slice %arg1 [0:5]
  // HALO-NEXT:   %[[PAD:.*]] = stablehlo.pad %[[S_SRC]], %arg2, low = [0], high = [1], interior = [0]
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[SHARD:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[BCAST_PAD:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<2xi32>
  // HALO-NEXT:   %[[PIECE0:.*]] = stablehlo.slice %[[PAD]] [0:2]
  // HALO-NEXT:   %[[CP0:.*]] = "stablehlo.collective_permute"(%[[PIECE0]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[4, 0], [5, 1]]> : tensor<2x2xi64>
  // HALO-NEXT:   %[[C_TARGET0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET0:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET0]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED0:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET0]], dims = [] : (tensor<i1>) -> tensor<2xi1>
  // HALO-NEXT:   %[[SEL0:.*]] = stablehlo.select %[[BCAST_PRED0]], %[[CP0]], %[[BCAST_PAD]] : tensor<2xi1>, tensor<2xi32>
  // HALO-NEXT:   %[[PIECE1:.*]] = stablehlo.slice %[[PAD]] [2:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[PIECE1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[4, 2], [5, 3]]> : tensor<2x2xi64>
  // HALO-NEXT:   %[[C_TARGET1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET1:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED1:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET1]], dims = [] : (tensor<i1>) -> tensor<2xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BCAST_PRED1]], %[[CP1]], %[[SEL0]] : tensor<2xi1>, tensor<2xi32>
  // HALO-NEXT:   %[[PIECE2:.*]] = stablehlo.slice %[[PAD]] [4:6]
  // HALO-NEXT:   %[[C_TARGET2:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET2:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED2:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET2]], dims = [] : (tensor<i1>) -> tensor<2xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BCAST_PRED2]], %[[PIECE2]], %[[SEL1]] : tensor<2xi1>, tensor<2xi32>
  // HALO-NEXT:   sdy.return %[[SEL2]] : tensor<2xi32>
  // HALO-NEXT: } : (tensor<24xi32>, tensor<i32>) -> tensor<8xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:5]
  %0 = stablehlo.slice %arg0 [12:17] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}]>]>
  } : (tensor<24xi32>) -> tensor<5xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh_a4, [{}]> : tensor<5xi32>
  %1 = sdy.reshard %0 <@mesh_a4, [{}]> : tensor<5xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<5xi32>
  return %1 : tensor<5xi32>
}

// The sliced element is already on shard 0, the only shard that holds output,
// so nothing is sent. The hop analysis still counts 2 hops because it also
// walks the output shards that hold only padding.
// CHECK-LABEL: func @slice_single_source_already_on_target
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16xi32>
func.func @slice_single_source_already_on_target(
  %arg0: tensor<16xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}]>})
  -> tensor<1xi32> {
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}]> : tensor<16xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [1:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a4, [{"a"}]> : tensor<1xi32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[S_SRC:.*]] = stablehlo.slice %arg1 [1:2]
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[SHARD:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[BCAST_PAD:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<1xi32>
  // HALO-NEXT:   %[[C_TARGET:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET]], dims = [] : (tensor<i1>) -> tensor<1xi1>
  // HALO-NEXT:   %[[SEL:.*]] = stablehlo.select %[[BCAST_PRED]], %[[S_SRC]], %[[BCAST_PAD]] : tensor<1xi1>, tensor<1xi32>
  // HALO-NEXT:   sdy.return %[[SEL]] : tensor<1xi32>
  // HALO-NEXT: } : (tensor<16xi32>, tensor<i32>) -> tensor<4xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:1]
  %0 = stablehlo.slice %arg0 [1:2] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}]>]>
  } : (tensor<16xi32>) -> tensor<1xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh_a4, [{}]> : tensor<1xi32>
  %1 = sdy.reshard %0 <@mesh_a4, [{}]> : tensor<1xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<1xi32>
  return %1 : tensor<1xi32>
}

// Indivisible input (15 on 8 shards) is padded to 16 first, then the single
// element on partially padded shard 7 is sliced and sent to shard 0.
// CHECK-LABEL: func @slice_single_source_indivisible_input
// CHECK-SAME: (%[[ARG0:.*]]: tensor<15xi32>
func.func @slice_single_source_indivisible_input(
  %arg0: tensor<15xi32> {sdy.sharding = #sdy.sharding<@mesh_a8, [{"a"}]>})
  -> tensor<1xi32> {
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a8, [{}]> : tensor<15xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [14:15] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a8, [{}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a8, [{"a"}]> : tensor<1xi32>

  // HALO-NEXT: %[[CST0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[PAD_IN:.*]] = stablehlo.pad %[[ARG0]], %[[CST0]], low = [0], high = [1], interior = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a8, [{"a"}]>]>} : (tensor<15xi32>, tensor<i32>) -> tensor<16xi32>
  // HALO-NEXT: %[[CST1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[PAD_IN]], %[[CST1]]) in_shardings=[<@mesh_a8, [{"a"}]>, <@mesh_a8, []>] out_shardings=[<@mesh_a8, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<2xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[S_SRC:.*]] = stablehlo.slice %arg1 [0:1]
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[SHARD:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[BCAST_PAD:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<1xi32>
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[S_SRC]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[7, 0]]> : tensor<1x2xi64>
  // HALO-NEXT:   %[[C_TARGET:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET]], dims = [] : (tensor<i1>) -> tensor<1xi1>
  // HALO-NEXT:   %[[SEL:.*]] = stablehlo.select %[[BCAST_PRED]], %[[CP]], %[[BCAST_PAD]] : tensor<1xi1>, tensor<1xi32>
  // HALO-NEXT:   sdy.return %[[SEL]] : tensor<1xi32>
  // HALO-NEXT: } : (tensor<16xi32>, tensor<i32>) -> tensor<8xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:1]
  %0 = stablehlo.slice %arg0 [14:15] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a8, [{"a"}]>]>
  } : (tensor<15xi32>) -> tensor<1xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh_a8, [{}]> : tensor<1xi32>
  %1 = sdy.reshard %0 <@mesh_a8, [{}]> : tensor<1xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<1xi32>
  return %1 : tensor<1xi32>
}

// All 8 elements on shard 3 split evenly across all 4 output shards (pieces of
// 2), so neither local padding nor post-manual_computation trimming is needed.
// CHECK-LABEL: func @slice_single_source_all_targets_no_pad
// CHECK-SAME: (%[[ARG0:.*]]: tensor<32xi32>
func.func @slice_single_source_all_targets_no_pad(
  %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}]>})
  -> tensor<8xi32> {
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}]> : tensor<32xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [24:32] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a4, [{"a"}]> : tensor<8xi32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<8xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[SHARD:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[BCAST_PAD:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<2xi32>
  // HALO-NEXT:   %[[PIECE0:.*]] = stablehlo.slice %arg1 [0:2]
  // HALO-NEXT:   %[[CP0:.*]] = "stablehlo.collective_permute"(%[[PIECE0]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[6, 0], [7, 1]]> : tensor<2x2xi64>
  // HALO-NEXT:   %[[C_TARGET0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET0:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET0]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED0:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET0]], dims = [] : (tensor<i1>) -> tensor<2xi1>
  // HALO-NEXT:   %[[SEL0:.*]] = stablehlo.select %[[BCAST_PRED0]], %[[CP0]], %[[BCAST_PAD]] : tensor<2xi1>, tensor<2xi32>
  // HALO-NEXT:   %[[PIECE1:.*]] = stablehlo.slice %arg1 [2:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[PIECE1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[6, 2], [7, 3]]> : tensor<2x2xi64>
  // HALO-NEXT:   %[[C_TARGET1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET1:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED1:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET1]], dims = [] : (tensor<i1>) -> tensor<2xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BCAST_PRED1]], %[[CP1]], %[[SEL0]] : tensor<2xi1>, tensor<2xi32>
  // HALO-NEXT:   %[[PIECE2:.*]] = stablehlo.slice %arg1 [4:6]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[PIECE2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[6, 4], [7, 5]]> : tensor<2x2xi64>
  // HALO-NEXT:   %[[C_TARGET2:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET2:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED2:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET2]], dims = [] : (tensor<i1>) -> tensor<2xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BCAST_PRED2]], %[[CP2]], %[[SEL1]] : tensor<2xi1>, tensor<2xi32>
  // HALO-NEXT:   %[[PIECE3:.*]] = stablehlo.slice %arg1 [6:8]
  // HALO-NEXT:   %[[C_TARGET3:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET3:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET3]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED3:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET3]], dims = [] : (tensor<i1>) -> tensor<2xi1>
  // HALO-NEXT:   %[[SEL3:.*]] = stablehlo.select %[[BCAST_PRED3]], %[[PIECE3]], %[[SEL2]] : tensor<2xi1>, tensor<2xi32>
  // HALO-NEXT:   sdy.return %[[SEL3]] : tensor<2xi32>
  // HALO-NEXT: } : (tensor<32xi32>, tensor<i32>) -> tensor<8xi32>
  %0 = stablehlo.slice %arg0 [24:32] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}]>]>
  } : (tensor<32xi32>) -> tensor<8xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh_a4, [{}]> : tensor<8xi32>
  %1 = sdy.reshard %0 <@mesh_a4, [{}]> : tensor<8xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<8xi32>
  return %1 : tensor<8xi32>
}

// The 3 sliced elements are on shard 7 while output lands on shards 0, 1, and 2
// (sourceShard >= numTargets), so every target requires a collective-permute.
// CHECK-LABEL: func @slice_single_source_all_targets_remote
// CHECK-SAME: (%[[ARG0:.*]]: tensor<32xi32>
func.func @slice_single_source_all_targets_remote(
  %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh_a8, [{"a"}]>})
  -> tensor<3xi32> {
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a8, [{}]> : tensor<32xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [28:31] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a8, [{}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a8, [{"a"}]> : tensor<3xi32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a8, [{"a"}]>, <@mesh_a8, []>] out_shardings=[<@mesh_a8, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[S_SRC:.*]] = stablehlo.slice %arg1 [0:3]
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[SHARD:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[BCAST_PAD:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<1xi32>
  // HALO-NEXT:   %[[PIECE0:.*]] = stablehlo.slice %[[S_SRC]] [0:1]
  // HALO-NEXT:   %[[CP0:.*]] = "stablehlo.collective_permute"(%[[PIECE0]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[7, 0]]> : tensor<1x2xi64>
  // HALO-NEXT:   %[[C_TARGET0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET0:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET0]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED0:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET0]], dims = [] : (tensor<i1>) -> tensor<1xi1>
  // HALO-NEXT:   %[[SEL0:.*]] = stablehlo.select %[[BCAST_PRED0]], %[[CP0]], %[[BCAST_PAD]] : tensor<1xi1>, tensor<1xi32>
  // HALO-NEXT:   %[[PIECE1:.*]] = stablehlo.slice %[[S_SRC]] [1:2]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[PIECE1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[7, 1]]> : tensor<1x2xi64>
  // HALO-NEXT:   %[[C_TARGET1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET1:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED1:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET1]], dims = [] : (tensor<i1>) -> tensor<1xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BCAST_PRED1]], %[[CP1]], %[[SEL0]] : tensor<1xi1>, tensor<1xi32>
  // HALO-NEXT:   %[[PIECE2:.*]] = stablehlo.slice %[[S_SRC]] [2:3]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[PIECE2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[7, 2]]> : tensor<1x2xi64>
  // HALO-NEXT:   %[[C_TARGET2:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[IS_TARGET2:.*]] = stablehlo.compare EQ, %[[SHARD]], %[[C_TARGET2]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PRED2:.*]] = stablehlo.broadcast_in_dim %[[IS_TARGET2]], dims = [] : (tensor<i1>) -> tensor<1xi1>
  // HALO-NEXT:   %[[SEL2:.*]] = stablehlo.select %[[BCAST_PRED2]], %[[CP2]], %[[SEL1]] : tensor<1xi1>, tensor<1xi32>
  // HALO-NEXT:   sdy.return %[[SEL2]] : tensor<1xi32>
  // HALO-NEXT: } : (tensor<32xi32>, tensor<i32>) -> tensor<8xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:3]
  %0 = stablehlo.slice %arg0 [28:31] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a8, [{"a"}]>]>
  } : (tensor<32xi32>) -> tensor<3xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh_a8, [{}]> : tensor<3xi32>
  %1 = sdy.reshard %0 <@mesh_a8, [{}]> : tensor<3xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<3xi32>
  return %1 : tensor<3xi32>
}

// 2D slice where both sharded dims take the single-source multi-hop lowering
// back-to-back inside manual_computation.
// CHECK-LABEL: func @slice_single_source_2d_both_dims
// CHECK-SAME: (%[[ARG0:.*]]: tensor<9x20xi32>
func.func @slice_single_source_2d_both_dims(
  %arg0: tensor<9x20xi32> {sdy.sharding = #sdy.sharding<@mesh_a3_b4, [{"a"}, {"b"}]>})
  -> tensor<2x1xi32> {
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a3_b4, [{}, {}]> : tensor<9x20xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [6:8, 16:17] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a3_b4, [{}, {}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a3_b4, [{"a"}, {"b"}]> : tensor<2x1xi32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a3_b4, [{"a"}, {"b"}]>, <@mesh_a3_b4, []>] out_shardings=[<@mesh_a3_b4, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<3x5xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[S_DIM0:.*]] = stablehlo.slice %arg1 [0:2, 0:5]
  // HALO-NEXT:   %[[PID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV0:.*]] = stablehlo.constant dense<4> : tensor<ui32>
  // HALO-NEXT:   %[[DIV0:.*]] = stablehlo.divide %[[PID0]], %[[C_DIV0]] : tensor<ui32>
  // HALO-NEXT:   %[[SHARD0:.*]] = stablehlo.convert %[[DIV0]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[BCAST_PAD0:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<1x5xi32>
  // HALO-NEXT:   %[[P0_0:.*]] = stablehlo.slice %[[S_DIM0]] [0:1, 0:5]
  // HALO-NEXT:   %[[CP0_0:.*]] = "stablehlo.collective_permute"(%[[P0_0]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[8, 0], [9, 1], [10, 2], [11, 3]]> : tensor<4x2xi64>
  // HALO-NEXT:   %[[C_T0_0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[IS_T0_0:.*]] = stablehlo.compare EQ, %[[SHARD0]], %[[C_T0_0]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BP0_0:.*]] = stablehlo.broadcast_in_dim %[[IS_T0_0]], dims = [] : (tensor<i1>) -> tensor<1x5xi1>
  // HALO-NEXT:   %[[SEL0_0:.*]] = stablehlo.select %[[BP0_0]], %[[CP0_0]], %[[BCAST_PAD0]] : tensor<1x5xi1>, tensor<1x5xi32>
  // HALO-NEXT:   %[[P0_1:.*]] = stablehlo.slice %[[S_DIM0]] [1:2, 0:5]
  // HALO-NEXT:   %[[CP0_1:.*]] = "stablehlo.collective_permute"(%[[P0_1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[8, 4], [9, 5], [10, 6], [11, 7]]> : tensor<4x2xi64>
  // HALO-NEXT:   %[[C_T0_1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[IS_T0_1:.*]] = stablehlo.compare EQ, %[[SHARD0]], %[[C_T0_1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BP0_1:.*]] = stablehlo.broadcast_in_dim %[[IS_T0_1]], dims = [] : (tensor<i1>) -> tensor<1x5xi1>
  // HALO-NEXT:   %[[SEL0_1:.*]] = stablehlo.select %[[BP0_1]], %[[CP0_1]], %[[SEL0_0]] : tensor<1x5xi1>, tensor<1x5xi32>
  // HALO-NEXT:   %[[S_DIM1:.*]] = stablehlo.slice %[[SEL0_1]] [0:1, 1:2]
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_MOD1:.*]] = stablehlo.constant dense<4> : tensor<ui32>
  // HALO-NEXT:   %[[REM1:.*]] = stablehlo.remainder %[[PID1]], %[[C_MOD1]] : tensor<ui32>
  // HALO-NEXT:   %[[SHARD1:.*]] = stablehlo.convert %[[REM1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[BCAST_PAD1:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<1x1xi32>
  // HALO-NEXT:   %[[CP1_0:.*]] = "stablehlo.collective_permute"(%[[S_DIM1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[3, 0], [7, 4], [11, 8]]> : tensor<3x2xi64>
  // HALO-NEXT:   %[[C_T1_0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[IS_T1_0:.*]] = stablehlo.compare EQ, %[[SHARD1]], %[[C_T1_0]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BP1_0:.*]] = stablehlo.broadcast_in_dim %[[IS_T1_0]], dims = [] : (tensor<i1>) -> tensor<1x1xi1>
  // HALO-NEXT:   %[[SEL1_0:.*]] = stablehlo.select %[[BP1_0]], %[[CP1_0]], %[[BCAST_PAD1]] : tensor<1x1xi1>, tensor<1x1xi32>
  // HALO-NEXT:   sdy.return %[[SEL1_0]] : tensor<1x1xi32>
  // HALO-NEXT: } : (tensor<9x20xi32>, tensor<i32>) -> tensor<3x4xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:2, 0:1]
  %0 = stablehlo.slice %arg0 [6:8, 16:17] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a3_b4, [{"a"}, {"b"}]>]>
  } : (tensor<9x20xi32>) -> tensor<2x1xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh_a3_b4, [{}, {}]> : tensor<2x1xi32>
  %1 = sdy.reshard %0 <@mesh_a3_b4, [{}, {}]> : tensor<2x1xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<2x1xi32>
  return %1 : tensor<2x1xi32>
}

// The slice reads shards 2 and 3 and shard 0 is 2 hops from its data, so whole
// shards are exchanged one hop at a time.
// CHECK-LABEL: func @slice_two_sources_multiple_hops
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xi32>
func.func @slice_two_sources_multiple_hops(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}]>})
  -> tensor<3xi32> {
  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}]> : tensor<8xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [5:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a4, [{"a"}]> : tensor<3xi32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a"}]>] manual_axes={"a"} (%arg1: tensor<2xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[SHARD:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [4, 2], [5, 3], [6, 4], [7, 5]]> : tensor<6x2xi64>
  // HALO-NEXT:   %[[C_LAST:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[HAS_HOP1:.*]] = stablehlo.compare LT, %[[SHARD]], %[[C_LAST]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // HALO-NEXT:   %[[BCAST_PAD:.*]] = stablehlo.broadcast_in_dim %arg2, dims = [] : (tensor<i32>) -> tensor<2xi32>
  // HALO-NEXT:   %[[BCAST_PRED:.*]] = stablehlo.broadcast_in_dim %[[HAS_HOP1]], dims = [] : (tensor<i1>) -> tensor<2xi1>
  // HALO-NEXT:   %[[SEL1:.*]] = stablehlo.select %[[BCAST_PRED]], %[[CP1]], %[[BCAST_PAD]] : tensor<2xi1>, tensor<2xi32>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[4, 0], [5, 1], [6, 2], [7, 3]]> : tensor<4x2xi64>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %arg1, %[[SEL1]], %[[CP2]], dim = 0
  // HALO-NEXT:   %[[C_STRIDE:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // HALO-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[SHARD]], %[[C_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[DYN_SLICE:.*]] = stablehlo.dynamic_slice %[[CONCAT]], %[[ADD]], sizes = [1]
  // HALO-NEXT:   sdy.return %[[DYN_SLICE]] : tensor<1xi32>
  // HALO-NEXT: } : (tensor<8xi32>, tensor<i32>) -> tensor<4xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:3]
  %0 = stablehlo.slice %arg0 [5:8] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}]>]>
  } : (tensor<8xi32>) -> tensor<3xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh_a4, [{}]> : tensor<3xi32>
  %1 = sdy.reshard %0 <@mesh_a4, [{}]> : tensor<3xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<3xi32>
  return %1 : tensor<3xi32>
}

// The 2 sliced elements are on the last shard, 7 hops from shard 0, but the
// slice is strided, so its operand is still replicated.
// CHECK-LABEL: func @strided_slice_single_source_far_end
// CHECK-SAME: (%[[ARG0:.*]]: tensor<32xi32>
func.func @strided_slice_single_source_far_end(
  %arg0: tensor<32xi32> {sdy.sharding = #sdy.sharding<@mesh_a8, [{"a"}]>})
  -> tensor<2xi32> {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh_a8, [{}]> : tensor<32xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD1]] [28:32:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a8, [{}]>]>} : (tensor<32xi32>) -> tensor<2xi32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh_a8, [{"a"}]> : tensor<2xi32>
  %0 = stablehlo.slice %arg0 [28:32:2] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a8, [{"a"}]>]>
  } : (tensor<32xi32>) -> tensor<2xi32>

  // CHECK-NEXT: %[[RS:.*]] = sdy.reshard %[[RES]] <@mesh_a8, [{}]> : tensor<2xi32>
  %1 = sdy.reshard %0 <@mesh_a8, [{}]> : tensor<2xi32>

  // CHECK-NEXT: return %[[RS]] : tensor<2xi32>
  return %1 : tensor<2xi32>
}

// CHECK-LABEL: func @slice_non_communication_free_zero_hops
func.func @slice_non_communication_free_zero_hops(
    %arg0: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
    -> (tensor<6x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  // REPL-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<8x4xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD]] [1:7, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh, [{"a"}, {}]> : tensor<6x4xi32>

  // HALO-NEXT: %[[CST_0:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%arg0, %[[CST_0]]) in_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<4x4xi32>, %arg2: tensor<i32>) {
  // HALO-NOT:    collective_permute
  // HALO:        %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO:        %[[SLICE:.*]] = stablehlo.dynamic_slice %arg1, {{.*}}, sizes = [3, 4]
  // HALO-NEXT:   sdy.return %[[SLICE]] : tensor<3x4xi32>
  // HALO-NEXT: }
  %0 = stablehlo.slice %arg0 [1:7, 0:4] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {}]>]>
  } : (tensor<8x4xi32>) -> tensor<6x4xi32>
  return %0 : tensor<6x4xi32>
}

// CHECK-LABEL: func @slice_comm_free_to_one_and_comm
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4xi32>
func.func @slice_comm_free_to_one_and_comm(
  %arg0: tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<1x3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // REPL-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{"a"}, {}]> : tensor<4x4xi32>
  // REPL-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD]] [0:1, 1:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[SLICE]] <@mesh, [{"a"}, {"b"}]> : tensor<1x3xi32>

  // HALO-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%[[ARG1:.*]]: tensor<2x2xi32>, %[[ARG2:.*]]: tensor<i32>) {
  // HALO:        %[[SLICE0:.*]] = stablehlo.dynamic_slice %[[ARG1]], {{.*}}, sizes = [1, 2]
  // HALO:        %[[CP:.*]] = "stablehlo.collective_permute"
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>
  // HALO:        %[[CONCAT:.*]] = stablehlo.concatenate {{.*}}, dim = 1
  // HALO:        %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT]], {{.*}}, sizes = [1, 2]
  // HALO-NEXT:   sdy.return %[[SLICE1]] : tensor<1x2xi32>
  // HALO-NEXT: } : (tensor<4x4xi32>, tensor<i32>) -> tensor<2x4xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:1, 0:3]
  %0 = stablehlo.slice %arg0 [0:1, 1:4] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<4x4xi32>) -> tensor<1x3xi32>

  // CHECK-NEXT: return %[[RES]] : tensor<1x3xi32>
  return %0 : tensor<1x3xi32>
}
