// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s --check-prefixes=CHECK,HALO

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>
// CHECK: @mesh_a4 = <["a"=4, "b"=2]>
sdy.mesh @mesh_a4 = <["a"=4, "b"=2]>

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
  // HALO:      %[[CP:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]> : tensor<6x2xi64>
  // HALO-NEXT: %[[SELF_SLICE:.*]] = stablehlo.slice %arg1 [0:2, 0:4]
  // HALO-NEXT: %[[CONCAT:.*]] = stablehlo.concatenate %[[CP]], %[[SELF_SLICE]], dim = 0
  // HALO: %[[DYN_SLICE:.*]] = stablehlo.dynamic_slice %[[CONCAT]], {{.*}}, sizes = [2, 4]
  // HALO-NEXT: sdy.return %[[DYN_SLICE]] : tensor<2x4xi32>
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
  // HALO:        %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV_MH:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV_MH:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV_MH]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV_MH]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [4, 2], [5, 3], [6, 4], [7, 5]]> : tensor<6x2xi64>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO:        %[[CP2:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[4, 0], [5, 1], [6, 2], [7, 3]]> : tensor<4x2xi64>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %arg1, %[[SEL1]], %[[CP2]], dim = 0
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<5> : tensor<i32>
  // HALO-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[CONV1]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[C_OFFSET1]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[CONCAT1]], %[[ADD1]], %[[C_ZERO1]], sizes = [1, 4]
  // HALO-NEXT:   sdy.return %[[SLICE1]] : tensor<1x4xi32>
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
  // HALO:        %[[CONCAT:.*]] = stablehlo.concatenate %arg1, dim = 0
  // HALO:        %[[SLICE:.*]] = stablehlo.dynamic_slice %[[CONCAT]], {{.*}}, sizes = [3, 4]
  // HALO-NEXT:   sdy.return %[[SLICE]] : tensor<3x4xi32>
  // HALO-NEXT: }
  %0 = stablehlo.slice %arg0 [1:7, 0:4] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {}]>]>
  } : (tensor<8x4xi32>) -> tensor<6x4xi32>
  return %0 : tensor<6x4xi32>
}
