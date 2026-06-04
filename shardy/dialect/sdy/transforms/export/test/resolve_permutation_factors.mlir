// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true" | FileCheck %s --check-prefixes=CHECK,HALO

// HALO-DAG: sdy.mesh @mesh_abc_reversed_1 = <["a"=2, "b"=2, "c"=4], device_ids=[9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6]>
// HALO-DAG: sdy.mesh @mesh_abc_reversed_0 = <["a"=2, "b"=2, "c"=4], device_ids=[15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]>
// HALO-DAG: sdy.mesh @mesh_abc_reversed = <["a"=2, "b"=2, "c"=4], device_ids=[7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8]>
// HALO-DAG: sdy.mesh @mesh_reversed = <["a"=2, "b"=2], device_ids=[2, 3, 0, 1]>

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>
// CHECK: sdy.mesh @mesh_abc = <["a"=2, "b"=2, "c"=4]>
sdy.mesh @mesh_abc = <["a"=2, "b"=2, "c"=4]>

// CHECK-LABEL: func @reduce_window_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
// CHECK-SAME: -> (tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
func.func @reduce_window_permutation(%arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
    // CHECK: %[[CST:.*]] = stablehlo.constant
  %cst = stablehlo.constant dense<0> : tensor<i32>
  // REPL: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<8x8xi32>
  // REPL: %[[RW:.*]] = "stablehlo.reduce_window"(%[[RESHARD_IN]], %[[CST]])
  // REPL: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  // REPL: %[[RES:.*]] = sdy.reshard %[[RW]] <@mesh, [{"a"}, {"b"}]> : tensor<6x8xi32>
  %0 = "stablehlo.reduce_window"(%arg0, %cst) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
      stablehlo.return %1 : tensor<i32>
  }) {
    window_dimensions = array<i64: 3, 1>,
    window_strides = array<i64: 1, 1>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x8xi32>, tensor<i32>) -> tensor<6x8xi32>
  // REPL: return %[[RES]] : tensor<6x8xi32>
  return %0 : tensor<6x8xi32>
}

// CHECK-LABEL: func @reverse_divisible
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
// CHECK-SAME: -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
func.func @reverse_divisible(
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {

  // REPL: %[[REPL_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<4x8xi32>
  // REPL-NEXT: %[[REPL_REV:.*]] = stablehlo.reverse %[[REPL_IN]], dims = [0]
  // REPL-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[REPL_REV]] <@mesh, [{"a"}, {"b"}]> : tensor<4x8xi32>

  // HALO: %[[HALO_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_reversed, [{"a"}, {"b"}]> : tensor<4x8xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.reverse %[[HALO_IN]], dims = [0]
  // HALO-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
  %0 = stablehlo.reverse %arg0, dims = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
    : tensor<4x8xi32>

  // CHECK-NEXT: return %[[RES]] : tensor<4x8xi32>
  return %0 : tensor<4x8xi32>
}

// CHECK-LABEL: func @reverse_indivisible
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
// CHECK-SAME:  -> (tensor<5x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}) {
func.func @reverse_indivisible(
  %arg0: tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<5x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}) {

  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:5, 0:8]
  %0 = stablehlo.slice %arg0 [0:5, 0:8]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}, {"b"}]>]>}
    : (tensor<6x8xi32>) -> tensor<5x8xi32>

  // REPL: %[[REPL_IN:.*]] = sdy.reshard %[[SLICE]] <@mesh, [{}, {"b"}]>
  // REPL: %[[REPL_REV:.*]] = stablehlo.reverse %[[REPL_IN]], dims = [0]
  // REPL-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  // REPL: %[[RES:.*]] = sdy.reshard %[[REPL_REV]] <@mesh, [{"a"}, {"b"}]>

  // HALO: %[[CST:.*]] = stablehlo.constant dense<0>
  // HALO: %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]]
  // HALO-SAME: low = [0, 0], high = [1, 0]

  // HALO-NEXT: %[[HALO_SHIFT:.*]] = sdy.manual_computation(%[[PAD]])
  // HALO-SAME: in_shardings=[<@mesh, [{"a"}, {"b"}]>]
  // HALO-SAME: out_shardings=[<@mesh, [{"a"}, {"b"}]>]
  // HALO-SAME: manual_axes={"a"}
  // HALO-SAME: (%[[LOCAL_IN:.*]]: tensor<3x8xi32>) {
  // HALO-NEXT:   %[[HALO_SLICE:.*]] = stablehlo.slice %[[LOCAL_IN]] [2:3, 0:8]
  // HALO-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  // HALO-NEXT:   %[[RECV:.*]] = "stablehlo.collective_permute"(%[[HALO_SLICE]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1]]> : tensor<1x2xi64>
  // HALO-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  // HALO-NEXT:   %[[OWN_SLICE:.*]] = stablehlo.slice %[[LOCAL_IN]] [0:2, 0:8]
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[RECV]], %[[OWN_SLICE]], dim = 0
  // HALO-NEXT:   sdy.return %[[CONCAT]]
  // HALO-NEXT: } : (tensor<6x8xi32>) -> tensor<6x8xi32>

  // HALO: %[[RESHARD:.*]] = sdy.reshard %[[HALO_SHIFT]] <@mesh_reversed, [{"a"}, {"b"}]>
  // HALO: %[[REV:.*]] = stablehlo.reverse %[[RESHARD]], dims = [0]
  // HALO: %[[RES:.*]] = stablehlo.slice %[[REV]] [0:5, 0:8]
  %1 = stablehlo.reverse %0, dims = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
    : tensor<5x8xi32>

  // CHECK: %[[AG:.*]] = sdy.all_gather [{"a"}, {}] %[[RES]]
  %2 = sdy.all_gather [{"a"}, {}] %1 out_sharding=<@mesh, [{}, {"b"}]> : tensor<5x8xi32>
  // CHECK: return %[[AG]] : tensor<5x8xi32>
  return %2 : tensor<5x8xi32>
}

// CHECK-LABEL: func @reverse_indivisible_padding_exceed_single_shard
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4x6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>})
// CHECK-SAME:  -> (tensor<4x6x5xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {}]>}) {
func.func @reverse_indivisible_padding_exceed_single_shard(
  %arg0: tensor<4x6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>})
  -> (tensor<4x6x5xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {}]>}) {

  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:4, 0:6, 0:5]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
  %0 = stablehlo.slice %arg0 [0:4, 0:6, 0:5]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
    : (tensor<4x6x8xi32>) -> tensor<4x6x5xi32>

  // CHECK: %[[REPL_IN:.*]] = sdy.reshard %[[SLICE]] <@mesh_abc, [{}, {"a"}, {}]>
  // CHECK-NEXT: %[[REPL_REV:.*]] = stablehlo.reverse %[[REPL_IN]], dims = [0, 2]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{}, {"a"}, {}]>]>}
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[REPL_REV]] <@mesh_abc, [{"b"}, {"a"}, {"c"}]>
  %1 = stablehlo.reverse %0, dims = [0, 2]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
    : tensor<4x6x5xi32>

  // CHECK: %[[AG:.*]] = sdy.all_gather [{}, {}, {"c"}] %[[RES]]
  // CHECK-SAME: out_sharding=<@mesh_abc, [{"b"}, {"a"}, {}]>
  %2 = sdy.all_gather [{}, {}, {"c"}] %1 out_sharding=<@mesh_abc, [{"b"}, {"a"}, {}]> : tensor<4x6x5xi32>

  // CHECK: return %[[AG]]
  return %2 : tensor<4x6x5xi32>
}

// CHECK-LABEL: func @reverse_divisible_indivisible_irrelevant_dim_indivisible
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4x6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>})
// CHECK-SAME:  -> (tensor<4x5x7xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {}, {}]>}) {
func.func @reverse_divisible_indivisible_irrelevant_dim_indivisible(
  %arg0: tensor<4x6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>})
  -> (tensor<4x5x7xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {}, {}]>}) {

  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:4, 0:5, 0:7]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
  %0 = stablehlo.slice %arg0 [0:4, 0:5, 0:7]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
    : (tensor<4x6x8xi32>) -> tensor<4x5x7xi32>

  // REPL: %[[REPL_IN:.*]] = sdy.reshard %[[SLICE]] <@mesh_abc, [{}, {"a"}, {}]>
  // REPL-NEXT: %[[REPL_REV:.*]] = stablehlo.reverse %[[REPL_IN]], dims = [0, 2]
  // REPL-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{}, {"a"}, {}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[REPL_REV]] <@mesh_abc, [{"b"}, {"a"}, {"c"}]>

  // HALO: %[[CST:.*]] = stablehlo.constant dense<0>
  // HALO: %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]]
  // HALO-SAME: low = [0, 0, 0], high = [0, 0, 1]

  // HALO-NEXT: %[[HALO_SHIFT:.*]] = sdy.manual_computation(%[[PAD]])
  // HALO-SAME: in_shardings=[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]
  // HALO-SAME: out_shardings=[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]
  // HALO-SAME: manual_axes={"c"}
  // HALO-SAME: (%[[LOCAL_IN:.*]]: tensor<4x5x2xi32>) {
  // HALO-NEXT:   %[[HALO_SLICE:.*]] = stablehlo.slice %[[LOCAL_IN]] [0:4, 0:5, 1:2]
  // HALO-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b"}, {"a"}, {}]>]>}
  // HALO-NEXT:   %[[RECV:.*]] = "stablehlo.collective_permute"(%[[HALO_SLICE]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  // HALO-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b"}, {"a"}, {}]>]>}
  // HALO-NEXT:   %[[OWN_SLICE:.*]] = stablehlo.slice %[[LOCAL_IN]] [0:4, 0:5, 0:1]
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[RECV]], %[[OWN_SLICE]], dim = 2
  // HALO-NEXT:   sdy.return %[[CONCAT]]
  // HALO-NEXT: } : (tensor<4x5x8xi32>) -> tensor<4x5x8xi32>

  // HALO-NEXT: %[[HALO_RESHARD:.*]] = sdy.reshard %[[HALO_SHIFT]] <@mesh_abc_reversed, [{"b"}, {"a"}, {"c"}]>
  // HALO: %[[HALO_REV:.*]] = stablehlo.reverse %[[HALO_RESHARD]], dims = [0, 2]
  // HALO: %[[RES:.*]] = stablehlo.slice %[[HALO_REV]] [0:4, 0:5, 0:7]
  %1 = stablehlo.reverse %0, dims = [0, 2]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
    : tensor<4x5x7xi32>

  // CHECK: %[[AG:.*]] = sdy.all_gather [{}, {"a"}, {"c"}] %[[RES]]
  // CHECK-SAME: out_sharding=<@mesh_abc, [{"b"}, {}, {}]>
  %2 = sdy.all_gather [{}, {"a"}, {"c"}] %1 out_sharding=<@mesh_abc, [{"b"}, {}, {}]> : tensor<4x5x7xi32>

  // CHECK: return %[[AG]]
  return %2 : tensor<4x5x7xi32>
}

// CHECK-LABEL: func @reverse_3_indivisible_with_padding_equal_shard_size
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4x6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>})
// CHECK-SAME:  -> tensor<3x5x6xi32> {
func.func @reverse_3_indivisible_with_padding_equal_shard_size(
  %arg0: tensor<4x6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>})
  -> tensor<3x5x6xi32> {

  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:3, 0:5, 0:6]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
  %0 = stablehlo.slice %arg0 [0:3, 0:5, 0:6]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
    : (tensor<4x6x8xi32>) -> tensor<3x5x6xi32>

  // REPL: %[[REPL_IN:.*]] = sdy.reshard %[[SLICE]] <@mesh_abc, [{}, {}, {}]>
  // REPL-NEXT: %[[REPL_REV:.*]] = stablehlo.reverse %[[REPL_IN]], dims = [0, 1, 2]
  // REPL-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{}, {}, {}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[REPL_REV]] <@mesh_abc, [{"b"}, {"a"}, {"c"}]>

  // HALO: %[[CST:.*]] = stablehlo.constant dense<0>
  // HALO: %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]]
  // HALO-SAME: low = [0, 0, 0], high = [1, 1, 2]
  // HALO-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}

  // HALO:      %[[HALO_SHIFTED:.*]] = sdy.manual_computation(%[[PAD]])
  // HALO-SAME: in_shardings=[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]
  // HALO-SAME: out_shardings=[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]
  // HALO-SAME: manual_axes={"a", "b", "c"}
  // HALO-SAME: (%[[LOCAL_IN:.*]]: tensor<2x3x2xi32>) {
  // HALO-NEXT:   %[[HALO1:.*]] = stablehlo.slice %[[LOCAL_IN]] [1:2, 0:3, 0:2]
  // HALO-NEXT:   %[[RECV1:.*]] = "stablehlo.collective_permute"(%[[HALO1]])
  // HALO-SAME{LITERAL}:   source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]]>
  // HALO-NEXT:   %[[DATA1:.*]] = stablehlo.slice %[[LOCAL_IN]] [0:1, 0:3, 0:2]
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[RECV1]], %[[DATA1]], dim = 0

  // HALO-NEXT:   %[[HALO2:.*]] = stablehlo.slice %[[CONCAT1]] [0:2, 2:3, 0:2]
  // HALO-NEXT:   %[[RECV2:.*]] = "stablehlo.collective_permute"(%[[HALO2]])
  // HALO-SAME{LITERAL}:   source_target_pairs = dense<[[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]>
  // HALO-NEXT:   %[[DATA2:.*]] = stablehlo.slice %[[CONCAT1]] [0:2, 0:2, 0:2]
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[RECV2]], %[[DATA2]], dim = 1

  // HALO-NEXT:   %[[HALO3:.*]] = "stablehlo.collective_permute"(%[[CONCAT2]])
  // HALO-SAME{LITERAL}:   source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [12, 13], [13, 14], [14, 15]]>
  // HALO-NEXT:   sdy.return %[[HALO3]] : tensor<2x3x2xi32>
  // HALO:      } : (tensor<4x6x8xi32>) -> tensor<4x6x8xi32>

  // HALO: %[[HALO_IN:.*]] = sdy.reshard %[[HALO_SHIFTED]] <@mesh_abc_reversed_0, [{"b"}, {"a"}, {"c"}]>
  // HALO: %[[HALO_REV:.*]] = stablehlo.reverse %[[HALO_IN]], dims = [0, 1, 2]
  // HALO: %[[RES:.*]] = stablehlo.slice %[[HALO_REV]] [0:3, 0:5, 0:6]
  %1 = stablehlo.reverse %0, dims = [0, 1, 2]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
    : tensor<3x5x6xi32>

  // CHECK: %[[AG:.*]] = sdy.all_gather [{"b"}, {"a"}, {"c"}] %[[RES]]
  // CHECK-SAME: out_sharding=<@mesh_abc, [{}, {}, {}]>
  %2 = sdy.all_gather [{"b"}, {"a"}, {"c"}] %1 out_sharding=<@mesh_abc, [{}, {}, {}]> : tensor<3x5x6xi32>

  // CHECK: return %[[AG]]
  return %2 : tensor<3x5x6xi32>
}

// CHECK-LABEL: func @reverse_indivisible_multiple_axes
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{}, {"a", "c":(2)2}]>})
// CHECK-SAME:  -> tensor<6x7xi32> {
func.func @reverse_indivisible_multiple_axes(
  %arg0: tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{}, {"a", "c":(2)2}]>})
  -> tensor<6x7xi32> {

  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:6, 0:7]
  %0 = stablehlo.slice %arg0 [0:6, 0:7]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_abc, [{}, {"a", "c":(2)2}]>]>}
    : (tensor<6x8xi32>) -> tensor<6x7xi32>

  // REPL: %[[REPL_IN:.*]] = sdy.reshard %[[SLICE]] <@mesh_abc, [{}, {}]>
  // REPL: %[[REPL_REV:.*]] = stablehlo.reverse %[[REPL_IN]], dims = [1]
  // REPL-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{}, {}]>]>}
  // REPL: %[[RES:.*]] = sdy.reshard %[[REPL_REV]] <@mesh_abc, [{}, {"a", "c":(2)2}]>

  // HALO: %[[CST:.*]] = stablehlo.constant dense<0>
  // HALO: %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]]
  // HALO-SAME: low = [0, 0], high = [0, 1]

  // HALO:      %[[HALO_SHIFTED:.*]] = sdy.manual_computation(%[[PAD]])
  // HALO-SAME: manual_axes={"a", "c"}
  // HALO-SAME: (%[[LOCAL_IN:.*]]: tensor<6x2xi32>) {
  // HALO-NEXT:   %[[HALO_SLICE:.*]] = stablehlo.slice %[[LOCAL_IN]] [0:6, 1:2]
  // HALO:        %[[HALO_RECV:.*]] = "stablehlo.collective_permute"(%[[HALO_SLICE]])
  // HALO-SAME{LITERAL}:   source_target_pairs = dense<[[0, 1], [1, 4], [2, 3], [3, 6], [4, 5], [6, 7]]>
  // HALO-NEXT:   %[[OWN_SLICE:.*]] = stablehlo.slice %[[LOCAL_IN]] [0:6, 0:1]
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[HALO_RECV]], %[[OWN_SLICE]], dim = 1
  // HALO-NEXT:   sdy.return %[[CONCAT]]
  // HALO-NEXT: } : (tensor<6x8xi32>) -> tensor<6x8xi32>

  // HALO:      %[[RESHARD:.*]] = sdy.reshard %[[HALO_SHIFTED]] <@mesh_abc_reversed_1, [{}, {"a", "c":(2)2}]>
  // HALO:      %[[REV:.*]] = stablehlo.reverse %[[RESHARD]], dims = [1]
  // HALO:      %[[RES:.*]] = stablehlo.slice %[[REV]] [0:6, 0:7]
  %1 = stablehlo.reverse %0, dims = [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{}, {"a", "c":(2)2}]>]>}
    : tensor<6x7xi32>

  // CHECK: %[[AG:.*]] = sdy.all_gather [{}, {"a", "c":(2)2}] %[[RES]]
  %2 = sdy.all_gather [{}, {"a", "c":(2)2}] %1 out_sharding=<@mesh_abc, [{}, {}]> : tensor<6x7xi32>
  // CHECK: return %[[AG]] : tensor<6x7xi32>
  return %2 : tensor<6x7xi32>
}

// CHECK-LABEL: func @convolution_spatial_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}, %arg1: tensor<3x3x1x1xf32>)
// CHECK-SAME: -> (tensor<1x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>})
func.func @convolution_spatial_permutation(
    %arg0: tensor<1x1x16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>},
    %arg1: tensor<3x3x1x1xf32>)
    -> (tensor<1x1x14x14xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}, {}]>}) {
  // REPL: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}, {}]> : tensor<1x1x16x16xf32>
  // REPL: %[[CONV:.*]] = stablehlo.convolution(%[[RESHARD_IN]], %arg1)
  // REPL: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>
  // REPL: %[[RES:.*]] = sdy.reshard %[[CONV]] <@mesh, [{}, {}, {"a"}, {}]> : tensor<1x1x14x14xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, f, 0, 1] x [0, 1, i, o] -> [b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"a"}, {}]>]>
    }
    : (tensor<1x1x16x16xf32>, tensor<3x3x1x1xf32>) -> tensor<1x1x14x14xf32>
   // REPL: return %[[RES]] : tensor<1x1x14x14xf32>
  return %0 : tensor<1x1x14x14xf32>
}

// CHECK-LABEL: func @select_and_scatter_permutation
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<1x8xi32>)
// CHECK-SAME: -> (tensor<1x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>})
func.func @select_and_scatter_permutation(
    %arg0: tensor<1x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>},
    %arg1: tensor<1x8xi32>)
    -> (tensor<1x16xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>}) {
  // REPL: %[[CST:.*]] = stablehlo.constant
  %cst = stablehlo.constant dense<0> : tensor<i32>
  // REPL: %[[RESHARD_OP:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<1x16xi32>
  // REPL: %[[SS:.*]] = "stablehlo.select_and_scatter"(%[[RESHARD_OP]], %[[ARG1]], %[[CST]])
  // REPL: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // REPL: %[[RES:.*]] = sdy.reshard %[[SS]] <@mesh, [{}, {"b"}]> : tensor<1x16xi32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %cst) ({
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.compare GT, %arg3, %arg4 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %2 : tensor<i1>
  }, {
    ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
      %2 = stablehlo.add %arg3, %arg4 : tensor<i32>
      stablehlo.return %2 : tensor<i32>
  }) {
    window_dimensions = array<i64: 1, 2>,
    window_strides = array<i64: 1, 2>,
    padding = dense<0> : tensor<2x2xi64>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}, {"b"}]>]>
  } : (tensor<1x16xi32>, tensor<1x8xi32>, tensor<i32>) -> tensor<1x16xi32>
  // REPL: return %[[RES]] : tensor<1x16xi32>
  return %0 : tensor<1x16xi32>
}

// CHECK-LABEL: func @slice_partition_partial_dim_without_communication
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func @slice_partition_partial_dim_without_communication(
  %arg0: tensor<4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
  -> tensor<3xi32> {
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:3]
  %0 = stablehlo.slice %arg0 [0:3] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}]>]>
  } : (tensor<4xi32>) -> tensor<3xi32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.all_gather [{"a"}] %[[SLICE]]
  %1 = sdy.all_gather [{"a"}] %0 out_sharding=<@mesh, [{}]> : tensor<3xi32>
  // CHECK-NEXT: return %[[RES]]
  return %1 : tensor<3xi32>
}

// CHECK-LABEL: func @slice_partition_partial_dim_with_communication
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func @slice_partition_partial_dim_with_communication(
  %arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
  -> tensor<3xi32> {
  // CHECK-NEXT:  %[[RESHARD_0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD_0]] [0:3]
  // CHECK-NEXT:  %[[RESHARD_1:.*]] = sdy.reshard %[[SLICE]] <@mesh, [{"a"}]>
  %0 = stablehlo.slice %arg0 [0:3] {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a"}]>]>
  } : (tensor<8xi32>) -> tensor<3xi32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.all_gather [{"a"}] %[[RESHARD_1]]
  %1 = sdy.all_gather [{"a"}] %0 out_sharding=<@mesh, [{}]> : tensor<3xi32>
  // CHECK-NEXT: return %[[RES]]
  return %1 : tensor<3xi32>
}
