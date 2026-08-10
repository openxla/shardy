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
// CHECK: @mesh_a4 = <["a"=4, "b"=2]>
sdy.mesh @mesh_a4 = <["a"=4, "b"=2]>
// CHECK: @mesh_a6 = <["a"=6]>
sdy.mesh @mesh_a6 = <["a"=6]>
// CHECK: @mesh_a_4 = <["a"=4]>
sdy.mesh @mesh_a_4 = <["a"=4]>
// CHECK: @mesh_a_8 = <["a"=8]>
sdy.mesh @mesh_a_8 = <["a"=8]>
// CHECK: @mesh_xy_8 = <["x"=8, "y"=8]>
sdy.mesh @mesh_xy_8 = <["x"=8, "y"=8]>
// CHECK: @mesh_custom = <["b"=2, "c"=2], device_ids=[3, 2, 1, 0]>
sdy.mesh @mesh_custom = <["b"=2, "c"=2], device_ids=[3, 2, 1, 0]>


//===----------------------------------------------------------------------===//
// stablehlo.convolution tests
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// stablehlo.pad tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @pad_comm_free
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @pad_comm_free(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> (tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // CHECK-NOT: sdy.manual_computation
  // CHECK:      %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %[[CST]]
  %0 = stablehlo.pad %arg0, %c, low = [4, 0], high = [4, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>} : (tensor<8x8xi32>, tensor<i32>) -> tensor<16x8xi32>

  // CHECK-NEXT: return %[[PAD]]
  return %0 : tensor<16x8xi32>
}

// CHECK-LABEL: func @pad_single_left_hop
// CHECK-SAME: (%[[ARG0:.*]]: {{.*}}) -> {{.*}}
func.func @pad_single_left_hop(
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a":(2)2}, {"b"}]>})
  -> tensor<7x8xi32> {
   %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // REPL: %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}, {"b"}]> : tensor<4x8xi32>
  // REPL: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]], %[[CST]], low ={{.*}}
  // REPL: %[[RES:.*]] = sdy.reshard %[[PAD]] <@mesh_a4, [{"a":(2)2}, {"b"}]> : tensor<7x8xi32>

  // HALO: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>] manual_axes={"a", "b"} (%[[ARG1:.*]]: tensor<2x4xi32>, %[[ARG2:.*]]: tensor<i32>) {
  // HALO: %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>, source_target_pairs ={{.*}}
  // HALO: %[[CONCAT:.*]] = stablehlo.concatenate %[[CP]], %[[ARG1]], dim = 0 {sdy.sharding ={{.*}}
  // HALO: %[[PAD_1:.*]] = stablehlo.pad %[[CONCAT]], %[[ARG2]], low ={{.*}}
  // HALO: %[[PART_ID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO: %[[CONVERT:.*]] = stablehlo.convert %[[PART_ID]] : (tensor<ui32>) -> tensor<i64>
  // HALO: %[[RESHAPE:.*]] = stablehlo.reshape %[[CONVERT]] : (tensor<i64>) -> tensor<i64>
  // HALO: %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO: %[[DIV:.*]] = stablehlo.divide %[[RESHAPE]], %[[C_DIV]] : tensor<i64>
  // HALO: %[[C_REM:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO: %[[REM:.*]] = stablehlo.remainder %[[DIV]], %[[C_REM]] : tensor<i64>
  // HALO: %[[CST_3:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO: %[[CST_4:.*]] = stablehlo.constant dense<3> : tensor<i64>
  // HALO: %[[MUL:.*]] = stablehlo.multiply %[[REM]], %[[CST_3]] : tensor<i64>
  // HALO: %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[CST_4]] : tensor<i64>
  // HALO: %[[CST_5:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // HALO: %[[MAX:.*]] = stablehlo.maximum %[[ADD]], %[[CST_5]] : tensor<i64>
  // HALO: %[[V_RES:.*]] = stablehlo.dynamic_slice %[[PAD_1]], %[[MAX]], %[[CST_5]], sizes = [4, 4] {sdy.sharding ={{.*}}
  // HALO: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:7, 0:8] {sdy.sharding ={{.*}}
  %0 = stablehlo.pad %arg0, %c, low = [3, 0], high = [0, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a":(2)2}, {"b"}]>]>} : (tensor<4x8xi32>, tensor<i32>) -> tensor<7x8xi32>

  // CHECK: %[[RSD:.*]] = sdy.reshard %[[RES]] <@mesh_a4, [{}, {}]> : tensor<7x8xi32>
  %1 = sdy.reshard %0 <@mesh_a4, [{}, {}]> : tensor<7x8xi32>

  // CHECK: return %[[RSD]] : {{.*}}
  return %1 : tensor<7x8xi32>
}

// CHECK-LABEL: func @pad_multiple_left_hops
// CHECK-SAME: (%[[ARG0:.*]]: {{.*}}) -> {{.*}}
func.func @pad_multiple_left_hops(
  %arg0: tensor<4x8x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {}, {"b"}]>})
  -> tensor<13x8x3xi32> {
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // REPL: %[[SLICE:.*]] = stablehlo.slice {{.*}} [0:4, 0:8, 0:3] {sdy.sharding ={{.*}}
  // REPL: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_a4, [{}, {}, {"b"}]> : tensor<4x8x3xi32>
  // REPL: %[[PAD_1:.*]] = stablehlo.pad %[[RESHARD]], %[[CST]], low ={{.*}}
  // REPL: %[[RES:.*]] = sdy.reshard %[[PAD_1]] <@mesh_a4, [{"a"}, {}, {"b"}]> : tensor<13x8x3xi32>

  // HALO: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO: %[[SLICE:.*]] = stablehlo.slice {{.*}} [0:4, 0:8, 0:3] {sdy.sharding ={{.*}}
  // HALO: %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]], low ={{.*}}
  // HALO: %[[MC:.*]] = sdy.manual_computation(%[[PAD]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a"}, {}, {"b"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a"}, {}, {"b"}]>] manual_axes={"a", "b"} (%[[ARG1:.*]]: tensor<1x8x2xi32>, %[[ARG2:.*]]: tensor<i32>) {
  // HALO: %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>, source_target_pairs ={{.*}}
  // HALO: %[[CP_1:.*]] = "stablehlo.collective_permute"(%[[ARG1]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>, source_target_pairs ={{.*}}
  // HALO: %[[CONCAT:.*]] = stablehlo.concatenate %[[CP]], %[[CP_1]], %[[ARG1]], dim = 0 {sdy.sharding ={{.*}}
  // HALO: %[[PAD_2:.*]] = stablehlo.pad %[[CONCAT]], %[[ARG2]], low ={{.*}}
  // HALO: %[[V_RES:.*]] = stablehlo.dynamic_slice %[[PAD_2]], {{.*}}, sizes = [4, 8, 2] {sdy.sharding ={{.*}}
  // HALO: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:13, 0:8, 0:3] {sdy.sharding ={{.*}}
  %0 = stablehlo.slice %arg0 [0:4, 0:8, 0:3]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}, {}, {"b"}]>]>}
    : (tensor<4x8x4xi32>) -> tensor<4x8x3xi32>
  %1 = stablehlo.pad %0, %c, low = [9, 0, 0], high = [0, 0, 0], interior = [0, 0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}, {}, {"b"}]>]>}
    : (tensor<4x8x3xi32>, tensor<i32>) -> tensor<13x8x3xi32>

  // CHECK: %[[RSD:.*]] = sdy.reshard %[[RES]] <@mesh_a4, [{"a"}, {}, {"b"}]> : tensor<13x8x3xi32>
  %2 = sdy.reshard %1 <@mesh_a4, [{"a"}, {}, {"b"}]> : tensor<13x8x3xi32>

  // CHECK: return %[[RSD]] : {{.*}}
  return %2 : tensor<13x8x3xi32>
}

// CHECK-LABEL: func @pad_single_right_hop
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xi32>
func.func @pad_single_right_hop(
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> tensor<7x8xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL-NEXT:     %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<4x8xi32>
  // REPL-NEXT:     %[[PAD:.*]] = stablehlo.pad %[[RESHARD1]], %[[CST]], low = [-1, 0], high = [4, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  // REPL-NEXT:     %[[MC_SLICE:.*]] = sdy.reshard %[[PAD]] <@mesh, [{"a"}, {"b"}]>

  // HALO-NEXT:     %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<2x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:       %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %arg1, %[[CP1]], dim = 0
  // HALO-NEXT:       %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [4, 0], high = [4, 0], interior = [0, 0]
  // HALO:       %[[SLICE:.*]] = stablehlo.dynamic_slice %[[PAD]], {{.*}}, sizes = [4, 4]
  // HALO-NEXT:       sdy.return %[[SLICE]] : tensor<4x4xi32>
  // HALO-NEXT:     }
  // HALO-NEXT:     %[[MC_SLICE:.*]] = stablehlo.slice %[[RES]] [0:7, 0:8]
  %0 = stablehlo.pad %arg0, %c, low = [-1, 0], high = [4, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>} : (tensor<4x8xi32>, tensor<i32>) -> tensor<7x8xi32>

  // CHECK-NEXT:     %[[RSD:.*]] = sdy.reshard %[[MC_SLICE]] <@mesh, [{}, {}]> : tensor<7x8xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<7x8xi32>

  // CHECK-NEXT:   return %[[RSD]] : tensor<7x8xi32>
  return %1 : tensor<7x8xi32>
}

// CHECK-LABEL: func @pad_multiple_right_hops_beyond_halo_limit
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xi32>
func.func @pad_multiple_right_hops_beyond_halo_limit(
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>})
  -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh_a4, [{}, {"b"}]> : tensor<4x8xi32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]], %[[CST]], low = [-3, 0], high = [7, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {"b"}]>]>} : (tensor<4x8xi32>, tensor<i32>) -> tensor<8x8xi32>
  // CHECK-NEXT: %[[RES:.*]] = sdy.reshard %[[PAD]] <@mesh_a4, [{"a"}, {"b"}]> : tensor<8x8xi32>
  %0 = stablehlo.pad %arg0, %c, low = [-3, 0], high = [7, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}, {"b"}]>]>} : (tensor<4x8xi32>, tensor<i32>) -> tensor<8x8xi32>

  // CHECK-NEXT:   return %[[RES]] : tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// CHECK-LABEL: func @pad_two_direction_hops
// CHECK-SAME: (%[[ARG0:.*]]: tensor<12x8xi32>
func.func @pad_two_direction_hops(
  %arg0: tensor<12x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>})
  -> tensor<13x8xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
   %c = stablehlo.constant dense<0> : tensor<i32>

  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice {{.*}} [0:11, 0:8] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}, {"b"}]>]>} : (tensor<12x8xi32>) -> tensor<11x8xi32>
  %slice = stablehlo.slice %arg0 [0:11, 0:8]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>]>}
    : (tensor<12x8xi32>) -> tensor<11x8xi32>

  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[SLICE]] <@mesh_a4, [{}, {"b"}]> : tensor<11x8xi32>
  // REPL-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD1]], %[[CST]], low = [2, 0], high = [0, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {"b"}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[PAD]] <@mesh_a4, [{"a"}, {"b"}]>

  // HALO-NEXT: %[[PAD_HIGH:.*]] = stablehlo.pad %[[SLICE]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}, {"b"}]>]>} : (tensor<11x8xi32>, tensor<i32>) -> tensor<12x8xi32>
  // HALO-NEXT:   %[[MC:.*]] = sdy.manual_computation(%[[PAD_HIGH]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a"}, {"b"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<3x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]> : tensor<6x2xi64>}>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [4, 2], [5, 3], [6, 4], [7, 5]]> : tensor<6x2xi64>}>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[CP1]], %arg1, %[[CP2]], dim = 0
  // HALO-NEXT:   %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [4, 0], high = [4, 0], interior = [0, 0]
  // HALO:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[PAD]], {{.*}}, sizes = [4, 4]
  // HALO-NEXT:   sdy.return %[[SLICE]] : tensor<4x4xi32>
  // HALO-NEXT: }
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:13, 0:8]
  %0 = stablehlo.pad %slice, %c, low = [2, 0], high = [0, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>]>} : (tensor<11x8xi32>, tensor<i32>) -> tensor<13x8xi32>

  // CHECK-NEXT: %[[RSD:.*]] = sdy.reshard %[[RES]] <@mesh_a4, [{}, {}]> : tensor<13x8xi32>
  %1 = sdy.reshard %0 <@mesh_a4, [{}, {}]> : tensor<13x8xi32>

  // CHECK-NEXT: return %[[RSD]] : tensor<13x8xi32>
  return %1 : tensor<13x8xi32>
}

// CHECK-LABEL: func @pad_multidim_with_hops
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4x2xi32>
func.func @pad_multidim_with_hops(
  %arg0: tensor<4x4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}, {}]>})
  -> tensor<7x6x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}]> : tensor<4x4x2xi32>
  // REPL-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD1]], %[[CST]], low = [-1, 2, 1], high = [4, 0, 1], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[PAD]] <@mesh, [{"a"}, {"b"}, {}]>

  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}, {}]>] manual_axes={"a", "b"} (%arg1: tensor<2x2x2xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %arg1, %[[CP1]], dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [4, 0, 0], high = [4, 0, 0], interior = [0, 0, 0]
  // HALO:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], {{.*}}, sizes = [4, 2, 2]
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}>
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[CP3]], %[[SLICE1]], dim = 1
  // HALO-NEXT:   %[[PAD2:.*]] = stablehlo.pad %[[CONCAT2]], %arg2, low = [0, 3, 0], high = [0, 3, 0], interior = [0, 0, 0]
  // HALO:        %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[PAD2]], {{.*}}, sizes = [4, 3, 2]
  // HALO-NEXT:   %[[PAD3:.*]] = stablehlo.pad %[[SLICE2]], %arg2, low = [0, 0, 1], high = [0, 0, 1], interior = [0, 0, 0]
  // HALO-NEXT:   sdy.return %[[PAD3]] : tensor<4x3x4xi32>
  // HALO-NEXT: }
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:7, 0:6, 0:4]

  %0 = stablehlo.pad %arg0, %c, low = [-1, 2, 1], high = [4, 0, 1], interior = [0, 0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}, {}]>]>} : (tensor<4x4x2xi32>, tensor<i32>) -> tensor<7x6x4xi32>

  // CHECK-NEXT: %[[RSD:.*]] = sdy.reshard %[[RES]] <@mesh, [{}, {}, {}]> : tensor<7x6x4xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}, {}]> : tensor<7x6x4xi32>

  // CHECK-NEXT:  return %[[RSD]] : tensor<7x6x4xi32>
  return %1 : tensor<7x6x4xi32>
}

// CHECK-LABEL: func @pad_replicated_negative_low_padding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8x3xi32>
func.func @pad_replicated_negative_low_padding(
  %arg0: tensor<8x8x3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}, {}]>})
  -> tensor<11x8x2xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL: %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}, {}]> : tensor<8x8x3xi32>
  // REPL: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]], %[[CST]], low ={{.*}}
  // REPL: %[[RES:.*]] = sdy.reshard %[[PAD]] <@mesh, [{"a"}, {"b"}, {}]> : tensor<11x8x2xi32>

  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}, {}]>] manual_axes={"a", "b"} (%arg1: tensor<4x4x3xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[CP1]], %arg1, dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [6, 0, 0], high = [6, 0, 0], interior = [0, 0, 0]
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[PID1]] : (tensor<ui32>) -> tensor<i64>
  // HALO-NEXT:   %[[RESHAPE1:.*]] = stablehlo.reshape %[[CONV1]] : (tensor<i64>) -> tensor<i64>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[RESHAPE1]], %[[C_DIV1]] : tensor<i64>
  // HALO-NEXT:   %[[C_MOD1:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO-NEXT:   %[[REM_PR:.*]] = stablehlo.remainder %[[DIV1]], %[[C_MOD1]] : tensor<i64>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<7> : tensor<i64>
  // HALO-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[REM_PR]], %[[C_STRIDE1]] : tensor<i64>
  // HALO-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[C_OFFSET1]] : tensor<i64>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // HALO-NEXT:   %[[MAX1:.*]] = stablehlo.maximum %[[ADD1]], %[[C_ZERO1]] : tensor<i64>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[MAX1]], %[[C_ZERO1]], %[[C_ZERO1]], sizes = [6, 4, 3]
  // HALO-NEXT:   %[[PAD_OUT:.*]] = stablehlo.pad %[[SLICE1]], %arg2, low = [0, 0, -1], high = [0, 0, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<6x4x3xi32>, tensor<i32>) -> tensor<6x4x2xi32>
  // HALO-NEXT:   sdy.return %[[PAD_OUT]] : tensor<6x4x2xi32>
  // HALO-NEXT: }
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:11, 0:8, 0:2]
  %0 = stablehlo.pad %arg0, %c, low = [3, 0, -1], high = [0, 0, 0], interior = [0, 0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}, {}]>]>} : (tensor<8x8x3xi32>, tensor<i32>) -> tensor<11x8x2xi32>

  // CHECK: %[[RSD:.*]] = sdy.reshard %[[RES]] <@mesh, [{}, {}, {}]> : tensor<11x8x2xi32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}, {}]> : tensor<11x8x2xi32>

  // CHECK: return %[[RSD]] : {{.*}}
  return %1 : tensor<11x8x2xi32>
}

// CHECK-LABEL: func @pad_large_low_pad_within_one_hop
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xi32>
func.func @pad_large_low_pad_within_one_hop(
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
  -> (tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<4x8xi32>
  // REPL-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD_IN]], %[[CST]], low = [12, 0], high = [0, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[PAD]] <@mesh, [{"a"}, {}]>

  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[CP1]], %arg1, dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [8, 0], high = [8, 0], interior = [0, 0]
  // HALO:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], {{.*}}, sizes = [8, 8]
  // HALO-NEXT:   sdy.return %[[SLICE1]] : tensor<8x8xi32>
  // HALO-NEXT: }
  %0 = stablehlo.pad %arg0, %c, low = [12, 0], high = [0, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<4x8xi32>, tensor<i32>) -> tensor<16x8xi32>

  // CHECK-NEXT: return %[[RES]] : tensor<16x8xi32>
  return %0 : tensor<16x8xi32>
}

// CHECK-LABEL: func @pad_replicated_negative_low_and_positive_high
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8x3xi32>
func.func @pad_replicated_negative_low_and_positive_high(
  %arg0: tensor<8x8x3xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}, {}]>})
  -> tensor<11x8x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}, {}]> : tensor<8x8x3xi32>
  // REPL-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD1]], %[[CST]], low = [3, 0, -1], high = [0, 0, 2], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}, {}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[PAD]] <@mesh, [{"a"}, {"b"}, {}]>

  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}, {}]>] manual_axes={"a", "b"} (%arg1: tensor<4x4x3xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[CP1]], %arg1, dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [6, 0, 0], high = [6, 0, 0], interior = [0, 0, 0]
  // HALO:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], {{.*}}, sizes = [6, 4, 3]
  // HALO-NEXT:   %[[PAD2:.*]] = stablehlo.pad %[[SLICE1]], %arg2, low = [0, 0, -1], high = [0, 0, 2], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<6x4x3xi32>, tensor<i32>) -> tensor<6x4x4xi32>
  // HALO-NEXT:   sdy.return %[[PAD2]] : tensor<6x4x4xi32>
  // HALO-NEXT: }
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:11, 0:8, 0:4]
  %0 = stablehlo.pad %arg0, %c, low = [3, 0, -1], high = [0, 0, 2], interior = [0, 0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}, {}]>]>} : (tensor<8x8x3xi32>, tensor<i32>) -> tensor<11x8x4xi32>

  // CHECK-NEXT: return %[[RES]] : tensor<11x8x4xi32>
  return %0 : tensor<11x8x4xi32>
}

// CHECK-LABEL: func @pad_sharded_indivisible_interior_pad
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3x8xi32>
func.func @pad_sharded_indivisible_interior_pad(
  %arg0: tensor<3x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> tensor<7x8xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<3x8xi32>
  // REPL-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD1]], %[[CST]], low = [0, 0], high = [0, 0], interior = [2, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[PAD]] <@mesh, [{"a"}, {"b"}]>

  // HALO-NEXT: %[[PAD_HIGH:.*]] = stablehlo.pad %[[ARG0]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>} : (tensor<3x8xi32>, tensor<i32>) -> tensor<4x8xi32>
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[PAD_HIGH]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%arg1: tensor<2x4xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>}>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[CP1]], %arg1, %[[CP2]], dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [12, 0], high = [12, 0], interior = [2, 0]
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[PID1]] : (tensor<ui32>) -> tensor<i64>
  // HALO-NEXT:   %[[RESHAPE1:.*]] = stablehlo.reshape %[[CONV1]] : (tensor<i64>) -> tensor<i64>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[RESHAPE1]], %[[C_DIV1]] : tensor<i64>
  // HALO-NEXT:   %[[C_MOD1:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO-NEXT:   %[[REM1:.*]] = stablehlo.remainder %[[DIV1]], %[[C_MOD1]] : tensor<i64>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<-2> : tensor<i64>
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<18> : tensor<i64>
  // HALO-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[REM1]], %[[C_STRIDE1]] : tensor<i64>
  // HALO-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[C_OFFSET1]] : tensor<i64>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // HALO-NEXT:   %[[MAX1:.*]] = stablehlo.maximum %[[ADD1]], %[[C_ZERO1]]
  // HALO:        %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[MAX1]], %[[C_ZERO1]], sizes = [4, 4]
  // HALO-NEXT:   sdy.return %[[SLICE1]] : tensor<4x4xi32>
  // HALO-NEXT: }
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:7, 0:8]
  %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [0, 0], interior = [2, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>} : (tensor<3x8xi32>, tensor<i32>) -> tensor<7x8xi32>

  // CHECK-NEXT: return %[[RES]] : tensor<7x8xi32>
  return %0 : tensor<7x8xi32>
}

// CHECK-LABEL: func @pad_replicated_negative_high_padding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @pad_replicated_negative_high_padding(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>})
  -> tensor<6x9xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x8xi32>
  // REPL-NEXT: %[[PAD1:.*]] = stablehlo.pad %[[RESHARD1]], %[[CST]], low = [0, 0], high = [-2, 1], interior = [0, 0]
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[PAD1]] <@mesh, [{}, {"b"}]>

  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{}, {"b"}]>] manual_axes={"b"} (%[[ARG1:.*]]: tensor<8x4xi32>, %[[ARG2:.*]]: tensor<i32>) {
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[ARG1]], %[[CP]], dim = 1
  // HALO-NEXT:   %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %[[ARG2]], low = [0, 5], high = [0, 5], interior = [0, 0]
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[CONV:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // HALO-NEXT:   %[[RESHAPE:.*]] = stablehlo.reshape %[[CONV]] : (tensor<i64>) -> tensor<i64>
  // HALO-NEXT:   %[[C_MOD:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.remainder %[[RESHAPE]], %[[C_MOD]] : tensor<i64>
  // HALO-NEXT:   %[[C_STRIDE:.*]] = stablehlo.constant dense<1> : tensor<i64>
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<5> : tensor<i64>
  // HALO-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[REM]], %[[C_STRIDE]] : tensor<i64>
  // HALO-NEXT:   %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_OFFSET]] : tensor<i64>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // HALO-NEXT:   %[[MAX:.*]] = stablehlo.maximum %[[ADD]], %[[C_ZERO]] : tensor<i64>
  // HALO-NEXT:   %[[SLICE_D:.*]] = stablehlo.dynamic_slice %[[PAD]], %[[C_ZERO]], %[[MAX]], sizes = [8, 5]
  // HALO-NEXT:   %[[SLICE_H:.*]] = stablehlo.pad %[[SLICE_D]], %[[ARG2]], low = [0, 0], high = [-2, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x5xi32>, tensor<i32>) -> tensor<6x5xi32>
  // HALO-NEXT:   sdy.return %[[SLICE_H]] : tensor<6x5xi32>
  // HALO-NEXT: }
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:6, 0:9]
  %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [-2, 1], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<8x8xi32>, tensor<i32>) -> tensor<6x9xi32>

  // CHECK: return %[[RES]] : tensor<6x9xi32>
  return %0 : tensor<6x9xi32>
}

// CHECK-LABEL: func @pad_sharded_indivisible_interior_low_and_high
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xi32>
func.func @pad_sharded_indivisible_interior_low_and_high(
  %arg0: tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {}]>})
  -> tensor<15x8xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // CHECK-NEXT: %[[PRE_SLICE:.*]] = stablehlo.slice {{.*}} [0:7, 0:8]
  %0 = stablehlo.slice %arg0 [0:7, 0:8] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}, {}]>]>} : (tensor<8x8xi32>) -> tensor<7x8xi32>

  // REPL-NEXT: %[[RESHARD_0:.*]] = sdy.reshard %[[PRE_SLICE]] <@mesh_a4, [{}, {}]> : tensor<7x8xi32>
  // REPL-NEXT: %[[PAD_0:.*]] = stablehlo.pad %[[RESHARD_0]], %[[CST]], low = [1, 0], high = [1, 0], interior = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}]>]>} : (tensor<7x8xi32>, tensor<i32>) -> tensor<15x8xi32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[PAD_0]] <@mesh_a4, [{"a"}, {}]> : tensor<15x8xi32>

  // HALO-NEXT: %[[PRE_PAD:.*]] = stablehlo.pad %[[PRE_SLICE]], %[[CST]], low = [0, 0], high = [1, 0], interior = [0, 0]
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[PRE_PAD]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a"}, {}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xi32>, %arg2: tensor<i32>) {
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]> : tensor<6x2xi64>}>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [4, 2], [5, 3], [6, 4], [7, 5]]> : tensor<6x2xi64>}>
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[CP1]], %arg1, %[[CP2]], dim = 0
  // HALO-NEXT:   %[[PAD_INNER:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [8, 0], high = [8, 0], interior = [1, 0]
  // HALO:        %[[MAX_INNER:.*]] = stablehlo.maximum %{{.*}}, %{{.*}}
  // HALO:        %[[SLICE_INNER:.*]] = stablehlo.dynamic_slice %[[PAD_INNER]], %[[MAX_INNER]], %{{.*}}, sizes = [4, 8]
  // HALO-NEXT:   sdy.return %[[SLICE_INNER]] : tensor<4x8xi32>
  // HALO-NEXT: }
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:15, 0:8]
  %1 = stablehlo.pad %0, %c, low = [1, 0], high = [1, 0], interior = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}, {}]>]>} : (tensor<7x8xi32>, tensor<i32>) -> tensor<15x8xi32>

  // CHECK-NEXT: %[[RSD:.*]] = sdy.reshard %[[RES]] <@mesh_a4, [{}, {}]> : tensor<15x8xi32>
  %2 = sdy.reshard %1 <@mesh_a4, [{}, {}]> : tensor<15x8xi32>

  // CHECK: return %[[RSD]] : tensor<15x8xi32>
  return %2 : tensor<15x8xi32>
}

//===----------------------------------------------------------------------===//
// stablehlo.reshape tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @reshape_1d_to_2d_non_divisible_comm_free(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<6xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>})
func.func @reshape_1d_to_2d_non_divisible_comm_free(%arg0: tensor<6xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>}) -> tensor<1x6xi32> {
  // CHECK-NEXT:     %[[RES:.*]] = stablehlo.reshape %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}, {"a"}]>]>} : (tensor<6xi32>) -> tensor<1x6xi32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{}, {"a"}]>]>} : (tensor<6xi32>) -> tensor<1x6xi32>

  // CHECK-NEXT:     return %[[RES]]
  return %0 : tensor<1x6xi32>
}

// CHECK-LABEL: func @reshape_2d_to_1d_non_divisible_comm_free(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1x6xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}, {"a"}]>})
func.func @reshape_2d_to_1d_non_divisible_comm_free(%arg0: tensor<1x6xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}, {"a"}]>}) -> tensor<6xi32> {
  // CHECK-NEXT:     %[[RES:.*]] = stablehlo.reshape %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>} : (tensor<1x6xi32>) -> tensor<6xi32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"a"}]>]>} : (tensor<1x6xi32>) -> tensor<6xi32>

  // CHECK-NEXT:     return %[[RES]]
  return %0 : tensor<6xi32>
}

// CHECK-LABEL: func @reshape_single_dim_split_comm_free(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}, {}]>})
func.func @reshape_single_dim_split_comm_free(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}, {}]>}) -> (tensor<3x2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}, {}, {}]>}) {
  // CHECK:         %[[SLICE_IN:.*]] = stablehlo.slice %[[ARG0]] [0:6, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<6x2xi32>
  %0 = stablehlo.slice %arg0 [0:6, 0:2] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<6x2xi32>

  // CHECK-NEXT:     %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE_IN]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}, {}, {}]>]>} : (tensor<6x2xi32>) -> tensor<3x2x2xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"a"}, {}, {}]>]>} : (tensor<6x2xi32>) -> tensor<3x2x2xi32>

  // CHECK-NEXT:     %[[RES:.*]] = sdy.reshard %[[RESHAPE]] <@mesh_a_4, [{}, {}, {}]> : tensor<3x2x2xi32>
  %2 = sdy.reshard %1 <@mesh_a_4, [{}, {}, {}]> : tensor<3x2x2xi32>

  // CHECK-NEXT:     return %[[RES]]
  return %2 : tensor<3x2x2xi32>
}

// CHECK-LABEL: func @reshape_single_dim_combine_comm_free(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<4x2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}, {}, {}]>})
func.func @reshape_single_dim_combine_comm_free(%arg0: tensor<4x2x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}, {}, {}]>}) -> (tensor<6x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}, {}]>}) {
  // CHECK:         %[[SLICE_IN:.*]] = stablehlo.slice %[[ARG0]] [0:3, 0:2, 0:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}, {}, {}]>]>} : (tensor<4x2x2xi32>) -> tensor<3x2x2xi32>
  %0 = stablehlo.slice %arg0 [0:3, 0:2, 0:2] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"a"}, {}, {}]>]>} : (tensor<4x2x2xi32>) -> tensor<3x2x2xi32>

  // CHECK-NEXT:     %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE_IN]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}, {}]>]>} : (tensor<3x2x2xi32>) -> tensor<6x2xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"a"}, {}]>]>} : (tensor<3x2x2xi32>) -> tensor<6x2xi32>

  // CHECK-NEXT:     %[[RES:.*]] = sdy.reshard %[[RESHAPE]] <@mesh_a_4, [{}, {}]> : tensor<6x2xi32>
  %2 = sdy.reshard %1 <@mesh_a_4, [{}, {}]> : tensor<6x2xi32>

  // CHECK-NEXT:     return %[[RES]]
  return %2 : tensor<6x2xi32>
}

// CHECK-LABEL: func @reshape_indivisible_cross_dims(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<6x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}, {}]>})
func.func @reshape_indivisible_cross_dims(%arg0: tensor<6x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}, {}]>}) -> tensor<4x3xi32> {
  // CHECK-NEXT:     %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh_a_4, [{}, {}]> : tensor<6x2xi32>
  // CHECK-NEXT:     %[[RES:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}, {}]>]>} : (tensor<6x2xi32>) -> tensor<4x3xi32>
  // CHECK-NEXT:     %[[RESHARD_OUT:.*]] = sdy.reshard %[[RES]] <@mesh_a_4, [{"a"}, {}]> : tensor<4x3xi32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"a"}, {}]>]>} : (tensor<6x2xi32>) -> tensor<4x3xi32>

  // CHECK-NEXT:     return %[[RESHARD_OUT]]
  return %0 : tensor<4x3xi32>
}

// CHECK-LABEL: func @reshape_2x3x5_to_30_group_padded_size_mismatch(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x3x5xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}, {}]>})
func.func @reshape_2x3x5_to_30_group_padded_size_mismatch(%arg0: tensor<2x3x5xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}, {}]>}) -> tensor<30xi32> {
  // CHECK-NEXT:     %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}, {}]> : tensor<2x3x5xi32>
  // CHECK-NEXT:     %[[RES:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<2x3x5xi32>) -> tensor<30xi32>
  // CHECK-NEXT:     %[[RESHARD_OUT:.*]] = sdy.reshard %[[RES]] <@mesh, [{"a", "b"}]> : tensor<30xi32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"a", "b"}]>]>} : (tensor<2x3x5xi32>) -> tensor<30xi32>

  // CHECK-NEXT:     return %[[RESHARD_OUT]]
  return %0 : tensor<30xi32>
}

// CHECK-LABEL: func @reshape_1d_to_2d_split(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>})
func.func @reshape_1d_to_2d_split(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}]>}) -> (tensor<2x3xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}, {}]>}) {
  // CHECK:         %[[SLICE_IN:.*]] = stablehlo.slice %[[ARG0]] [0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>} : (tensor<8xi32>) -> tensor<6xi32>
  %0 = stablehlo.slice %arg0 [0:6] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"a"}]>]>} : (tensor<8xi32>) -> tensor<6xi32>

  // REPL:          %[[RESHARD_IN:.*]] = sdy.reshard %[[SLICE_IN]] <@mesh_a_4, [{}]> : tensor<6xi32>
  // REPL-NEXT:     %[[RESHAPE_REPL:.*]] = stablehlo.reshape %[[RESHARD_IN]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}, {}]>]>} : (tensor<6xi32>) -> tensor<2x3xi32>
  // REPL-NEXT:     %[[RES_SHARDED:.*]] = sdy.reshard %[[RESHAPE_REPL]] <@mesh_a_4, [{"a":(1)2}, {"a":(2)2}]> : tensor<2x3xi32>

  // HALO:          %[[PAD:.*]] = stablehlo.pad %[[SLICE_IN]], %{{.*}}, low = [0], high = [2], interior = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a"}]>]>} : (tensor<6xi32>, tensor<i32>) -> tensor<8xi32>
  // HALO-NEXT:     %[[MC:.*]] = sdy.manual_computation(%[[PAD]], %{{.*}}) in_shardings=[<@mesh_a_4, [{"a"}]>, <@mesh_a_4, []>] out_shardings=[<@mesh_a_4, [{"a":(1)2}, {"a":(2)2}]>] manual_axes={"a"} (%[[ARG1:.*]]: tensor<2xi32>, %[[ARG2:.*]]: tensor<i32>) {
  // HALO-NEXT:       %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %[[CP]], %[[ARG1]], dim = 0
  // HALO-NEXT:       %[[PAD_BUF:.*]] = stablehlo.pad %[[CONCAT]], %[[ARG2]], low = [2], high = [2], interior = [0]
  // HALO:            sdy.return %{{.*}} : tensor<1x2xi32>
  // HALO-NEXT:     } : (tensor<8xi32>, tensor<i32>) -> tensor<2x4xi32>
  // HALO-NEXT:     %[[RES_SHARDED:.*]] = stablehlo.slice %[[MC]] [0:2, 0:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{"a":(1)2}, {"a":(2)2}]>]>} : (tensor<2x4xi32>) -> tensor<2x3xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"a":(1)2}, {"a":(2)2}]>]>} : (tensor<6xi32>) -> tensor<2x3xi32>

  // CHECK:         %[[RES:.*]] = sdy.reshard %[[RES_SHARDED]] <@mesh_a_4, [{}, {}]> : tensor<2x3xi32>
  %2 = sdy.reshard %1 <@mesh_a_4, [{}, {}]> : tensor<2x3xi32>

  // CHECK-NEXT:     return %[[RES]]
  return %2 : tensor<2x3xi32>
}

// CHECK-LABEL: func @reshape_2d_split_with_unrelated_axis(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>})
func.func @reshape_2d_split_with_unrelated_axis(%arg0: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>}) -> (tensor<2x3x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}]>}) {
  // CHECK:         %[[SLICE_IN:.*]] = stablehlo.slice %[[ARG0]] [0:6, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}, {"b"}]>]>} : (tensor<8x4xi32>) -> tensor<6x4xi32>
  %0 = stablehlo.slice %arg0 [0:6, 0:4] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}, {"b"}]>]>} : (tensor<8x4xi32>) -> tensor<6x4xi32>

  // REPL:          %[[RESHARD_IN:.*]] = sdy.reshard %[[SLICE_IN]] <@mesh_a4, [{}, {"b"}]> : tensor<6x4xi32>
  // REPL-NEXT:     %[[RESHAPE_REPL:.*]] = stablehlo.reshape %[[RESHARD_IN]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {"b"}]>]>} : (tensor<6x4xi32>) -> tensor<2x3x4xi32>
  // REPL-NEXT:     %[[RES_SHARDED:.*]] = sdy.reshard %[[RESHAPE_REPL]] <@mesh_a4, [{"a":(1)2}, {"a":(2)2}, {"b"}]> : tensor<2x3x4xi32>

  // HALO:          %[[PAD:.*]] = stablehlo.pad %[[SLICE_IN]], %{{.*}}, low = [0, 0], high = [2, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}, {"b"}]>]>} : (tensor<6x4xi32>, tensor<i32>) -> tensor<8x4xi32>
  // HALO-NEXT:     %[[MC:.*]] = sdy.manual_computation(%[[PAD]], %{{.*}}) in_shardings=[<@mesh_a4, [{"a"}, {"b"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a":(1)2}, {"a":(2)2}, {"b"}]>] manual_axes={"a"} (%[[ARG1:.*]]: tensor<2x4xi32>, %[[ARG2:.*]]: tensor<i32>) {
  // HALO-NEXT:       %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]> : tensor<6x2xi64>
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %[[CP]], %[[ARG1]], dim = 0
  // HALO-NEXT:       %[[PAD_BUF:.*]] = stablehlo.pad %[[CONCAT]], %[[ARG2]], low = [2, 0], high = [2, 0], interior = [0, 0]
  // HALO:            sdy.return %{{.*}} : tensor<1x2x4xi32>
  // HALO-NEXT:     }
  // HALO-NEXT:     %[[RES_SHARDED:.*]] = stablehlo.slice %[[MC]] [0:2, 0:3, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a":(1)2}, {"a":(2)2}, {"b"}]>]>} : (tensor<2x4x4xi32>) -> tensor<2x3x4xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a":(1)2}, {"a":(2)2}, {"b"}]>]>} : (tensor<6x4xi32>) -> tensor<2x3x4xi32>

  // CHECK:         %[[RES:.*]] = sdy.reshard %[[RES_SHARDED]] <@mesh_a4, [{}, {}, {}]> : tensor<2x3x4xi32>
  %2 = sdy.reshard %1 <@mesh_a4, [{}, {}, {}]> : tensor<2x3x4xi32>

  // CHECK-NEXT:     return %[[RES]]
  return %2 : tensor<2x3x4xi32>
}

// CHECK-LABEL: func @reshape_1d_to_2d_split_gap_2(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<14xi32> {sdy.sharding = #sdy.sharding<@mesh_a6, [{"a"}]>})
func.func @reshape_1d_to_2d_split_gap_2(%arg0: tensor<14xi32> {sdy.sharding = #sdy.sharding<@mesh_a6, [{"a"}]>}) -> (tensor<2x7xi32> {sdy.sharding = #sdy.sharding<@mesh_a6, [{}, {}]>}) {
  // REPL:          %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_a6, [{}]> : tensor<14xi32>
  // REPL-NEXT:     %[[RESHAPE_REPL:.*]] = stablehlo.reshape %[[RESHARD_IN]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a6, [{}, {}]>]>} : (tensor<14xi32>) -> tensor<2x7xi32>
  // REPL-NEXT:     %[[RES_SHARDED:.*]] = sdy.reshard %[[RESHAPE_REPL]] <@mesh_a6, [{"a":(1)2}, {"a":(2)3}]> : tensor<2x7xi32>

  // HALO:          %[[PAD:.*]] = stablehlo.pad %[[ARG0]], %{{.*}}, low = [0], high = [4], interior = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a6, [{"a"}]>]>} : (tensor<14xi32>, tensor<i32>) -> tensor<18xi32>
  // HALO-NEXT:     %[[MC:.*]] = sdy.manual_computation(%[[PAD]], %{{.*}}) in_shardings=[<@mesh_a6, [{"a"}]>, <@mesh_a6, []>] out_shardings=[<@mesh_a6, [{"a":(1)2}, {"a":(2)3}]>] manual_axes={"a"} (%[[ARG1:.*]]: tensor<3xi32>, %[[ARG2:.*]]: tensor<i32>) {
  // HALO-NEXT:       %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]> : tensor<5x2xi64>
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %[[CP]], %[[ARG1]], dim = 0
  // HALO-NEXT:       %[[PAD_BUF:.*]] = stablehlo.pad %[[CONCAT]], %[[ARG2]], low = [3], high = [3], interior = [0]
  // HALO:            sdy.return %{{.*}} : tensor<1x3xi32>
  // HALO-NEXT:     } : (tensor<18xi32>, tensor<i32>) -> tensor<2x9xi32>
  // HALO-NEXT:     %[[RES_SHARDED:.*]] = stablehlo.slice %[[MC]] [0:2, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a6, [{"a":(1)2}, {"a":(2)3}]>]>} : (tensor<2x9xi32>) -> tensor<2x7xi32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a6, [{"a":(1)2}, {"a":(2)3}]>]>} : (tensor<14xi32>) -> tensor<2x7xi32>

  // CHECK:         %[[RES:.*]] = sdy.reshard %[[RES_SHARDED]] <@mesh_a6, [{}, {}]> : tensor<2x7xi32>
  %1 = sdy.reshard %0 <@mesh_a6, [{}, {}]> : tensor<2x7xi32>

  // CHECK-NEXT:     return %[[RES]]
  return %1 : tensor<2x7xi32>
}

// CHECK-LABEL: func @reshape_1d_to_3d_split(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x16xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}, {"a"}]>})
func.func @reshape_1d_to_3d_split(%arg0: tensor<2x16xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}, {"a"}]>}) -> (tensor<2x2x7xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}, {}, {}]>}) {
  // CHECK:         %[[SLICE_IN:.*]] = stablehlo.slice %[[ARG0]] [0:2, 0:14] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}, {"a"}]>]>} : (tensor<2x16xi32>) -> tensor<2x14xi32>
  %0 = stablehlo.slice %arg0 [0:2, 0:14] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{}, {"a"}]>]>} : (tensor<2x16xi32>) -> tensor<2x14xi32>

  // REPL:          %[[RESHARD_IN:.*]] = sdy.reshard %[[SLICE_IN]] <@mesh_a_4, [{}, {}]> : tensor<2x14xi32>
  // REPL-NEXT:     %[[RESHAPE_REPL:.*]] = stablehlo.reshape %[[RESHARD_IN]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}, {}, {}]>]>} : (tensor<2x14xi32>) -> tensor<2x2x7xi32>
  // REPL-NEXT:     %[[RES_SHARDED:.*]] = sdy.reshard %[[RESHAPE_REPL]] <@mesh_a_4, [{}, {"a":(1)2}, {"a":(2)2}]> : tensor<2x2x7xi32>

  // HALO:          %[[PAD:.*]] = stablehlo.pad %[[SLICE_IN]], %{{.*}}, low = [0, 0], high = [0, 2], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}, {"a"}]>]>} : (tensor<2x14xi32>, tensor<i32>) -> tensor<2x16xi32>
  // HALO-NEXT:     %[[MC:.*]] = sdy.manual_computation(%[[PAD]], %{{.*}}) in_shardings=[<@mesh_a_4, [{}, {"a"}]>, <@mesh_a_4, []>] out_shardings=[<@mesh_a_4, [{}, {"a":(1)2}, {"a":(2)2}]>] manual_axes={"a"} (%[[ARG1:.*]]: tensor<2x4xi32>, %[[ARG2:.*]]: tensor<i32>) {
  // HALO-NEXT:       %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>}>
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %[[CP]], %[[ARG1]], dim = 1
  // HALO:            sdy.return %{{.*}} : tensor<2x1x4xi32>
  // HALO-NEXT:     } : (tensor<2x16xi32>, tensor<i32>) -> tensor<2x2x8xi32>
  // HALO-NEXT:     %[[RES_SHARDED:.*]] = stablehlo.slice %[[MC]] [0:2, 0:2, 0:7] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}, {"a":(1)2}, {"a":(2)2}]>]>} : (tensor<2x2x8xi32>) -> tensor<2x2x7xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{}, {"a":(1)2}, {"a":(2)2}]>]>} : (tensor<2x14xi32>) -> tensor<2x2x7xi32>

  // CHECK:         %[[RES:.*]] = sdy.reshard %[[RES_SHARDED]] <@mesh_a_4, [{}, {}, {}]> : tensor<2x2x7xi32>
  %2 = sdy.reshard %1 <@mesh_a_4, [{}, {}, {}]> : tensor<2x2x7xi32>

  // CHECK-NEXT:     return %[[RES]]
  return %2 : tensor<2x2x7xi32>
}

// CHECK-LABEL: func @reshape_two_splitting_groups(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<16x32xi32> {sdy.sharding = #sdy.sharding<@mesh_xy_8, [{"x"}, {"y"}]>})
func.func @reshape_two_splitting_groups(%arg0: tensor<16x32xi32> {sdy.sharding = #sdy.sharding<@mesh_xy_8, [{"x"}, {"y"}]>}) -> (tensor<4x3x4x6xi32> {sdy.sharding = #sdy.sharding<@mesh_xy_8, [{}, {}, {}, {}]>}) {
  // REPL-NEXT:     %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_xy_8, [{"x"}, {}]> : tensor<16x32xi32>
  // REPL-NEXT:     %[[SLICE_IN:.*]] = stablehlo.slice %[[RESHARD_IN]] [0:12, 0:24] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy_8, [{"x"}, {}]>]>} : (tensor<16x32xi32>) -> tensor<12x24xi32>
  // REPL-NEXT:     %[[RESHARD_1:.*]] = sdy.reshard %[[SLICE_IN]] <@mesh_xy_8, [{"x"}, {"y"}]> : tensor<12x24xi32>
  // REPL-NEXT:     %[[RESHARD_2:.*]] = sdy.reshard %[[RESHARD_1]] <@mesh_xy_8, [{}, {"y"}]> : tensor<12x24xi32>
  // REPL-NEXT:     %[[RESHAPE_REPL:.*]] = stablehlo.reshape %[[RESHARD_2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy_8, [{}, {}, {"y":(1)4}, {"y":(4)2}]>]>} : (tensor<12x24xi32>) -> tensor<4x3x4x6xi32>
  // REPL-NEXT:     %[[RES_SHARDED:.*]] = sdy.reshard %[[RESHAPE_REPL]] <@mesh_xy_8, [{"x":(1)4}, {"x":(4)2}, {"y":(1)4}, {"y":(4)2}]> : tensor<4x3x4x6xi32>
  // REPL-NEXT:     %[[RES:.*]] = sdy.reshard %[[RES_SHARDED]] <@mesh_xy_8, [{}, {}, {}, {}]> : tensor<4x3x4x6xi32>

  // HALO-NEXT:     %[[C:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:     %[[MC1:.*]] = sdy.manual_computation(%[[ARG0]], %[[C]]) in_shardings=[<@mesh_xy_8, [{"x"}, {"y"}]>, <@mesh_xy_8, []>] out_shardings=[<@mesh_xy_8, [{"x"}, {"y"}]>] manual_axes={"x", "y"} (%[[ARG1:.*]]: tensor<2x4xi32>, %arg2: tensor<i32>) {
  // HALO:            sdy.return %{{.*}} : tensor<2x3xi32>
  // HALO-NEXT:     } : (tensor<16x32xi32>, tensor<i32>) -> tensor<16x24xi32>
  // HALO-NEXT:     %[[SLICE1:.*]] = stablehlo.slice %[[MC1]] [0:12, 0:24] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy_8, [{"x"}, {"y"}]>]>} : (tensor<16x24xi32>) -> tensor<12x24xi32>
  // HALO-NEXT:     %[[RESHARD2:.*]] = sdy.reshard %[[SLICE1]] <@mesh_xy_8, [{}, {"y"}]> : tensor<12x24xi32>
  // HALO-NEXT:     %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy_8, [{}, {}, {"y":(1)4}, {"y":(4)2}]>]>} : (tensor<12x24xi32>) -> tensor<4x3x4x6xi32>
  // HALO-NEXT:     %[[RES_SHARDED:.*]] = sdy.reshard %[[RESHAPE]] <@mesh_xy_8, [{"x":(1)4}, {"x":(4)2}, {"y":(1)4}, {"y":(4)2}]> : tensor<4x3x4x6xi32>
  // HALO-NEXT:     %[[RES:.*]] = sdy.reshard %[[RES_SHARDED]] <@mesh_xy_8, [{}, {}, {}, {}]> : tensor<4x3x4x6xi32>
  %0 = stablehlo.slice %arg0 [0:12, 0:24] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_xy_8, [{"x"}, {"y"}]>]>} : (tensor<16x32xi32>) -> tensor<12x24xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_xy_8, [{"x":(1)4}, {"x":(4)2}, {"y":(1)4}, {"y":(4)2}]>]>} : (tensor<12x24xi32>) -> tensor<4x3x4x6xi32>
  %2 = sdy.reshard %1 <@mesh_xy_8, [{}, {}, {}, {}]> : tensor<4x3x4x6xi32>

  // CHECK-NEXT:     return %[[RES]]
  return %2 : tensor<4x3x4x6xi32>
}

// CHECK-LABEL: func @reshape_1d_to_2d_split_custom_device_ids
// CHECK-SAME:      %[[ARG0:.*]]: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{"b", "c"}]>})
func.func @reshape_1d_to_2d_split_custom_device_ids(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{"b", "c"}]>}) -> (tensor<2x3xi32> {sdy.sharding = #sdy.sharding<@mesh_custom, [{}, {}]>}) {
  // CHECK:         %[[SLICE_IN:.*]] = stablehlo.slice %[[ARG0]] [0:6] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_custom, [{"b", "c"}]>]>} : (tensor<8xi32>) -> tensor<6xi32>
  %0 = stablehlo.slice %arg0 [0:6] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_custom, [{"b", "c"}]>]>} : (tensor<8xi32>) -> tensor<6xi32>

  // REPL:          %[[RESHARD_IN:.*]] = sdy.reshard %[[SLICE_IN]] <@mesh_custom, [{}]> : tensor<6xi32>
  // REPL-NEXT:     %[[RESHAPE_REPL:.*]] = stablehlo.reshape %[[RESHARD_IN]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_custom, [{}, {}]>]>} : (tensor<6xi32>) -> tensor<2x3xi32>
  // REPL-NEXT:     %[[RES_SHARDED:.*]] = sdy.reshard %[[RESHAPE_REPL]] <@mesh_custom, [{"b"}, {"c"}]> : tensor<2x3xi32>

  // HALO:          %[[PAD:.*]] = stablehlo.pad %[[SLICE_IN]], %{{.*}}, low = [0], high = [2], interior = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_custom, [{"b", "c"}]>]>} : (tensor<6xi32>, tensor<i32>) -> tensor<8xi32>
  // HALO-NEXT:     %[[MC:.*]] = sdy.manual_computation(%[[PAD]], %{{.*}}) in_shardings=[<@mesh_custom, [{"b", "c"}]>, <@mesh_custom, []>] out_shardings=[<@mesh_custom, [{"b"}, {"c"}]>] manual_axes={"b", "c"} (%[[ARG1:.*]]: tensor<2xi32>, %[[ARG2:.*]]: tensor<i32>) {
  // HALO-NEXT:       %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[3, 2], [2, 1], [1, 0]]> : tensor<3x2xi64>}>
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %[[CP]], %[[ARG1]], dim = 0
  // HALO-NEXT:       %[[PAD_BUF:.*]] = stablehlo.pad %[[CONCAT]], %[[ARG2]], low = [2], high = [2], interior = [0]
  // HALO-NEXT:       %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:       %[[CONV:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // HALO-NEXT:       %[[RESHAPE:.*]] = stablehlo.reshape %[[CONV]] : (tensor<i64>) -> tensor<i64>
  // HALO-NEXT:       %[[C_TABLE:.*]] = stablehlo.constant dense<[3, 2, 1, 0]> : tensor<4xi64>
  // HALO-NEXT:       %[[DS_ID:.*]] = stablehlo.dynamic_slice %[[C_TABLE]], %[[RESHAPE]], sizes = [1]
  // HALO:            sdy.return %{{.*}} : tensor<1x2xi32>
  // HALO-NEXT:     } : (tensor<8xi32>, tensor<i32>) -> tensor<2x4xi32>
  // HALO-NEXT:     %[[RES_SHARDED:.*]] = stablehlo.slice %[[MC]] [0:2, 0:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_custom, [{"b"}, {"c"}]>]>} : (tensor<2x4xi32>) -> tensor<2x3xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_custom, [{"b"}, {"c"}]>]>} : (tensor<6xi32>) -> tensor<2x3xi32>

  // CHECK:         %[[RES:.*]] = sdy.reshard %[[RES_SHARDED]] <@mesh_custom, [{}, {}]> : tensor<2x3xi32>
  %2 = sdy.reshard %1 <@mesh_custom, [{}, {}]> : tensor<2x3xi32>

  // CHECK-NEXT:     return %[[RES]]
  return %2 : tensor<2x3xi32>
}

// CHECK-LABEL: func @reshape_mix_split_combine_halo_impossible(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}, {}]>})
func.func @reshape_mix_split_combine_halo_impossible(%arg0: tensor<6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{"a"}, {}]>}) -> (tensor<4x12xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}, {}]>}) {
  // CHECK-NEXT:     %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh_a_4, [{}, {}]> : tensor<6x8xi32>
  // CHECK-NEXT:     %[[RESHAPE_REPL:.*]] = stablehlo.reshape %[[RESHARD_IN]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_4, [{}, {}]>]>} : (tensor<6x8xi32>) -> tensor<4x12xi32>
  // CHECK-NEXT:     %[[RES_SHARDED:.*]] = sdy.reshard %[[RESHAPE_REPL]] <@mesh_a_4, [{"a"}, {}]> : tensor<4x12xi32>
  // CHECK-NEXT:     %[[RES:.*]] = sdy.reshard %[[RES_SHARDED]] <@mesh_a_4, [{}, {}]> : tensor<4x12xi32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a_4, [{"a"}, {}]>]>} : (tensor<6x8xi32>) -> tensor<4x12xi32>
  %1 = sdy.reshard %0 <@mesh_a_4, [{}, {}]> : tensor<4x12xi32>

  // CHECK-NEXT:     return %[[RES]]
  return %1 : tensor<4x12xi32>
}

//===----------------------------------------------------------------------===//
// stablehlo.reduce_window tests
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// stablehlo.reverse tests
//===----------------------------------------------------------------------===//

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

  // HALO: %[[REV:.*]] = stablehlo.reverse %[[ARG0]], dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_reversed, [{"a"}, {"b"}]>]>} : tensor<4x8xi32>
  // HALO-NEXT: %[[RES:.*]] = sdy.reshard %[[REV]] <@mesh, [{"a"}, {"b"}]> : tensor<4x8xi32>
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
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[LOCAL_IN]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>
  // HALO-NEXT:   %[[OWN_SLICE:.*]] = stablehlo.slice %[[CP]] [2:3, 0:8]
  // HALO-NEXT:   %[[HALO_SLICE:.*]] = stablehlo.slice %[[LOCAL_IN]] [0:2, 0:8]
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[OWN_SLICE]], %[[HALO_SLICE]], dim = 0
  // HALO-NEXT:   sdy.return %[[CONCAT]]
  // HALO-NEXT: } : (tensor<6x8xi32>) -> tensor<6x8xi32>

  // HALO: %[[REV:.*]] = stablehlo.reverse %[[HALO_SHIFT]], dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_reversed, [{"a"}, {"b"}]>]>} : tensor<6x8xi32>
  // HALO-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[REV]] <@mesh, [{"a"}, {"b"}]> : tensor<6x8xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[RESHARD]] [0:5, 0:8]
  %1 = stablehlo.reverse %0, dims = [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
    : tensor<5x8xi32>

  // CHECK: %[[AG:.*]] = sdy.all_gather [{"a"}, {}] %[[RES]]
  %2 = sdy.all_gather [{"a"}, {}] %1 out_sharding=<@mesh, [{}, {"b"}]> : tensor<5x8xi32>
  // CHECK: return %[[AG]] : tensor<5x8xi32>
  return %2 : tensor<5x8xi32>
}

// CHECK-LABEL: func @reverse_multiple_hops
// CHECK-SAME:  (%[[ARG0:.*]]: tensor<4x6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>})
// CHECK-SAME:  -> (tensor<4x6x5xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {}]>}) {
func.func @reverse_multiple_hops(
  %arg0: tensor<4x6x8xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>})
  -> (tensor<4x6x5xi32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {}]>}) {

  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[ARG0]] [0:4, 0:6, 0:5]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
  %0 = stablehlo.slice %arg0 [0:4, 0:6, 0:5]
    {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}
    : (tensor<4x6x8xi32>) -> tensor<4x6x5xi32>

  // REPL: %[[REPL_IN:.*]] = sdy.reshard %[[SLICE]] <@mesh_abc, [{}, {"a"}, {}]>
  // REPL-NEXT: %[[REPL_REV:.*]] = stablehlo.reverse %[[REPL_IN]], dims = [0, 2]
  // REPL-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{}, {"a"}, {}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[REPL_REV]] <@mesh_abc, [{"b"}, {"a"}, {"c"}]>

  // HALO:      %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[PAD:.*]] = stablehlo.pad %[[SLICE]], %[[CST]], low = [0, 0, 0], high = [0, 0, 3]
  // HALO-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b"}, {"a"}, {"c"}]>]>}

  // HALO:      %[[SHIFTED:.*]] = sdy.manual_computation(%[[PAD]])
  // HALO-SAME: manual_axes={"c"}
  // HALO-SAME: (%[[LOCAL_IN:.*]]: tensor<4x6x2xi32>) {
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[LOCAL_IN]])
  // HALO-SAME{LITERAL}:   source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [12, 13], [13, 14], [14, 15]]> : tensor<12x2xi64>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[LOCAL_IN]])
  // HALO-SAME{LITERAL}:   source_target_pairs = dense<[[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]]> : tensor<8x2xi64>
  // HALO-NEXT:   %[[L_SLICE:.*]] = stablehlo.slice %[[CP2]] [0:4, 0:6, 1:2]
  // HALO-NEXT:   %[[H_SLICE:.*]] = stablehlo.slice %[[CP1]] [0:4, 0:6, 0:1]
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[L_SLICE]], %[[H_SLICE]], dim = 2
  // HALO-NEXT:   sdy.return %[[CONCAT]] : tensor<4x6x2xi32>
  // HALO:      } : (tensor<4x6x8xi32>) -> tensor<4x6x8xi32>
  // HALO-NEXT: %[[HALO_REV:.*]] = stablehlo.reverse %[[SHIFTED]], dims = [0, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc_reversed, [{"b"}, {"a"}, {"c"}]>]>} : tensor<4x6x8xi32>
  // HALO-NEXT: %[[HALO_RESHARD:.*]] = sdy.reshard %[[HALO_REV]] <@mesh_abc, [{"b"}, {"a"}, {"c"}]> : tensor<4x6x8xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[HALO_RESHARD]] [0:4, 0:6, 0:5]
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
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[LOCAL_IN]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [12, 13], [13, 14], [14, 15]]> : tensor<12x2xi64>
  // HALO-NEXT:   %[[OWN_SLICE:.*]] = stablehlo.slice %[[CP]] [0:4, 0:5, 1:2]
  // HALO-NEXT:   %[[HALO_SLICE:.*]] = stablehlo.slice %[[LOCAL_IN]] [0:4, 0:5, 0:1]
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[OWN_SLICE]], %[[HALO_SLICE]], dim = 2
  // HALO-NEXT:   sdy.return %[[CONCAT]]
  // HALO:      } : (tensor<4x5x8xi32>) -> tensor<4x5x8xi32>
  // HALO-NEXT: %[[HALO_REV:.*]] = stablehlo.reverse %[[HALO_SHIFT]], dims = [0, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc_reversed, [{"b"}, {"a"}, {"c"}]>]>} : tensor<4x5x8xi32>
  // HALO-NEXT: %[[HALO_RESHARD:.*]] = sdy.reshard %[[HALO_REV]] <@mesh_abc, [{"b"}, {"a"}, {"c"}]> : tensor<4x5x8xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[HALO_RESHARD]] [0:4, 0:5, 0:7]
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
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[LOCAL_IN]])
  // HALO-SAME{LITERAL}:   source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7], [8, 12], [9, 13], [10, 14], [11, 15]]>
  // HALO-NEXT:   %[[S1_0:.*]] = stablehlo.slice %[[CP1]] [1:2, 0:3, 0:2]
  // HALO-NEXT:   %[[S1_1:.*]] = stablehlo.slice %[[LOCAL_IN]] [0:1, 0:3, 0:2]
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[S1_0]], %[[S1_1]], dim = 0
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[CONCAT1]])
  // HALO-SAME{LITERAL}:   source_target_pairs = dense<[[0, 8], [1, 9], [2, 10], [3, 11], [4, 12], [5, 13], [6, 14], [7, 15]]>
  // HALO-NEXT:   %[[S2_0:.*]] = stablehlo.slice %[[CP2]] [0:2, 2:3, 0:2]
  // HALO-NEXT:   %[[S2_1:.*]] = stablehlo.slice %[[CONCAT1]] [0:2, 0:2, 0:2]
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[S2_0]], %[[S2_1]], dim = 1
  // HALO-NEXT:   %[[RECV3_0:.*]] = "stablehlo.collective_permute"(%[[CONCAT2]])
  // HALO-SAME{LITERAL}:   source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [12, 13], [13, 14], [14, 15]]>
  // HALO-NEXT:   %[[RECV3_1:.*]] = "stablehlo.collective_permute"(%[[CONCAT2]])
  // HALO-SAME{LITERAL}:   source_target_pairs = dense<[[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]]>
  // HALO-NEXT:   %[[S3_0:.*]] = stablehlo.slice %[[RECV3_1]] [0:2, 0:3, 2:2]
  // HALO-NEXT:   %[[S3_1:.*]] = stablehlo.slice %[[RECV3_0]] [0:2, 0:3, 0:2]
  // HALO-NEXT:   %[[CONCAT3:.*]] = stablehlo.concatenate %[[S3_0]], %[[S3_1]], dim = 2
  // HALO-NEXT:   sdy.return %[[CONCAT3]] : tensor<2x3x2xi32>
  // HALO:      } : (tensor<4x6x8xi32>) -> tensor<4x6x8xi32>

  // HALO: %[[HALO_REV:.*]] = stablehlo.reverse %[[HALO_SHIFTED]], dims = [0, 1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc_reversed_0, [{"b"}, {"a"}, {"c"}]>]>} : tensor<4x6x8xi32>
  // HALO-NEXT: %[[HALO_IN:.*]] = sdy.reshard %[[HALO_REV]] <@mesh_abc, [{"b"}, {"a"}, {"c"}]> : tensor<4x6x8xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[HALO_IN]] [0:3, 0:5, 0:6]
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
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[LOCAL_IN]])
  // HALO-SAME{LITERAL}:   source_target_pairs = dense<[[0, 1], [1, 8], [2, 3], [3, 10], [4, 5], [5, 12], [6, 7], [7, 14], [8, 9], [10, 11], [12, 13], [14, 15]]> : tensor<12x2xi64>
  // HALO-NEXT:   %[[OWN_SLICE:.*]] = stablehlo.slice %[[CP]] [0:6, 1:2]
  // HALO-NEXT:   %[[HALO_SLICE:.*]] = stablehlo.slice %[[LOCAL_IN]] [0:6, 0:1]
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[OWN_SLICE]], %[[HALO_SLICE]], dim = 1
  // HALO-NEXT:   sdy.return %[[CONCAT]]
  // HALO-NEXT: } : (tensor<6x8xi32>) -> tensor<6x8xi32>

  // HALO:      %[[REV:.*]] = stablehlo.reverse %[[HALO_SHIFTED]], dims = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc_reversed_1, [{}, {"a", "c":(2)2}]>]>} : tensor<6x8xi32>
  // HALO-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[REV]] <@mesh_abc, [{}, {"a", "c":(2)2}]> : tensor<6x8xi32>
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[RESHARD]] [0:6, 0:7]
  %1 = stablehlo.reverse %0, dims = [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{}, {"a", "c":(2)2}]>]>}
    : tensor<6x7xi32>

  // CHECK: %[[AG:.*]] = sdy.all_gather [{}, {"a", "c":(2)2}] %[[RES]]
  %2 = sdy.all_gather [{}, {"a", "c":(2)2}] %1 out_sharding=<@mesh_abc, [{}, {}]> : tensor<6x7xi32>
  // CHECK: return %[[AG]] : tensor<6x7xi32>
  return %2 : tensor<6x7xi32>
}

//===----------------------------------------------------------------------===//
// stablehlo.select_and_scatter tests
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// stablehlo.slice tests
//===----------------------------------------------------------------------===//

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
  // HALO-NEXT:    %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>}>
  // HALO-NEXT:    %[[CONCAT:.*]] = stablehlo.concatenate %[[CP1]], %arg1, dim = 0
  // HALO-NEXT:    %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [2], high = [2], interior = [0]
  // HALO-NEXT:    %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:    %[[CONV:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // HALO-NEXT:    %[[RESHAPE:.*]] = stablehlo.reshape %[[CONV]] : (tensor<i64>) -> tensor<i64>
  // HALO-NEXT:    %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO-NEXT:    %[[DIV:.*]] = stablehlo.divide %[[RESHAPE]], %[[C_DIV]] : tensor<i64>
  // HALO-NEXT:    %[[C_MOD:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO-NEXT:    %[[REM:.*]] = stablehlo.remainder %[[DIV]], %[[C_MOD]] : tensor<i64>
  // HALO-NEXT:    %[[C_STRIDE:.*]] = stablehlo.constant dense<-2> : tensor<i64>
  // HALO-NEXT:    %[[C_OFFSET:.*]] = stablehlo.constant dense<6> : tensor<i64>
  // HALO-NEXT:    %[[MUL:.*]] = stablehlo.multiply %[[REM]], %[[C_STRIDE]] : tensor<i64>
  // HALO-NEXT:    %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_OFFSET]] : tensor<i64>
  // HALO-NEXT:    %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // HALO-NEXT:    %[[MAX:.*]] = stablehlo.maximum %[[ADD]], %[[C_ZERO]] : tensor<i64>
  // HALO-NEXT:    %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD]], %[[MAX]], sizes = [2]
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
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>}>
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %arg1, %[[CP3]], dim = 1
  // HALO-NEXT:   %[[PAD2:.*]] = stablehlo.pad %[[CONCAT2]], %arg2, low = [0, 2, 0], high = [0, 2, 0], interior = [0, 0, 0]
  // HALO-NEXT:   %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[CONV2:.*]] = stablehlo.convert %[[PID2]] : (tensor<ui32>) -> tensor<i64>
  // HALO-NEXT:   %[[RESHAPE2:.*]] = stablehlo.reshape %[[CONV2]] : (tensor<i64>) -> tensor<i64>
  // HALO-NEXT:   %[[C_MOD_S:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO-NEXT:   %[[REM_S:.*]] = stablehlo.remainder %[[RESHAPE2]], %[[C_MOD_S]] : tensor<i64>
  // HALO-NEXT:   %[[C_STRIDE2:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // HALO-NEXT:   %[[C_OFFSET2:.*]] = stablehlo.constant dense<3> : tensor<i64>
  // HALO-NEXT:   %[[MUL2:.*]] = stablehlo.multiply %[[REM_S]], %[[C_STRIDE2]] : tensor<i64>
  // HALO-NEXT:   %[[ADD2:.*]] = stablehlo.add %[[MUL2]], %[[C_OFFSET2]] : tensor<i64>
  // HALO-NEXT:   %[[C_ZERO2:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // HALO-NEXT:   %[[MAX2:.*]] = stablehlo.maximum %[[ADD2]], %[[C_ZERO2]] : tensor<i64>
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.dynamic_slice %[[PAD2]], %[[C_ZERO2]], %[[MAX2]], %[[C_ZERO2]], sizes = [2, 2, 4]
  // HALO-NEXT:   %[[SLICE3:.*]] = stablehlo.slice %[[SLICE2]] [0:2, 0:2, 1:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<2x2x4xi32>) -> tensor<2x2x2xi32>
  // HALO-NEXT:   sdy.return %[[SLICE3]] : tensor<2x2x2xi32>
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
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [4, 2], [5, 3], [6, 4], [7, 5]]> : tensor<6x2xi64>}>
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%arg1)
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[4, 0], [5, 1], [6, 2], [7, 3]]> : tensor<4x2xi64>}>
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %arg1, %[[CP1]], %[[CP2]], dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [1, 0], high = [1, 0], interior = [0, 0]
  // HALO-NEXT:   %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[PID1]] : (tensor<ui32>) -> tensor<i64>
  // HALO-NEXT:   %[[RESHAPE1:.*]] = stablehlo.reshape %[[CONV1]] : (tensor<i64>) -> tensor<i64>
  // HALO-NEXT:   %[[C_DIV_MH:.*]] = stablehlo.constant dense<2> : tensor<i64>
  // HALO-NEXT:   %[[DIV_MH:.*]] = stablehlo.divide %[[RESHAPE1]], %[[C_DIV_MH]] : tensor<i64>
  // HALO-NEXT:   %[[C_MOD_MH:.*]] = stablehlo.constant dense<4> : tensor<i64>
  // HALO-NEXT:   %[[REM_MH:.*]] = stablehlo.remainder %[[DIV_MH]], %[[C_MOD_MH]] : tensor<i64>
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<-1> : tensor<i64>
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<6> : tensor<i64>
  // HALO-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[REM_MH]], %[[C_STRIDE1]] : tensor<i64>
  // HALO-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[C_OFFSET1]] : tensor<i64>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // HALO-NEXT:   %[[MAX1:.*]] = stablehlo.maximum %[[ADD1]], %[[C_ZERO1]] : tensor<i64>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[MAX1]], %[[C_ZERO1]], sizes = [1, 4]
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
