// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s --check-prefixes=CHECK,HALO

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>
// CHECK: @mesh_a4 = <["a"=4, "b"=2]>
sdy.mesh @mesh_a4 = <["a"=4, "b"=2]>

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
  // HALO-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a":(2)2}, {"b"}]>] manual_axes={"a":(2)2, "b"} (%[[ARG1:.*]]: tensor<2x4xi32>, %[[ARG2:.*]]: tensor<i32>) {
  // HALO-NEXT: %[[PART_ID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT: %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT: %[[DIV:.*]] = stablehlo.divide %[[PART_ID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT: %[[C_REM:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT: %[[REM:.*]] = stablehlo.remainder %[[DIV]], %[[C_REM]] : tensor<ui32>
  // HALO-NEXT: %[[CONVERT:.*]] = stablehlo.convert %[[REM]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT: %[[HALO_SLICE:.*]] = stablehlo.slice %[[ARG1]] [1:2, 0:4]
  // HALO-NEXT: %[[CP:.*]] = "stablehlo.collective_permute"(%[[HALO_SLICE]]) <{channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>, source_target_pairs ={{.*}}
  // HALO: %[[SELECT:.*]] = stablehlo.select {{.*}}, %[[CP]], {{.*}}
  // HALO-NEXT: %[[CONCAT:.*]] = stablehlo.concatenate %[[SELECT]], %[[ARG1]], dim = 0 {sdy.sharding ={{.*}}
  // HALO-NEXT: %[[PAD_1:.*]] = stablehlo.pad %[[CONCAT]], %[[ARG2]], low = [3, 0], high = [1, 0]
  // HALO-NEXT: %[[CST_3:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT: %[[CST_4:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT: %[[MUL:.*]] = stablehlo.multiply %[[CONVERT]], %[[CST_3]] : tensor<i32>
  // HALO-NEXT: %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[CST_4]] : tensor<i32>
  // HALO-NEXT: %[[CST_5:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT: %[[V_RES:.*]] = stablehlo.dynamic_slice %[[PAD_1]], %[[ADD]], %[[CST_5]], sizes = [4, 4] {sdy.sharding ={{.*}}
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
  // HALO:            %[[HALO_SLICE:.*]] = stablehlo.slice %arg1 [0:2, 0:4]
  // HALO-NEXT:       %[[CP1:.*]] = "stablehlo.collective_permute"(%[[HALO_SLICE]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>
  // HALO:            %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %arg1, %[[SEL1]], dim = 0
  // HALO-NEXT:       %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [0, 0], high = [5, 0], interior = [0, 0]
  // HALO:            %[[SLICE:.*]] = stablehlo.dynamic_slice %[[PAD]], {{.*}}, sizes = [4, 4]
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
  // HALO:        %[[SLICE1:.*]] = stablehlo.slice %arg1 [2:3, 0:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]> : tensor<6x2xi64>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.slice %arg1 [0:1, 0:4]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [4, 2], [5, 3], [6, 4], [7, 5]]> : tensor<6x2xi64>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, %[[SEL2]], dim = 0
  // HALO-NEXT:   %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [2, 0], high = [2, 0], interior = [0, 0]
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
  // HALO:        %[[SLICE0:.*]] = stablehlo.slice %arg1 [0:2, 0:2, 0:2]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE0]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %arg1, %[[SEL1]], dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [0, 0, 0], high = [5, 0, 0], interior = [0, 0, 0]
  // HALO:        %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], {{.*}}, sizes = [4, 2, 2]
  // HALO-NEXT:   %[[PID2:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_MOD2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[REM2:.*]] = stablehlo.remainder %[[PID2]], %[[C_MOD2]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV2:.*]] = stablehlo.convert %[[REM2]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_DIM1:.*]] = stablehlo.slice %[[SLICE1]] [0:4, 1:2, 0:2]
  // HALO-NEXT:   %[[CP3:.*]] = "stablehlo.collective_permute"(%[[SLICE_DIM1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>
  // HALO:        %[[SEL3:.*]] = stablehlo.select {{.*}}, %[[CP3]], {{.*}}
  // HALO-NEXT:   %[[CONCAT2:.*]] = stablehlo.concatenate %[[SEL3]], %[[SLICE1]], dim = 1
  // HALO-NEXT:   %[[PAD2:.*]] = stablehlo.pad %[[CONCAT2]], %arg2, low = [0, 2, 0], high = [0, 0, 0], interior = [0, 0, 0]
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
  // HALO:        %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV1]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE_L:.*]] = stablehlo.slice %arg1 [3:4, 0:4, 0:3]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [3, 0, 0], high = [1, 0, 0], interior = [0, 0, 0]
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<2> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[CONV1]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[C_OFFSET1]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[ADD1]], %[[C_ZERO1]], %[[C_ZERO1]], sizes = [6, 4, 3]
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
  // HALO:        %[[HALO_SLICE:.*]] = stablehlo.slice %arg1 [0:2, 0:8]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[HALO_SLICE]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [12, 0], high = [0, 0], interior = [0, 0]
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
  // HALO:        %[[SLICE_L:.*]] = stablehlo.slice %arg1 [3:4, 0:4, 0:3]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE_L]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [3, 0, 0], high = [1, 0, 0], interior = [0, 0, 0]
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
  // HALO:        %[[PID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV1:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV1:.*]] = stablehlo.divide %[[PID1]], %[[C_DIV1]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV1:.*]] = stablehlo.convert %[[DIV1]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.slice %arg1 [1:2, 0:4]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.slice %arg1 [0:2, 0:4]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, %[[SEL2]], dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [0, 0], high = [5, 0], interior = [2, 0]
  // HALO-NEXT:   %[[C_STRIDE1:.*]] = stablehlo.constant dense<-2> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET1:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[MUL1:.*]] = stablehlo.multiply %[[CONV1]], %[[C_STRIDE1]] : tensor<i32>
  // HALO-NEXT:   %[[ADD1:.*]] = stablehlo.add %[[MUL1]], %[[C_OFFSET1]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO1:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], %[[ADD1]], %[[C_ZERO1]], sizes = [4, 4]
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
  // HALO:        %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_MOD:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[REM:.*]] = stablehlo.remainder %[[PID]], %[[C_MOD]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV:.*]] = stablehlo.convert %[[REM]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[HALO_SLICE:.*]] = stablehlo.slice %[[ARG1]] [0:8, 0:1]
  // HALO-NEXT:   %[[CP:.*]] = "stablehlo.collective_permute"(%[[HALO_SLICE]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[1, 0], [3, 2]]> : tensor<2x2xi64>
  // HALO:        %[[SEL:.*]] = stablehlo.select {{.*}}, %[[CP]], {{.*}}
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[ARG1]], %[[SEL]], dim = 1
  // HALO-NEXT:   %[[PAD:.*]] = stablehlo.pad %[[CONCAT]], %[[ARG2]], low = [0, 0], high = [0, 2], interior = [0, 0]
  // HALO-NEXT:   %[[C_STRIDE:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[CONV]], %[[C_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE_D:.*]] = stablehlo.dynamic_slice %[[PAD]], %[[C_ZERO]], %[[ADD]], sizes = [8, 5]
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
  // HALO:        %[[SLICE1:.*]] = stablehlo.slice %arg1 [1:2, 0:8]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[SLICE1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]> : tensor<6x2xi64>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[SLICE2:.*]] = stablehlo.slice %arg1 [0:1, 0:8]
  // HALO-NEXT:   %[[CP2:.*]] = "stablehlo.collective_permute"(%[[SLICE2]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1], [4, 2], [5, 3], [6, 4], [7, 5]]> : tensor<6x2xi64>
  // HALO:        %[[SEL2:.*]] = stablehlo.select {{.*}}, %[[CP2]], {{.*}}
  // HALO-NEXT:   %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL1]], %arg1, %[[SEL2]], dim = 0
  // HALO-NEXT:   %[[PAD_INNER:.*]] = stablehlo.pad %[[CONCAT]], %arg2, low = [2, 0], high = [3, 0], interior = [1, 0]
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<3> : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE_INNER:.*]] = stablehlo.dynamic_slice %[[PAD_INNER]], %[[C_OFFSET]], %[[C_ZERO]], sizes = [4, 8]
  // HALO-NEXT:   sdy.return %[[SLICE_INNER]] : tensor<4x8xi32>
  // HALO-NEXT: }
  // HALO-NEXT: %[[RES:.*]] = stablehlo.slice %[[MC]] [0:15, 0:8]
  %1 = stablehlo.pad %0, %c, low = [1, 0], high = [1, 0], interior = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}, {}]>]>} : (tensor<7x8xi32>, tensor<i32>) -> tensor<15x8xi32>

  // CHECK-NEXT: %[[RSD:.*]] = sdy.reshard %[[RES]] <@mesh_a4, [{}, {}]> : tensor<15x8xi32>
  %2 = sdy.reshard %1 <@mesh_a4, [{}, {}]> : tensor<15x8xi32>

  // CHECK: return %[[RSD]] : tensor<15x8xi32>
  return %2 : tensor<15x8xi32>
}
// CHECK-LABEL: func @pad_large_high_padding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xi32>
func.func @pad_large_high_padding(
  %arg0: tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
  -> tensor<14x8xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>

  // REPL-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<4x8xi32>
  // REPL-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD_IN]], %[[CST]], low = [0, 0], high = [10, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[PAD]] <@mesh, [{"a"}, {}]>

  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<2x8xi32>, %arg2: tensor<i32>) {
  // HALO:        %[[HALO_SLICE:.*]] = stablehlo.slice %arg1 [0:2, 0:8]
  // HALO-NEXT:   %[[CP1:.*]] = "stablehlo.collective_permute"(%[[HALO_SLICE]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[2, 0], [3, 1]]> : tensor<2x2xi64>
  // HALO:        %[[SEL1:.*]] = stablehlo.select {{.*}}, %[[CP1]], {{.*}}
  // HALO-NEXT:   %[[CONCAT1:.*]] = stablehlo.concatenate %arg1, %[[SEL1]], dim = 0
  // HALO-NEXT:   %[[PAD1:.*]] = stablehlo.pad %[[CONCAT1]], %arg2, low = [0, 0], high = [10, 0], interior = [0, 0]
  // HALO:        %[[SLICE1:.*]] = stablehlo.dynamic_slice %[[PAD1]], {{.*}}, sizes = [7, 8]
  // HALO-NEXT:   sdy.return %[[SLICE1]] : tensor<7x8xi32>
  // HALO-NEXT: }
  %0 = stablehlo.pad %arg0, %c, low = [0, 0], high = [10, 0], interior = [0, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<4x8xi32>, tensor<i32>) -> tensor<14x8xi32>

  // CHECK-NEXT: return %[[RES]] : tensor<14x8xi32>
  return %0 : tensor<14x8xi32>
}

// CHECK-LABEL: func @pad_sharded_negative_low_with_interior_pad
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x4xi32>
func.func @pad_sharded_negative_low_with_interior_pad(
  %arg0: tensor<2x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
  -> tensor<2x4xi32> {
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  %c = stablehlo.constant dense<-1> : tensor<i32>

  // REPL-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {"b"}]> : tensor<2x4xi32>
  // REPL-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]], %[[CST]], low = [-1, 0], high = [0, 0], interior = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>} : (tensor<2x4xi32>, tensor<i32>) -> tensor<2x4xi32>
  // REPL-NEXT: %[[RES:.*]] = sdy.reshard %[[PAD]] <@mesh, [{"a"}, {"b"}]> : tensor<2x4xi32>

  // HALO-NEXT: %[[RES:.*]] = sdy.manual_computation(%[[ARG0]], %[[CST]]) in_shardings=[<@mesh, [{"a"}, {"b"}]>, <@mesh, []>] out_shardings=[<@mesh, [{"a"}, {"b"}]>] manual_axes={"a", "b"} (%[[ARG1:.*]]: tensor<1x2xi32>, %[[ARG2:.*]]: tensor<i32>) {
  // HALO-NEXT:   %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:   %[[C_DIV:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // HALO-NEXT:   %[[DIV:.*]] = stablehlo.divide %[[PID]], %[[C_DIV]] : tensor<ui32>
  // HALO-NEXT:   %[[CONV:.*]] = stablehlo.convert %[[DIV]] : (tensor<ui32>) -> tensor<i32>
  // HALO-NEXT:   %[[PAD_INNER:.*]] = stablehlo.pad %[[ARG1]], %[[ARG2]], low = [0, 0], high = [1, 0], interior = [1, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<1x2xi32>, tensor<i32>) -> tensor<2x2xi32>
  // HALO-NEXT:   %[[C_STRIDE:.*]] = stablehlo.constant dense<-1> : tensor<i32>
  // HALO-NEXT:   %[[C_OFFSET:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // HALO-NEXT:   %[[MUL:.*]] = stablehlo.multiply %[[CONV]], %[[C_STRIDE]] : tensor<i32>
  // HALO-NEXT:   %[[ADD:.*]] = stablehlo.add %[[MUL]], %[[C_OFFSET]] : tensor<i32>
  // HALO-NEXT:   %[[C_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // HALO-NEXT:   %[[SLICE:.*]] = stablehlo.dynamic_slice %[[PAD_INNER]], %[[ADD]], %[[C_ZERO]], sizes = [1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<2x2xi32>, tensor<i32>, tensor<i32>) -> tensor<1x2xi32>
  // HALO-NEXT:   sdy.return %[[SLICE]] : tensor<1x2xi32>
  // HALO-NEXT: } : (tensor<2x4xi32>, tensor<i32>) -> tensor<2x4xi32>
  %0 = stablehlo.pad %arg0, %c, low = [-1, 0], high = [0, 0], interior = [1, 0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>} : (tensor<2x4xi32>, tensor<i32>) -> tensor<2x4xi32>

  // CHECK-NEXT: return %[[RES]] : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}
