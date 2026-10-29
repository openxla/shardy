// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s --check-prefixes=CHECK,REPL
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s --check-prefixes=CHECK,HALO

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>
// CHECK: @mesh_a4 = <["a"=4, "b"=2]>
sdy.mesh @mesh_a4 = <["a"=4, "b"=2]>
// CHECK: @mesh_a6 = <["a"=6]>
sdy.mesh @mesh_a6 = <["a"=6]>
// CHECK: @mesh_a_4 = <["a"=4]>
sdy.mesh @mesh_a_4 = <["a"=4]>
// CHECK: @mesh_xy_8 = <["x"=8, "y"=8]>
sdy.mesh @mesh_xy_8 = <["x"=8, "y"=8]>
// CHECK: @mesh_custom = <["b"=2, "c"=2], device_ids=[3, 2, 1, 0]>
sdy.mesh @mesh_custom = <["b"=2, "c"=2], device_ids=[3, 2, 1, 0]>

// CHECK-LABEL: func @reshape_replicated_operand_missing_result_sharding(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<6xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}]>})
func.func @reshape_replicated_operand_missing_result_sharding(%arg0: tensor<6xi32> {sdy.sharding = #sdy.sharding<@mesh_a_4, [{}]>}) -> tensor<2x3xi32> {
  // CHECK-NEXT:     %[[RES:.*]] = stablehlo.reshape %[[ARG0]] : (tensor<6xi32>) -> tensor<2x3xi32>
  %0 = stablehlo.reshape %arg0 : (tensor<6xi32>) -> tensor<2x3xi32>

  // CHECK-NEXT:     return %[[RES]]
  return %0 : tensor<2x3xi32>
}

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
  // HALO:            %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  // HALO:            %[[SEL:.*]] = stablehlo.select {{.*}}, %[[CP]], {{.*}}
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL]], %[[ARG1]], dim = 0
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
  // HALO:            %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]> : tensor<6x2xi64>
  // HALO:            %[[SEL:.*]] = stablehlo.select {{.*}}, %[[CP]], {{.*}}
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL]], %[[ARG1]], dim = 0
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

// CHECK-LABEL: func @reshape_2d_split_with_unreduced_max(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {}], unreduced=max{"b"}>})
func.func @reshape_2d_split_with_unreduced_max(%arg0: tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{"a"}, {}], unreduced=max{"b"}>}) -> (tensor<2x3x4xi32> {sdy.sharding = #sdy.sharding<@mesh_a4, [{}, {}, {}], unreduced=max{"b"}>}) {
  // CHECK:         %[[SLICE_IN:.*]] = stablehlo.slice %[[ARG0]] [0:6, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}, {}], unreduced=max{"b"}>]>} : (tensor<8x4xi32>) -> tensor<6x4xi32>
  %0 = stablehlo.slice %arg0 [0:6, 0:4] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a"}, {}], unreduced=max{"b"}>]>} : (tensor<8x4xi32>) -> tensor<6x4xi32>

  // REPL:          %[[RESHARD_IN:.*]] = sdy.reshard %[[SLICE_IN]] <@mesh_a4, [{}, {}], unreduced=max{"b"}> : tensor<6x4xi32>
  // REPL-NEXT:     %[[RESHAPE_REPL:.*]] = stablehlo.reshape %[[RESHARD_IN]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}], unreduced=max{"b"}>]>} : (tensor<6x4xi32>) -> tensor<2x3x4xi32>
  // REPL-NEXT:     %[[RES_SHARDED:.*]] = sdy.reshard %[[RESHAPE_REPL]] <@mesh_a4, [{"a":(1)2}, {"a":(2)2}, {}], unreduced=max{"b"}> : tensor<2x3x4xi32>

  // HALO:          %[[PAD:.*]] = stablehlo.pad %[[SLICE_IN]], %{{.*}}, low = [0, 0], high = [2, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a"}, {}], unreduced=max{"b"}>]>} : (tensor<6x4xi32>, tensor<i32>) -> tensor<8x4xi32>
  // HALO-NEXT:     %[[MC:.*]] = sdy.manual_computation(%[[PAD]], %{{.*}}) in_shardings=[<@mesh_a4, [{"a"}, {}], unreduced=max{"b"}>, <@mesh_a4, []>] out_shardings=[<@mesh_a4, [{"a":(1)2}, {"a":(2)2}, {}], unreduced=max{"b"}>] manual_axes={"a"} (%[[ARG1:.*]]: tensor<2x4xi32>, %[[ARG2:.*]]: tensor<i32>) {
  // HALO:            %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]> : tensor<6x2xi64>
  // HALO:            %[[SEL:.*]] = stablehlo.select {{.*}}, %[[CP]], {{.*}}
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL]], %[[ARG1]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}], unreduced=max{"b"}>]>}
  // HALO-NEXT:       %[[PAD_BUF:.*]] = stablehlo.pad %[[CONCAT]], %[[ARG2]], low = [2, 0], high = [2, 0], interior = [0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}], unreduced=max{"b"}>]>}
  // HALO:            %[[SLICE_BUF:.*]] = stablehlo.dynamic_slice %[[PAD_BUF]], {{.*}} {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}], unreduced=max{"b"}>]>}
  // HALO-NEXT:       %[[RESHAPE_LOCAL:.*]] = stablehlo.reshape %[[SLICE_BUF]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{}, {}, {}], unreduced=max{"b"}>]>} : (tensor<2x4xi32>) -> tensor<1x2x4xi32>
  // HALO-NEXT:       sdy.return %[[RESHAPE_LOCAL]] : tensor<1x2x4xi32>
  // HALO-NEXT:     }
  // HALO-NEXT:     %[[RES_SHARDED:.*]] = stablehlo.slice %[[MC]] [0:2, 0:3, 0:4] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a4, [{"a":(1)2}, {"a":(2)2}, {}], unreduced=max{"b"}>]>} : (tensor<2x4x4xi32>) -> tensor<2x3x4xi32>
  %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_a4, [{"a":(1)2}, {"a":(2)2}, {}], unreduced=max{"b"}>]>} : (tensor<6x4xi32>) -> tensor<2x3x4xi32>

  // CHECK:         %[[RES:.*]] = sdy.reshard %[[RES_SHARDED]] <@mesh_a4, [{}, {}, {}], unreduced=max{"b"}> : tensor<2x3x4xi32>
  %2 = sdy.reshard %1 <@mesh_a4, [{}, {}, {}], unreduced=max{"b"}> : tensor<2x3x4xi32>

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
  // HALO:            %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]> : tensor<5x2xi64>
  // HALO:            %[[SEL:.*]] = stablehlo.select {{.*}}, %[[CP]], {{.*}}
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL]], %[[ARG1]], dim = 0
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
  // HALO:            %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  // HALO:            %[[SEL:.*]] = stablehlo.select {{.*}}, %[[CP]], {{.*}}
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL]], %[[ARG1]], dim = 1
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
  // HALO:            %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // HALO-NEXT:       %[[C_TABLE:.*]] = stablehlo.constant dense<[3, 2, 1, 0]> : tensor<4xi32>
  // HALO-NEXT:       %[[DS_ID:.*]] = stablehlo.dynamic_slice %[[C_TABLE]], %[[PID]], sizes = [1]
  // HALO-NEXT:       %[[RESHAPE_ID:.*]] = stablehlo.reshape %[[DS_ID]]
  // HALO-NEXT:       %[[CP:.*]] = "stablehlo.collective_permute"(%[[ARG1]])
  // HALO-SAME{LITERAL}: source_target_pairs = dense<[[3, 2], [2, 1], [1, 0]]> : tensor<3x2xi64>
  // HALO:            %[[SEL:.*]] = stablehlo.select {{.*}}, %[[CP]], {{.*}}
  // HALO-NEXT:       %[[CONCAT:.*]] = stablehlo.concatenate %[[SEL]], %[[ARG1]], dim = 0
  // HALO-NEXT:       %[[PAD_BUF:.*]] = stablehlo.pad %[[CONCAT]], %[[ARG2]], low = [2], high = [2], interior = [0]
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
