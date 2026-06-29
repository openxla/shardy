// RUN: sdy_opt %s -sdy-split-resharding-dimensions | FileCheck %s

sdy.mesh @mesh2d = <["x"=2, "y"=2]>
sdy.mesh @mesh2d_2x8 = <["x"=2, "y"=8]>
sdy.mesh @mesh3d = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh3d_4x2x4 = <["x"=4, "y"=2, "z"=4]>
sdy.mesh @mesh4d = <["x"=2, "y"=2, "z"=2, "w"=2]>
sdy.mesh @mesh3d_2x2x8 = <["x"=2, "y"=2, "z"=8]>

// CHECK-LABEL: func @input_dim_split_and_swapped_to_non_adjacent_dims
func.func @input_dim_split_and_swapped_to_non_adjacent_dims(%arg0 : tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d_4x2x4, [{}, {"y", "x"}, {}]>}) -> tensor<16x8x8xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d_4x2x4, [{}, {"y"}, {"x"}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<16x2x4x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE_0]] <@mesh3d_4x2x4, [{"x"}, {}, {}, {"y"}]> : tensor<16x2x4x8xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d_4x2x4, [{"x"}, {}, {"y"}]>]>} : (tensor<16x2x4x8xf32>) -> tensor<16x8x8xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.reshard %arg0 <@mesh3d_4x2x4, [{"x"}, {}, {"y"}]> : tensor<16x8x8xf32>
  return %0 : tensor<16x8x8xf32>
}

// CHECK-LABEL: func @input_dim_split_into_four_groups
func.func @input_dim_split_into_four_groups(%arg0: tensor<32x2x2x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh4d, [{"x", "y", "z", "w"}, {}, {}, {}, {}]>}) -> tensor<32x2x2x2x2xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh4d, [{"x"}, {"y"}, {"z"}, {"w"}, {}, {}, {}, {}]>]>} : (tensor<32x2x2x2x2xf32>) -> tensor<2x2x2x4x2x2x2x2xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE_0]] <@mesh4d, [{}, {}, {}, {}, {"x"}, {"y"}, {"z"}, {"w"}]> : tensor<2x2x2x4x2x2x2x2xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh4d, [{}, {"x"}, {"y"}, {"z"}, {"w"}]>]>} : (tensor<2x2x2x4x2x2x2x2xf32>) -> tensor<32x2x2x2x2xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.reshard %arg0 <@mesh4d, [{}, {"x"}, {"y"}, {"z"}, {"w"}]> : tensor<32x2x2x2x2xf32>
  return %0 : tensor<32x2x2x2x2xf32>
}

// CHECK-LABEL: func @input_dim_split_into_three_groups
func.func @input_dim_split_into_three_groups(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"x", "y", "z"}, {}, {}]>}) -> tensor<16x8x8xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d, [{"x"}, {"y"}, {"z"}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x2x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE_0]] <@mesh3d, [{}, {"y"}, {}, {"x"}, {"z"}]> : tensor<2x2x4x8x8xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d, [{"y"}, {"x"}, {"z"}]>]>} : (tensor<2x2x4x8x8xf32>) -> tensor<16x8x8xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.reshard %arg0 <@mesh3d, [{"y"}, {"x"}, {"z"}]> : tensor<16x8x8xf32>
  return %0 : tensor<16x8x8xf32>
}

// CHECK-LABEL: func @input_dim_split_with_subaxes
func.func @input_dim_split_with_subaxes(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d_2x8, [{"y"}, {}, {}]>}) -> tensor<16x8x8xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2d_2x8, [{"y":(1)2}, {"y":(2)4}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<2x8x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE_0]] <@mesh2d_2x8, [{}, {}, {"y":(1)2}, {"y":(2)4}]> : tensor<2x8x8x8xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2d_2x8, [{}, {"y":(1)2}, {"y":(2)4}]>]>} : (tensor<2x8x8x8xf32>) -> tensor<16x8x8xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.reshard %arg0 <@mesh2d_2x8, [{}, {"y":(1)2}, {"y":(2)4}]> : tensor<16x8x8xf32>
  return %0 : tensor<16x8x8xf32>
}

// CHECK-LABEL: func @input_dim_split_and_swapped
func.func @input_dim_split_and_swapped(%arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{"x", "y"}, {}, {}]>}) -> tensor<8x8x8xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh2d, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<8x8x8xf32>) -> tensor<2x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE_0]] <@mesh2d, [{}, {"y"}, {"x"}, {}]> : tensor<2x4x8x8xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2d, [{"y"}, {"x"}, {}]>]>} : (tensor<2x4x8x8xf32>) -> tensor<8x8x8xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.reshard %arg0 <@mesh2d, [{"y"}, {"x"}, {}]> : tensor<8x8x8xf32>
  return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func @corresponding_dim_has_cross_dim_subaxes_not_split
func.func @corresponding_dim_has_cross_dim_subaxes_not_split(%arg0 : tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d_2x2x8, [{"x", "y"}, {"z"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh3d_2x2x8, [{"z":(1)2}, {"y", "z":(2)4}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh3d_2x2x8, [{"z":(1)2}, {"y", "z":(2)4}]> : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @equivalent_sharding_no_split
func.func @equivalent_sharding_no_split(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{"x", "y"}]>}) -> tensor<4xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh2d, [{"x", "y"}]> : tensor<4xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh2d, [{"x", "y"}]> : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @axes_map_to_same_target_dim_no_split
func.func @axes_map_to_same_target_dim_no_split(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh2d, [{"x", "y"}, {}]>}) -> tensor<4x4xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh2d, [{}, {"x", "y"}]> : tensor<4x4xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh2d, [{}, {"x", "y"}]> : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @output_dim_split_into_two_groups_with_subaxes
func.func @output_dim_split_into_two_groups_with_subaxes(%arg0: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh3d_2x2x8, [{"z":(1)2}, {"y"}]>}) -> tensor<4x2xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d_2x2x8, [{"z":(1)2}, {}, {"y"}]>]>} : (tensor<4x2xf32>) -> tensor<2x2x2xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE_0]] <@mesh3d_2x2x8, [{"z":(1)2}, {"y"}, {}]> : tensor<2x2x2xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d_2x2x8, [{"z":(1)2, "y"}, {}]>]>} : (tensor<2x2x2xf32>) -> tensor<4x2xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.reshard %arg0 <@mesh3d_2x2x8, [{"z":(1)2, "y"}, {}]> : tensor<4x2xf32>
  return %0 : tensor<4x2xf32>
}

// CHECK-LABEL: func @output_dim_split_into_two_groups_with_axis_merge
func.func @output_dim_split_into_two_groups_with_axis_merge(%arg0: tensor<8x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh3d_2x2x8, [{"z":(1)2}, {"y"}, {"z":(2)4}]>}) -> tensor<8x2x4xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d_2x2x8, [{"z":(1)2}, {}, {"y"}, {"z":(2)4}]>]>} : (tensor<8x2x4xf32>) -> tensor<2x4x2x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE_0]] <@mesh3d_2x2x8, [{"z":(1)2}, {"z":(2)4}, {"y"}, {}]> : tensor<2x4x2x4xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d_2x2x8, [{"z"}, {"y"}, {}]>]>} : (tensor<2x4x2x4xf32>) -> tensor<8x2x4xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.reshard %arg0 <@mesh3d_2x2x8, [{"z"}, {"y"}, {}]> : tensor<8x2x4xf32>
  return %0 : tensor<8x2x4xf32>
}

// CHECK-LABEL: func @output_dim_split_into_two_groups
func.func @output_dim_split_into_two_groups(%arg0: tensor<4x2x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"y", "z"}, {"x"}, {}]>}) -> tensor<4x2x8xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d, [{"y", "z"}, {"x"}, {}, {}]>]>} : (tensor<4x2x8xf32>) -> tensor<4x2x2x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE_0]] <@mesh3d, [{}, {}, {"x"}, {"y", "z"}]> : tensor<4x2x2x4xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d, [{}, {}, {"x", "y", "z"}]>]>} : (tensor<4x2x2x4xf32>) -> tensor<4x2x8xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.reshard %arg0 <@mesh3d, [{}, {}, {"x", "y", "z"}]> : tensor<4x2x8xf32>
  return %0 : tensor<4x2x8xf32>
}

// CHECK-LABEL: func @target_dim_has_extra_axes_divisible_split
func.func @target_dim_has_extra_axes_divisible_split(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh3d_4x2x4, [{"x", "y"}, {}]>}) -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d_4x2x4, [{"x"}, {"y"}, {}, {}]>]>} : (tensor<8x16xf32>) -> tensor<4x2x4x4xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE_0]] <@mesh3d_4x2x4, [{}, {}, {"x"}, {"z"}]> : tensor<4x2x4x4xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d_4x2x4, [{}, {"x", "z"}]>]>} : (tensor<4x2x4x4xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.reshard %arg0 <@mesh3d_4x2x4, [{}, {"x", "z"}]> : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @target_dim_has_extra_axes_indivisible_not_split
func.func @target_dim_has_extra_axes_indivisible_not_split(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d_4x2x4, [{"x"}, {"z"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh3d_4x2x4, [{"z"}, {"x", "y"}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh3d_4x2x4, [{"z"}, {"x", "y"}]> : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @target_dim_has_no_cross_dim_axes_split
func.func @target_dim_has_no_cross_dim_axes_split(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"x", "y"}, {"z"}]>}) -> tensor<4x4xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d, [{"x", "y"}, {}, {"z"}]>]>} : (tensor<4x4xf32>) -> tensor<4x2x2xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE_0]] <@mesh3d, [{}, {"x"}, {"z"}]> : tensor<4x2x2xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d, [{}, {"x", "z"}]>]>} : (tensor<4x2x2xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.reshard %arg0 <@mesh3d, [{}, {"x", "z"}]> : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @target_dim_has_cross_dim_axes_not_split
func.func @target_dim_has_cross_dim_axes_not_split(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"x", "y"}, {"z"}]>}) -> tensor<4x4xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh3d, [{"y"}, {"x", "z"}]> : tensor<4x4xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh3d, [{"y"}, {"x", "z"}]> : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @corresponding_dim_has_extra_axes_divisible_split
func.func @corresponding_dim_has_extra_axes_divisible_split(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d_4x2x4, [{"x", "z"}, {}, {}]>}) -> tensor<16x8x8xf32> {
  // CHECK-NEXT: %[[RESHAPE_0:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d_4x2x4, [{"x"}, {"z"}, {}, {}]>]>} : (tensor<16x8x8xf32>) -> tensor<4x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE_0]] <@mesh3d_4x2x4, [{}, {"y"}, {"x"}, {}]> : tensor<4x4x8x8xf32>
  // CHECK-NEXT: %[[RESHAPE_1:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh3d_4x2x4, [{"y"}, {"x"}, {}]>]>} : (tensor<4x4x8x8xf32>) -> tensor<16x8x8xf32>
  // CHECK-NEXT: return %[[RESHAPE_1]]
  %0 = sdy.reshard %arg0 <@mesh3d_4x2x4, [{"y"}, {"x"}, {}]> : tensor<16x8x8xf32>
  return %0 : tensor<16x8x8xf32>
}

// CHECK-LABEL: func @corresponding_dim_has_extra_axes_indivisible_not_split
func.func @corresponding_dim_has_extra_axes_indivisible_not_split(%arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh3d_4x2x4, [{"x", "y"}, {}, {}]>}) -> tensor<8x8x8xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh3d_4x2x4, [{"z"}, {"x"}, {}]> : tensor<8x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh3d_4x2x4, [{"z"}, {"x"}, {}]> : tensor<8x8x8xf32>
  return %0 : tensor<8x8x8xf32>
}

// CHECK-LABEL: func @corresponding_dim_has_cross_dim_axes_not_split
func.func @corresponding_dim_has_cross_dim_axes_not_split(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh3d, [{"x", "y"}, {"z"}]>}) -> tensor<4x4xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh3d, [{"z"}, {"y"}]> : tensor<4x4xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh3d, [{"z"}, {"y"}]> : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @corresponding_dim_has_extra_axes_but_no_replicated_groups_not_split
func.func @corresponding_dim_has_extra_axes_but_no_replicated_groups_not_split(%arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh4d, [{"x", "y"}, {"z"}]>}) -> tensor<4x4xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh4d, [{"w"}, {"x", "y"}]> : tensor<4x4xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh4d, [{"w"}, {"x", "y"}]> : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
