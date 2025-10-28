// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>

// CHECK-LABEL: func @reshape
func.func @reshape(%arg0: tensor<16x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}, {"x"}]> : tensor<16x2x4xf32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<16x2x4xf32>) -> tensor<16x8xf32>
  // CHECK-NEXT: return %[[RESHAPE]] : tensor<16x8xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<16x2x4xf32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_x_to_x_and_x_fits_exactly_to_first_dim
func.func @reshape_simple_merge_sharding_is_from_x_to_x_and_x_fits_exactly_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_x_to_y_and_x_fits_exactly_to_first_dim
func.func @reshape_simple_merge_sharding_is_from_x_to_y_and_x_fits_exactly_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{"y"}]> : tensor<32xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_y_to_y_and_y_underfits_to_first_dim
func.func @reshape_simple_merge_sharding_is_from_y_to_y_and_y_underfits_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_y_to_x_and_y_underfits_to_first_dim
func.func @reshape_simple_merge_sharding_is_from_y_to_x_and_y_underfits_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: return %[[RESHAPE]] : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_xy_to_xy_and_x_fits_exactly_to_first_dim
func.func @reshape_simple_merge_sharding_is_from_xy_to_xy_and_x_fits_exactly_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_xy_to_yx_and_x_fits_exactly_to_first_dim
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_simple_merge_sharding_is_from_xy_to_yx_and_x_fits_exactly_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}]>}) {
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{"y", "x"}]> : tensor<32xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_xy_to_x_and_x_fits_exactly_to_first_dim
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_simple_merge_sharding_is_from_xy_to_x_and_x_fits_exactly_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{"x"}]> : tensor<32xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_sharding_is_from_yx_to_yx_and_y_underfits_to_first_dim
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_simple_merge_sharding_is_from_sharding_is_from_yx_to_yx_and_y_underfits_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"y", "x":(1)2}, {"x":(2)2}]> : tensor<4x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: return %1 : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_yx_to_y_and_y_underfits_to_first_dim
func.func @reshape_simple_merge_sharding_is_from_yx_to_y_and_y_underfits_to_first_dim(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"y"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: return %1 : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_only_second_dim_is_sharded
func.func @reshape_simple_merge_sharding_is_from_only_second_dim_is_sharded(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: return %1 : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_only_second_dim_is_sharded_and_result_is_sharded
func.func @reshape_simple_merge_sharding_is_from_only_second_dim_is_sharded_and_result_is_sharded(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  // CHECK-NEXT: return %1 : tensor<32xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_of_subaxes
func.func @reshape_simple_merge_sharding_is_from_of_subaxes(%arg0: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"x":(2)2, "y"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<2x8xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @reshape_simple_merge_sharding_is_from_of_subaxes_only_result_is_sharded
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_simple_merge_sharding_is_from_of_subaxes_only_result_is_sharded(%arg0: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x":(1)2}, {"x":(2)2, "y"}]> : tensor<2x8xf32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<2x8xf32>) -> tensor<16xf32>
  // CHECK-NEXT: return %[[RESHAPE]] : tensor<16xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}]>]>} : (tensor<2x8xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}

// CHECK-LABEL: func.func @reshape_simple_split_sharding_is_to_x_from_x_and_x_fits_exactly_to_first_dim
func.func @reshape_simple_split_sharding_is_to_x_from_x_and_x_fits_exactly_to_first_dim(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @reshape_simple_split_sharding_is_to_y_from_y_and_y_underfits_to_first_dim
func.func @reshape_simple_split_sharding_is_to_y_from_y_and_y_underfits_to_first_dim(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @reshape_simple_split_sharding_is_to_xy_from_xy_and_x_fits_exactly_to_first_dim
func.func @reshape_simple_split_sharding_is_to_xy_from_xy_and_x_fits_exactly_to_first_dim(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @reshape_simple_split_sharding_is_to_yx_from_yx_and_y_underfits_to_first_dim
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_simple_split_sharding_is_to_yx_from_yx_and_y_underfits_to_first_dim(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {"x":(2)2}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{"y"}, {"x"}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %1 : tensor<4x8xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @reshape_simple_split_sharding_is_to_only_second_dim_is_sharded
func.func @reshape_simple_split_sharding_is_to_only_second_dim_is_sharded(%arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x"}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %1 : tensor<4x8xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<32xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @reshape_simple_split_sharding_is_to_to_subaxes
func.func @reshape_simple_split_sharding_is_to_to_subaxes(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}]>}) -> (tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {"x":(2)2, "y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"x":(2)2, "y"}]>]>} : (tensor<16xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// CHECK-LABEL: func @reshape_strided_view_on_both_operand_and_result
func.func @reshape_strided_view_on_both_operand_and_result(%arg0: tensor<2x2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) -> (tensor<4x2x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"y"}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{}, {}, {}]> : tensor<2x2x4xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<2x2x4xf32>) -> tensor<4x2x2xf32>
  // CHECK-NEXT: %2 = sdy.reshard %1 <@mesh, [{}, {}, {"y"}]> : tensor<4x2x2xf32>
  // CHECK-NEXT: return %2 : tensor<4x2x2xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"y"}]>]>} : (tensor<2x2x4xf32>) -> tensor<4x2x2xf32>
  return %0 : tensor<4x2x2xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_x_to_x_merged_dimensions_are_sharded
func.func @reshape_ij_k_to_i_jk_and_x_to_x_merged_dimensions_are_sharded(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_x_to_x_singleton_dimensions_are_sharded
func.func @reshape_ij_k_to_i_jk_and_x_to_x_singleton_dimensions_are_sharded(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {"x"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_xy_to_xy
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_xy_to_xy(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x", "y"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_xy_to_yx
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_xy_to_yx(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"y", "x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_yx_to_xy
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_yx_to_xy(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {"x":(2)2}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x", "y"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_yx_to_yx
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_yx_to_yx(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {"x":(2)2}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"y", "x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_xy_to_x
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_xy_to_x(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {"x"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_xy_to_y
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_xy_to_y(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_yx_to_x
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_yx_to_x(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {"x":(2)2}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_yx_to_y
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_yx_to_y(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y", "x":(1)2}, {"x":(2)2}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_x_to_z_singleton_dims_are_sharded
func.func @reshape_ij_k_to_i_jk_and_x_to_z_singleton_dims_are_sharded(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x"}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh_xyz, [{"z"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_x_to_z_factor_j_is_sharded
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_x_to_z_factor_j_is_sharded(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"x"}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {"y"}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh_xyz, [{"z", "y"}, {"x"}]> : tensor<8x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {"y", "x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %2 = sdy.reshard %1 <@mesh_xyz, [{"z"}, {"y"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %2 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {"y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_x_to_xy
func.func @reshape_ij_k_to_i_jk_and_x_to_xy(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x", "y"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_x_to_yx
// NOTE: It reshards this way because the dependencies are dropped as factors are fully-sharded.
func.func @reshape_ij_k_to_i_jk_and_x_to_yx(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}]>}) {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{"x", "y"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %1 = stablehlo.reshape %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %2 = sdy.reshard %1 <@mesh, [{}, {"y", "x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %2 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_y_to_xy
func.func @reshape_ij_k_to_i_jk_and_y_to_xy(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x", "y"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_ij_k_to_i_jk_and_y_to_yx
func.func @reshape_ij_k_to_i_jk_and_y_to_yx(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"y", "x"}]> : tensor<4x16xf32>
  // CHECK-NEXT: return %1 : tensor<4x16xf32>
  %0 = stablehlo.reshape %arg0  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y", "x"}]>]>} : (tensor<8x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @reshape_size_1_dimensions_1
func.func @reshape_size_1_dimensions_1(
    %arg0: tensor<1x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>})
    -> (tensor<4x1xi32>  {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<1x4xi32>) -> tensor<4x1xi32>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x"}]> : tensor<4x1xi32>
  // CHECK-NEXT: return %1 : tensor<4x1xi32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<1x4xi32>) -> tensor<4x1xi32>
  return %0 : tensor<4x1xi32>
}

// CHECK-LABEL: func @reshape_size_1_dimensions_2
func.func @reshape_size_1_dimensions_2(
    %arg0: tensor<1x4xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<4x1xi32>  {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {"x"}]> : tensor<4x1xi32>
  // CHECK-NEXT: return %1 : tensor<4x1xi32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : (tensor<1x4xi32>) -> tensor<4x1xi32>
  return %0 : tensor<4x1xi32>
}

// TODO(enver): Add a unit test for overflow axes on reshapes.
