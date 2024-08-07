// RUN: sdy_opt %s -sdy-update-non-divisible-input-output-shardings | FileCheck %s

sdy.mesh @mesh_x_4_y_2 = <"x"=4, "y"=2>
sdy.mesh @mesh_x_8_y_3 = <"x"=8, "y"=3>
sdy.mesh @mesh_x_16 = <"x"=16>
sdy.mesh @mesh_x_24 = <"x"=24>

// CHECK-LABEL: func @only_one_dim_modified
// CHECK-SAME:    %arg0: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x":(1)2}, {"y"}]>}
func.func @only_one_dim_modified(%arg0: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x"}, {"y"}]>}) -> tensor<2x2xf32> {
  return %arg0 : tensor<2x2xf32>
}

// CHECK-LABEL: func @multiple_dims_modified
// CHECK-SAME:    %arg0: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x":(1)2}, {}]>}
func.func @multiple_dims_modified(%arg0: tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x"}, {"y"}]>}) -> tensor<2x3xf32> {
  return %arg0 : tensor<2x3xf32>
}

// CHECK-LABEL: func @intermediate_with_sharding
// CHECK-SAME:    %arg0: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x":(1)2}, {"y"}]>}
func.func @intermediate_with_sharding(%arg0: tensor<2x2xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x"}, {"y"}]>}) -> tensor<2x2xf32> {
  // CHECK  stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x_4_y_2, [{"x"}, {"y"}]>]>} : tensor<2x2xf32>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x_4_y_2, [{"x"}, {"y"}]>]>} : tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: func @preserve_replicated_axes
// CHECK-SAME:    %arg0: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x":(1)2}], replicated={"y"}>}
func.func @preserve_replicated_axes(%arg0: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x"}], replicated={"y"}>}) -> tensor<2xf32> {
  return %arg0 : tensor<2xf32>
}

// CHECK-LABEL: func @result_sharding
// CHECK-SAME:    %arg0: tensor<2xf32>)
// CHECK-SAME:  -> (tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x":(1)2}], replicated={"y"}>})
func.func @result_sharding(%arg0: tensor<2xf32>) -> (tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x"}], replicated={"y"}>}) {
  // CHECK: return %arg0 : tensor<2xf32>
  return %arg0 : tensor<2xf32>
}

// CHECK-LABEL: func @no_gcd
// CHECK-SAME:    %arg0: tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{}]>}
func.func @no_gcd(%arg0: tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x"}]>}) -> tensor<3xf32> {
  return %arg0 : tensor<3xf32>
}

// CHECK-LABEL: func @matching_axis_size
// CHECK-SAME:    %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x"}]>}
func.func @matching_axis_size(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x"}]>}) -> tensor<4xf32> {
  return %arg0 : tensor<4xf32>
}

// CHECK-LABEL: func @open_dim_preserved
// CHECK-SAME:    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x", ?}]>}
func.func @open_dim_preserved(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x", ?}]>}) -> tensor<8xf32> {
  return %arg0 : tensor<8xf32>
}

// CHECK-LABEL: func @cant_use_further_axes_axis_not_fully_used
// CHECK-SAME:    %arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_x_8_y_3, [{"x":(1)2}]>}
func.func @cant_use_further_axes_axis_not_fully_used(%arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_x_8_y_3, [{"x", "y"}]>}) -> tensor<6xf32> {
  return %arg0 : tensor<6xf32>
}

// CHECK-LABEL: func @cant_use_further_axes_dim_not_divisible
// CHECK-SAME:    %arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_x_8_y_3, [{"y", "x":(1)2}]>}
func.func @cant_use_further_axes_dim_not_divisible(%arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_x_8_y_3, [{"y", "x"}]>}) -> tensor<6xf32> {
  return %arg0 : tensor<6xf32>
}

// CHECK-LABEL: func @subaxis_smaller
// CHECK-SAME:    %arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_x_8_y_3, [{"x":(1)2}]>}
func.func @subaxis_smaller(%arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_x_8_y_3, [{"x":(1)4}]>}) -> tensor<6xf32> {
  return %arg0 : tensor<6xf32>
}

// CHECK-LABEL: func @subaxis_bigger
// CHECK-SAME:    %arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_x_16, [{"x":(1)2}]>}
func.func @subaxis_bigger(%arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_x_16, [{"x":(1)8}]>}) -> tensor<6xf32> {
  return %arg0 : tensor<6xf32>
}

// CHECK-LABEL: func @pre_axis_preserved
// CHECK-SAME:    %arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_x_24, [{"x":(6)2}]>}
func.func @pre_axis_preserved(%arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_x_24, [{"x":(6)2}]>}) -> tensor<6xf32> {
  return %arg0 : tensor<6xf32>
}

// CHECK-LABEL: func @smaller_axis_with_pre_axis
// CHECK-SAME:    %arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_x_24, [{"x":(6)2}]>}
func.func @smaller_axis_with_pre_axis(%arg0: tensor<6xf32> {sdy.sharding = #sdy.sharding<@mesh_x_24, [{"x":(6)4}]>}) -> tensor<6xf32> {
  return %arg0 : tensor<6xf32>
}
