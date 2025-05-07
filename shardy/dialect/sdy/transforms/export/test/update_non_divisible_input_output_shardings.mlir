// RUN: sdy_opt %s -sdy-update-non-divisible-input-output-shardings | FileCheck %s

sdy.mesh @mesh_x_4_y_2 = <["x"=4, "y"=2]>
sdy.mesh @mesh_x_8_y_3 = <["x"=8, "y"=3]>
sdy.mesh @mesh_x_16 = <["x"=16]>
sdy.mesh @mesh_x_24 = <["x"=24]>
sdy.mesh @mesh3d = <["a"=4, "b"=4, "c"=4]>

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

// CHECK-LABEL: func @manual_computation_free_axes_non_divisible
func.func @manual_computation_free_axes_non_divisible(
    %arg0: tensor<4xf32>, %arg1: tensor<12xf32>, %arg2: tensor<24xf32>,
    %arg3: tensor<48xf32>, %arg4: tensor<96xf32>, %arg5: tensor<192xf32>)
    -> (tensor<4xf32>, tensor<12xf32>, tensor<24xf32>,
        tensor<48xf32>, tensor<96xf32>, tensor<192xf32>) {
  // CHECK-NEXT: sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
  // CHECK-SAME:   in_shardings=[<@mesh3d, [{"a"}]>, <@mesh3d, [{"a"}]>,
  // CHECK-SAME:                 <@mesh3d, [{"a", "b":(1)2}]>, <@mesh3d, [{"a", "b"}]>,
  // CHECK-SAME:                 <@mesh3d, [{"a", "b", "c":(1)2}]>, <@mesh3d, [{"a", "b", "c"}]>]
  // CHECK-SAME:   out_shardings=[<@mesh3d, [{"a"}]>, <@mesh3d, [{"a"}]>,
  // CHECK-SAME:                  <@mesh3d, [{"a", "b":(1)2}]>, <@mesh3d, [{"a", "b"}]>,
  // CHECK-SAME:                  <@mesh3d, [{"a", "b", "c":(1)2}]>, <@mesh3d, [{"a", "b", "c"}]>]
  // CHECK-SAME:   manual_axes={"a"}
  %0:6 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5)
    in_shardings=[<@mesh3d, [{"a", "b", "c"}]>, <@mesh3d, [{"a", "b", "c"}]>,
                  <@mesh3d, [{"a", "b", "c"}]>, <@mesh3d, [{"a", "b", "c"}]>,
                  <@mesh3d, [{"a", "b", "c"}]>, <@mesh3d, [{"a", "b", "c"}]>]
    out_shardings=[<@mesh3d, [{"a", "b", "c"}]>, <@mesh3d, [{"a", "b", "c"}]>,
                   <@mesh3d, [{"a", "b", "c"}]>, <@mesh3d, [{"a", "b", "c"}]>,
                   <@mesh3d, [{"a", "b", "c"}]>, <@mesh3d, [{"a", "b", "c"}]>]
    manual_axes={"a"} (%arg6: tensor<1xf32>, %arg7: tensor<3xf32>, %arg8: tensor<6xf32>, %arg9: tensor<12xf32>, %arg10: tensor<24xf32>, %arg11: tensor<48xf32>) {
    sdy.return %arg6, %arg7, %arg8, %arg9, %arg10, %arg11 : tensor<1xf32>, tensor<3xf32>, tensor<6xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>
  } : (tensor<4xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>, tensor<96xf32>, tensor<192xf32>) -> (tensor<4xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>, tensor<96xf32>, tensor<192xf32>)
  return %0#0, %0#1, %0#2, %0#3, %0#4, %0#5 : tensor<4xf32>, tensor<12xf32>, tensor<24xf32>, tensor<48xf32>, tensor<96xf32>, tensor<192xf32>
}

// CHECK-LABEL: func @optimization_barrier_non_divisible_shardings_unchanged(
// CHECK-SAME:    %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x":(1)2}, {}]>})
// CHECK-SAME:  -> (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x":(1)2}, {}]>})
func.func @optimization_barrier_non_divisible_shardings_unchanged(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x"}, {}]>})
    -> (tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_x_4_y_2, [{"x"}, {}]>}) {
  // CHECK-NEXT: stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x_4_y_2, [{"x"}, {}]>]>}
  %1 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_x_4_y_2, [{"x"}, {}]>]>} %arg0 : tensor<2x4xf32>
  return %1 : tensor<2x4xf32>
}
