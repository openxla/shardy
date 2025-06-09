// RUN: sdy_opt %s -sdy-manual-axes-cleanup | FileCheck %s

sdy.mesh @empty_mesh = <[]>

sdy.mesh @mesh = <["c"=2, "a"=2, "b"=2]>
sdy.mesh @mesh_xyz = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh_x_8_y_8 = <["x"=8, "y"=8]>

// CHECK-LABEL: @add_new_replicated
func.func @add_new_replicated(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c", ?}]>] out_shardings=[<@mesh, [{"c", ?}]>] manual_axes={"c", "a"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// Basic appending to existing replicated axes list.
// CHECK-LABEL: @add_to_existing_replicated
func.func @add_to_existing_replicated(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"c", "a", "b"} (%arg1: tensor<4xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>] out_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>] manual_axes={"c", "a", "b"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// Similar to add_to_existing_replicated but must sort the replicated axes.
// CHECK-LABEL: @add_to_existing_replicated_requires_sorting
func.func @add_to_existing_replicated_requires_sorting(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"c", "a", "b"} (%arg1: tensor<4xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c", ?}], replicated={"b"}>] out_shardings=[<@mesh, [{"c", ?}], replicated={"a"}>] manual_axes={"c", "a", "b"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// Need to sort the manual axes in addition to adding to replicated axes of
// out shardings.
// CHECK-LABEL: @sort_manual_axes_and_add_out_shardings
func.func @sort_manual_axes_and_add_out_shardings(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"c", "a", "b"} (%arg1: tensor<4xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"c", ?}], replicated={"a", "b"}>] out_shardings=[<@mesh, [{"c", ?}], replicated={"b"}>] manual_axes={"b", "a", "c"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @inlined_mesh_add_replicated_and_sort
func.func @inlined_mesh_add_replicated_and_sort(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_xyz, [{"y", ?}], replicated={"x", "z"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh_xyz, [{"z", ?}], replicated={"x", "y"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"x", "y", "z"} (%arg1: tensor<4xf32>) {
  %0 = sdy.manual_computation(%arg0)
      in_shardings=[<@mesh_xyz, [{"y", ?}]>]
      out_shardings=[<mesh<["x"=2, "y"=2, "z"=2]>, [{"z", ?}], replicated={"y"}>]
      manual_axes={"y", "x", "z"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @empty_mesh_by_name_result
func.func @empty_mesh_by_name_result(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_xyz, [{"x"}], replicated={"y"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh_xyz, [{}], replicated={"x", "y"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"x", "y"} (%arg1: tensor<4xf32>) {
  %0 = sdy.manual_computation(%arg0)
      in_shardings=[<@mesh_xyz, [{"x"}]>]
      out_shardings=[<@empty_mesh, [{}]>]
      manual_axes={"y", "x"} (%arg1: tensor<4xf32>) {
    %1 = "stablehlo.all_gather"(%arg1) {
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<4xf32>) -> tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @inlined_empty_mesh_result
func.func @inlined_empty_mesh_result(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<mesh<["x"=2, "y"=2]>, [{"x"}], replicated={"y"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<mesh<["x"=2, "y"=2]>, [{}], replicated={"x", "y"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"x", "y"} (%arg1: tensor<4xf32>) {
  %0 = sdy.manual_computation(%arg0)
      in_shardings=[<mesh<["x"=2, "y"=2]>, [{"x"}]>]
      out_shardings=[<mesh<[]>, [{}]>]
      manual_axes={"y", "x"} (%arg1: tensor<4xf32>) {
    %1 = "stablehlo.all_gather"(%arg1) {
      all_gather_dim = 0 : i64,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<4xf32>) -> tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @empty_mesh_operand
func.func @empty_mesh_operand(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_xyz, [{}], replicated={"x", "y"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh_xyz, [{"x"}], replicated={"y"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"x", "y"} (%arg1: tensor<8xf32>) {
  %0 = sdy.manual_computation(%arg0)
      in_shardings=[<@empty_mesh, [{}]>]
      out_shardings=[<@mesh_xyz, [{"x"}]>]
      manual_axes={"y", "x"} (%arg1: tensor<8xf32>) {
    %1 = stablehlo.slice %arg1 [0:4]: (tensor<8xf32>) -> tensor<4xf32>
    sdy.return %1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @dont_add_manual_axes_to_non_shaped_types
func.func @dont_add_manual_axes_to_non_shaped_types(%arg0: !stablehlo.token, %arg1: tensor<8xf32>) -> (!stablehlo.token, tensor<8xf32>) {
  // CHECK-NEXT: sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh, []>, <@mesh, [{?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh, []>, <@mesh, [{?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"c"} (%arg2: !stablehlo.token, %arg3: tensor<8xf32>) {
  %0:2 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh, []>, <@mesh, [{?}]>]
      out_shardings=[<@mesh, []>, <@mesh, [{?}]>]
      manual_axes={"c"} (%arg2: !stablehlo.token, %arg3: tensor<8xf32>) {
    sdy.return %arg2, %arg3 : !stablehlo.token, tensor<8xf32>
  } : (!stablehlo.token, tensor<8xf32>) -> (!stablehlo.token, tensor<8xf32>)
  return %0#0, %0#1 : !stablehlo.token, tensor<8xf32>
}

// CHECK-LABEL: @sub_axes_in_out_shardings
func.func @sub_axes_in_out_shardings(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_x_8_y_8, [{"y":(2)2}], replicated={"x", "y":(1)2, "y":(4)2}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<@mesh_x_8_y_8, [{"y":(1)2}], replicated={"x", "y":(2)4}>]
  // CHECK-SAME{LITERAL}: manual_axes={"x", "y"} (%arg1: tensor<4xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_x_8_y_8, [{"y":(2)2}]>] out_shardings=[<@mesh_x_8_y_8, [{"y":(1)2}], replicated={"x":(2)2}>] manual_axes={"x", "y"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
