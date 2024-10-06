// RUN: sdy_opt %s -sdy-manual-axes-cleanup | FileCheck %s

sdy.mesh @mesh = <["c"=2, "a"=2, "b"=2]>

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

sdy.mesh @mesh_xyz = <["z"=2, "x"=2, "y"=2]>

// CHECK-LABEL: @inlined_mesh_add_replicated_and_sort
func.func @inlined_mesh_add_replicated_and_sort(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}: in_shardings=[<@mesh_xyz, [{"x", ?}], replicated={"z", "y"}>]
  // CHECK-SAME{LITERAL}: out_shardings=[<mesh<["z"=2, "x"=2, "y"=2]>, [{"z", ?}], replicated={"x", "y"}>]
  // CHECK-SAME{LITERAL}: manual_axes={"z", "x", "y"} (%arg1: tensor<4xf32>) {
  %0 = sdy.manual_computation(%arg0)
      in_shardings=[<@mesh_xyz, [{"x", ?}]>]
      out_shardings=[<mesh<["z"=2, "x"=2, "y"=2]>, [{"z", ?}], replicated={"y"}>]
      manual_axes={"y", "x", "z"} (%arg1: tensor<4xf32>) {
    sdy.return %arg1 : tensor<4xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}
