// RUN: mpmd_opt %s -mpmd-infer-mesh-assign-using-input-output-constraints='constraints=0:0' 2>&1 | FileCheck %s

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>>
!m2_tensor = !mpmd.mesh_tensor<"m2", tensor<8x16xf32>>

// CHECK-LABEL: func @output_with_empty_src_set(%arg0: !mpmd.mesh_tensor<"m2"
func.func @output_with_empty_src_set(
  %arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m3", "m2">})
   -> tensor<4x8xf32> attributes {topology=#topology} {
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign  {origin = "io_constraint_in"}  %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[UNASSIGN]], %[[UNASSIGN]]
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign  {origin = "io_constraint_out"} %[[ADD]]
  // CHECK-NEXT: return %[[ASSIGN]] : !mpmd.mesh_tensor<"m2"
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}


// CHECK-LABEL: func @output_with_use_set(%arg0: !mpmd.mesh_tensor<"m3"
func.func @output_with_use_set(
  %arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2", "m3">})
  -> tensor<4x8xf32> attributes {topology=#topology} {

  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign  {origin = "io_constraint_in"} %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[UNASSIGN]], %[[UNASSIGN]]
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign  {origin = "io_constraint_out"} %[[ADD]]
  // CHECK-NEXT: return %[[ASSIGN]] : !mpmd.mesh_tensor<"m3"
  %0 = stablehlo.add %arg0, %arg0  {mpmd.src_set = #mpmd.meshes_with_origins<"m2">, mpmd.use_set = #mpmd.meshes_with_origins<"m3">}: tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @output_with_src_set_only(%arg0: !mpmd.mesh_tensor<"m2"
func.func @output_with_src_set_only(
  %arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m2", "m3">})
  -> tensor<4x8xf32> attributes {topology=#topology} {
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign  {origin = "io_constraint_in"} %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[UNASSIGN]], %[[UNASSIGN]]
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign  {origin = "io_constraint_out"} %[[ADD]]
  // CHECK-NEXT: return %[[ASSIGN]] : !mpmd.mesh_tensor<"m2"
  %0 = stablehlo.add %arg0, %arg0  {mpmd.src_set = #mpmd.meshes_with_origins<"m2">}: tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @input_no_use_set_but_use_set_on_output
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
func.func @input_no_use_set_but_use_set_on_output(
  %arg0: tensor<4x8xf32>
)
  -> tensor<4x8xf32> attributes {topology=#topology} {
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign {origin = "io_constraint_in"} %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[UNASSIGN]], %[[UNASSIGN]]
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "io_constraint_out"} %[[ADD]]
  // CHECK-NEXT: return %[[ASSIGN]]
  // CHECK-SAME:   !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m3", "m2">}: tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @input_no_use_set_but_src_set_on_output
// CHECK-SAME: %arg0: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
func.func @input_no_use_set_but_src_set_on_output(
  %arg0: tensor<4x8xf32>
)
  -> tensor<4x8xf32> attributes {topology=#topology} {
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign {origin = "io_constraint_in"} %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[UNASSIGN]], %[[UNASSIGN]]
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign {origin = "io_constraint_out"} %[[ADD]]
  // CHECK-NEXT: return %[[ASSIGN]]
  // CHECK-SAME:   !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2">}: tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @input_multi_use_set(%arg0: !mpmd.mesh_tensor<"m2"
func.func @input_multi_use_set(
  %arg0: tensor<4x8xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2", "m3">})
  -> tensor<4x8xf32> attributes {topology=#topology} {
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign  {origin = "io_constraint_in"} %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[UNASSIGN]], %[[UNASSIGN]]
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign  {origin = "io_constraint_out"} %[[ADD]]
  // CHECK-NEXT: return %[[ASSIGN]] : !mpmd.mesh_tensor<"m2"
  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2">}: tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @input_is_mesh_tensor_with_compatible_mesh(%arg0: !mpmd.mesh_tensor<"m2"
func.func @input_is_mesh_tensor_with_compatible_mesh(
  %arg0: !m2_tensor {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2", "m3">})
  -> tensor<8x16xf32> attributes {topology=#topology} {
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign  {origin = "io_constraint_in"} %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[UNASSIGN]], %[[UNASSIGN]]
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign  {origin = "io_constraint_out"} %[[ADD]]
  // CHECK-NEXT: return %[[ASSIGN]] : !mpmd.mesh_tensor<"m2"
  %1 = mpmd.unassign  {origin = "io_constraint_in"} %arg0 : (!m2_tensor) -> tensor<8x16xf32>
  %2 = stablehlo.add %1, %1 : tensor<8x16xf32>
  func.return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @input_is_mesh_tensor_with_incompatible_mesh(%arg0: !mpmd.mesh_tensor<"m2"
func.func @input_is_mesh_tensor_with_incompatible_mesh(
  %arg0: !m2_tensor {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2", "m3">})
  -> tensor<8x16xf32> attributes {topology=#topology} {
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign  {origin = "io_constraint_in"} %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[UNASSIGN]], %[[UNASSIGN]]
  // CHECK-NEXT: return %[[ADD]] : tensor<8x16xf32>
  // Note it is a no-op. We will introduce the transfer later.
  %1 = mpmd.unassign  {origin = "io_constraint_in"} %arg0 : (!m2_tensor) -> tensor<8x16xf32>
  %2 = stablehlo.add %1, %1 {mpmd.src_set = #mpmd.meshes_with_origins<"m1">} : tensor<8x16xf32>
  func.return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @output_already_has_mesh(%arg0: !mpmd.mesh_tensor<"m2"
func.func @output_already_has_mesh(
  %arg0: tensor<8x16xf32> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2", "m3">})
  -> !m2_tensor attributes {topology=#topology} {
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign  {origin = "io_constraint_in"} %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[UNASSIGN]], %[[UNASSIGN]]
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign  {origin = "io_constraint_out"} %[[ADD]]
  // CHECK-NEXT: return %[[ASSIGN]] : !mpmd.mesh_tensor<"m2"
  %1 = stablehlo.add %arg0, %arg0 : tensor<8x16xf32>
  %3 = mpmd.assign  {origin = "io_constraint_out"} %1 : (tensor<8x16xf32>) -> !m2_tensor
  func.return %3 : !m2_tensor
}

// CHECK-LABEL: func @noop_if_both_already_assigned(%arg0: !mpmd.mesh_tensor<"m2"
func.func @noop_if_both_already_assigned(
  %arg0: !m2_tensor {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2", "m3">})
  -> !m2_tensor attributes {topology=#topology} {
  // CHECK-NEXT: %[[UNASSIGN:.*]] = mpmd.unassign  {origin = "io_constraint_in"} %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[UNASSIGN]], %[[UNASSIGN]]
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign  {origin = "io_constraint_out"} %[[ADD]]
  // CHECK-NEXT: return %[[ASSIGN]] : !mpmd.mesh_tensor<"m2"
  %1 = mpmd.unassign  {origin = "io_constraint_in"} %arg0 : (!m2_tensor) -> tensor<8x16xf32>
  %2 = stablehlo.add %1, %1 : tensor<8x16xf32>
  %3 = mpmd.assign  {origin = "io_constraint_out"} %2 : (tensor<8x16xf32>) -> !m2_tensor
  func.return %3 : !m2_tensor
}
