// RUN: mpmd_opt %s -mpmd-infer-mesh-validate-no-additional-transfers-needed 2>&1 | FileCheck %s

!mesh_1_tensor_ui32 = !mpmd.mesh_tensor<"m1", tensor<ui32>>
!mesh_1_tensor_1_ui32 = !mpmd.mesh_tensor<"m1", tensor<1xui32>>
!mesh_1_tensor_2_ui32 = !mpmd.mesh_tensor<"m1", tensor<2xui32>>
!mesh_1_tensor_5_5_ui32 = !mpmd.mesh_tensor<"m1", tensor<5x5xui32>>
!mesh_1_tensor_4_4_f32 = !mpmd.mesh_tensor<"m1", tensor<4x4xf32>>
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_tensor_8_16_f32 = !mpmd.mesh_tensor<"m1", tensor<8x16xf32>>
!mesh_1_tensor_4_16_f32 = !mpmd.mesh_tensor<"m1", tensor<4x16xf32>>
!mesh_1_tensor_16_8_f32 = !mpmd.mesh_tensor<"m1", tensor<16x8xf32>>

!mesh_2_tensor_ui32 = !mpmd.mesh_tensor<"m2", tensor<ui32>>
!mesh_2_tensor_1_ui32 = !mpmd.mesh_tensor<"m2", tensor<1xui32>>
!mesh_2_tensor_2_ui32 = !mpmd.mesh_tensor<"m2", tensor<2xui32>>
!mesh_2_tensor_5_5_ui32 = !mpmd.mesh_tensor<"m2", tensor<5x5xui32>>
!mesh_2_tensor_4_4_f32 = !mpmd.mesh_tensor<"m2", tensor<4x4xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
!mesh_2_tensor_4_16_f32 = !mpmd.mesh_tensor<"m2", tensor<4x16xf32>>
!mesh_2_tensor_8_16_f32 = !mpmd.mesh_tensor<"m2", tensor<8x16xf32>>
!mesh_2_tensor_16_8_f32 = !mpmd.mesh_tensor<"m2", tensor<16x8xf32>>

// The majority of these tests verify only that validation passes and assume
// that the DAG unchanged aside from additional attributes. There is one test
// `dag_unchanged` that verifies that the DAG
// remains unchanged.

// CHECK-LABEL: func @op_without_src_set_or_use_set(%arg0: tensor<4x8xf32>)
func.func @op_without_src_set_or_use_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @op_with_src_set_only(%arg0: tensor<4x8xf32>)
func.func @op_with_src_set_only(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m3">} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @op_with_src_set_multiple_only(%arg0: tensor<4x8xf32>)
func.func @op_with_src_set_multiple_only(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m2", "m3">} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @output_with_use_set_only(%arg0: tensor<4x8xf32>)
func.func @output_with_use_set_only(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m3">} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @op_with_use_set_multiple_only(%arg0: tensor<4x8xf32>)
func.func @op_with_use_set_multiple_only(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<"m2", "m3">} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @op_with_src_set_and_use_set(%arg0: tensor<4x8xf32>)
func.func @op_with_src_set_and_use_set(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m3">, mpmd.use_set = #mpmd.meshes_with_origins<"m3">} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @op_with_src_set_and_use_set_multiple(%arg0: tensor<4x8xf32>)
func.func @op_with_src_set_and_use_set_multiple(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
  attributes {"topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>
  >} {
  %0 = stablehlo.add %arg0, %arg0 {mpmd.src_set = #mpmd.meshes_with_origins<"m3", "m2">, mpmd.use_set = #mpmd.meshes_with_origins<"m2", "m3">} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @dag_unchanged
// CHECK-SAME:     %arg0: tensor<4x8xf32>
// CHECK-SAME:     %arg1: tensor<8x16xf32>
// CHECK-SAME:     %arg2: tensor<16x8xf32>
// CHECK-SAME:     %arg3: tensor<4x16xf32>
// CHECK-SAME:     %arg4: tensor<16x8xf32>
func.func @dag_unchanged(
  %arg0: tensor<4x8xf32>,
  %arg1: tensor<8x16xf32>,
  %arg2: tensor<16x8xf32>,
  %arg3: tensor<4x16xf32>,
  %arg4: tensor<16x8xf32>)
  -> (tensor<4x16xf32>, tensor<16x8xf32>) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {
  // CHECK-NEXT: %0 = stablehlo.add %arg2, %arg4
  %0 = stablehlo.add %arg2, %arg4 : tensor<16x8xf32>

  // CHECK-NEXT: %1 = mpmd.assign %arg0
  %1 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  // CHECK-NEXT: %2 = mpmd.assign %arg1
  %2 = mpmd.assign %arg1 : (tensor<8x16xf32>) -> !mesh_1_tensor_8_16_f32

  // CHECK-NEXT: %3 = mpmd.fragment<mesh="m1", origin=["f1"]> (%1, %2)
  // CHECK-SAME:  (%arg5: tensor<4x8xf32>, %arg6: tensor<8x16xf32>) {
  %3 = mpmd.fragment<mesh="m1", origin=["f1"]> (%1, %2)
    (%arg5: tensor<4x8xf32>, %arg6: tensor<8x16xf32>) {

    // CHECK-NEXT: %6 = stablehlo.dot %arg5, %arg6 : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %6 = "stablehlo.dot"(%arg5, %arg6) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    // CHECK-NEXT: mpmd.return %6 : tensor<4x16xf32>
    mpmd.return %6 : tensor<4x16xf32>
    // CHECK-NEXT: }
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_8_16_f32) -> (!mesh_1_tensor_4_16_f32)

  // CHECK-NEXT: %4 = mpmd.transfer %3
  %4 = mpmd.transfer %3 : (!mesh_1_tensor_4_16_f32) -> !mesh_2_tensor_4_16_f32

  // CHECK-NEXT: %5 = mpmd.unassign %4
  %5 = mpmd.unassign %4 : (!mesh_2_tensor_4_16_f32) -> tensor<4x16xf32>

  // CHECK-NEXT: return %5, %0 : tensor<4x16xf32>, tensor<16x8xf32>
  func.return %5, %0: tensor<4x16xf32>, tensor<16x8xf32>
}

// CHECK-LABEL: func @func_with_no_topology_does_not_error_because_skipped
func.func @func_with_no_topology_does_not_error_because_skipped(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %0 = stablehlo.add
  // CHECK-NEXT: return %0
  %0 = stablehlo.add %arg0, %arg0 {mpmd.use_set = #mpmd.meshes_with_origins<>} : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
