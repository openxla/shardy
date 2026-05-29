// RUN: mpmd_opt %s -mpmd-export-pipeline -split-input-file 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func.func @main
func.func @main(%arg0: !mesh_1_tensor_4_8_f32 {tf.aliasing_output = 0: i32}, %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK: mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[CALLEE:.*]](%arg0, %arg1)
// CHECK: func.func @[[CALLEE]](%arg0: tensor<4x8xf32> {tf.aliasing_output = 0 : i32}, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {mesh_shape = #sdy.mesh<["x"=2]>, reserved_hbm_bytes = 0 : i64}
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %0 = stablehlo.add %arg2, %arg3: tensor<4x8xf32>
    mpmd.return %0 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32)
  func.return %0 : !mesh_1_tensor_4_8_f32
}

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// This test verifies that an explicit fragment and an inferred fragment
// (created by the UniquifyFunctionInputsOutputsPass for the duplicated return
// of the transfer result) are merged sideways. Without sideways merge, the
// transfer result would produce a separate inferred fragment call on m1.
// The function-level returns remain unique SSA values (%[[RES]]#0, #1, #2),
// preserving the invariant established by the uniquify pass, even though the
// fragment body internally returns the same value in multiple positions.
// CHECK-LABEL: func.func @test_sideways_merge
func.func @test_sideways_merge(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
      "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK: %[[RES:.*]]:3 = mpmd.fragment_call<mesh="m1", origin=["f1"]> @[[CALLEE_M1:.*]]
  // CHECK-NOT: mpmd.fragment_call<mesh="m1"
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2

  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.transfer %arg1 : (!mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %0, %1, %1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}
