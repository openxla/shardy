// RUN: mpmd_opt %s -mpmd-merge-inferred-fragments=merge-any-consumer=true 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-LABEL: func @inferred_fragment_is_last_user
func.func @inferred_fragment_is_last_user(
  %arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
  // CHECK-NEXT:   stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
  // CHECK-NEXT:   stablehlo.subtract %3, %3 : tensor<4x8xf32>
  // CHECK-NEXT:   mpmd.return
  // CHECK-NEXT: }
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // This is not a mergeable consumer as both fragments are user-defined fragments.
  // CHECK-NEXT:  mpmd.fragment<mesh="m1", origin=["g"]>
  %1 = mpmd.fragment<mesh="m1", origin=["g"]> (%0)
    (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // The transfer won't impact merging as this is the merge-inferred pass
  %2 = mpmd.transfer %0 : (!mesh_1_tensor_4_8_f32) -> !mesh_2_tensor_4_8_f32

  // An inferred fragment that can be merged with the producer of its operand.
  // CHECK-NOT: mpmd.fragment<mesh="m1", origin=[]>
  %3 = mpmd.fragment<mesh="m1", origin=[]> (%0)
    (%arg2: tensor<4x8xf32>) {
    %5 = stablehlo.subtract %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %5 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %3, %2, %1 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}
