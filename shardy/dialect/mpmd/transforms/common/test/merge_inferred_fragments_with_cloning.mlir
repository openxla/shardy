// RUN: mpmd_opt %s -mpmd-merge-inferred-fragments=clone-inferred-fragments=true 2>&1 | FileCheck %s

!mesh_1_tensor_f32 = !mpmd.mesh_tensor<"m1", tensor<f32>>
!mesh_1_tensor_2_8_f32 = !mpmd.mesh_tensor<"m1", tensor<2x8xf32>>
!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @small_inferred_fragment_is_cloned
func.func @small_inferred_fragment_is_cloned(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
// CHECK-NEXT:     stablehlo.add
// CHECK-NEXT:     stablehlo.multiply
// CHECK-NEXT:     return
// CHECK-NEXT: }
// CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["g"]> (%arg0)
// CHECK-NEXT:     stablehlo.add
// CHECK-NEXT:     stablehlo.add
// CHECK-NEXT:     return
// CHECK-NEXT: }
// CHECK-NEXT: return
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"]> (%0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %2 = mpmd.fragment<mesh="m1", origin=["g"]> (%0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %1, %2 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @return_operand_is_replaced_with_clone
func.func @return_operand_is_replaced_with_clone(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK-NEXT: %[[F:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
// CHECK-NEXT:     stablehlo.add
// CHECK-NEXT:     stablehlo.multiply
// CHECK-NEXT:     return
// CHECK-NEXT: }
// CHECK-NEXT: return %[[F]]#0, %[[F]]#1
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"]> (%0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %0, %1 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @chain_of_inferred_fragments
func.func @chain_of_inferred_fragments()
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]> ()
// CHECK-NEXT:     stablehlo.const
// CHECK-NEXT:     stablehlo.add
// CHECK-NEXT:     stablehlo.multiply
// CHECK-NEXT:     stablehlo.multiply
// CHECK-NEXT:     return
// CHECK-NEXT: }
// CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["g"]> ()
// CHECK-NEXT:     stablehlo.const
// CHECK-NEXT:     stablehlo.add
// CHECK-NEXT:     stablehlo.multiply
// CHECK-NEXT:     stablehlo.add
// CHECK-NEXT:     return
// CHECK-NEXT: }
// CHECK-NEXT: return
  %0 = mpmd.fragment<mesh="m1", origin=[]> () () {
    %3 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : () -> !mesh_1_tensor_4_8_f32

  %10 = mpmd.fragment<mesh="m1", origin=[]> (%0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %11 = mpmd.fragment<mesh="m1", origin=[]> (%10)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"]> (%11)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %2 = mpmd.fragment<mesh="m1", origin=["g"]> (%11)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %1, %2 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @large_inferred_fragment_is_not_cloned
func.func @large_inferred_fragment_is_not_cloned(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
// CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f"]>
// CHECK-NEXT:     stablehlo.add
// CHECK-NEXT:     stablehlo.add
// CHECK-NEXT:     stablehlo.multiply
// CHECK-NEXT:     return
// CHECK-NEXT: }
// CHECK-NEXT:  mpmd.fragment<mesh="m1", origin=["g"]>
// CHECK-NEXT:     stablehlo.add
// CHECK-NEXT:     return
// CHECK-NEXT: }
// CHECK-NEXT: return
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
    %4 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    %3 = stablehlo.add %4, %4 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"]> (%0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  %2 = mpmd.fragment<mesh="m1", origin=["g"]> (%0)
    (%arg2: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  func.return %1, %2 : !mesh_1_tensor_4_8_f32, !mesh_1_tensor_4_8_f32
}
