// RUN: mpmd_opt %s -mpmd-verify-stage-merging -split-input-file 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// The fragments are assigned to different stages and therefore this is a valid
// program.

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>}
{
  %0 = mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"], stage=1> (%0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// One of the fragments is not assigned to a stage. This is a valid program.

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>}
{
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"], stage=1> (%0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// -----

// Although both fragments are assigned to the same stage, they have different
// transpose_counts, and therefore this is a valid program.

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>}
{
  %0 = mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%arg0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"(1)], stage=0> (%0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// -----

// Although both fragments are assigned to the same stage and have the same
// transpose_count, they have different call_counter, and therefore this is a
// valid program.

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>}
{
  %0 = mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%arg0) {call_counter = 0 : ui32}
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%0) {call_counter = 1 : ui32}
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}
