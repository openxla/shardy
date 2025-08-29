// RUN: mpmd_opt %s -mpmd-verify-stage-merging -split-input-file -verify-diagnostics 2>&1

// One of the fragments does not have a call_counter while the other does. This
// is not a valid program.

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

  // expected-error@+1 {{A valid program cannot have more than one fragment with the same mesh, stage, transpose, call counts, split type, and remat flag but found multiple fragments with the same attributes: [mesh=m1, stage_id=0, transpose_count=0, call_count=1, split_type=nullopt, is_remat=0] for current fragment with origin: [#mpmd.user_origin<"f">]}}
  %1 = mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%0) {call_counter = 1 : ui32}
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// -----

// Both fragments have the same call counter. This is not a valid program.

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @main
func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>}
{
  %0 = mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%arg0) {call_counter = 1 : ui32}
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  // expected-error@+1 {{A valid program cannot have more than one fragment with the same mesh, stage, transpose, call counts, split type, and remat flag but found multiple fragments with the same attributes: [mesh=m1, stage_id=0, transpose_count=0, call_count=1, split_type=nullopt, is_remat=0] for current fragment with origin: [#mpmd.user_origin<"f">]}}
  %1 = mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%0) {call_counter = 1 : ui32}
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// -----

// The fragments do not have have a call counter but they are both assigned to
// the same stage. This is not a valid program.

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

  // expected-error@+1 {{A valid program cannot have more than one fragment with the same mesh, stage, transpose, call counts, split type, and remat flag but found multiple fragments with the same attributes: [mesh=m1, stage_id=0, transpose_count=0, call_count=-1, split_type=nullopt, is_remat=0] for current fragment with origin: [#mpmd.user_origin<"f">]}}
  %1 = mpmd.fragment<mesh="m1", origin=["f"], stage=0> (%0)
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}
