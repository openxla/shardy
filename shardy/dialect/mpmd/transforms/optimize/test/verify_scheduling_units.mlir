// RUN: mpmd_opt %s -mpmd-scheduling-units-verifier -verify-diagnostics -split-input-file 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// The topology has two meshes, but m2 is unused. Moreover, m1 doesn't
// have any fragment which is user-defined.
// CHECK-LABEL: func @main
// expected-warning@+3 {{Number of backward scheduling units in mesh m2 does not match expected number for 1 microbatches. Got 0.}}
// expected-warning@+2 {{Number of forward scheduling units in mesh m2 does not match expected number for 1 microbatches. Got 0.}}
// expected-warning@+1 {{Number of forward scheduling units in mesh m1 does not match expected number for 1 microbatches. Got 0.}}
func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>
    >} {
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) {call_counter = 0 : ui32}
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"(1)]> (%0) {call_counter = 0 : ui32}
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %1 : !mesh_1_tensor_4_8_f32
}

// -----

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @main
// This test has too many fwd scheduling units
// expected-warning@+1 {{Number of forward scheduling units in mesh m1 does not match expected number for 1 microbatches. Got 2.}}
func.func @main(%arg0: !mesh_1_tensor_4_8_f32)
  -> (!mesh_1_tensor_4_8_f32) attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>
    >} {
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) {call_counter = 0 : ui32}
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %1 = mpmd.fragment<mesh="m1", origin=["f"]> (%0) {call_counter = 0 : ui32}
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  %2 = mpmd.fragment<mesh="m1", origin=["f"(1)]> (%1) {call_counter = 0 : ui32}
    (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32

  func.return %2 : !mesh_1_tensor_4_8_f32
}
