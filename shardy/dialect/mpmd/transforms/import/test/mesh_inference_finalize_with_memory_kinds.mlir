// RUN: mpmd_opt %s -mpmd-infer-mesh-finalize -split-input-file -verify-diagnostics 2>&1 | FileCheck %s

// CHECK-LABEL: func @single_assign_to_pinned_host
// CHECK-SAME: !mpmd.mesh_tensor<"m", tensor<4x8xi32>, memory_kind="pinned_host">
func.func @single_assign_to_pinned_host(%arg0: tensor<4x8xi32>) -> !mpmd.mesh_tensor<"m", tensor<4x8xi32>, memory_kind="pinned_host">
  attributes {topology = #mpmd.topology<<"m" : <["devices"=8]>>>}
{
  %0 = mpmd.assign %arg0 : (tensor<4x8xi32>) -> !mpmd.mesh_tensor<"m", tensor<4x8xi32>, memory_kind="pinned_host">
  return %0 : !mpmd.mesh_tensor<"m", tensor<4x8xi32>, memory_kind="pinned_host">
}

// -----

// CHECK-LABEL: func @multiple_assign_to_pinned_host
// CHECK-SAME: !mpmd.mesh_tensor<"m", tensor<4x8xi32>, memory_kind="pinned_host">
func.func @multiple_assign_to_pinned_host(%arg0: tensor<4x8xi32>)
  attributes {topology = #mpmd.topology<<"m" : <["devices"=8]>>>}
{
  %0 = mpmd.assign %arg0 : (tensor<4x8xi32>) -> !mpmd.mesh_tensor<"m", tensor<4x8xi32>, memory_kind="pinned_host">
  %1 = mpmd.assign %arg0 : (tensor<4x8xi32>) -> !mpmd.mesh_tensor<"m", tensor<4x8xi32>, memory_kind="pinned_host">
  return
}

// -----

// expected-error @+1 {{Argument 0 has different memory kinds assigned to it.}}
func.func @multiple_assign_users_with_different_memory_kinds(%arg0: tensor<4x8xi32>)
  attributes {topology = #mpmd.topology<<"m" : <["devices"=8]>>>}
{
  %0 = mpmd.assign %arg0 : (tensor<4x8xi32>) -> !mpmd.mesh_tensor<"m", tensor<4x8xi32>, memory_kind="pinned_host">
  %1 = mpmd.assign %arg0 : (tensor<4x8xi32>) -> !mpmd.mesh_tensor<"m", tensor<4x8xi32>, memory_kind="device">
  return
}

// -----

// expected-error @+1 {{Argument 0 has different memory kinds assigned to it. Found at least one user with undefined memory kind and at least one user with a memory kind.}}
func.func @multiple_assign_with_memory_kinds_defined_and_undefined(%arg0: tensor<4x8xi32>)
  attributes {topology = #mpmd.topology<<"m" : <["devices"=8]>>>}
{
  %0 = mpmd.assign %arg0 : (tensor<4x8xi32>) -> !mpmd.mesh_tensor<"m", tensor<4x8xi32>>
  %1 = mpmd.assign %arg0 : (tensor<4x8xi32>) -> !mpmd.mesh_tensor<"m", tensor<4x8xi32>, memory_kind="device">
  return
}
