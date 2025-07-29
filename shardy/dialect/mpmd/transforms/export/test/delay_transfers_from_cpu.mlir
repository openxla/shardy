// RUN: mpmd_opt %s -mpmd-delay-transfers-from-cpu 2>&1 | FileCheck %s

!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4x8xf32>>
!pinned_host_mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4x8xf32>, memory_kind="pinned_host">
!host_mesh_tensor = !mpmd.mesh_tensor<"m/cpu", tensor<4x8xf32>>

// CHECK-LABEL: func @transfer_of_arg_is_delayed
func.func @transfer_of_arg_is_delayed(%arg0: !mesh_tensor, %arg1: !host_mesh_tensor)
  -> (!mesh_tensor, !mesh_tensor) attributes {
    "topology"=#mpmd.topology<<"m": <["x"=2, "y"=2]>>, <"m/cpu": <["x"=2, "y"=2]>>>} {
  // CHECK: mpmd.fragment
  // CHECK: mpmd.transfer
  // This transfer isn't needed until later and it's operand lives on host. So,
  // we will delay it.
  %t = mpmd.transfer %arg1 : (!host_mesh_tensor) -> !mesh_tensor
  %f = mpmd.fragment<mesh="m", origin=["f"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  func.return %f, %t : !mesh_tensor, !mesh_tensor
}

// CHECK-LABEL: func @interleave_transfers_with_fragments
func.func @interleave_transfers_with_fragments(%arg0: !host_mesh_tensor, %arg1: !host_mesh_tensor)
  -> (!mesh_tensor, !mesh_tensor) attributes {
    "topology"=#mpmd.topology<<"m": <["x"=2, "y"=2]>>, <"m/cpu": <["x"=2, "y"=2]>>>} {
  // CHECK: mpmd.transfer %arg0
  // CHECK: mpmd.fragment<mesh="m", origin=["f0"]>
  // CHECK: mpmd.transfer %arg1
  // CHECK: mpmd.fragment<mesh="m", origin=["f1"]>
  %t0 = mpmd.transfer %arg0 : (!host_mesh_tensor) -> !mesh_tensor
  %t1 = mpmd.transfer %arg1 : (!host_mesh_tensor) -> !mesh_tensor
  %f0 = mpmd.fragment<mesh="m", origin=["f0"]> (%t0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  %f1 = mpmd.fragment<mesh="m", origin=["f1"]> (%t1) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  func.return %f0, %f1 : !mesh_tensor, !mesh_tensor
}

// We shouldn't see this pattern in the wild (at least for now). But there's no
// reason not to support it and it might be useful in the future.
// CHECK-LABEL: func @transfer_of_produced_value_is_delayed
func.func @transfer_of_produced_value_is_delayed(%arg0: !mesh_tensor, %arg1: !host_mesh_tensor)
  -> (!mesh_tensor, !mesh_tensor) attributes {
    "topology"=#mpmd.topology<<"m": <["x"=2, "y"=2]>>, <"m/cpu": <["x"=2, "y"=2]>>>} {
  // CHECK: mpmd.fragment
  // CHECK: mpmd.fragment
  // CHECK: mpmd.transfer
  %x = mpmd.fragment<mesh="m/cpu", origin=["f"]> (%arg1) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!host_mesh_tensor) -> !host_mesh_tensor
  // This transfer isn't needed until later and it's operand lives on host. So,
  // we will delay it.
  %t = mpmd.transfer %x : (!host_mesh_tensor) -> !mesh_tensor
  %f = mpmd.fragment<mesh="m", origin=["f"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  func.return %f, %t : !mesh_tensor, !mesh_tensor
}

// CHECK-LABEL: func @transfer_without_users_is_not_delayed
func.func @transfer_without_users_is_not_delayed(%arg0: !host_mesh_tensor)
  attributes {"topology"=#mpmd.topology<<"m": <["x"=2, "y"=2]>>, <"m/cpu": <["x"=2, "y"=2]>>>}
{
  // CHECK: mpmd.transfer
  // CHECK: return
  %t = mpmd.transfer %arg0 : (!host_mesh_tensor) -> !mesh_tensor
  func.return
}

// Typo in the mesh name means this is not a host mesh.
!typo_host_mesh_tensor = !mpmd.mesh_tensor<"m/cpu_", tensor<4x8xf32>>

// CHECK-LABEL: func @transfer_of_arg_is_not_delayed_because_typo
func.func @transfer_of_arg_is_not_delayed_because_typo(%arg0: !mesh_tensor, %arg1: !typo_host_mesh_tensor)
  -> (!mesh_tensor, !mesh_tensor) attributes {
    "topology"=#mpmd.topology<<"m": <["x"=2, "y"=2]>>, <"m/cpu_": <["x"=2, "y"=2]>>>} {
  // CHECK: mpmd.transfer
  // CHECK-NEXT: mpmd.fragment
  // This transfer isn't needed until later and it's operand lives on host. So,
  // we will delay it.
  %t = mpmd.transfer %arg1 : (!typo_host_mesh_tensor) -> !mesh_tensor
  %f = mpmd.fragment<mesh="m", origin=["f"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  func.return %f, %t : !mesh_tensor, !mesh_tensor
}

// CHECK-LABEL: func @transfer_of_arg_pinned_to_host_is_delayed
func.func @transfer_of_arg_pinned_to_host_is_delayed(%arg0: !mesh_tensor, %arg1: !pinned_host_mesh_tensor)
  -> (!mesh_tensor, !mesh_tensor) attributes {
    "topology"=#mpmd.topology<<"m": <["x"=2, "y"=2]>>>} {
  // CHECK: mpmd.fragment
  // CHECK: mpmd.transfer
  // This transfer isn't needed until later and it's operand is pinned to host.
  // So, we will delay it.
  %t = mpmd.transfer %arg1 : (!pinned_host_mesh_tensor) -> !mesh_tensor
  %f = mpmd.fragment<mesh="m", origin=["f"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  func.return %f, %t : !mesh_tensor, !mesh_tensor
}

func.func @transfer_of_produced_value_pinned_to_host_is_delayed(%arg0: !mesh_tensor, %arg1: !pinned_host_mesh_tensor)
  -> (!mesh_tensor, !mesh_tensor) attributes {
    "topology"=#mpmd.topology<<"m": <["x"=2, "y"=2]>>>} {
  // CHECK: mpmd.fragment
  // CHECK: mpmd.fragment
  // CHECK: mpmd.transfer
  %x = mpmd.fragment<mesh="m", origin=["f"]> (%arg1) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!pinned_host_mesh_tensor) -> !pinned_host_mesh_tensor
  // This transfer isn't needed until later and it's operand is pinned to host.
  // So, we will delay it.
  %t = mpmd.transfer %x : (!pinned_host_mesh_tensor) -> !mesh_tensor
  %f = mpmd.fragment<mesh="m", origin=["f"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  func.return %f, %t : !mesh_tensor, !mesh_tensor
}
