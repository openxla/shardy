// RUN: mpmd_opt %s -mpmd-introduce-transfers 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-LABEL: func @push_assign_through_single_add
func.func @push_assign_through_single_add(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
// CHECK-DAG:  %[[ARG1_TRANSFER:.*]] = mpmd.transfer %arg1
// CHECK-DAG:  %[[ADD_FRAG:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %[[ARG1_TRANSFER]])
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-DAG:  return %[[ADD_FRAG]]
  %m1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %m2 = mpmd.unassign %arg1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>
  %add = stablehlo.add %m1, %m2 : tensor<4x8xf32>
  %add_m1 = mpmd.assign %add : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  func.return %add_m1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @push_assign_through_multiple_add
func.func @push_assign_through_multiple_add(%arg0: !mesh_1_tensor_4_8_f32,
  %arg1: !mesh_2_tensor_4_8_f32, %arg2: !mesh_1_tensor_4_8_f32,
  %arg3: !mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {

// CHECK-DAG:  %[[ARG1_TRANSFER:.*]] = mpmd.transfer %arg1
// CHECK-DAG:  %[[ADD_FRAG_0:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %[[ARG1_TRANSFER]])
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg4, %arg5
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-DAG:  %[[ADD_FRAG_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ADD_FRAG_0]], %arg2)
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg4, %arg5
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-DAG:  %[[ARG3_TRANSFER:.*]] = mpmd.transfer %arg3
// CHECK-DAG:  %[[ADD_FRAG_2:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ARG3_TRANSFER]], %[[ADD_FRAG_1]])
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg4, %arg5
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-DAG: return %[[ADD_FRAG_2]]
  %arg0_m1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %arg1_m2 = mpmd.unassign %arg1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>
  %arg2_m1 = mpmd.unassign %arg2 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %arg3_m2 = mpmd.unassign %arg3 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>

  %add_0 = stablehlo.add %arg0_m1, %arg1_m2 : tensor<4x8xf32>
  %add_1 = stablehlo.add %add_0, %arg2_m1 : tensor<4x8xf32>
  %add_2 = stablehlo.add %arg3_m2, %add_1 : tensor<4x8xf32>

  %add_m1 = mpmd.assign %add_2 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  func.return %add_m1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @push_assign_only_if_adding_unassign
func.func @push_assign_only_if_adding_unassign(%arg0: !mesh_1_tensor_4_8_f32,
  %arg1: !mesh_2_tensor_4_8_f32, %arg2: !mesh_1_tensor_4_8_f32,
  %arg3: !mesh_2_tensor_4_8_f32) -> !mesh_1_tensor_4_8_f32
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  // No AssignOps are pushed through since one of the operands is neither an
  // UnassignOp nor an AddOp.
  // CHECK-NOT: fragment
  %arg0_m1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %arg0_modified = stablehlo.negate %arg0_m1 : tensor<4x8xf32>
  %arg1_m2 = mpmd.unassign %arg1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>
  %arg2_m1 = mpmd.unassign %arg2 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %arg3_m2 = mpmd.unassign %arg3 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>

  %add_0 = stablehlo.add %arg0_modified, %arg1_m2 : tensor<4x8xf32>
  %add_1 = stablehlo.add %add_0, %arg2_m1 : tensor<4x8xf32>
  %add_2 = stablehlo.add %arg3_m2, %add_1 : tensor<4x8xf32>

  %add_m1 = mpmd.assign %add_2 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  func.return %add_m1 : !mesh_1_tensor_4_8_f32
}

// CHECK-LABEL: func @transfers_are_deduped
func.func @transfers_are_deduped(%arg0: !mesh_1_tensor_4_8_f32) -> (!mesh_2_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_2_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %arg0
  // CHECK-NEXT: return %[[TRANSFER]], %[[TRANSFER]], %[[TRANSFER]]
  %arg0_m1_0 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %arg0_m2_0 = mpmd.assign %arg0_m1_0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32

  %arg0_m1_1 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %arg0_m2_1 = mpmd.assign %arg0_m1_1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32

  %arg0_m1_2 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %arg0_m1_2_self = mpmd.assign %arg0_m1_2 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %arg0_m1_3 = mpmd.unassign %arg0_m1_2_self : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %arg0_m2_2 = mpmd.assign %arg0_m1_3 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32

  func.return %arg0_m2_0, %arg0_m2_1, %arg0_m2_2 : !mesh_2_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_2_tensor_4_8_f32
}

// CHECK-LABEL: func @assign_of_unassign_with_memory_kinds_becomes_transfer_on_same_mesh
func.func @assign_of_unassign_with_memory_kinds_becomes_transfer_on_same_mesh(
    %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="pinned_host">
  ) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="device">
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
// CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %arg0
// CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="pinned_host">)
// CHECK-SAME: -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="device">
  %m1 = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="pinned_host">) -> tensor<4x8xf32>
  %m2 = mpmd.assign %m1 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="device">
  func.return %m2 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="device">
}

// CHECK-LABEL: func @assign_of_unassign_with_memory_kinds_becomes_transfer_on_different_mesh
func.func @assign_of_unassign_with_memory_kinds_becomes_transfer_on_different_mesh(
    %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="pinned_host">
  ) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, memory_kind="device">
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
// CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %arg0
// CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="pinned_host">)
// CHECK-SAME: -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, memory_kind="device">
  %m1 = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="pinned_host">) -> tensor<4x8xf32>
  %m2 = mpmd.assign %m1 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, memory_kind="device">
  func.return %m2 : !mpmd.mesh_tensor<"m2", tensor<4x8xf32>, memory_kind="device">
}

// CHECK-LABEL: func @no_memory_kind_to_pinned_host_is_transfer
func.func @no_memory_kind_to_pinned_host_is_transfer(
    %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  ) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="pinned_host">
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
// CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %arg0
// CHECK-SAME: (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
// CHECK-SAME: -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="pinned_host">
  %m1 = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %m2 = mpmd.assign %m1 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="pinned_host">
  func.return %m2 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, memory_kind="pinned_host">
}
