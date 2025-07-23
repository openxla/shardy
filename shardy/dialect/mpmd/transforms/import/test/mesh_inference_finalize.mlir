// RUN: mpmd_opt %s -mpmd-infer-mesh-finalize 2>&1 | FileCheck -implicit-check-not use_set -implicit-check-not src_set %s

// CHECK-LABEL: func @assign_of_unassign_same_mesh_no_other_users
func.func @assign_of_unassign_same_mesh_no_other_users(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">},
  %arg1: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["y"=2]>>>}
{
  %0 = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %1 = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %2 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %3 = mpmd.assign %1 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT: return %arg0, %arg1
  return %2, %3 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
}

// CHECK-LABEL: func @assign_of_unassign_same_mesh_with_other_users
func.func @assign_of_unassign_same_mesh_with_other_users(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">},
  %arg1: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["y"=2]>>>}
{
  %0 = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %1 = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %2 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %3 = mpmd.assign %1 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %arg0 :  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  %4 = mpmd.assign %0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

  // CHECK-NEXT: return %arg0, %arg1, %[[TRANSFER]]
  return %2, %3, %4 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
}

// CHECK-LABEL: func @duplicate_assign
func.func @duplicate_assign(%arg0: tensor<4x8xf32>)
  -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["y"=2]>>>}
{
  %2 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %3 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT: return %arg0, %arg0
  return %2, %3 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
}

// CHECK-LABEL: func @different_mesh_assign
// CHECK-SAME:   %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
func.func @different_mesh_assign(%arg0: tensor<4x8xf32>)
  -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["y"=2]>>>}
{
  %2 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %arg0 :  (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  %3 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT: return %arg0, %[[TRANSFER]]
  return %2, %3 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
}

// CHECK-LABEL: func @dedup_assign_of_unassign_and_transfer
func.func @dedup_assign_of_unassign_and_transfer(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">}
) -> (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["y"=2]>>>}
{
  // CHECK-NEXT: %[[FRAGMENT_M1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
  // CHECK:      %[[TRANSFER:.*]] = mpmd.transfer %[[FRAGMENT_M1]] : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  // CHECK-NEXT: %[[FRAGMENT_M2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[TRANSFER]])
  // CHECK:      return %[[TRANSFER]], %[[FRAGMENT_M2]]

  %f = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg1: tensor<4x8xf32>) {
    %13 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %13 : tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %y = mpmd.unassign %f : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %z = mpmd.assign %y : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  %w = mpmd.transfer %f : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  %user_z = mpmd.fragment<mesh="m2", origin=[]> (%z) (%arg1: tensor<4x8xf32>) {
    %13 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %13 : tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  return %w, %user_z : !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
}


// CHECK-LABEL: func @rewrite_broadcast
func.func @rewrite_broadcast(%arg0: tensor<4x8xf32>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["x"=2]>>>} {
  // CHECK-NEXT: %[[FRAG:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
  // CHECK-NEXT:   stablehlo.add
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[TRANSFER:.*]] = mpmd.transfer %[[FRAG]]
  // CHECK-NEXT: return %[[FRAG]], %[[TRANSFER]]
  %0 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %1 = mpmd.fragment<mesh="m1", origin=[]> (%0) (%arg1: tensor<4x8xf32>) {
    %8 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %8 : tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %2 = mpmd.unassign %1 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %3 = mpmd.assign %2 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %4 = mpmd.unassign %3 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %5 = mpmd.broadcast %4 : tensor<4x8xf32>
  %6 = mpmd.assign %5 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %7 = mpmd.assign %5 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  return %6, %7 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
}

// CHECK-LABEL: func @lower_mpmd_reduce_single_user(%arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:                     %arg1: !mpmd.mesh_tensor<"m2"
func.func @lower_mpmd_reduce_single_user(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, %arg1: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m3", tensor<4x8xf32>> attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["y"=2]>>, <"m3" : <["z"=2]>>>} {
// CHECK-NEXT:  %[[ADD_LOCAL_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg0
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[ADD_LOCAL_2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%arg1, %arg1
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[TRANSFER_1:.*]] = mpmd.transfer %[[ADD_LOCAL_1]]
// CHECK-NEXT:  %[[TRANSFER_2:.*]] = mpmd.transfer %[[ADD_LOCAL_2]]
// CHECK-NEXT:  %[[ADD_FINAL:.*]] = mpmd.fragment<mesh="m3", origin=[]> (%[[TRANSFER_1]], %[[TRANSFER_2]])
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[ADD_FINAL]]
  %0 = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %1 = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %2 = mpmd.reduce<add>  {mpmd.use_set = #mpmd.meshes_with_origins<"m3">} %0, %1, %0, %1 : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %3 = mpmd.assign %2 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m3", tensor<4x8xf32>>
  return %3 : !mpmd.mesh_tensor<"m3", tensor<4x8xf32>>
}

// CHECK-LABEL: func @lower_mpmd_reduce_multiple_users(%arg0: !mpmd.mesh_tensor<"m1"
// CHECK-SAME:                     %arg1: !mpmd.mesh_tensor<"m2"
func.func @lower_mpmd_reduce_multiple_users(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, %arg1: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["y"=2]>>, <"m3" : <["z"=2]>>>} {
// CHECK-NEXT:  %[[ADD_LOCAL_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg0
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[ADD_LOCAL_2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%arg1, %arg1
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[TRANSFER_2:.*]] = mpmd.transfer %[[ADD_LOCAL_2]]
// CHECK-NEXT:  %[[ADD_1:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ADD_LOCAL_1]], %[[TRANSFER_2]])
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[TRANSFER_1:.*]] = mpmd.transfer %[[ADD_LOCAL_1]]
// CHECK-NEXT:  %[[ADD_2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[TRANSFER_1]], %[[ADD_LOCAL_2]])
// CHECK-NEXT:    %[[ADD:.*]] = stablehlo.add %arg2, %arg3
// CHECK-NEXT:    mpmd.return %[[ADD]]
// CHECK-NEXT:  }
// CHECK-NEXT:  return %[[ADD_1]], %[[ADD_2]]
  %0 = mpmd.unassign %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %1 = mpmd.unassign %arg1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
  %2 = mpmd.reduce<add>  {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} %0, %1, %0, %1 : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %3 = mpmd.assign %2 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  %4 = mpmd.assign %2 : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
  return %3, %4 : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
}

// CHECK-LABEL: func @erases_leftover_use_set_and_src_set
// CHECK-NOT: meshes_with_origins
// CHECK-NOT: use_set
// CHECK-NOT: src_set
func.func @erases_leftover_use_set_and_src_set(
  %arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>> {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">}
) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2]>>, <"m2" : <["y"=2]>>>}
{
  %f = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg1: tensor<4x8xf32>) {
    %13 = stablehlo.add %arg1, %arg1 {mpmd.use_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
    %14 = stablehlo.add %arg1, %arg1 {mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
    mpmd.return %13 : tensor<4x8xf32>
  } : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
  return %f : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
}
