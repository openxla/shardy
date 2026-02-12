// RUN: mpmd_opt %s -mpmd-map-input-output-to-mesh='input-assignment=0@m1,1@m2 output-assignment=0@m1,1@m2' 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

sdy.mesh @m1 = <["x"=2]>
sdy.mesh @m2 = <["y"=2]>


// CHECK: func.func @multiple_input_output_mesh_assignment(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, %arg1: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>)
// CHECK-DAG: %[[UNASSIGN_1:.*]] = mpmd.unassign {origin = "user_in"} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK-DAG: %[[UNASSIGN_2:.*]] = mpmd.unassign {origin = "user_in"} %arg1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK: %[[ADD_1:.*]] = stablehlo.add %[[UNASSIGN_1]], %[[UNASSIGN_2]] : tensor<4x8xf32>
// CHECK: %[[ADD_2:.*]] = stablehlo.add %[[ADD_1]], %[[ADD_1]] : tensor<4x8xf32>
// CHECK-DAG: %[[ASSIGN_1:.*]] = mpmd.assign {origin = "user_out"} %[[ADD_1]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-DAG: %[[ASSIGN_2:.*]] = mpmd.assign {origin = "user_out"} %[[ADD_2]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
// CHECK: return %[[ASSIGN_1]], %[[ASSIGN_2]] : !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
func.func @multiple_input_output_mesh_assignment(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) attributes {
    "topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
 %1 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
 %2 = stablehlo.add %1, %1 : tensor<4x8xf32>
 func.return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func private @private_function_should_not_be_mapped
// CHECK-NOT: mpmd.mesh_tensor
// CHECK-NOT: mpmd.unassign
// CHECK-NOT: mpmd.assign
func.func private @private_function_should_not_be_mapped(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  %1 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @same_output_mapped_to_different_meshes(%arg0: !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, %arg1: !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>, !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>)
// CHECK-DAG: %[[UNASSIGN_3:.*]]= mpmd.unassign {origin = "user_in"} %arg0 : (!mpmd.mesh_tensor<"m1", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK-DAG: %[[UNASSIGN_4:.*]] = mpmd.unassign {origin = "user_in"} %arg1 : (!mpmd.mesh_tensor<"m2", tensor<4x8xf32>>) -> tensor<4x8xf32>
// CHECK: %[[ADD_3:.*]] = stablehlo.add
// CHECK-DAG: %[[ASSIGN_MESH_1:.*]] = mpmd.assign {origin = "user_out"} %[[ADD_3]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
// CHECK-DAG: %[[ASSIGN_MESH_2:.*]] = mpmd.assign {origin = "user_out"} %[[ADD_3]] : (tensor<4x8xf32>) -> !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
// CHECK:  return %[[ASSIGN_MESH_1]], %[[ASSIGN_MESH_2]]
func.func @same_output_mapped_to_different_meshes(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) attributes {
     "topology"=#mpmd.topology<
    <"m1": <["x"=2]>>, <"m2": <["y"=2]>>>} {
  %1 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  func.return %1, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}
