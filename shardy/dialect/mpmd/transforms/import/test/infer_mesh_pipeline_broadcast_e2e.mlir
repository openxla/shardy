// RUN: mpmd_opt %s -mpmd-infer-mesh-pipeline 2>&1 | FileCheck %s

!mesh_1_tensor_4_8_f32 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor_4_8_f32 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
!mesh_3_tensor_4_8_f32 = !mpmd.mesh_tensor<"m3", tensor<4x8xf32>>


// In the next test, the result of the broadcast are assigned to m1, through the
// matmul. There are two options to assign the add to:
// 1. m2, in which case we would need to transfer its result to m1 so it's used
// by the matmul.
// 2. m1, in which case no transfer is needed. We pick this option.

// CHECK-LABEL: func @broadcast_only_user_assigned
func.func @broadcast_only_user_assigned(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>)
  ->!mesh_1_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["y"=2]>>
    >} {

// CHECK-NEXT: %[[ADD_FRAG:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg1)
// CHECK-NEXT:   stablehlo.add
// CHECK-NEXT:   return
// CHECK-NEXT: }
// CHECK-NEXT: %[[MUL_F:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%[[ADD_FRAG]], %arg2)
// CHECK-NEXT:   stablehlo.multiply
// CHECK-NEXT:   return
// CHECK-NEXT: }
  %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  %1 = mpmd.broadcast %0 : tensor<4x8xf32>
  %2 = stablehlo.multiply %1, %arg2 : tensor<4x8xf32>
  %3 = mpmd.assign %2 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  func.return %3 : !mesh_1_tensor_4_8_f32
}

// In the next test, we have a single computational op, which is clearly
// assigned to m1 (as its operands are assigned to m1 too).
// The result of this computation is then used in all meshes, so we introduce
// chained transfers.

// CHECK-LABEL: func @chained_broadcast
func.func @chained_broadcast(%arg0: !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_3_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>, <"m3": <["x"=2]>>>}
{
// CHECK-NEXT: %[[ADD_FRAG:.*]] = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
// CHECK-NEXT:   stablehlo.add
// CHECK-NEXT:   return
// CHECK-NEXT: }
// CHECK-NEXT: %[[T1:.*]] = mpmd.transfer %[[ADD_FRAG]] : {{.*}}m1{{.*}} -> {{.*}}m2{{.*}}
// CHECK-NEXT: %[[T2:.*]] = mpmd.transfer %[[T1]] : {{.*}}m2{{.*}} -> {{.*}}m3{{.*}}
  %u = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %add = stablehlo.add %u, %u : tensor<4x8xf32>
  %b0 = mpmd.broadcast %add : tensor<4x8xf32>
  %b1 = mpmd.broadcast %b0 : tensor<4x8xf32>
  %a1 = mpmd.assign %b1 : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %a2 = mpmd.assign %b1 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  %a3 = mpmd.assign %b1 : (tensor<4x8xf32>) -> !mesh_3_tensor_4_8_f32
  func.return %a1, %a2, %a3 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_3_tensor_4_8_f32
}

// The next test illustrates when a value is indirectly used by a broadcast and by a
// user of a broadcast.

// CHECK-LABEL: func @escape_broadcast
func.func @escape_broadcast(%arg0: !mesh_1_tensor_4_8_f32, %arg1: !mesh_2_tensor_4_8_f32)
  ->!mesh_3_tensor_4_8_f32 attributes {
    "topology"=#mpmd.topology<
      <"m1": <["x"=2]>>,
      <"m2": <["x"=2]>>,
      <"m3": <["x"=2]>>
    >} {

// CHECK-NEXT: %[[T0:.*]] = mpmd.transfer %arg0 : {{.*}}m1{{.*}} -> {{.*}}m2{{.*}}
// CHECK-NEXT: %[[MUL_FRAG:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[T0]], %arg1)
// CHECK-NEXT:   multiply
// CHECK-NEXT:   return
// CHECK-NEXT: }
// CHECK-DAG:  %[[T1:.*]] = mpmd.transfer %[[MUL_FRAG]] : {{.*}}m2{{.*}} -> {{.*}}m3{{.*}}
// CHECK-DAG:  %[[T2:.*]] = mpmd.transfer %arg0 : {{.*}}m1{{.*}} -> {{.*}}m3{{.*}}
// CHECK-DAG:  mpmd.fragment<mesh="m3", origin=[]> (%[[T2]], %[[T1]])

  %u0 = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>

  %a0 = mpmd.assign %u0 : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  %u1 = mpmd.unassign %a0 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>

  %u2 = mpmd.unassign %arg1 : (!mesh_2_tensor_4_8_f32) -> tensor<4x8xf32>
  %mul = stablehlo.multiply %u1, %u2 : tensor<4x8xf32>
  %bcast = mpmd.broadcast %mul : tensor<4x8xf32>
  %a = mpmd.assign %bcast : (tensor<4x8xf32>) -> !mesh_3_tensor_4_8_f32
  %t = mpmd.assign %u0 : (tensor<4x8xf32>) -> !mesh_3_tensor_4_8_f32
  %m3_frag = mpmd.fragment<mesh="m3", origin=[]> (%t, %a) (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
    %3 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
    mpmd.return %3 : tensor<4x8xf32>
  } : (!mesh_3_tensor_4_8_f32, !mesh_3_tensor_4_8_f32) -> !mesh_3_tensor_4_8_f32
  func.return %m3_frag : !mesh_3_tensor_4_8_f32
}

// CHECK-LABEL: func @arg_broadcast
func.func @arg_broadcast(%arg0: !mesh_1_tensor_4_8_f32) -> (!mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_3_tensor_4_8_f32)
  attributes {"topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>, <"m3": <["x"=2]>>>}
{
// CHECK-NEXT: %[[T1:.*]] = mpmd.transfer %arg0 : {{.*}}m1{{.*}} -> {{.*}}m2{{.*}}
// CHECK-NEXT: %[[T2:.*]] = mpmd.transfer %[[T1]] : {{.*}}m2{{.*}} -> {{.*}}m3{{.*}}
  %u = mpmd.unassign %arg0 : (!mesh_1_tensor_4_8_f32) -> tensor<4x8xf32>
  %b = mpmd.broadcast %u : tensor<4x8xf32>

  %a1 = mpmd.assign %b : (tensor<4x8xf32>) -> !mesh_1_tensor_4_8_f32
  %a2 = mpmd.assign %b : (tensor<4x8xf32>) -> !mesh_2_tensor_4_8_f32
  %a3 = mpmd.assign %b : (tensor<4x8xf32>) -> !mesh_3_tensor_4_8_f32
  func.return %a1, %a2, %a3 : !mesh_1_tensor_4_8_f32, !mesh_2_tensor_4_8_f32, !mesh_3_tensor_4_8_f32
}
