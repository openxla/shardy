// RUN: mpmd_opt %s -mpmd-infer-mesh-pipeline='infer-cross-mesh-reductions=true' 2>&1 | FileCheck %s



!m1_4x8 = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!m2_4x8 = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
!m3_4x8 = !mpmd.mesh_tensor<"m3", tensor<4x8xf32>>

!m2_4x1x8 = !mpmd.mesh_tensor<"m2", tensor<4x1x8xf32>>
!m3_4x1x8 = !mpmd.mesh_tensor<"m3", tensor<4x1x8xf32>>

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>>

// CHECK-LABEL: func @assign_outputs_with_empty_src_set(
func.func @assign_outputs_with_empty_src_set(%arg0: !m3_4x8, %arg1: !m2_4x8)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
// CHECK:       %[[T0:.*]] = mpmd.transfer %arg0 {{.*}}m3{{.*}} -> {{.*}}m2
// CHECK-NEXT:  %[[ADD:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[T0]], %arg1)
// CHECK:       return %[[ADD]]
  %4 = mpmd.unassign %arg0 : (!m3_4x8) -> tensor<4x8xf32>
  %5 = mpmd.unassign %arg1 : (!m2_4x8) -> tensor<4x8xf32>

  %6 = stablehlo.add %4, %5 : tensor<4x8xf32>

  // The return value will have empty src_set. If
  // `infer-cross-mesh-reductions = false`, we would have an error
  // here. But instead, we assign it to the first mesh and introduce transfers.
  // Note that this isn't optimal, but inferring the right mesh is hard. Until
  // we improve the inference algorithm, users will need to adjust their
  // assignments manually if they want optimal results.
  func.return %6 : tensor<4x8xf32>
}

// CHECK-LABEL: func @concat_reduce(%arg0: !mpmd.mesh_tensor<"m2"
// CHECK-SAME:    %arg1: !mpmd.mesh_tensor<"m3"
func.func @concat_reduce(%arg0: !m2_4x1x8, %arg1: !m3_4x1x8)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
  %init = stablehlo.constant dense<1.0> : tensor<f32>
  %1 = mpmd.unassign %arg0 : (!m2_4x1x8) -> tensor<4x1x8xf32>
  %2 = mpmd.unassign %arg1 : (!m3_4x1x8) -> tensor<4x1x8xf32>
  // CHECK-NEXT:  %[[RESHAPE_M2:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%arg0)
  // CHECK-NEXT:     stablehlo.reshape
  // CHECK:       %[[RESHAPE_M3:.*]] = mpmd.fragment<mesh="m3", origin=[]> (%arg1)
  // CHECK-NEXT:     stablehlo.reshape
  // CHECK:       %[[T0:.*]] = mpmd.transfer %[[RESHAPE_M3]] {{.*}}m3{{.*}} -> {{.*}}m2
  // CHECK-NEXT:  %[[ADD:.*]] = mpmd.fragment<mesh="m2", origin=[]> (%[[RESHAPE_M2]], %[[T0]])
  // CHECK:       return %[[ADD]]
  %concat = "stablehlo.concatenate"(%1, %2) <{dimension = 1 : i64}> :
    (tensor<4x1x8xf32>, tensor<4x1x8xf32>) -> tensor<4x2x8xf32>
  %reduce = stablehlo.reduce(%concat init: %init) applies stablehlo.maximum across dimensions = [1] :
    (tensor<4x2x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  func.return %reduce : tensor<4x8xf32>
}
