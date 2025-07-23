// RUN: mpmd_opt %s -mpmd-infer-mesh-rewrite-using-analysis -mpmd-infer-mesh-finalize -verify-diagnostics

// These failures should never happen when running the entire pass pipeline,
// since validation should catch them first. But they act as a sanity check that
// all ops are indeed assigned properly.

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>>
!m1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!m2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @assign_unassign_still_exist(%arg0: !m1_tensor)
  -> !m2_tensor attributes {topology=#topology} {
  %1 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!m1_tensor) -> !m1_tensor
  // expected-error @+1 {{assigns, unassigns or broadcasts are not allowed after mesh inference}}
  %3 = mpmd.unassign %1 : (!m1_tensor) -> tensor<4x8xf32>
  // expected-error @+1 {{assigns, unassigns or broadcasts are not allowed after mesh inference}}
  %5 = mpmd.assign %3 : (tensor<4x8xf32>) -> !m2_tensor

  func.return %5 : !m2_tensor
}
