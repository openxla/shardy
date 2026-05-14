// RUN: mpmd_opt %s -mpmd-validate-no-inferred-fragments -split-input-file -verify-diagnostics 2>&1

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4xf32>>

func.func @inferred_not_merged(%arg0: !mesh_1_tensor) -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>> } {
  // expected-warning@+1 {{Inferred fragment has not been merged (inferred by pass1, pass2)}}
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) {mpmd.inferred_by = ["pass1", "pass2"]} (%arg1: tensor<4xf32>) {

    mpmd.return %arg1 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4xf32>>

func.func @no_inferred(%arg0: !mesh_1_tensor) -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>> } {
  // No error expected
  %0 = mpmd.fragment<mesh="m1", origin=["user"]> (%arg0) (%arg1: tensor<4xf32>) {
    mpmd.return %arg1 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}
