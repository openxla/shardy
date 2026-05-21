// RUN: mpmd_opt %s -mpmd-validate-no-param-transfers='fail-on-param-transfers=true' -split-input-file -verify-diagnostics 2>&1

// Test: With fail-on-param-transfers=true, a matching transfer should emit
// an error.

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @has_param_transformer_transfer(
    %arg0: !mesh_1_tensor loc("params['transformer/layer_0/attention']['w']")) -> !mesh_2_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  // expected-error@+1 {{Detected cross-mesh transfer of a parameter matching "params['transformer" from mesh "m1" to "m2". JAX locations: ["params['transformer/layer_0/attention']['w']"]}}
  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_2_tensor
  func.return %0 : !mesh_2_tensor
}
