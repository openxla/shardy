// RUN: mpmd_opt %s -mpmd-validate-no-param-transfers -split-input-file -verify-diagnostics 2>&1

// Test: With default options (fail-on-param-transfers=false), a matching
// transfer should emit a warning.

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @has_param_transformer_transfer_warning(
    %arg0: !mesh_1_tensor loc("params['transformer/layer_0/attention']['w']")) -> !mesh_2_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  // expected-warning@+1 {{Detected cross-mesh transfer of a parameter matching "params['transformer" from mesh "m1" to "m2". JAX locations: ["params['transformer/layer_0/attention']['w']"]}}
  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_2_tensor
  func.return %0 : !mesh_2_tensor
}

// -----

// Test: TransferOp whose operand location has "transformer" but NOT
// "params['transformer" should pass without warning.

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @non_param_transformer_transfer(
    %arg0: !mesh_1_tensor loc("transformer/compute_activations")) -> !mesh_2_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_2_tensor
  func.return %0 : !mesh_2_tensor
}

// -----

// Test: TransferOp whose operand location has "params['embedding" should
// pass without warning (only transformer params are flagged by default).

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @embedding_param_transfer(
    %arg0: !mesh_1_tensor loc("params['embedding/lookup']['w']")) -> !mesh_2_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_2_tensor
  func.return %0 : !mesh_2_tensor
}

// -----

// Test: Intra-mesh transfer should NOT warn even with matching location.

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

func.func @intra_mesh_param_transfer(
    %arg0: !mesh_1_tensor loc("params['transformer/layer_0']['w']")) -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>} {
  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}

// -----

// Test: Non-MPMD function should be skipped.

func.func @non_mpmd_function(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  func.return %arg0 : tensor<4x8xf32>
}
