// RUN: mpmd_opt %s -mpmd-validate-no-reshards -split-input-file -verify-diagnostics 2>&1

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_tensor_sharded_x = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@m1, [{"x"}, {?}]>>

func.func @has_reshard_only_fragment(%arg0: !mesh_1_tensor loc("x")) -> (!mesh_1_tensor_sharded_x {jax.result_info = "result"}) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
  // expected-error@+1 {{Detected reshard-only fragment. This usually indicates an unexpected reshard. Operands: Replicated loc("x"). Results: #sdy.sharding<@m1, [{"x"}, {?}]> loc("result")}}
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor_sharded_x
  func.return %0 : !mesh_1_tensor_sharded_x
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_tensor_sharded_x = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@m1, [{"x"}, {?}]>>

func.func @has_non_reshard_only_fragment(%arg0: !mesh_1_tensor) -> !mesh_1_tensor_sharded_x attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg1: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor_sharded_x
  func.return %0 : !mesh_1_tensor_sharded_x
}
