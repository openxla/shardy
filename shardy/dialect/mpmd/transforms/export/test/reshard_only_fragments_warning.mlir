// RUN: mpmd_opt %s -mpmd-validate-no-reshards -split-input-file -verify-diagnostics 2>&1

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_tensor_sharded_x = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@m1, [{"x"}, {?}]>>

func.func private @reshard_only_callee(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=2]>} {
  return %arg0 : tensor<4x8xf32>
}

func.func @has_reshard_only_fragment(%arg0: !mesh_1_tensor loc("x")) -> (!mesh_1_tensor_sharded_x {jax.result_info = "result"}) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>> } {
  // expected-warning@+1 {{Detected reshard-only fragment 'reshard_only_callee'. This usually indicates an unexpected reshard. Operands: Replicated loc("x"). Results: #sdy.sharding<@m1, [{"x"}, {?}]> loc("result")}}
  %0 = mpmd.fragment_call<mesh="m1", origin=[]> @reshard_only_callee(%arg0) : (!mesh_1_tensor) -> !mesh_1_tensor_sharded_x
  func.return %0 : !mesh_1_tensor_sharded_x
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_tensor_sharded_x = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@m1, [{"x"}, {?}]>>

func.func private @non_reshard_callee(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {mesh_shape = #sdy.mesh<["x"=2, "y"=2]>} {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

func.func @has_non_reshard_only_fragment(%arg0: !mesh_1_tensor) -> !mesh_1_tensor_sharded_x attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>> } {
  %0 = mpmd.fragment_call<mesh="m1", origin=[]> @non_reshard_callee(%arg0) : (!mesh_1_tensor) -> !mesh_1_tensor_sharded_x
  func.return %0 : !mesh_1_tensor_sharded_x
}
