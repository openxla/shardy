// RUN: mpmd_opt %s -mpmd-export-pipeline='fail-on-reshard-only-fragments=true fail-on-backward-deps=true' -split-input-file -verify-diagnostics 2>&1

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_tensor_sharded_x = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@m1, [{"x"}, {?}]>>

func.func @has_reshard_only_fragment(%arg0: !mesh_1_tensor) -> !mesh_1_tensor_sharded_x attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
  // expected-error@+1 {{Detected reshard-only fragment 'p0_inferred.main'. This usually indicates an unexpected reshard. Operands:}}
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

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @has_backward_dep(%arg0: !mesh_1_tensor) -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg1: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %2 = mpmd.fragment<mesh="m2", origin=[]> (%1) (%arg1: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_2_tensor) -> !mesh_2_tensor
  %3 = mpmd.transfer %2 : (!mesh_2_tensor) -> !mesh_1_tensor
  // expected-error@+1 {{Detected backward dependency in forward-only program}}
  %4 = mpmd.fragment<mesh="m1", origin=[]> (%3) (%arg1: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %4 : !mesh_1_tensor
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @has_forward_deps_only(%arg0: !mesh_1_tensor) -> !mesh_2_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg1: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %2 = mpmd.fragment<mesh="m2", origin=[]> (%1) (%arg1: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_2_tensor) -> !mesh_2_tensor
  func.return %2 : !mesh_2_tensor
}
