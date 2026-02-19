// RUN: mpmd_opt %s -mpmd-validate-no-backward-deps -split-input-file -verify-diagnostics 2>&1

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @has_backward_dep(%arg0: !mesh_1_tensor) -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.fragment<mesh="m1", origin=["c1"(0)], stage=0> (%arg0) (%arg1: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %2 = mpmd.fragment<mesh="m2", origin=["c2"(0)], stage=1> (%1) (%arg1: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_2_tensor) -> !mesh_2_tensor
  %3 = mpmd.transfer %2 : (!mesh_2_tensor) -> !mesh_1_tensor
  // expected-error@+1 {{Detected backward dependency in forward-only program}}
  %4 = mpmd.fragment<mesh="m1", origin=["c3"(0)], stage=0> (%3) (%arg1: tensor<4x8xf32>) {
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
  %0 = mpmd.fragment<mesh="m1", origin=["c1"(0)], stage=0> (%arg0) (%arg1: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %2 = mpmd.fragment<mesh="m2", origin=["c2"(0)], stage=1> (%1) (%arg1: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_2_tensor) -> !mesh_2_tensor
  func.return %2 : !mesh_2_tensor
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @not_forward_only_program(%arg0: !mesh_1_tensor) -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.fragment<mesh="m1", origin=["c1"(0)], stage=0> (%arg0) (%arg1: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %2 = mpmd.fragment<mesh="m2", origin=["c2"(1)], stage=1> (%1) (%arg1: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_2_tensor) -> !mesh_2_tensor
  %3 = mpmd.transfer %2 : (!mesh_2_tensor) -> !mesh_1_tensor
  %4 = mpmd.fragment<mesh="m1", origin=["c3"(0)], stage=0> (%3) (%arg1: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %4 : !mesh_1_tensor
}
