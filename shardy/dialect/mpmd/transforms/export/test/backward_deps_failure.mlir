// RUN: mpmd_opt %s -mpmd-validate-no-backward-deps='fail-on-backward-deps=true' -split-input-file -verify-diagnostics 2>&1

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @fragment_m1(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

func.func @fragment_m2(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

func.func @has_backward_dep(%arg0: !mesh_1_tensor) -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.fragment_call<mesh="m1", origin=["f1"(0)]> @fragment_m1(%arg0) : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %2 = mpmd.fragment_call<mesh="m2", origin=["f2"(0)]> @fragment_m2(%1) : (!mesh_2_tensor) -> !mesh_2_tensor
  %3 = mpmd.transfer %2 : (!mesh_2_tensor) -> !mesh_1_tensor
  // expected-error@+1 {{Detected backward dependency but expected forward-only pipeline since there are no transpose fragments}}
  %4 = mpmd.fragment_call<mesh="m1", origin=["f1"(0)]> @fragment_m1(%3) : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %4 : !mesh_1_tensor
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @fragment_m1(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

func.func @fragment_m2(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

func.func @has_forward_deps_only(%arg0: !mesh_1_tensor) -> !mesh_2_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.fragment_call<mesh="m1", origin=["f1"(0)]> @fragment_m1(%arg0) : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %2 = mpmd.fragment_call<mesh="m2", origin=["f2"(0)]> @fragment_m2(%1) : (!mesh_2_tensor) -> !mesh_2_tensor
  func.return %2 : !mesh_2_tensor
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func @fragment_m1(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

func.func @fragment_m2(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
  %0 = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// expected-error@+1 {{Expected forward-only program but found non-forward fragments.}}
func.func @not_forward_only_program(%arg0: !mesh_1_tensor) -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.fragment_call<mesh="m1", origin=["f"(1)]> @fragment_m1(%arg0) : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %2 = mpmd.fragment_call<mesh="m2", origin=["f2"(0)]> @fragment_m2(%1) : (!mesh_2_tensor) -> !mesh_2_tensor
  %3 = mpmd.transfer %2 : (!mesh_2_tensor) -> !mesh_1_tensor
  %4 = mpmd.fragment_call<mesh="m1", origin=["f1"(1)]> @fragment_m1(%3) : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %4 : !mesh_1_tensor
}
