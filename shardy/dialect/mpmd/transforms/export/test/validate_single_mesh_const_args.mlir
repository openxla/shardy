// RUN: mpmd_opt %s -mpmd-validate-single-mesh-const-args -split-input-file -verify-diagnostics 2>&1

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// Test 1: Closed-over constant used on a single mesh passes validation.
func.func @const_on_single_mesh(
    %arg0: !mesh_1_tensor {jax.const = true},
    %arg1: !mesh_1_tensor) -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg1) (%a0: tensor<4x8xf32>, %a1: tensor<4x8xf32>) {
    %1 = stablehlo.add %a0, %a1 : tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// Test 2: Unused closed-over constant argument emits a warning.
// expected-warning@+1 {{Closed-over constant (argument 0) is unused.}}
func.func @unused_const_arg_emits_warning(
    %arg0: !mesh_1_tensor {jax.const = true},
    %arg1: !mesh_1_tensor) -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  func.return %arg1 : !mesh_1_tensor
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// Test 3: Closed-over constant placed on m1 and transferred to m2 emits an error.
// expected-error@+1 {{Closed-over constant (argument 0) is used across multiple meshes: ['m1', 'm2']. Constants cannot be shared across multiple meshes in MPMD. Please duplicate the constant in each mesh/stage where it is needed, or avoid closing over constants.}}
func.func @const_transferred_across_meshes(
    %arg0: !mesh_1_tensor {jax.const = true}) -> !mesh_2_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_2_tensor
  func.return %0 : !mesh_2_tensor
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// Test 4: Closed-over constant consumed by fragments on both m1 and m2 emits an error.
// expected-error@+1 {{Closed-over constant (argument 0) is used across multiple meshes: ['m1', 'm2']. Constants cannot be shared across multiple meshes in MPMD. Please duplicate the constant in each mesh/stage where it is needed, or avoid closing over constants.}}
func.func @const_used_in_fragments_on_multiple_meshes(
    %arg0: !mesh_1_tensor {jax.const = true},
    %arg1: !mesh_1_tensor) -> !mesh_2_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.fragment<mesh="m1", origin=[]> (%arg0, %arg1) (%a0: tensor<4x8xf32>, %a1: tensor<4x8xf32>) {
    %10 = stablehlo.add %a0, %a1 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %2 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %3 = mpmd.fragment<mesh="m2", origin=[]> (%1, %2) (%b0: tensor<4x8xf32>, %b1: tensor<4x8xf32>) {
    %20 = stablehlo.add %b0, %b1 : tensor<4x8xf32>
    mpmd.return %20 : tensor<4x8xf32>
  } : (!mesh_2_tensor, !mesh_2_tensor) -> !mesh_2_tensor
  func.return %3 : !mesh_2_tensor
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func private @f1(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

func.func private @f2(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {mesh_shape = #sdy.mesh<["x"=2]>} {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// Test 5: Post-lowered fragment_call across m1 and m2 emits an error.
// expected-error@+1 {{Closed-over constant (argument 0) is used across multiple meshes: ['m1', 'm2']. Constants cannot be shared across multiple meshes in MPMD. Please duplicate the constant in each mesh/stage where it is needed, or avoid closing over constants.}}
func.func @const_used_in_fragment_calls_on_multiple_meshes(
    %arg0: !mesh_1_tensor {jax.const = true},
    %arg1: !mesh_1_tensor) -> !mesh_2_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>} {
  %0 = mpmd.fragment_call<mesh="m1", origin=[]> @f1(%arg0, %arg1) : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %2 = mpmd.transfer %arg0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %3 = mpmd.fragment_call<mesh="m2", origin=[]> @f2(%1, %2) : (!mesh_2_tensor, !mesh_2_tensor) -> !mesh_2_tensor
  func.return %3 : !mesh_2_tensor
}
