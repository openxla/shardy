// RUN: mpmd_opt %s -mpmd-delay-inferred-fragments 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>


// CHECK-LABEL: func @user_fragment_is_not_delayed
func.func @user_fragment_is_not_delayed(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
  // CHECK: mpmd.fragment<mesh="m1", origin=["f"]>
  // CHECK: mpmd.fragment<mesh="m1", origin=["g"]>
  %f = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %g = mpmd.fragment<mesh="m1", origin=["g"]> (%arg1) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %f : !mesh_1_tensor
}

// CHECK-LABEL: func @inferred_fragment_is_delayed_to_before_first_consumer
func.func @inferred_fragment_is_delayed_to_before_first_consumer(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
  // CHECK: mpmd.fragment<mesh="m1", origin=["g"]>
  // CHECK: mpmd.fragment<mesh="m1", origin=[]>
  // CHECK: mpmd.fragment<mesh="m1", origin=["f"]>
  // CHECK: return
  %inf = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %g = mpmd.fragment<mesh="m1", origin=["g"]> (%arg1) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %first_consumer = mpmd.fragment<mesh="m1", origin=["f"]> (%inf) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %inf : !mesh_1_tensor
}

// CHECK-LABEL: func @multiple_inferred_same_consumer
func.func @multiple_inferred_same_consumer(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
  // Since we visit the ops backwards, the `add` fragment must appear
  // before the `multiply` fragment as it's moved first.
  // CHECK: mpmd.fragment<mesh="m1", origin=["g"]>
  // CHECK: mpmd.fragment<mesh="m1", origin=[]>
  // CHECK-NEXT: add
  // CHECK: mpmd.fragment<mesh="m1", origin=[]>
  // CHECK-NEXT: multiply
  // CHECK: return
  %inf1 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %0 = stablehlo.multiply %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %0 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %inf2 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    %0 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %0 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %g = mpmd.fragment<mesh="m1", origin=["g"]> (%arg1) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %first_consumer = mpmd.fragment<mesh="m1", origin=["g"]> (%inf1, %inf2) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor
  func.return %first_consumer : !mesh_1_tensor
}

// CHECK-LABEL: func @chain_of_inferred_is_moved
func.func @chain_of_inferred_is_moved(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
  // CHECK: mpmd.fragment<mesh="m1", origin=["g"]>
  // CHECK: mpmd.fragment<mesh="m1", origin=[]>
  // CHECK: mpmd.fragment<mesh="m1", origin=[]>
  // CHECK: return
  %inf1 = mpmd.fragment<mesh="m1", origin=[]> (%arg0) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %inf2 = mpmd.fragment<mesh="m1", origin=[]> (%inf1) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %g = mpmd.fragment<mesh="m1", origin=["g"]> (%arg1) (%arg2: tensor<4x8xf32>) {
    mpmd.return %arg2 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  func.return %inf2 : !mesh_1_tensor
}
