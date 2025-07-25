// RUN: mpmd_opt %s -canonicalize 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_1_sharded_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>

module {

// CHECK-LABEL: func @one_used_one_unused_pass_through
func.func @one_used_one_unused_pass_through(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor)
  attributes { "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>> } {

  // CHECK-NEXT: %[[F:.*]]:3 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1) (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
  // CHECK-NEXT:   mpmd.return %[[ADD]], %arg2, %arg3 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[F]]#0, %arg0

  %0:3 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
      %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
      mpmd.return %1, %arg2, %arg3 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor)
  func.return %0#0, %0#1 : !mesh_1_tensor, !mesh_1_tensor
}

// CHECK-LABEL: func @canonicalize_is_noop
func.func @canonicalize_is_noop(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor) -> !mesh_1_tensor
  attributes { "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>> } {

  // CHECK-NEXT: %[[F:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
  // CHECK-NEXT:   mpmd.return %[[ADD]] : tensor<4x8xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[F]]

  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
      %1 = stablehlo.add %arg2, %arg3 : tensor<4x8xf32>
      mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}

// CHECK-LABEL: func @canonicalize_is_noop_because_of_different_types
func.func @canonicalize_is_noop_because_of_different_types(%arg0: !mesh_1_tensor) -> !mesh_1_sharded_tensor
  attributes { "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>> } {
  // CHECK-NEXT: %[[F:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
  // CHECK-NEXT:   mpmd.return %arg1 : tensor<4x8xf32>
  // CHECK-NEXT: } : ({{.*}}tensor<4x8xf32>>) -> {{.*}}tensor<4x8xf32>, sharding=<@mesh, [{"x"}, {?}]>>
  // CHECK-NEXT: return %[[F]]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1 : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_sharded_tensor
  func.return %0 : !mesh_1_sharded_tensor
}

}
