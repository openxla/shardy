// RUN: mpmd_opt %s -mpmd-uniquify-and-merge-returns -split-input-file 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4xf32>>

// CHECK-LABEL: func @no_work_needed
func.func @no_work_needed(%arg0: !mesh_1_tensor, %arg1: !mesh_2_tensor) -> (!mesh_1_tensor, !mesh_2_tensor) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>
} {
  // CHECK-NEXT: %[[F1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK:      %[[F2:.*]] = mpmd.fragment<mesh="m2", origin=["f2"]>
  // CHECK:      return %[[F1]], %[[F2]]
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<4xf32>
    mpmd.return %1 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.fragment<mesh="m2", origin=["f2"]> (%arg1) (%arg2: tensor<4xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<4xf32>
    mpmd.return %1 : tensor<4xf32>
  } : (!mesh_2_tensor) -> !mesh_2_tensor
  return %0, %1 : !mesh_1_tensor, !mesh_2_tensor
}


// Test: single mesh, one return operand used multiple times.
// Instead of creating a separate inferred fragment, the producing fragment
// directly gets extra results.
// CHECK-LABEL: func @single_mesh_one_return_operand
func.func @single_mesh_one_return_operand(%arg0: !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2]>>>
} {
  // %0 appears twice in the return -> f1 gets 1 extra result for the copy.
  // CHECK-NEXT: %[[F1:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK-SAME:   (%arg0) (%arg1: tensor<4xf32>) {
  // CHECK-NEXT:     %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<4xf32>
  // CHECK-NEXT:     mpmd.return %[[ADD]], %[[ADD]] : tensor<4xf32>, tensor<4xf32>
  // CHECK-NEXT:   }
  // CHECK:      %[[F2:.*]] = mpmd.fragment<mesh="m1", origin=["f2"]>
  // CHECK:      return %[[F2]], %[[F1]]#0, %[[F1]]#1
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg1: tensor<4xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<4xf32>
    mpmd.return %1 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%0) (%arg1: tensor<4xf32>) {
    mpmd.return %arg1 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  return %1, %0, %0 : !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
}

// Test: multiple meshes, values from different fragments duplicated.
// CHECK-LABEL: func @needs_extra_results_for_m1_with_many_values
func.func @needs_extra_results_for_m1_with_many_values(%arg0: !mesh_1_tensor, %arg1: !mesh_2_tensor
) -> (!mesh_2_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>
} {
  // f1 returns %0, used at return[1] and return[3] -> f1 gets 1 extra result
  // f3 returns %2, used at return[2], return[4], return[5] -> f3 gets 2 extra results
  // f2 returns %1, used at return[0] -> no extra (single use)
  // CHECK: %[[F1:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK: %[[F2:.*]] = mpmd.fragment<mesh="m2", origin=["f2"]>
  // CHECK: %[[F3:.*]]:3 = mpmd.fragment<mesh="m1", origin=["f3"]>
  // CHECK: return %[[F2]], %[[F1]]#0, %[[F3]]#0, %[[F1]]#1, %[[F3]]#1, %[[F3]]#2
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4xf32>) {
    mpmd.return %arg2 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.fragment<mesh="m2", origin=["f2"]> (%arg1) (%arg2: tensor<4xf32>) {
    mpmd.return %arg2 : tensor<4xf32>
  } : (!mesh_2_tensor) -> !mesh_2_tensor
  %2 = mpmd.fragment<mesh="m1", origin=["f3"]> (%arg0) (%arg2: tensor<4xf32>) {
    mpmd.return %arg2 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  return %1, %0, %2, %0, %2, %2 : !mesh_2_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
}

// -----

!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4xui32>, sharding=<@mesh, [{"x"}]>>

// Test: block argument returned directly -> merged as passthrough into an
// existing fragment on the same mesh.
// CHECK-LABEL: func @f
func.func @f(%arg0: !mesh_tensor) -> (!mesh_tensor, !mesh_tensor, !mesh_tensor)
  attributes {"topology"=#mpmd.topology<<"m": <["x"=2]>>>}
{
  // The block arg %arg0 is used at return[0] and return[2].
  // The fragment result %0 is at return[1].
  // %arg0 should be merged as a passthrough into the existing fragment.
  // CHECK: %[[F:.*]]:3 = mpmd.fragment<mesh="m", origin=["f"]> (%arg0, %arg0)
  // CHECK-SAME:   (%arg1: tensor<4xui32>, %arg2: tensor<4xui32>) {
  // CHECK-NEXT:     mpmd.return %arg1, %arg2, %arg2 : tensor<4xui32>, tensor<4xui32>, tensor<4xui32>
  // CHECK-NEXT:   }
  // CHECK-NEXT: return %[[F]]#1, %[[F]]#0, %[[F]]#2
  %0 = mpmd.fragment<mesh="m", origin=["f"]>(%arg0) (%arg1: tensor<4xui32>) {
    mpmd.return %arg1 : tensor<4xui32>
  } : (!mesh_tensor) -> !mesh_tensor
  func.return %arg0, %0, %arg0 : !mesh_tensor, !mesh_tensor, !mesh_tensor
}

// -----

!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4xui32>, sharding=<@mesh, [{"x"}]>>

// Test: identity function with no existing fragment -> must create a fallback.
// CHECK-LABEL: func @identity_function
func.func @identity_function(%arg0: !mesh_tensor) -> !mesh_tensor
  attributes {"topology"=#mpmd.topology<<"m": <["x"=2]>>>}
{
  // CHECK-NEXT: %[[F:.*]] = mpmd.fragment<mesh="m", origin=[]> (%arg0) (%arg1: tensor<4xui32>) {
  // CHECK-NEXT:   mpmd.return %arg1 : tensor<4xui32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[F]]
  func.return %arg0 : !mesh_tensor
}
