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
// The identity fragment for the extra copies merges into the producing fragment.
// CHECK-LABEL: func @single_mesh_duplicate_return
func.func @single_mesh_duplicate_return(%arg0: !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2]>>>
} {
  // f1 originally returns 1 result. After merge it returns 2 (original + copy).
  // CHECK:      %[[F1:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f1"]>
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

// -----

!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4xui32>, sharding=<@mesh, [{"x"}]>>

// Test: block argument returned directly -> merged into an existing fragment
// on the same mesh.
// CHECK-LABEL: func @block_arg_passthrough
func.func @block_arg_passthrough(%arg0: !mesh_tensor) -> (!mesh_tensor, !mesh_tensor, !mesh_tensor)
  attributes {"topology"=#mpmd.topology<<"m": <["x"=2]>>>}
{
  // The block arg %arg0 is used at return[0] and return[2].
  // The identity fragment for %arg0 merges into fragment "f".
  // CHECK: %[[F:.*]]:3 = mpmd.fragment<mesh="m", origin=["f"]> (%arg0)
  // CHECK-SAME:   (%arg1: tensor<4xui32>) {
  // CHECK-NEXT:     mpmd.return %arg1, %arg1, %arg1 : tensor<4xui32>, tensor<4xui32>, tensor<4xui32>
  // CHECK-NEXT:   }
  // CHECK: return %[[F]]#1, %[[F]]#0, %[[F]]#2
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

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4xf32>>

// Regression test: multi-result fragment with multiple duplicated return values.
// This exercises the case where MergeRegionOps erases the producing fragment.
// Processing the first result merges and erases the original fragment; the
// second result must be handled from the new merged fragment, not the stale one.
// CHECK-LABEL: func @multi_result_duplicate_returns
func.func @multi_result_duplicate_returns(%arg0: !mesh_1_tensor) -> (
    !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2]>>>
} {
  // f1 produces two results, each returned twice.
  // After uniquification, f1 returns 4 values: a, b, a_copy, b_copy.
  // CHECK:      %[[F:.*]]:4 = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK-SAME:   (%arg0) (%arg1: tensor<4xf32>) {
  // CHECK-NEXT:     %[[A:.*]] = stablehlo.add %arg1, %arg1
  // CHECK-NEXT:     %[[B:.*]] = stablehlo.multiply %arg1, %arg1
  // CHECK-NEXT:     mpmd.return %[[A]], %[[B]], %[[A]], %[[B]]
  // CHECK-NEXT:   }
  // CHECK:      return %[[F]]#0, %[[F]]#2, %[[F]]#1, %[[F]]#3
  %0:2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg1: tensor<4xf32>) {
    %a = stablehlo.add %arg1, %arg1 : tensor<4xf32>
    %b = stablehlo.multiply %arg1, %arg1 : tensor<4xf32>
    mpmd.return %a, %b : tensor<4xf32>, tensor<4xf32>
  } : (!mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor)
  return %0#0, %0#0, %0#1, %0#1 : !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
}

// -----

!mesh_1_tensor_v2 = !mpmd.mesh_tensor<"m1", tensor<4xf32>>

// Regression test: chained fragments where both producers have duplicated
// return values.  F1 feeds into F2, and both F1's result and F2's result
// appear twice in the return.  The pass must uniquify all duplicates without
// leaving any behind.
// CHECK-LABEL: func @chained_fragments_duplicate_returns
func.func @chained_fragments_duplicate_returns(
    %arg0: !mesh_1_tensor_v2) -> (
    !mesh_1_tensor_v2, !mesh_1_tensor_v2, !mesh_1_tensor_v2, !mesh_1_tensor_v2
  ) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2]>>>
} {
  // CHECK:      mpmd.fragment<mesh="m1"
  // CHECK:      mpmd.fragment<mesh="m1"
  // CHECK:      return
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg1: tensor<4xf32>) {
    %a = stablehlo.add %arg1, %arg1 : tensor<4xf32>
    mpmd.return %a : tensor<4xf32>
  } : (!mesh_1_tensor_v2) -> !mesh_1_tensor_v2
  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%0) (%arg1: tensor<4xf32>) {
    %b = stablehlo.multiply %arg1, %arg1 : tensor<4xf32>
    mpmd.return %b : tensor<4xf32>
  } : (!mesh_1_tensor_v2) -> !mesh_1_tensor_v2
  return %0, %0, %1, %1 : !mesh_1_tensor_v2, !mesh_1_tensor_v2, !mesh_1_tensor_v2, !mesh_1_tensor_v2
}
