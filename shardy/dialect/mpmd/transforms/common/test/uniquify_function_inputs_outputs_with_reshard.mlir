// RUN: mpmd_opt %s -mpmd-uniquify-function-inputs-outputs -split-input-file | FileCheck %s

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


// CHECK-LABEL: func @single_mesh_one_return_operand
func.func @single_mesh_one_return_operand(%arg0: !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2]>>>
} {
  // CHECK-NEXT: %[[F1:.*]]:3 = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK:        %[[ADD:.*]] = stablehlo.add
  // CHECK:        mpmd.return %[[ADD]], %[[ADD]], %[[ADD]]
  // CHECK:      %[[F2:.*]] = mpmd.fragment<mesh="m1", origin=["f2"]> (%[[F1]]#0)
  // CHECK-NOT:  mpmd.inferred_by
  // CHECK:      return %[[F2]], %[[F1]]#1, %[[F1]]#2
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg1: tensor<4xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<4xf32>
    mpmd.return %1 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%0) (%arg1: tensor<4xf32>) {
    mpmd.return %arg1 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  return %1, %0, %0 : !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
}

// CHECK-LABEL: func @needs_fragment_for_m1_with_many_values
func.func @needs_fragment_for_m1_with_many_values(%arg0: !mesh_1_tensor, %arg1: !mesh_2_tensor
) -> (!mesh_2_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>
} {
  // CHECK-NEXT: %[[F1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK:      %[[F2:.*]] = mpmd.fragment<mesh="m2", origin=["f2"]>
  // CHECK:      %[[F3:.*]]:5 = mpmd.fragment<mesh="m1", origin=["f3"]> (%[[F1]], %arg0)
  // CHECK-SAME:   (%[[A1:.*]]: tensor<4xf32>, %[[A2:.*]]: tensor<4xf32>)
  // CHECK-NEXT:   mpmd.return %[[A1]], %[[A1]], %[[A2]], %[[A2]], %[[A2]]
  // CHECK-NEXT: }
  // CHECK-NOT:  mpmd.inferred_by
  // CHECK-NEXT: return %[[F2]], %[[F3]]#0, %[[F3]]#2, %[[F3]]#1, %[[F3]]#3, %[[F3]]#4
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

// CHECK-LABEL: func @needs_fragment_for_m1_and_m2
func.func @needs_fragment_for_m1_and_m2(%arg0: !mesh_1_tensor, %arg1: !mesh_2_tensor
) -> (!mesh_1_tensor, !mesh_2_tensor, !mesh_2_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>
} {
  // CHECK:     %[[F1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK:     %[[F2:.*]]:2 = mpmd.fragment<mesh="m2", origin=["f2"]>
  // CHECK:     %[[F3:.*]]:4 = mpmd.fragment<mesh="m1", origin=["f3"]> (%[[F1]], %arg0)
  // CHECK-NOT: mpmd.inferred_by
  // CHECK:     return %[[F3]]#0, %[[F2]]#0, %[[F2]]#1, %[[F3]]#2, %[[F3]]#1, %[[F3]]#3
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4xf32>) {
    mpmd.return %arg2 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %1 = mpmd.fragment<mesh="m2", origin=["f2"]> (%arg1) (%arg2: tensor<4xf32>) {
    mpmd.return %arg2 : tensor<4xf32>
  } : (!mesh_2_tensor) -> !mesh_2_tensor
  %2 = mpmd.fragment<mesh="m1", origin=["f3"]> (%arg0) (%arg2: tensor<4xf32>) {
    mpmd.return %arg2 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  return %0, %1, %1, %2, %0, %2 : !mesh_1_tensor, !mesh_2_tensor, !mesh_2_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
}

// This test verifies that an explicit fragment and an inferred fragment
// (created by the UniquifyFunctionInputsOutputsPass for the duplicated return
// of the transfer result) are merged sideways. Without sideways merge, the
// transfer result would produce a separate inferred fragment call on m1.
// CHECK-LABEL: func @test_sideways_merge
func.func @test_sideways_merge(%arg0: !mesh_1_tensor, %arg1: !mesh_2_tensor)
  -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) attributes {
      "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>
} {
  // CHECK:     %[[TRANSFER:.*]] = mpmd.transfer %arg1
  // CHECK:     %[[RES:.*]]:3 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0, %[[TRANSFER]])
  // CHECK-NOT: mpmd.fragment<mesh="m1"
  // CHECK-NOT: mpmd.inferred_by
  // CHECK:     return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4xf32>) {
    %4 = stablehlo.add %arg2, %arg2 : tensor<4xf32>
    mpmd.return %4 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor

  %1 = mpmd.transfer %arg1 : (!mesh_2_tensor) -> !mesh_1_tensor

  func.return %0, %1, %1 : !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
}

// -----

!dist_mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4xf32>, sharding=<@mesh, [{"x"}]>>

module {

// CHECK-LABEL: func @single_mesh_one_return_operand
func.func @single_mesh_one_return_operand_with_global_view(%arg0: !dist_mesh_tensor) -> (!dist_mesh_tensor, !dist_mesh_tensor, !dist_mesh_tensor) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2]>>>
} {
  // CHECK-NEXT: %[[F1:.*]]:3 = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK:        %[[ADD:.*]] = stablehlo.add
  // CHECK:        mpmd.return %[[ADD]], %[[ADD]], %[[ADD]]
  // CHECK:      %[[F2:.*]] = mpmd.fragment<mesh="m1", origin=["f2"]> (%[[F1]]#0)
  // CHECK-NOT:  mpmd.inferred_by
  // CHECK:      return %[[F2]], %[[F1]]#1, %[[F1]]#2
  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg1: tensor<4xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<4xf32>
    mpmd.return %1 : tensor<4xf32>
  } : (!dist_mesh_tensor) -> !dist_mesh_tensor
  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%0) (%arg1: tensor<4xf32>) {
    mpmd.return %arg1 : tensor<4xf32>
  } : (!dist_mesh_tensor) -> !dist_mesh_tensor
  return %1, %0, %0 : !dist_mesh_tensor, !dist_mesh_tensor, !dist_mesh_tensor
}
}

// -----

!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4xui32>, sharding=<@mesh, [{"x"}]>>

// CHECK-LABEL: func @f
func.func @f(%arg0: !mesh_tensor) -> (!mesh_tensor, !mesh_tensor, !mesh_tensor)
  attributes {"topology"=#mpmd.topology<<"m": <["x"=2]>>>}
{
  // CHECK-NEXT: %[[F:.*]]:3 = mpmd.fragment<mesh="m", origin=["f"]> (%arg0) (%arg1: tensor<4xui32>) {
  // CHECK-NEXT:   return %arg1, %arg1, %arg1
  // CHECK-NEXT: }
  // CHECK-NOT:  mpmd.inferred_by
  // CHECK-NEXT: return %[[F]]#1, %[[F]]#0, %[[F]]#2
  %0 = mpmd.fragment<mesh="m", origin=["f"]>(%arg0) (%arg1: tensor<4xui32>) {
    mpmd.return %arg1 : tensor<4xui32>
  } : (!mesh_tensor) -> !mesh_tensor
  func.return %arg0, %0, %arg0 : !mesh_tensor, !mesh_tensor, !mesh_tensor
}

// -----

!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4xui32>, sharding=<@mesh, [{"x"}]>>

// CHECK-LABEL: func @identity_function
func.func @identity_function(%arg0: !mesh_tensor) -> !mesh_tensor
  attributes {"topology"=#mpmd.topology<<"m": <["x"=2]>>>}
{
  // CHECK-NEXT: %[[F:.*]] = mpmd.fragment<mesh="m", origin=[]> (%arg0) {mpmd.inferred_by = ["uniquify"]} (%arg1: tensor<4xui32>) {
  // CHECK-NEXT:   return %arg1
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[F]]
  func.return %arg0 : !mesh_tensor
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4xf32>>

// This test verifies that the pass does NOT merge with an unrelated earlier
// fragment (f1) when the nearest same-mesh fragment (f2) can't be moved past
// the transfer. Merging with f1 would introduce a false dependency — f1's
// computation (which only needs %arg0) would be blocked on the cross-mesh
// transfer completing.
// CHECK-LABEL: func @test_complex_reordering_merge
func.func @test_complex_reordering_merge(%arg0: !mesh_1_tensor, %arg1: !mesh_2_tensor)
  -> (!mesh_1_tensor, !mesh_2_tensor, !mesh_1_tensor, !mesh_1_tensor) attributes {
      "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>
} {
  // CHECK: %[[F1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK: %[[F2:.*]] = mpmd.fragment<mesh="m1", origin=["f2"]>
  // CHECK: %[[T3:.*]] = mpmd.transfer %[[F2]]
  // CHECK: %[[T2:.*]] = mpmd.transfer %arg1
  // CHECK: %[[INF:.*]]:2 = mpmd.fragment<mesh="m1", origin=[]> (%[[T2]]) {mpmd.inferred_by = ["uniquify"]}
  // CHECK: return %[[F1]], %[[T3]], %[[INF]]#0, %[[INF]]#1

  %0 = mpmd.fragment<mesh="m1", origin=["f1"]> (%arg0) (%arg2: tensor<4xf32>) {
    %4 = stablehlo.add %arg2, %arg2 : tensor<4xf32>
    mpmd.return %4 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor

  %1 = mpmd.fragment<mesh="m1", origin=["f2"]> (%arg0) (%arg2: tensor<4xf32>) {
    mpmd.return %arg2 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor

  %3 = mpmd.transfer %1 : (!mesh_1_tensor) -> !mesh_2_tensor

  %2 = mpmd.transfer %arg1 : (!mesh_2_tensor) -> !mesh_1_tensor

  func.return %0, %3, %2, %2 : !mesh_1_tensor, !mesh_2_tensor, !mesh_1_tensor, !mesh_1_tensor
}

// -----

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4xf32>>

// This test verifies that the inferred fragment (created for the duplicated
// return of %0) merges into f_a (the producer of %0) rather than f_b (the
// nearest preceding same-mesh fragment). Merging into f_b would introduce a
// false dependency: f_b's multiply — which only needs %arg0 — would be
// blocked on %0 (the add result) becoming available.
// CHECK-LABEL: func @test_producer_preferred_over_bystander
func.func @test_producer_preferred_over_bystander(%arg0: !mesh_1_tensor)
  -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>>
} {
  // CHECK:     %[[FA:.*]]:3 = mpmd.fragment<mesh="m1", origin=["f_a"]> (%arg0)
  // CHECK:       %[[ADD:.*]] = stablehlo.add
  // CHECK:       mpmd.return %[[ADD]], %[[ADD]], %[[ADD]]
  // CHECK:     %[[FB:.*]] = mpmd.fragment<mesh="m1", origin=["f_b"]> (%arg0)
  // CHECK:       %[[MUL:.*]] = stablehlo.multiply
  // CHECK:       mpmd.return %[[MUL]]
  // CHECK-NOT: mpmd.inferred_by
  // CHECK:     return %[[FA]]#0, %[[FB]], %[[FA]]#1, %[[FA]]#2
  %0 = mpmd.fragment<mesh="m1", origin=["f_a"]> (%arg0) (%arg1: tensor<4xf32>) {
    %2 = stablehlo.add %arg1, %arg1 : tensor<4xf32>
    mpmd.return %2 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor

  %1 = mpmd.fragment<mesh="m1", origin=["f_b"]> (%arg0) (%arg1: tensor<4xf32>) {
    %2 = stablehlo.multiply %arg1, %arg1 : tensor<4xf32>
    mpmd.return %2 : tensor<4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor

  func.return %0, %1, %0, %0 : !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
}
