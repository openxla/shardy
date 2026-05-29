// RUN: mpmd_opt %s -mpmd-uniquify-function-inputs-outputs -split-input-file 2>&1 | FileCheck %s

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
  // CHECK:      %[[F2:.*]] = mpmd.fragment<mesh="m1", origin=["f2"]> (%[[F1]]#0)
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
  // CHECK:      return %[[F2]], %[[F3]]#0, %[[F3]]#2, %[[F3]]#1, %[[F3]]#3, %[[F3]]#4
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
  // CHECK: %[[F1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK: %[[F2:.*]]:2 = mpmd.fragment<mesh="m2", origin=["f2"]>
  // CHECK: %[[F3:.*]]:4 = mpmd.fragment<mesh="m1", origin=["f3"]> (%[[F1]], %arg0)
  // CHECK: return %[[F3]]#0, %[[F2]]#0, %[[F2]]#1, %[[F3]]#2, %[[F3]]#1, %[[F3]]#3
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

// -----

!dist_mesh_tensor = !mpmd.mesh_tensor<"m1", tensor<4xf32>, sharding=<@mesh, [{"x"}]>>

module {

// CHECK-LABEL: func @single_mesh_one_return_operand
func.func @single_mesh_one_return_operand_with_global_view(%arg0: !dist_mesh_tensor) -> (!dist_mesh_tensor, !dist_mesh_tensor, !dist_mesh_tensor) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2]>>>
} {
  // CHECK-NEXT: %[[F1:.*]]:3 = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK:      %[[F2:.*]] = mpmd.fragment<mesh="m1", origin=["f2"]> (%[[F1]]#0)
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
  // CHECK-NEXT: %[[F1:.*]] = mpmd.fragment<mesh="m", origin=["f"]> (%arg0) (%arg1: tensor<4xui32>) {
  // CHECK-NEXT:   return %arg1
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[F2:.*]]:2 = mpmd.fragment<mesh="m", origin=[]> (%arg0) {mpmd.inferred_by = ["uniquify"]} (%arg1: tensor<4xui32>) {
  // CHECK-NEXT:   return %arg1, %arg1
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[F2]]#0, %[[F1]], %[[F2]]#1
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

// Block-argument-only inferred fragments are not merged inline. When all
// operands of the inferred fragment are block arguments, `latest_operand_producer`
// remains null and the inline merge bails out, leaving a separate fragment.
// See TODO(petebu) in uniquify_function_inputs_outputs.cc.

!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4xf32>>

// CHECK-LABEL: func @block_arg_only_no_merge
func.func @block_arg_only_no_merge(%arg0: !mesh_tensor, %arg1: !mesh_tensor)
  -> (!mesh_tensor, !mesh_tensor, !mesh_tensor) attributes {
    "topology"=#mpmd.topology<<"m": <["x"=2]>>>
} {
  // The existing fragment uses %arg0 and produces %0.
  // The return uses %0, %arg1, %arg1 — so uniquify creates an inferred fragment
  // for the duplicated %arg1 returns. Since %arg1 is a block argument (no
  // defining op), the inferred fragment is NOT merged into f1.
  // CHECK: %[[F1:.*]] = mpmd.fragment<mesh="m", origin=["f1"]>
  // CHECK: mpmd.fragment<mesh="m", origin=[]> {{.*}}{mpmd.inferred_by = ["uniquify"]}
  %0 = mpmd.fragment<mesh="m", origin=["f1"]> (%arg0) (%arg2: tensor<4xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<4xf32>
    mpmd.return %1 : tensor<4xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  func.return %0, %arg1, %arg1 : !mesh_tensor, !mesh_tensor, !mesh_tensor
}

// -----

// When the inferred fragment's operand comes from a same-mesh fragment (f2),
// the inline merge merges the inferred fragment into f2.

!mesh_tensor = !mpmd.mesh_tensor<"m", tensor<4xf32>>

// CHECK-LABEL: func @merge_into_operand_producer
func.func @merge_into_operand_producer(%arg0: !mesh_tensor)
  -> (!mesh_tensor, !mesh_tensor, !mesh_tensor) attributes {
    "topology"=#mpmd.topology<<"m": <["x"=2]>>>
} {
  // f1 produces %0.
  // f2 consumes %0 and produces %1.
  // The return duplicates %1, creating an inferred fragment for uniquify.
  // The inferred fragment's operand is %1 (produced by f2), so f2 is the
  // merge target. The inferred fragment is merged into f2.
  // CHECK: %[[F1:.*]] = mpmd.fragment<mesh="m", origin=["f1"]>
  // CHECK: %[[F2:.*]]:2 = mpmd.fragment<mesh="m", origin=["f2"]>
  // CHECK-NOT: mpmd.fragment<mesh="m", origin=[]>
  // CHECK: return %[[F1]], %[[F2]]#0, %[[F2]]#1
  %0 = mpmd.fragment<mesh="m", origin=["f1"]> (%arg0) (%arg2: tensor<4xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<4xf32>
    mpmd.return %1 : tensor<4xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  %1 = mpmd.fragment<mesh="m", origin=["f2"]> (%0) (%arg2: tensor<4xf32>) {
    %2 = stablehlo.add %arg2, %arg2 : tensor<4xf32>
    mpmd.return %2 : tensor<4xf32>
  } : (!mesh_tensor) -> !mesh_tensor
  func.return %0, %1, %1 : !mesh_tensor, !mesh_tensor, !mesh_tensor
}
