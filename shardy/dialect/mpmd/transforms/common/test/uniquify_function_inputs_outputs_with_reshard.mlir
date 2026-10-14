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
  // The producer f1 is extended with extra results instead of creating a
  // separate uniquify fragment.
  // CHECK-NEXT: %[[F1:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK:        %[[V:.*]] = stablehlo.add
  // CHECK:        mpmd.return %[[V]], %[[V]]
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

// CHECK-LABEL: func @needs_fragment_for_m1_with_many_values
func.func @needs_fragment_for_m1_with_many_values(%arg0: !mesh_1_tensor, %arg1: !mesh_2_tensor
) -> (!mesh_2_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor) attributes {
  "topology"=#mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>
} {
  // f1 is extended with 1 extra result (used at positions 1 and 3).
  // CHECK-NEXT: %[[F1:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK:        mpmd.return %arg2, %arg2
  // CHECK:      %[[F2:.*]] = mpmd.fragment<mesh="m2", origin=["f2"]>
  // f3 is extended with 2 extra results (used at positions 2, 4, and 5).
  // CHECK:      %[[F3:.*]]:3 = mpmd.fragment<mesh="m1", origin=["f3"]>
  // CHECK:        mpmd.return %arg2, %arg2, %arg2
  // CHECK:      return %[[F2]], %[[F1]]#0, %[[F3]]#0, %[[F1]]#1, %[[F3]]#1, %[[F3]]#2
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
  // f1 is extended with 1 extra result.
  // CHECK: %[[F1:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK:   mpmd.return %arg2, %arg2
  // f2 is extended with 1 extra result.
  // CHECK: %[[F2:.*]]:2 = mpmd.fragment<mesh="m2", origin=["f2"]>
  // CHECK:   mpmd.return %arg2, %arg2
  // f3 is extended with 1 extra result.
  // CHECK: %[[F3:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f3"]>
  // CHECK:   mpmd.return %arg2, %arg2
  // CHECK: return %[[F1]]#0, %[[F2]]#0, %[[F2]]#1, %[[F3]]#0, %[[F1]]#1, %[[F3]]#1
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
  // CHECK-NEXT: %[[F1:.*]]:2 = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK:        %[[V:.*]] = stablehlo.add
  // CHECK:        mpmd.return %[[V]], %[[V]]
  // CHECK:      %[[F2:.*]] = mpmd.fragment<mesh="m1", origin=["f2"]>
  // CHECK:      return %[[F2]], %[[F1]]#0, %[[F1]]#1
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
  // The f fragment result appears once (position 1). Block arg %arg0 appears at
  // positions 0 and 2 — it gets a uniquify identity fragment.
  // CHECK-NEXT: %[[F1:.*]] = mpmd.fragment<mesh="m", origin=["f"]> (%arg0) (%arg1: tensor<4xui32>) {
  // CHECK-NEXT:   return %arg1
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[UF:.*]]:2 = mpmd.fragment<mesh="m", origin=[]> (%arg0) {mpmd.inferred_by = ["uniquify"]} (%arg1: tensor<4xui32>) {
  // CHECK-NEXT:   return %arg1, %arg1
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[UF]]#0, %[[F1]], %[[UF]]#1
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
