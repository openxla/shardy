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
  // CHECK-NEXT: %[[F1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK:      %[[F2:.*]] = mpmd.fragment<mesh="m1", origin=["f2"]>
  // CHECK:      %[[UF:.*]]:2 = mpmd.fragment<mesh="m1", origin=[]> (%[[F1]]) (%arg1: tensor<4xf32>) {
  // CHECK:         mpmd.return %arg1, %arg1 : tensor<4xf32>, tensor<4xf32>
  // CHECK:      %[[F2]], %[[UF]]#0, %[[UF]]#1
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
  // CHECK:      %[[F3:.*]] = mpmd.fragment<mesh="m1", origin=["f3"]>
  // CHECK:      %[[UF:.*]]:5 = mpmd.fragment<mesh="m1", origin=[]> (%[[F1]], %[[F3]]) (%[[A1:.*]]: tensor<4xf32>, %[[A2:.*]]: tensor<4xf32>)
  // CHECK-NEXT:   mpmd.return %[[A1]], %[[A1]], %[[A2]], %[[A2]], %[[A2]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[F2]], %[[UF]]#0, %[[UF]]#2, %[[UF]]#1, %[[UF]]#3, %[[UF]]#4
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
  // CHECK: %[[UF1:.*]]:4 = mpmd.fragment<mesh="m1", origin=[]>
  // CHECK: %[[UF2:.*]]:2 = mpmd.fragment<mesh="m2", origin=[]>
  // CHECK: return %[[UF1]]#0, %[[UF2]]#0, %[[UF2]]#1, %[[UF1]]#2, %[[UF1]]#1, %[[UF1]]#3
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
  // CHECK-NEXT: %[[F1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK:      %[[F2:.*]] = mpmd.fragment<mesh="m1", origin=["f2"]>
  // CHECK:      %[[UF:.*]]:2 = mpmd.fragment<mesh="m1", origin=[]> (%[[F1]]) (%arg1: tensor<4xf32>) {
  // CHECK:         mpmd.return %arg1, %arg1 : tensor<4xf32>, tensor<4xf32>
  // CHECK:      %[[F2]], %[[UF]]#0, %[[UF]]#1
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
  // CHECK-NEXT: %[[F2:.*]]:2 = mpmd.fragment<mesh="m", origin=[]> (%arg0) (%arg1: tensor<4xui32>) {
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
  // CHECK-NEXT: %[[F:.*]] = mpmd.fragment<mesh="m", origin=[]> (%arg0) (%arg1: tensor<4xui32>) {
  // CHECK-NEXT:   return %arg1
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[F]]
  func.return %arg0 : !mesh_tensor
}
