// RUN: mpmd_opt %s -mpmd-copy-constants 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

// CHECK-LABEL: func @fragment_to_fragments()
func.func @fragment_to_fragments() -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2, "y"=4]>>>}
{
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f0"]> ()
  // CHECK-NEXT:   constant
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f1"]> (%
  // CHECK-NEXT:   %[[C:.*]] = stablehlo.constant
  // CHECK-NEXT:   stablehlo.add %[[C]], %[[C]]
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f2"]> (%
  // CHECK-NEXT:   %[[C:.*]] = stablehlo.constant
  // CHECK-NEXT:   stablehlo.add %[[C]], %[[C]]
  // CHECK-NEXT:   return
  // CHECK-NEXT: }

  %0 = mpmd.fragment<mesh="m1", origin=["f0"]> () () {
    %c = stablehlo.constant dense<8.000000e+00> : tensor<4x8xf32>
    mpmd.return %c : tensor<4x8xf32>
  } : () -> !mesh_1_tensor
  %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%0) (%arg0: tensor<4x8xf32>) {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    mpmd.return %add : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %2 = mpmd.fragment<mesh="m1", origin=["f2"]> (%0) (%arg0: tensor<4x8xf32>) {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    mpmd.return %add : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  return %0, %1, %2 : !mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor
}

// CHECK-LABEL: func @through_transfers
func.func @through_transfers() -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_2_tensor)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=2, "y"=4]>>, <"m2" : <["x"=2, "y"=4]>>>}
{
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f0"]> ()
  // CHECK-NEXT:   constant
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: transfer
  // CHECK-NEXT: transfer
  // CHECK-NEXT: mpmd.fragment<mesh="m1", origin=["f1"]>
  // CHECK-NEXT:   %[[C:.*]] = stablehlo.constant
  // CHECK-NEXT:   stablehlo.add %[[C]], %[[C]]
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // CHECK-NEXT: mpmd.fragment<mesh="m2", origin=["f2"]>
  // CHECK-NEXT:   %[[C:.*]] = stablehlo.constant
  // CHECK-NEXT:   stablehlo.add %[[C]], %[[C]]
  // CHECK-NEXT:   return
  // CHECK-NEXT: }

  %0 = mpmd.fragment<mesh="m1", origin=["f0"]> () () {
    %c = stablehlo.constant dense<8.000000e+00> : tensor<4x8xf32>
    mpmd.return %c : tensor<4x8xf32>
  } : () -> !mesh_1_tensor
  %t1 = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_2_tensor
  %t2 = mpmd.transfer %t1 : (!mesh_2_tensor) -> !mesh_1_tensor
  %1 = mpmd.fragment<mesh="m1", origin=["f1"]> (%t2) (%arg0: tensor<4x8xf32>) {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    mpmd.return %add : tensor<4x8xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor
  %2 = mpmd.fragment<mesh="m2", origin=["f2"]> (%t1) (%arg0: tensor<4x8xf32>) {
    %add = stablehlo.add %arg0, %arg0 : tensor<4x8xf32>
    mpmd.return %add : tensor<4x8xf32>
  } : (!mesh_2_tensor) -> !mesh_2_tensor
  return %0, %1, %2 : !mesh_1_tensor, !mesh_1_tensor, !mesh_2_tensor
}
