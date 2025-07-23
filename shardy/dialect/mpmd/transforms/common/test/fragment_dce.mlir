// RUN: mpmd_opt %s -mpmd-fragment-dce 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>

// CHECK-LABEL: func @unused_operand
func.func @unused_operand(%arg0: !mesh_1_tensor, %arg1: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
// CHECK-SAME:   (%arg2: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FRAGMENT]]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %arg1)
    (%arg2: tensor<4x8xf32>, %arg3: tensor<4x8xf32>) {
    %1 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (!mesh_1_tensor, !mesh_1_tensor) -> !mesh_1_tensor
  func.return %0 : !mesh_1_tensor
}

// CHECK-LABEL: func @unused_result
func.func @unused_result(%arg0: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
// CHECK-SAME:   (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FRAGMENT]]
  %0:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
      %1 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      // This value is not used outside the fragment. It will be removed.
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
    } : (!mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor)
  func.return %0#0 : !mesh_1_tensor
}

// CHECK-LABEL: func @unused_result_causes_operand_to_be_removed
func.func @unused_result_causes_operand_to_be_removed(%arg0: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
// CHECK-SAME:   (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FRAGMENT]]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
      mpmd.return %arg2 : tensor<4x8xf32>
    } : (!mesh_1_tensor) -> !mesh_1_tensor
  %1:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %0)
    (%arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) {
      %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
      // This value is not used outside the fragment. It will be removed and
      // so it the fragment that produces the operand re to %arg2.
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
    } : (!mesh_1_tensor, !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor)
  func.return %1#0 : !mesh_1_tensor
}

// CHECK-LABEL: func @pure_ops_are_removed_but_side_effecting_ops_are_not
func.func @pure_ops_are_removed_but_side_effecting_ops_are_not(%arg0: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[F:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0) (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[CC:.*]] = stablehlo.custom_call @Sharding(%arg1)
// CHECK-NEXT:   mpmd.return %arg1
// CHECK-NEXT: }
  %1:3 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
    (%arg1: tensor<4x8xf32>) {
      // This is an unused pure op and will be removed.
      %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
      // This is an unused side-effecting op and will not be removed.
      %2 = stablehlo.custom_call @Sharding(%arg1) {has_side_effect = true} : (tensor<4x8xf32>) -> tensor<4x8xf32>
      mpmd.return %1, %2, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>
    } : (!mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor, !mesh_1_tensor)
  func.return %1#2 : !mesh_1_tensor
}

// CHECK-LABEL: func @transfer_becomes_unused
func.func @transfer_becomes_unused(%arg0: !mesh_1_tensor)
  -> !mesh_1_tensor attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2, "y"=2]>>>} {
// CHECK-NEXT: %[[FRAGMENT:.*]] = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
// CHECK-SAME:   (%arg1: tensor<4x8xf32>) {
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[FRAGMENT]]
  %0 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0)
    (%arg2: tensor<4x8xf32>) {
      mpmd.return %arg2 : tensor<4x8xf32>
    } : (!mesh_1_tensor) -> !mesh_1_tensor
  %t = mpmd.transfer %0 : (!mesh_1_tensor) -> !mesh_1_tensor
  %1:2 = mpmd.fragment<mesh="m1", origin=["f"]> (%arg0, %t)
    (%arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) {
      %1 = stablehlo.add %arg1, %arg1 : tensor<4x8xf32>
      // This value is not used outside the fragment. It will be removed and
      // so it the fragment that produces the operand re to %arg2.
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x8xf32>
      mpmd.return %1, %2 : tensor<4x8xf32>, tensor<4x8xf32>
    } : (!mesh_1_tensor, !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor)
  func.return %1#0 : !mesh_1_tensor
}
