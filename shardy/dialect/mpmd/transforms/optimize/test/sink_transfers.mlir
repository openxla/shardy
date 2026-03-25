// RUN: mpmd_opt %s -mpmd-sink-transfers | FileCheck %s

!mesh_0_tensor = !mpmd.mesh_tensor<"m0", tensor<4x4xf32>>
!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x4xf32>>

// Verify that a transfer is sunk past an unrelated fragment, placing it just
// before its first user.
//
// Before:  frag0, transfer(frag0), frag1(frag0), frag2(transfer)
// After:   frag0, frag1(frag0), transfer(frag0), frag2(transfer)

// CHECK-LABEL: func @sink_past_unrelated_fragment
func.func @sink_past_unrelated_fragment(%arg0: !mesh_0_tensor) -> !mesh_1_tensor
  attributes {topology = #mpmd.topology<
    <"m0" : <["x"=2, "y"=2]>>, <"m1" : <["x"=2, "y"=2]>>>}
{
  // CHECK: %[[FRAG0:.*]] = mpmd.fragment<mesh="m0", origin=["f0"]>
  %0 = mpmd.fragment<mesh="m0", origin=["f0"]> () () {
    %c = stablehlo.constant dense<1.0> : tensor<4x4xf32>
    mpmd.return %c : tensor<4x4xf32>
  } : () -> !mesh_0_tensor

  // The transfer should not appear here (it gets sunk below frag1).
  // CHECK-NOT: mpmd.transfer
  %1 = mpmd.transfer %0 : (!mesh_0_tensor) -> !mesh_1_tensor

  // CHECK: %[[FRAG1:.*]] = mpmd.fragment<mesh="m0", origin=["f1"]>
  %2 = mpmd.fragment<mesh="m0", origin=["f1"]> (%0) (%a: tensor<4x4xf32>) {
    %add = stablehlo.add %a, %a : tensor<4x4xf32>
    mpmd.return %add : tensor<4x4xf32>
  } : (!mesh_0_tensor) -> !mesh_0_tensor

  // The transfer reappears here, just before its consumer.
  // CHECK: %[[TRANSFER:.*]] = mpmd.transfer %[[FRAG0]]
  // CHECK: %[[FRAG2:.*]] = mpmd.fragment<mesh="m1", origin=["f2"]> (%[[TRANSFER]])
  %3 = mpmd.fragment<mesh="m1", origin=["f2"]> (%1) (%b: tensor<4x4xf32>) {
    %add = stablehlo.add %b, %b : tensor<4x4xf32>
    mpmd.return %add : tensor<4x4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor

  return %3 : !mesh_1_tensor
}

// Verify that a chained transfer (m0 -> m1 -> m2) is sunk as a unit.
// Both transfers should move past the unrelated fragment.

!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x4xf32>>

// CHECK-LABEL: func @sink_chained_transfers
func.func @sink_chained_transfers(%arg0: !mesh_0_tensor) -> !mesh_2_tensor
  attributes {topology = #mpmd.topology<
    <"m0" : <["x"=2, "y"=2]>>, <"m1" : <["x"=2, "y"=2]>>,
    <"m2" : <["x"=2, "y"=2]>>>}
{
  // CHECK: %[[FRAG0:.*]] = mpmd.fragment<mesh="m0", origin=["f0"]>
  %0 = mpmd.fragment<mesh="m0", origin=["f0"]> () () {
    %c = stablehlo.constant dense<1.0> : tensor<4x4xf32>
    mpmd.return %c : tensor<4x4xf32>
  } : () -> !mesh_0_tensor

  // CHECK-NOT: mpmd.transfer
  %1 = mpmd.transfer %0 : (!mesh_0_tensor) -> !mesh_1_tensor
  %2 = mpmd.transfer %1 : (!mesh_1_tensor) -> !mesh_2_tensor

  // CHECK: %[[FRAG1:.*]] = mpmd.fragment<mesh="m0", origin=["f1"]>
  %3 = mpmd.fragment<mesh="m0", origin=["f1"]> (%0) (%a: tensor<4x4xf32>) {
    %add = stablehlo.add %a, %a : tensor<4x4xf32>
    mpmd.return %add : tensor<4x4xf32>
  } : (!mesh_0_tensor) -> !mesh_0_tensor

  // Both transfers should appear here, in chain order, before the consumer.
  // CHECK: %[[T1:.*]] = mpmd.transfer %[[FRAG0]] : (!mpmd.mesh_tensor<"m0"
  // CHECK-NEXT: %[[T2:.*]] = mpmd.transfer %[[T1]] : (!mpmd.mesh_tensor<"m1"
  // CHECK: %[[FRAG2:.*]] = mpmd.fragment<mesh="m2", origin=["f2"]> (%[[T2]])
  %4 = mpmd.fragment<mesh="m2", origin=["f2"]> (%2) (%b: tensor<4x4xf32>) {
    %add = stablehlo.add %b, %b : tensor<4x4xf32>
    mpmd.return %add : tensor<4x4xf32>
  } : (!mesh_2_tensor) -> !mesh_2_tensor

  return %4 : !mesh_2_tensor
}

// Verify that a transfer already adjacent to its user is not moved.

// CHECK-LABEL: func @no_move_when_already_adjacent
func.func @no_move_when_already_adjacent(%arg0: !mesh_0_tensor) -> !mesh_1_tensor
  attributes {topology = #mpmd.topology<
    <"m0" : <["x"=2, "y"=2]>>, <"m1" : <["x"=2, "y"=2]>>>}
{
  // CHECK: %[[FRAG0:.*]] = mpmd.fragment<mesh="m0", origin=["f0"]>
  %0 = mpmd.fragment<mesh="m0", origin=["f0"]> () () {
    %c = stablehlo.constant dense<1.0> : tensor<4x4xf32>
    mpmd.return %c : tensor<4x4xf32>
  } : () -> !mesh_0_tensor

  // Transfer is already right before its user — should stay put.
  // CHECK: %[[TRANSFER:.*]] = mpmd.transfer %[[FRAG0]]
  %1 = mpmd.transfer %0 : (!mesh_0_tensor) -> !mesh_1_tensor
  // CHECK: %[[FRAG1:.*]] = mpmd.fragment<mesh="m1", origin=["f1"]> (%[[TRANSFER]])
  %2 = mpmd.fragment<mesh="m1", origin=["f1"]> (%1) (%b: tensor<4x4xf32>) {
    %add = stablehlo.add %b, %b : tensor<4x4xf32>
    mpmd.return %add : tensor<4x4xf32>
  } : (!mesh_1_tensor) -> !mesh_1_tensor

  return %2 : !mesh_1_tensor
}
