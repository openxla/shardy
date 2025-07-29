// RUN: mpmd_opt %s -mpmd-introduce-transfers 2>&1 | FileCheck %s

!mesh_1_tensor = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!mesh_2_tensor = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>
#topology=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>, <"m2": <["x"=2, "y"=4]>>>

// CHECK-LABEL: func @test_single_call_op_unassign_cancelled_out_with_assign
// CHECK-NEXT: %[[UNASSN:.*]] = mpmd.unassign %arg0 {{.*}}"m1"
// CHECK-NEXT: %[[CALL_RESULT:.*]] = mpmd.call @f(%[[UNASSN]])
func.func @test_single_call_op_unassign_cancelled_out_with_assign(%arg0 : !mesh_1_tensor) -> !mesh_1_tensor attributes {topology=#topology}
{
  %0 = mpmd.unassign %arg0: (!mesh_1_tensor) -> tensor<4x8xf32>
  %2 = mpmd.call @f(%0) : (tensor<4x8xf32>) ->  !mesh_1_tensor
  func.return %2 :  !mesh_1_tensor
}
// CHECK-LABEL: func private @f(%arg0: tensor<4x8xf32>)
// CHECK-NEXT: %[[ASSN:.*]] = mpmd.assign %arg0 {{.*}}"m1"
// CHECK-NEXT: %[[NEW_TRANSFER:.*]] = mpmd.transfer %[[ASSN]]
// CHECK-NEXT: %[[EXISTING_TRANSFER:.*]] = mpmd.transfer %[[NEW_TRANSFER]]
// CHECK-NEXT: return %[[EXISTING_TRANSFER]]
func.func private @f(%arg0 : tensor<4x8xf32>) -> !mesh_1_tensor
attributes {topology=#topology} {
  %assign = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_2_tensor
  %transfer_back_to_m1 = mpmd.transfer %assign : (!mesh_2_tensor) -> !mesh_1_tensor
  func.return %transfer_back_to_m1 : !mesh_1_tensor
}

// CHECK-LABEL: func @test_should_push_in_if_all_conditions_met_on_multiple_calls
// CHECK-NEXT: %[[UNASSN0:.*]] = mpmd.unassign %arg0 {{.*}}"m1"
// CHECK-NEXT: %[[UNASSN1:.*]] = mpmd.unassign %arg1 {{.*}}"m1"
// CHECK-NEXT: %[[CALL_RESULT_1:.*]] = mpmd.call @f_multiple_calls(%[[UNASSN0]])
// CHECK-NEXT: %[[CALL_RESULT_2:.*]] = mpmd.call @f_multiple_calls(%[[UNASSN1]])
func.func @test_should_push_in_if_all_conditions_met_on_multiple_calls(%arg0 : !mesh_1_tensor, %arg1 : !mesh_1_tensor) -> (!mesh_1_tensor, !mesh_1_tensor) attributes {
  topology=#topology}
{
  %0 = mpmd.unassign %arg0: (!mesh_1_tensor) -> tensor<4x8xf32>
  %1 = mpmd.unassign %arg1: (!mesh_1_tensor) -> tensor<4x8xf32>

  %2 = mpmd.call @f_multiple_calls(%0) : (tensor<4x8xf32>) ->  !mesh_1_tensor
  %3 = mpmd.call @f_multiple_calls(%1) : (tensor<4x8xf32>) ->  !mesh_1_tensor
  func.return %2, %3 :  !mesh_1_tensor, !mesh_1_tensor
}
// CHECK-LABEL: func private @f_multiple_calls(%arg0: tensor<4x8xf32>)
// CHECK-NEXT: %[[ASSN:.*]] = mpmd.assign %arg0 {{.*}}"m1"
// CHECK-NEXT: %[[NEW_TRANSFER:.*]] = mpmd.transfer %[[ASSN]]
// CHECK-NEXT: %[[EXISTING_TRANSFER:.*]] = mpmd.transfer %[[NEW_TRANSFER]]
// CHECK-NEXT: return %[[EXISTING_TRANSFER]]
func.func private @f_multiple_calls(%arg0 : tensor<4x8xf32>) -> !mesh_1_tensor
    attributes {topology=#topology}{
  %assign = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_2_tensor
  %transfer_back_to_m1 = mpmd.transfer %assign : (!mesh_2_tensor) -> !mesh_1_tensor
  func.return %transfer_back_to_m1 : !mesh_1_tensor
}

// CHECK-LABEL: func @test_should_not_push_if_mesh_different
// CHECK-NEXT: %[[UNASSIGN_RESULT:.*]] = mpmd.unassign %arg0
// CHECK-NEXT: %[[UNASSIGN_RESULT:.*]] = mpmd.unassign %arg1
func.func @test_should_not_push_if_mesh_different(%arg0 : !mesh_1_tensor, %arg1 : !mesh_2_tensor) -> (tensor<4x8xf32>, tensor<4x8xf32>) attributes {
  topology=#topology}
{
  %0 = mpmd.unassign %arg0: (!mesh_1_tensor) -> tensor<4x8xf32>
  %1 = mpmd.unassign %arg1: (!mesh_2_tensor) -> tensor<4x8xf32>

  %2 = mpmd.call @f2(%0) : (tensor<4x8xf32>) ->  tensor<4x8xf32>
  %3 = mpmd.call @f2(%1) : (tensor<4x8xf32>) ->  tensor<4x8xf32>
  func.return %2, %3 :  tensor<4x8xf32>, tensor<4x8xf32>
}

func.func private @f2(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {topology=#topology}  {
  func.return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @test_should_not_push_if_not_all_operand_are_from_unassign
// CHECK: %[[UNASSIGN_RESULT:.*]] = mpmd.unassign %arg0
func.func @test_should_not_push_if_not_all_operand_are_from_unassign(%arg0 : !mesh_1_tensor) -> (tensor<4x8xf32>, tensor<4x8xf32>) attributes {
  topology=#topology}
{
  %0 = mpmd.unassign %arg0: (!mesh_1_tensor) -> tensor<4x8xf32>
  %1 = stablehlo.constant dense<1.0> : tensor<4x8xf32>

  %2 = mpmd.call @f3(%0) : (tensor<4x8xf32>) ->  tensor<4x8xf32>
  %3 = mpmd.call @f3(%1) : (tensor<4x8xf32>) ->  tensor<4x8xf32>
  func.return %2, %3 :  tensor<4x8xf32>, tensor<4x8xf32>
}

func.func private @f3(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
    attributes {topology=#topology} {
  func.return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @test_should_not_push_if_arg_not_assigned_later
// CHECK-NEXT: %[[UNASSIGN_RESULT:.*]] = mpmd.unassign %arg0
// CHECK-NEXT: %[[CALL_RESULT:.*]] = mpmd.call @f4(%[[UNASSIGN_RESULT]])
func.func @test_should_not_push_if_arg_not_assigned_later(%arg0 : !mesh_1_tensor) -> tensor<4x8xf32> attributes {
  topology=#topology}
{
  %0 = mpmd.unassign %arg0: (!mesh_1_tensor) -> tensor<4x8xf32>
  %2 = mpmd.call @f4(%0) : (tensor<4x8xf32>) ->  tensor<4x8xf32>
  func.return %2 :  tensor<4x8xf32>
}
func.func private @f4(%arg0 : tensor<4x8xf32>) -> tensor<4x8xf32>
attributes {
  topology=#topology} {
  func.return %arg0 :tensor<4x8xf32>
}

// CHECK-LABEL: func @test_should_not_push_if_callee_has_no_input
// CHECK-NEXT: %[[CALL_RESULT:.*]] = mpmd.call @f5()
func.func @test_should_not_push_if_callee_has_no_input() -> tensor<4x8xf32> attributes {
  topology=#topology}
{
  %2 = mpmd.call @f5() : () ->  tensor<4x8xf32>
  func.return %2 :  tensor<4x8xf32>
}
func.func private @f5() -> tensor<4x8xf32>
attributes {topology=#topology} {
  %1 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
  func.return %1 :tensor<4x8xf32>
}

// CHECK-LABEL: func @test_should_not_push_if_no_defining_op_of_arg
// CHECK-NEXT: %[[CALL_RESULT:.*]] = mpmd.call @f6(%arg0)
func.func @test_should_not_push_if_no_defining_op_of_arg(%arg0 : !mesh_1_tensor) -> !mesh_1_tensor attributes {
  topology=#topology}
{
  %2 = mpmd.call @f6(%arg0) : (!mesh_1_tensor) ->  !mesh_1_tensor
  func.return %2 :  !mesh_1_tensor
}
func.func private @f6(%arg1 : !mesh_1_tensor) -> !mesh_1_tensor
attributes {
  topology=#topology} {
  func.return %arg1 :!mesh_1_tensor
}


// CHECK-LABEL: func @test_should_push_in_if_all_conditions_met_on_chained_calls
// CHECK-NEXT: %[[UNASSN0:.*]] = mpmd.unassign %arg0 {{.*}}"m1"
// CHECK-NEXT: %[[UNASSN1:.*]] = mpmd.unassign %arg1 {{.*}}"m1"
// CHECK-NEXT: %[[CALL_RESULT_1:.*]]:2 = mpmd.call @f_chained_calls(%[[UNASSN0]], %[[UNASSN1]])
// CHECK-NEXT: %[[CALL_RESULT_2:.*]]:2 = mpmd.call @f_chained_calls(%[[CALL_RESULT_1]]#1, %[[CALL_RESULT_1]]#0)
func.func @test_should_push_in_if_all_conditions_met_on_chained_calls(
  %arg0 : !mesh_1_tensor, %arg1 : !mesh_1_tensor
) -> (tensor<4x8xf32>, tensor<4x8xf32>) attributes {topology=#topology}
{
  %0 = mpmd.unassign %arg0: (!mesh_1_tensor) -> tensor<4x8xf32>
  %1 = mpmd.unassign %arg1: (!mesh_1_tensor) -> tensor<4x8xf32>

  %2:2 = mpmd.call @f_chained_calls(%0, %1) : (tensor<4x8xf32>, tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
  %3:2 = mpmd.call @f_chained_calls(%2#1, %2#0) : (tensor<4x8xf32>, tensor<4x8xf32>) ->  (tensor<4x8xf32>, tensor<4x8xf32>)
  func.return %3#0, %3#1 :  tensor<4x8xf32>, tensor<4x8xf32>
}
// CHECK-LABEL: func private @f_chained_calls
// CHECK-NEXT: %[[ASSN2:.*]] = mpmd.assign %arg1 {{.*}}"m1"
// CHECK-NEXT: %[[ASSN1:.*]] = mpmd.assign %arg0 {{.*}}"m1"
// CHECK-NEXT: %[[UNASSN1:.*]] = mpmd.unassign %[[ASSN1]] {{.*}}"m1"
// CHECK-NEXT: %[[NEW_TRANSFER:.*]] = mpmd.transfer %[[ASSN2]]
// CHECK-NEXT: %[[UNASSN2:.*]] = mpmd.unassign %[[NEW_TRANSFER]] {{.*}}"m2"
// CHECK-NEXT: return %[[UNASSN1]], %[[UNASSN2]]
func.func private @f_chained_calls(%arg0 : tensor<4x8xf32>, %arg1 : tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>)
    attributes {topology=#topology}{
  %assign1 = mpmd.assign %arg0 : (tensor<4x8xf32>) -> !mesh_1_tensor
  %unassign1 = mpmd.unassign %assign1 : (!mesh_1_tensor) -> tensor<4x8xf32>

  %assign2 = mpmd.assign %arg1 : (tensor<4x8xf32>) -> !mesh_2_tensor
  %unassign2 = mpmd.unassign %assign2 : (!mesh_2_tensor) -> tensor<4x8xf32>

  func.return %unassign1, %unassign2 : tensor<4x8xf32>, tensor<4x8xf32>
}
