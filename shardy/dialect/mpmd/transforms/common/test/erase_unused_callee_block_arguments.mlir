// RUN: mpmd_opt %s -mpmd-erase-unused-callee-block-arguments 2>&1 | FileCheck %s

!mesh2_t = !mpmd.mesh_tensor<"m2", tensor<3x5xf32>>

// CHECK-LABEL: func @used_by_return_only
func.func @used_by_return_only(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> !mesh2_t
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[CALL1:.*]] = mpmd.call @f(%arg0) {call_counter = 0 : ui32}
  // CHECK-NEXT: %[[CALL2:.*]] = mpmd.call @f(%[[CALL1]])
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign %[[CALL2]]
  // CHECK-NEXT: return %[[ASSIGN]]
  %0:2 = mpmd.call @f(%arg0, %arg1) {call_counter = 0 : ui32} : (tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
  %1:2 = mpmd.call @f(%0#0, %0#1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
  %2 = mpmd.assign %1#0 : (tensor<3x5xf32>) -> !mesh2_t
  return %2 : !mesh2_t
}

// CHECK-LABEL: func private @f
// CHECK-SAME: (%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
func.func private @f(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> (tensor<3x5xf32>, tensor<3x5xf32>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]] : tensor<3x5xf32>
  %1 = stablehlo.add %arg0, %arg0 : tensor<3x5xf32>
  // %arg1 is only used by the return.
  return %1, %arg1 : tensor<3x5xf32>, tensor<3x5xf32>
}

// CHECK-LABEL: func @call_op_is_not_needed
func.func @call_op_is_not_needed(%arg0: tensor<3x5xf32>)
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: return %arg0
  %0 = mpmd.call @g(%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  %1 = mpmd.call @g(%0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  return %1 : tensor<3x5xf32>
}

// CHECK-NOT: @g
func.func private @g(%arg0: tensor<3x5xf32>)
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // %arg1 is only used by the return.
  return %arg0 : tensor<3x5xf32>
}

// CHECK-LABEL: func @arg_is_completely_unused
func.func @arg_is_completely_unused(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[CALL:.*]] = mpmd.call @h(%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  // CHECK-NEXT: return %[[CALL]] : tensor<3x5xf32>
  %0 = mpmd.call @h(%arg0, %arg1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// CHECK-LABEL: func private @h
// CHECK-SAME: (%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
func.func private @h(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// CHECK-LABEL: func @duplicate_operand_in_call
func.func @duplicate_operand_in_call(%arg0: tensor<3x5xf32>)
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[CALL:.*]] = mpmd.call @i(%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  // CHECK-NEXT: return %[[CALL]] : tensor<3x5xf32>
  // Only the second use of %arg0 should be erased from the call op's operands.
  %0 = mpmd.call @i(%arg0, %arg0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// CHECK-LABEL: func private @i
// CHECK-SAME: (%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
func.func private @i(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// This test is a noop.
// CHECK-LABEL: func @arg_is_used_by_op
func.func @arg_is_used_by_op(%arg0: tensor<3x5xf32>)
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[CALL:.*]] = mpmd.call @j(%arg0, %arg0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  // CHECK-NEXT: return %[[CALL]] : tensor<3x5xf32>
  %0 = mpmd.call @j(%arg0, %arg0) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// CHECK-LABEL: func private @j
// CHECK-SAME: (%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>) -> tensor<3x5xf32>
func.func private @j(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[ADD0:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: %[[ADD1:.*]] = stablehlo.add %arg1, %arg1
  // CHECK-NEXT: return %[[ADD0]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<3x5xf32>
  // arg1 is actually used and therefore we cannot erase it.
  %1 = stablehlo.add %arg1, %arg1 : tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// CHECK-LABEL: func private @not_called
// CHECK-SAME: (%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>) -> tensor<3x5xf32>
func.func private @not_called(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[ADD0:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD0]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// CHECK-LABEL: func @call_w_many_results
func.func @call_w_many_results(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> !mesh2_t
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[CALL1:.*]]:2 = mpmd.call @k(%arg0)
  // CHECK-NEXT: %[[CALL2:.*]]:2 = mpmd.call @k(%[[CALL1]]#0)
  // CHECK-NEXT: %[[ASSIGN:.*]] = mpmd.assign %[[CALL2]]#0
  // CHECK-NEXT: return %[[ASSIGN]]
  %0:4 = mpmd.call @k(%arg0, %arg1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>)
  %1:4 = mpmd.call @k(%0#0, %0#1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>)
  %2 = mpmd.assign %1#0 : (tensor<3x5xf32>) -> !mesh2_t
  return %2 : !mesh2_t
}

// CHECK-LABEL: func private @k
// CHECK-SAME: (%arg0: tensor<3x5xf32>) -> (tensor<3x5xf32>, tensor<3x5xf32>)
func.func private @k(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>)
  -> (tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>)
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]], %[[ADD]]
  %1 = stablehlo.add %arg0, %arg0 : tensor<3x5xf32>
  // %arg1 is only used by the return.
  return %1, %arg1, %arg1, %1 : tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>, tensor<3x5xf32>
}

// CHECK-LABEL: func @call_inside_for_loop
func.func @call_inside_for_loop(%arg0: tensor<3x5xf32>)
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: mpmd.for (%arg0) {iterations = 3 : ui32, unroll_factor = 3 : ui32} (%arg1: tensor<3x5xf32>, %index: tensor<ui32>) {
  // CHECK-NEXT:   %[[CALL:.*]] = mpmd.call @l(%arg1) : (tensor<3x5xf32>) -> tensor<3x5xf32>
  // CHECK-NEXT:   return %[[CALL]] : tensor<3x5xf32>
  // CHECK-NEXT: } : tensor<3x5xf32>
  %0 = mpmd.for (%arg0) {iterations = 3 : ui32, unroll_factor = 3 : ui32} (%arg1: tensor<3x5xf32>, %index: tensor<ui32>) {
    %1 = mpmd.call @l(%arg1, %index) : (tensor<3x5xf32>, tensor<ui32>) -> tensor<3x5xf32>
    mpmd.return %1 : tensor<3x5xf32>
  } : tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}

// CHECK-LABEL: func private @l
// CHECK-SAME: (%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
func.func private @l(%arg0: tensor<3x5xf32>, %arg1: tensor<ui32>)
  -> tensor<3x5xf32>
  attributes {topology = #mpmd.topology<<"m1" : <["x"=1]>>, <"m2" : <["x"=1]>>>}
{
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<3x5xf32>
  return %0 : tensor<3x5xf32>
}
