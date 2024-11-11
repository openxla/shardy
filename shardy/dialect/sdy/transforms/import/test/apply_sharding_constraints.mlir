// RUN: sdy_opt %s -sdy-apply-sharding-constraints | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @input_already_has_sharding
func.func @input_already_has_sharding(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @input_has_one_use
func.func @input_has_one_use(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @input_produced_by_data_flow_edge
func.func @input_produced_by_data_flow_edge(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: sdy.data_flow_edge %arg0 : tensor<8x8xf32>
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = sdy.data_flow_edge %arg0 : tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @open_sharding_constraint
func.func @open_sharding_constraint(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b", ?}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @input_is_func_input_with_one_use(
// CHECK-SAMEL    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>})
func.func @input_is_func_input_with_one_use(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %1 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @no_other_sharding_constraint_users
func.func @no_other_sharding_constraint_users(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = stablehlo.add %0, %0 :  tensor<8x8xf32>
  return %0, %1, %2 : tensor<8x8xf32>,  tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @has_different_sharding_constraint_user
func.func @has_different_sharding_constraint_user(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  return %0, %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @has_other_identical_sharding_constraint_user
func.func @has_other_identical_sharding_constraint_user(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %0, %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @has_other_manual_computation_user_diff_sharding
func.func @has_other_manual_computation_user_diff_sharding(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"a"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>]
      manual_axes={"a"} (%arg2: tensor<4x8xf32>) {
    sdy.return %arg2 : tensor<4x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0, %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @has_other_manual_computation_user_same_sharding
func.func @has_other_manual_computation_user_same_sharding(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{}, {"b"}]>] out_shardings=[<@mesh, [{}, {"b"}]>]
      manual_axes={"b"} (%arg2: tensor<8x4xf32>) {
    sdy.return %arg2 : tensor<8x4xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0, %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @dangling_and_no_other_sharding_constraint_users
func.func @dangling_and_no_other_sharding_constraint_users(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = stablehlo.add %0, %0 :  tensor<8x8xf32>
  return %0, %2 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @dangling_and_has_other_sharding_constraint_user
func.func @dangling_and_has_other_sharding_constraint_user(%arg0: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @dangling_and_has_other_manual_computation_user
func.func @dangling_and_has_other_manual_computation_user(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"a"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>]
      manual_axes={"a"} (%arg2: tensor<4x8xf32>) {
    sdy.return %arg2 : tensor<4x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0, %2 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @manual_computation
func.func @manual_computation(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a"}, {}]>]>}
  // CHECK-NEXT: sdy.manual_computation
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = stablehlo.add %arg1, %arg1 :  tensor<8x8xf32>
  %2 = sdy.manual_computation(%0, %1) in_shardings=[<@mesh, [{"a", "b"}, {?}]>, <@mesh, [{"b", "a"}, {}]>] out_shardings=[<@mesh, [{"a", "b"}, {}]>]
      manual_axes={"a", "b"} (%arg2: tensor<2x8xf32>, %arg3: tensor<2x8xf32>) {
    %3 = stablehlo.add %arg2, %arg3 : tensor<2x8xf32>
    sdy.return %3 : tensor<2x8xf32>
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  func.return %0, %2: tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @chain_of_two_sharding_constraints
func.func @chain_of_two_sharding_constraints(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
  // CHECK-NEXT: %[[WSC_0:.*]] = sdy.sharding_constraint %[[ADD_0]] <@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[WSC_1]], %[[WSC_1]]
  // CHECK-NEXT: return %[[WSC_1]], %[[ADD_1]]
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %3 = stablehlo.add %0, %0 :  tensor<8x8xf32>
  return %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @chain_of_three_sharding_constraints
func.func @chain_of_three_sharding_constraints(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[WSC_0:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT: %[[WSC_2:.*]] = sdy.sharding_constraint %[[WSC_1]] <@mesh, [{}, {}]>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[WSC_2]], %[[WSC_2]]
  // CHECK-NEXT: return %[[WSC_2]], %[[ADD]]
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{}, {}]> :  tensor<8x8xf32>
  %3 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  return %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @chain_on_block_arg_after_other_user
func.func @chain_on_block_arg_after_other_user(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[WSC_0:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT: %[[WSC_2:.*]] = sdy.sharding_constraint %[[WSC_1]] <@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ADD]], %[[WSC_2]]
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %1 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %3 = sdy.sharding_constraint %2 <@mesh, [{}, {}]> :  tensor<8x8xf32>
  return %1, %3 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @chain_on_op_result_after_other_user
func.func @chain_on_op_result_after_other_user(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: %[[DIV:.*]] = stablehlo.divide %[[ADD]], %[[ADD]]
  // CHECK-NEXT: %[[WSC_0:.*]] = sdy.sharding_constraint %[[ADD]] <@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %[[ADD]], %[[DIV]]
  // CHECK-NEXT: %[[WSC_2:.*]] = sdy.sharding_constraint %[[WSC_1]] <@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[MUL]], %[[WSC_2]]
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = stablehlo.divide %0, %0 :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %3 = sdy.sharding_constraint %2 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %4 = stablehlo.multiply %0, %1 :  tensor<8x8xf32>
  %5 = sdy.sharding_constraint %3 <@mesh, [{}, {}]> :  tensor<8x8xf32>
  return %4, %5 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @first_constraint_in_chain_has_multiple_uses
func.func @first_constraint_in_chain_has_multiple_uses(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[WSC_0:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT: %[[WSC_2:.*]] = sdy.sharding_constraint %[[WSC_1]] <@mesh, [{}, {}]>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[WSC_0]], %[[WSC_2]], %[[ADD]]
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{}, {}]> :  tensor<8x8xf32>
  %3 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  return %0, %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @second_constraint_in_chain_has_multiple_uses
func.func @second_constraint_in_chain_has_multiple_uses(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[WSC_0:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT: %[[WSC_2:.*]] = sdy.sharding_constraint %[[WSC_1]] <@mesh, [{}, {}]>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[WSC_1]], %[[WSC_2]], %[[ADD]]
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{}, {}]> :  tensor<8x8xf32>
  %3 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  return %1, %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @chain_input_used_by_other_constraint
func.func @chain_input_used_by_other_constraint(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: %[[WSC_0:.*]] = sdy.sharding_constraint %[[ADD_0]] <@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[ADD_0]], %[[ADD_0]]
  // CHECK-NEXT: %[[WSC_2:.*]] = sdy.sharding_constraint %[[ADD_0]] <@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[WSC_1]], %[[ADD_1]], %[[WSC_2]]
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %3 = stablehlo.add %0, %0 :  tensor<8x8xf32>
  %4 = sdy.sharding_constraint %0 <@mesh, [{}, {}]> :  tensor<8x8xf32>
  return %2, %3, %4 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @last_constraint_in_chain_used_by_manual_computation
func.func @last_constraint_in_chain_used_by_manual_computation(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
  // CHECK-NEXT: %[[WSC_0:.*]] = sdy.sharding_constraint %[[ADD_0]] <@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[ADD_0]], %[[ADD_0]]
  // CHECK-NEXT: sdy.manual_computation(%[[WSC_1]]
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %3 = stablehlo.add %0, %0 :  tensor<8x8xf32>
  %4 = sdy.manual_computation(%2) in_shardings=[<@mesh, [{"a"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>]
      manual_axes={"a"} (%arg2: tensor<4x8xf32>) {
    sdy.return %arg2 : tensor<4x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %4, %3 : tensor<8x8xf32>, tensor<8x8xf32>
}
