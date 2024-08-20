// RUN: sdy_opt %s -sdy-propagation-pipeline 2>&1 | FileCheck %s

// Propagation tests for ops with data-flow edges like CaseOp and WhileOp

sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @case_single_result_func_args_single_sharding(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>}
// CHECK-SAME:      %arg2: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}]>})
// CHECK-SAME:      -> (tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}]>})
func.func @case_single_result_func_args_single_sharding(%arg0: tensor<i32>, %arg1: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>}, %arg2: tensor<4xi64>) -> (tensor<4xi64>) {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<4xi64>
  }, {
    stablehlo.return %arg2 : tensor<4xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}]>]>} :
  }) : (tensor<i32>) -> tensor<4xi64>
  return %0 : tensor<4xi64>
}

// CHECK-LABEL: func @case_token_result_skipped(
// CHECK-SAME:      %arg0: tensor<i32>, %arg1: !stablehlo.token, %arg2: !stablehlo.token,
// CHECK-SAME:      %arg3: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>}
// CHECK-SAME:      %arg4: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}]>})
// CHECK-SAME:      -> (!stablehlo.token, tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}]>})
func.func @case_token_result_skipped(%arg0: tensor<i32>, %arg1: !stablehlo.token, %arg2: !stablehlo.token,
                                     %arg3: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>},
                                     %arg4: tensor<4xi64>) -> (!stablehlo.token, tensor<4xi64>) {
  %0:2 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1, %arg3 : !stablehlo.token, tensor<4xi64>
  }, {
    stablehlo.return %arg2, %arg4 : !stablehlo.token, tensor<4xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, []>, <@mesh_a_2_b_2, [{"a", ?}]>]>} :
  }) : (tensor<i32>) -> (!stablehlo.token, tensor<4xi64>)
  return %0#0, %0#1 : !stablehlo.token, tensor<4xi64>
}

// This test makes sure we do not propagate any scalar value through the case op
// (even though we try it), since the OpShardingRuleAttr on scalars has no
// factors. Need to stick any sort of sharding on an argument to make sure this
// op has a bound mesh.
// CHECK-LABEL: func @case_scalars(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<i64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [], replicated={"a"}>},
// CHECK-SAME:      %arg2: tensor<i64>)
// CHECK-SAME:      -> tensor<i64>
func.func @case_scalars(%arg0: tensor<i32>, %arg1: tensor<i64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [], replicated={"a"}>}, %arg2: tensor<i64>) -> tensor<i64> {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<i64>
  }, {
    stablehlo.return %arg2 : tensor<i64>
  }) : (tensor<i32>) -> tensor<i64>
  return %0 : tensor<i64>
}

// CHECK-LABEL: func @case_single_result_func_args_conflict(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "c"}]>})
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}]>})
func.func @case_single_result_func_args_conflict(%arg0: tensor<i32>, %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>}, %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "c"}]>}) -> (tensor<8xi64>) {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", ?}]>]>} :
  }) : (tensor<i32>) -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// We find the most compatible major sharding axes, so the fact the first
// sharding is best shouldn't matter.
// CHECK-LABEL: func @case_single_result_func_first_sharding_best_ignored(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", "c"}]>}
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c"}]>})
func.func @case_single_result_func_first_sharding_best_ignored(
  %arg0: tensor<i32>,
  %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", "c"}]>},
  %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c"}]>}) -> (tensor<8xi64>) {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8xi64>
  // Make sure no sharding on case
  // CHECK: }) :
  }) : (tensor<i32>) -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// CHECK-LABEL: func @case_multiple_results_different_sharding(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c", ?}]>},
// CHECK-SAME:      %arg3: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>},
// CHECK-SAME:      %arg4: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c", ?}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>},
// CHECK-SAME:          tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c", ?}]>})
func.func @case_multiple_results_different_sharding(
  %arg0: tensor<i32>,

  %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}]>},
  %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c", ?}]>},

  %arg3: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>},
  %arg4: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c", ?}]>}
  ) -> (tensor<8xi64>, tensor<8xi64>) {
  %0:2 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1, %arg2 : tensor<8xi64>, tensor<8xi64>
  }, {
    stablehlo.return %arg3, %arg4 : tensor<8xi64>, tensor<8xi64>
  // CHECK:      }) {sdy.sharding = #sdy.sharding_per_value<[
  // CHECK-SAME:     <@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>,
  // CHECK-SAME      <@mesh_a_2_b_2_c_2, [{"c", ?}]>]>]>} :
  }) : (tensor<i32>) -> (tensor<8xi64>, tensor<8xi64>)
  return %0#0, %0#1 : tensor<8xi64>, tensor<8xi64>
}

// Make sure we account for when an axis is used in another dimension when
// finding the most compatible major sharding axes.
// CHECK-LABEL: func @case_multiple_dim_most_compatible(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8x8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {"b", ?}]>},
// CHECK-SAME:      %arg2: tensor<8x8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>}
// CHECK-SAME:      -> (tensor<8x8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}, {?}]>})
func.func @case_multiple_dim_most_compatible(
  %arg0: tensor<i32>,

  %arg1: tensor<8x8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{?}, {"b", ?}]>},
  %arg2: tensor<8x8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>}
  ) -> (tensor<8x8xi64>) {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8x8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8x8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", ?}, {?}]>]>} :
  }) : (tensor<i32>) -> (tensor<8x8xi64>)
  return %0 : tensor<8x8xi64>
}

// CHECK-LABEL: func @case_multiple_results_different_sharding_conflicts(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"b", ?}]>},
// CHECK-SAME:      %arg3: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>},
// CHECK-SAME:      %arg4: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>},
// CHECK-SAME:          tensor<8xi64>)
func.func @case_multiple_results_different_sharding_conflicts(
  %arg0: tensor<i32>,

  %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}]>},
  %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"b", ?}]>},

  %arg3: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>},
  %arg4: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}]>}
  ) -> (tensor<8xi64>, tensor<8xi64>) {
  %0:2 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1, %arg2 : tensor<8xi64>, tensor<8xi64>
  }, {
    stablehlo.return %arg3, %arg4 : tensor<8xi64>, tensor<8xi64>
  // CHECK:      }) {sdy.sharding = #sdy.sharding_per_value<[
  // CHECK-SAME:     <@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>,
  // CHECK-SAME:     <@mesh_a_2_b_2_c_2, [{?}]>]>} :
  }) : (tensor<i32>) -> (tensor<8xi64>, tensor<8xi64>)
  return %0#0, %0#1 : tensor<8xi64>, tensor<8xi64>
}

// CHECK-LABEL: func @case_closed_sharding(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>}
func.func @case_closed_sharding(
  %arg0: tensor<i32>,
  %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}]>},
  %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>}
  ) -> tensor<8xi64> {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>]>} :
  }) : (tensor<i32>) -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// Basic case where the shardings are read from an intermediate value and used
// by another op.
// CHECK-LABEL: func @case_not_func_args_or_directly_returned(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>}
func.func @case_not_func_args_or_directly_returned(
  %arg0: tensor<i32>,
  %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}]>},
  %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>}
  ) -> tensor<8xi64> {
  // CHECK: stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>]>} : tensor<8xi64>
  %0 = stablehlo.add %arg1, %arg1 : tensor<8xi64>
  // CHECK: stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>]>} : tensor<8xi64>
  %1 = stablehlo.add %arg2, %arg2 : tensor<8xi64>
  // CHECK: %[[CASE:.*]] = "stablehlo.case"
  %2 = "stablehlo.case"(%arg0) ({
    stablehlo.return %0 : tensor<8xi64>
  }, {
    stablehlo.return %1 : tensor<8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>]>} :
  }) : (tensor<i32>) -> tensor<8xi64>
  // CHECK: stablehlo.add %[[CASE]], %[[CASE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>]>} : tensor<8xi64>
  %3 = stablehlo.add %2, %2 :  tensor<8xi64>
  return %3 : tensor<8xi64>
}

// CHECK-LABEL: func @case_propagate_from_func_result(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>}
func.func @case_propagate_from_func_result(
    %arg0: tensor<i32>, %arg1: tensor<8xi64>,
    %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}]>})
    -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>}) {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}]>]>} :
  }) : (tensor<i32>) -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// CHECK-LABEL: func @case_propagate_from_case_op_result(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>}
func.func @case_propagate_from_case_op_result(
    %arg0: tensor<i32>, %arg1: tensor<8xi64>,
    %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}]>})
    -> tensor<8xi64> {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}]>]>} :
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}]>]>} : (tensor<i32>) -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// CHECK-LABEL: func @case_propagate_mulitple_uses(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>}
func.func @case_propagate_mulitple_uses(
    %arg0: tensor<i32>, %arg1: tensor<8xi64>,
    %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>})
    -> (tensor<8xi64>) {
  // CHECK: %[[CASE:.*]] = "stablehlo.case"(%arg0)
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}]>]>} :
  }) : (tensor<i32>) -> tensor<8xi64>
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[CASE]], %[[CASE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}]>]>} :
  %1 = stablehlo.add %0, %0 : tensor<8xi64>
  // CHECK: %[[SUB:.*]] = stablehlo.subtract %[[CASE]], %[[CASE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}]>]>} :
  %2 = stablehlo.subtract %0, %0 : tensor<8xi64>
  // CHECK: stablehlo.divide %[[SUB]], %[[ADD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}]>]>} :
  %3 = stablehlo.divide %2, %1 : tensor<8xi64>
  return %3 : tensor<8xi64>
}

// CHECK-LABEL: func @optimization_barrier(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b", ?}]>},
// CHECK-SAME:      %arg1: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b"}]>},
// CHECK-SAME:          tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b"}, {?}]>})
func.func @optimization_barrier(%arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {?}]>}, %arg1: tensor<32x96xf32>)
    -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b"}]>},
        tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b"}, {?}]>}) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>}
  %0 = stablehlo.add %arg0, %arg0 : tensor<32x96xf32>
  // CHECK-NEXT: stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>, <@mesh_a_2_b_2, [{"b", ?}, {?}]>]>}
  %1:2 = stablehlo.optimization_barrier %0, %arg1 : tensor<32x96xf32>, tensor<32x96xf32>
  return %1#0, %1#1 : tensor<32x96xf32>, tensor<32x96xf32>
}

// Check propagation starting at a return can go through a WhileOp.
// CHECK-LABEL: func @while_func_return(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
func.func @while_func_return(%arg0: tensor<32x96xf32>) -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) {
  // CHECK: %[[C0:.*]] = sdy.constant dense<0>
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  // CHECK: stablehlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]]) : tensor<32x96xf32>, tensor<i32>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>, <@mesh_a_2_b_2, []>]>}
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>} : tensor<32x96xf32>
    %5 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %5, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// Check propagation starting at a return can go through a WhileOp, but with it
// being based purely on the func block arg.
// CHECK-LABEL: func @while_func_return_directly_on_func_operand(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
func.func @while_func_return_directly_on_func_operand(
    %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>})
    -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b", ?}]>}) {
  // CHECK: %[[C0:.*]] = sdy.constant dense<0>
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  // CHECK: stablehlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]]) : tensor<32x96xf32>, tensor<i32>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>, <@mesh_a_2_b_2, []>]>}
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>} : tensor<32x96xf32>
    %5 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %5, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// Check propagation starting at a use of the while result can go through the
// WhileOp.
// CHECK-LABEL: func @while_result_use(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
func.func @while_result_use(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  // CHECK: %[[C0:.*]] = sdy.constant dense<0>
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  // CHECK: stablehlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]]) : tensor<32x96xf32>, tensor<i32>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>, <@mesh_a_2_b_2, []>]>}
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %5 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %5 : tensor<i1>
  } do {
    %5 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>} : tensor<32x96xf32>
    %6 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %6, %5 : tensor<32x96xf32>, tensor<i32>
  }
  %4 = stablehlo.add %3#0, %3#0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
  return %4 : tensor<32x96xf32>
}

// Check propagation starting inside the body of the while can go out of the
// body through the block argument of the body. We prevent forwards propagation
// in the body by making the use of the sharded op be not partitionable.
// CHECK-LABEL: func @while_body_propagate_block_arg(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
func.func @while_body_propagate_block_arg(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    %5 = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
    // Not partitionable!
    %6 = stablehlo.custom_call @sdy_testonly(%5) : (tensor<32x96xf32>) -> tensor<32x96xf32>
    stablehlo.return %6, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// CHECK-LABEL: func @while_body_token_block_arg_skipped(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>},
// CHECK-SAME:      %arg1: !stablehlo.token)
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>}, !stablehlo.token)
func.func @while_body_token_block_arg_skipped(%arg0: tensor<32x96xf32>, %arg1: !stablehlo.token)
    -> (tensor<32x96xf32>, !stablehlo.token) {
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  %3:3 = stablehlo.while(%iterArg = %arg1, %iterArg_0 = %arg0, %iterArg_1 = %0) : !stablehlo.token, tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_1, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_1, %1 : tensor<i32>
    %5 = stablehlo.add %iterArg_0, %iterArg_0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
    // Not partitionable!
    %6 = stablehlo.custom_call @sdy_testonly(%5) : (tensor<32x96xf32>) -> tensor<32x96xf32>
    stablehlo.return %iterArg, %6, %4 : !stablehlo.token, tensor<32x96xf32>, tensor<i32>
  }
  return %3#1, %3#0 : tensor<32x96xf32>, !stablehlo.token
}

// Similar test to while_body_propagate_block_arg, except this makes sure that
// propagation flows to the WhileOp operand through the return op - and not
// through the block argument. We prevent backwards propagation in the body by
// making the defining op of the sharded op's operand be not partitionable.
// CHECK-LABEL: func @while_body_propagate_return_op(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
func.func @while_body_propagate_return_op(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    // Not partitionable!
    %5 = stablehlo.custom_call @sdy_testonly(%iterArg) : (tensor<32x96xf32>) -> tensor<32x96xf32>
    %6 = stablehlo.add %5, %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
    stablehlo.return %6, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// Same as above, except the use is not a func operand nor is the result of the
// while directly returned from the func.
// CHECK-LABEL: func @while_body_non_func_operand_result_use(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
func.func @while_body_non_func_operand_result_use(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  // CHECK: %[[C0:.*]] = sdy.constant dense<0>
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>} : tensor<32x96xf32>
  %3 = stablehlo.add %arg0, %arg0 : tensor<32x96xf32>
  // CHECK: %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %[[ADD]], %iterArg_0 = %[[C0]]) : tensor<32x96xf32>, tensor<i32>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>, <@mesh_a_2_b_2, []>]>}
  %4:2 = stablehlo.while(%iterArg = %3, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %6 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %6 : tensor<i1>
  } do {
    %6 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    %7 = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
    stablehlo.return %7, %6 : tensor<32x96xf32>, tensor<i32>
  }
  // CHECK: stablehlo.add %[[WHILE]]#0, %[[WHILE]]#0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>} : tensor<32x96xf32>
  %5 = stablehlo.add %4#0, %4#0 : tensor<32x96xf32>
  return %5 : tensor<32x96xf32>
}

// Check propagation starting from a func arg can go through the WhileOp.
// CHECK-LABEL: func @while_func_operand(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>})
func.func @while_func_operand(%arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) -> tensor<32x96xf32> {
  // CHECK: %[[C0:.*]] = sdy.constant dense<0>
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  // CHECK: stablehlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]]) : tensor<32x96xf32>, tensor<i32>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>, <@mesh_a_2_b_2, []>]>}
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", ?}, {"b", ?}]>]>} : tensor<32x96xf32>
    %5 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %5, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}
