// RUN: sdy_opt %s -sdy-op-priority-propagate 2>&1 | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Without prioritizing element-wise ops first, the sharding on dim 0 would
// have been propagated first.
// CHECK-LABEL: func @element_wise_over_dot_general(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>}) {
func.func @element_wise_over_dot_general(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}, %arg1: tensor<8x8xf32>) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>}) {
  // CHECK:      %[[DOT:.*]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[DOT]], %[[DOT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[ADD_1]], %[[ADD_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: return %[[ADD_2]] : tensor<8x8xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x8xf32>
  %2 = stablehlo.add %1, %1 : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// Same as `element_wise_over_dot_general` but the dot_general is the last op.
// CHECK-LABEL: func @element_wise_over_dot_general_flipped_op_order(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>}) {
func.func @element_wise_over_dot_general_flipped_op_order(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}, %arg1: tensor<8x8xf32>) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>}) {
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[ADD_1]], %[[ADD_2]], contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: return %[[DOT]] : tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  %1 = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
  %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// If we propagated forward through element-wise ops with multiple uses in the
// first iteration, the sharding on dim 0 would have been propagated to the two
// add ops, which would result in two reshards instead of one.
// CHECK-LABEL: func @defer_forward_propagation_for_multi_use_ops
func.func @defer_forward_propagation_for_multi_use_ops(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>},
        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}) {
  // CHECK-NEXT: %[[SINE:.*]] = stablehlo.sine %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[SINE]], %[[SINE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[SINE]], %[[SINE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[COSINE_1:.*]] = stablehlo.cosine %[[ADD_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[COSINE_2:.*]] = stablehlo.cosine %[[ADD_2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: return %[[COSINE_1]], %[[COSINE_2]]
  %1 = stablehlo.sine %arg0 : tensor<8x8xf32>
  %2 = stablehlo.add %1, %1 : tensor<8x8xf32>
  %3 = stablehlo.add %1, %1 : tensor<8x8xf32>
  // Need the additinal ops since they will get the result sharding regardless
  // of op priority propagation.
  %4 = stablehlo.cosine %2 : tensor<8x8xf32>
  %5 = stablehlo.cosine %3 : tensor<8x8xf32>
  return %4, %5 : tensor<8x8xf32>, tensor<8x8xf32>
}

// If we propagated forward through dynamic-slice op with multiple uses in the
// first iteration, the sharding on dim 0 would have been propagated to the two
// add ops.
// CHECK-LABEL: func @defer_forward_propagation_for_multi_use_dynamic_slice
func.func @defer_forward_propagation_for_multi_use_dynamic_slice(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<i32>, %arg2: tensor<i32>)
    -> (tensor<8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}) {
  // CHECK-NEXT: %[[DS:.*]] = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [8, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[DS]], %[[DS]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[DS]], %[[DS]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %[[ADD_1]], %[[ADD_2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: return %[[MUL]]
  %1 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [8, 2] : (tensor<8x8xf32>, tensor<i32>, tensor<i32>) -> tensor<8x2xf32>
  %2 = stablehlo.add %1, %1 : tensor<8x2xf32>
  %3 = stablehlo.add %1, %1 : tensor<8x2xf32>
  // Need the additinal op since it will get the result sharding regardless of
  // op priority propagation.
  %4 = stablehlo.multiply %2, %3 : tensor<8x2xf32>
  return %4 : tensor<8x2xf32>
}

// If we propagated backwards through element-wise ops with a multi-use operand
// in the first iteration, the sharding on dim 1 would have been propagated to
// %arg0.
// CHECK-LABEL: func @defer_backwards_propagation_for_op_with_multi_use_operand(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>})
func.func @defer_backwards_propagation_for_op_with_multi_use_operand(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>},
        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[SINE:.*]] = stablehlo.sine %[[ADD_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  // CHECK-NEXT: return %[[ADD_2]], %[[SINE]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  %1 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  %2 = stablehlo.sine %0 : tensor<8x8xf32>
  return %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @defer_backwards_propagation_dynamic_slice(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>}
func.func @defer_backwards_propagation_dynamic_slice(
    %arg0: tensor<8x8xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>)
    -> (tensor<8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}) {
  // CHECK-NEXT: %[[DS:.*]] = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [8, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: return %[[DS]], %[[ADD]]
  %1 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [8, 2] : (tensor<8x8xf32>, tensor<i32>, tensor<i32>) -> tensor<8x2xf32>
  %2 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %1, %2 : tensor<8x2xf32>, tensor<8x8xf32>
}

// Verify that the element-wise ops are sharded on dim 1 due to the
// `sharding_constraint`. Without `sharding_constraint` haveing the
// `Elementwise` trait, then the element-wise ops would be sharded on dim 0
// first instead.
// CHECK-LABEL: func @sharding_constraint_propagated(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>}) {
func.func @sharding_constraint_propagated(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK:      %[[DOT:.*]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[DOT]], %[[DOT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[ADD_1]], %[[ADD_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[WSC:.*]] = sdy.sharding_constraint %[[ADD_2]] <@mesh, [{?}, {"a", ?}]> : tensor<8x8xf32>
  // CHECK-NEXT: return %[[WSC]] : tensor<8x8xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x8xf32>
  %2 = stablehlo.add %1, %1 : tensor<8x8xf32>
  %3 = sdy.sharding_constraint %2 <@mesh, [{?}, {"a", ?}]> : tensor<8x8xf32>
  return %3 : tensor<8x8xf32>
}

// Makes sure that dot_generals are propagated. If we were to accidentally only
// propagate just element-wise ops, this test would fail.
// CHECK-LABEL: func @chained_dot_generals(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32>)
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>}) {
func.func @chained_dot_generals(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}, %arg1: tensor<8x8xf32>) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>}) {
  // CHECK:      %[[DOT_1:.*]] = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  // CHECK:      %[[DOT_2:.*]] = stablehlo.dot_general %[[DOT_1]], %[[DOT_1]], contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[DOT_2]], %[[DOT_2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[ADD_1]], %[[ADD_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: return %[[ADD_2]] : tensor<8x8xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %0, contracting_dims = [1] x [0] : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  %2 = stablehlo.add %1, %1 : tensor<8x8xf32>
  %3 = stablehlo.add %2, %2 : tensor<8x8xf32>
  return %3 : tensor<8x8xf32>
}

// For an unknown op that isn't in the list of op strategies, make sure we can
// still propagate through it.
// CHECK-LABEL: func @unknown_op(
// CHECK-SAME:      %arg0: tensor<4x1x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}, {?}]>},
// CHECK-SAME:      %arg1: tensor<4x1x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}, {?}]>},
// CHECK-SAME:      %arg2: tensor<4x1x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}, {?}]>})
// CHECK-SAME:      -> (tensor<4x3x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}, {?}]>}) {
func.func @unknown_op(%arg0: tensor<4x1x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}, {?}]>}, %arg1: tensor<4x1x256xf32>, %arg2: tensor<4x1x256xf32>) -> tensor<4x3x256xf32> {
 %0 = stablehlo.concatenate %arg0, %arg1, %arg2, dim = 1 : (tensor<4x1x256xf32>, tensor<4x1x256xf32>, tensor<4x1x256xf32>) -> tensor<4x3x256xf32>
 return %0 : tensor<4x3x256xf32>
}

// Make sure that `DataFlowEdgeOp` can be propagated through. It isn't
// registered as an op based strategy, but is handled in utility functions used
// by basic propagation.
// CHECK-LABEL: func @data_flow_edge(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>},
// CHECK-SAME:      %arg1: tensor<32x96xf32>)
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>}, tensor<32x96xf32>)
func.func @data_flow_edge(%arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}, %arg1: tensor<32x96xf32>) -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>}, tensor<32x96xf32>) {
  // CHECK: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>} : tensor<32x96xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<32x96xf32>
  %1:2 = stablehlo.optimization_barrier %0, %arg1 : tensor<32x96xf32>, tensor<32x96xf32>
  %2 = sdy.data_flow_edge %1#0 : tensor<32x96xf32>
  %3 = sdy.data_flow_edge %1#1 : tensor<32x96xf32>
  return %2, %3 : tensor<32x96xf32>, tensor<32x96xf32>
}

// CHECK-LABEL: func @manual_computation(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
// CHECK-SAME:      %arg1: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>}) {
func.func @manual_computation(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}, %arg1: tensor<32x32xf32>) -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>}) {
  // CHECK:               %[[MC:.*]] = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh, [{?}, {?}]>, <@mesh, [{?}, {?}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh, [{"a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={}
  // CHECK-SAME:              (%arg2: tensor<32x32xf32>, %arg3: tensor<32x32xf32>) {
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{?}, {?}]>, <@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{"a", ?}, {?}]>] manual_axes={} (%arg2: tensor<32x32xf32>, %arg3: tensor<32x32xf32>) {
    // CHECK-NEXT: %[[EDGE_1:.*]] = sdy.data_flow_edge %arg2 sharding=<@mesh, [{"a", ?}, {?}]> : tensor<32x32xf32>
    // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %arg3 sharding=<@mesh, [{?}, {"b", ?}]> : tensor<32x32xf32>
    // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[EDGE_1]], %[[EDGE_2]], contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
    // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[DOT]], %[[DOT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
    // CHECK_NEXT sdy.return %[[ADD_1]]
    %1 = sdy.data_flow_edge %arg2 sharding=<@mesh, [{?}, {?}]> : tensor<32x32xf32>
    %2 = sdy.data_flow_edge %arg3 sharding=<@mesh, [{?}, {?}]> : tensor<32x32xf32>
    %3 = stablehlo.dot_general %1, %2, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = stablehlo.add %3, %3 : tensor<32x32xf32>
    sdy.return %4 : tensor<32x32xf32>
  } : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[EDGE_3:.*]] = sdy.data_flow_edge %0 sharding=<@mesh, [{"a", ?}, {"b", ?}]> : tensor<32x32xf32>
  // CHECK-NEXT: return %[[EDGE_3]] : tensor<32x32xf32>
  %4 = sdy.data_flow_edge %0 sharding=<@mesh, [{"a", ?}, {?}]> : tensor<32x32xf32>
  func.return %4: tensor<32x32xf32>
}

// CHECK-LABEL: func @pass_through_factor_higher_priority_than_reduction_factor(
// CHECK-SAME:      %arg0: tensor<32x1024xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
// CHECK-SAME:      %arg1: tensor<1024x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:      -> (tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>}) {
func.func @pass_through_factor_higher_priority_than_reduction_factor(
  %arg0: tensor<32x1024xf32>,
  %arg1: tensor<1024x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>}
) -> (tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: return %[[DOT]] : tensor<32x16xf32>
  %0 = stablehlo.dot %arg0, %arg1, precision = [DEFAULT, DEFAULT] : (tensor<32x1024xf32>, tensor<1024x16xf32>) -> tensor<32x16xf32>
  return %0 : tensor<32x16xf32>
}

// CHECK-LABEL: func @unreduced_axes_block_bwd_propagation(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
// CHECK-SAME:      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>})
// CHECK-SAME:  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}]>}) {
func.func @unreduced_axes_block_bwd_propagation(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<8x16xf32>)
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b"}]>}) {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}], unreduced={"b"}>]>}
  // CHECK-NEXT: stablehlo.add %[[DOT_GENERAL]], %[[DOT_GENERAL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?}], unreduced={"b"}>]>} :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @broadcast_forward_higher_priority_than_backwards
func.func @broadcast_forward_higher_priority_than_backwards(
  %arg0: tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}
) -> (tensor<32x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}, {}]>}) {
  // CHECK-NEXT: %[[BROADCAST_1:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>}
  // CHECK-NEXT: %[[BROADCAST_2:.*]] = stablehlo.broadcast_in_dim %[[BROADCAST_1]], dims = [0, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}, {?}]>]>}
  // CHECK-NEXT: return %[[BROADCAST_2]]
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<32xf32>) -> tensor<32x16xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1] : (tensor<32x16xf32>) -> tensor<32x16x8xf32>
  return %1 : tensor<32x16x8xf32>
}
