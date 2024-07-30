// RUN: sdy_opt %s -sdy-op-priority-propagate 2>&1 | FileCheck %s

sdy.mesh @mesh = <"a"=2, "b"=2>

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
  // CHECK-SAME{LITERAL}:     in_shardings=[<@mesh, [{"a", ?}, {?}]>, <@mesh, [{?}, {"b", ?}]>]
  // CHECK-SAME{LITERAL}:     out_shardings=[<@mesh, [{"a", ?}, {"b", ?}]>]
  // CHECK-SAME{LITERAL}:     manual_axes={}
  // CHECK-SAME:              (%arg2: tensor<32x32xf32>, %arg3: tensor<32x32xf32>) {
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{?}, {?}]>, <@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{"a", ?}, {?}]>] manual_axes={} (%arg2: tensor<32x32xf32>, %arg3: tensor<32x32xf32>) {
    // CHECK:      %[[DOT:.*]] = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
    // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[DOT]], %[[DOT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
    // CHECK_NEXT sdy.return %[[ADD_1]]
    %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %2 = stablehlo.add %1, %1 : tensor<32x32xf32>
    sdy.return %2 : tensor<32x32xf32>
  } : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[RET:.*]] = stablehlo.add %[[MC]], %[[MC]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: return %[[RET]] : tensor<32x32xf32>
  %3 = stablehlo.add %0, %0 : tensor<32x32xf32>
  func.return %3: tensor<32x32xf32>
}
