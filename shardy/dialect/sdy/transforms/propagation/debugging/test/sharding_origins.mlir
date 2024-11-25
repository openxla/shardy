// RUN: sdy_opt %s -sdy-aggressive-propagate="debug-sharding-origins=true" 2>&1 | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2, "c"=8]>

// CHECK-LABEL: input_output_source_sharding
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.origin_sharding = {a = "self", c = "input: 1"},
// CHECK-SAME:                            sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"c", ?}]>}
// CHECK-SAME:    %arg1: tensor<8x8xf32> {sdy.origin_sharding = {a = "input: 0", c = "self"},
// CHECK-SAME:                            sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"c", ?}]>}
// CHECK-SAME:    %arg2: tensor<8x16xf32> {sdy.origin_sharding = {b = "output: 0", c = "input: 1"},
// CHECK-SAME:                             sdy.sharding = #sdy.sharding<@mesh, [{"c", ?}, {"b", ?}]>}
// CHECK-SAME:    -> (tensor<8x16xf32> {sdy.origin_sharding = {a = "input: 0", b = "self"},
// CHECK-SAME:                          sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>}) {
func.func @input_output_source_sharding(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
  %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c", ?}]>},
  %arg2: tensor<8x16xf32>
  ) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {
  // CHECK-SAME:   sdy.origin_sharding = {a = "input: 0", c = "input: 1"},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>}
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[ADD]], %arg2, contracting_dims = [1] x [0] {
  // CHECK-SAME:   sdy.origin_sharding = {a = "input: 0", b = "output: 0"},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: return %[[DOT]]
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// TODO(b/379627411): This should say the output sharding on axis `a` came from
// self.
// CHECK-LABEL: direct_returned_arg_new_axis_input
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.origin_sharding = {a = "self", b = "self"},
// CHECK-SAME:                            sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.origin_sharding = {a = "input: 0", b = "input: 0"}
// CHECK-SAME:                         sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}) {
func.func @direct_returned_arg_new_axis_input(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>})
  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) {
  return %arg0 : tensor<8x8xf32>
}

// TODO(b/379627411): This should say the output sharding on axis `a` came from
// self.
// CHECK-LABEL: direct_returned_arg_new_axis_output
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.origin_sharding = {a = "self", b = "output: 0"},
// CHECK-SAME:                            sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.origin_sharding = {a = "input: 0", b = "self"}
// CHECK-SAME:                         sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}) {
func.func @direct_returned_arg_new_axis_output(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>})
  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}) {
  return %arg0 : tensor<8x8xf32>
}


// CHECK-LABEL: single_sharding_constraint
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.origin_sharding = {a = "constraint_0"},
// CHECK-SAME:                            sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.origin_sharding = {a = "constraint_0"},
// CHECK-SAME:                         sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) {
func.func @single_sharding_constraint(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[SC:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> {
  // CHECK-SAME:   sdy.origin_sharding = {a = "self"}, sdy.origin_sharding_name = "constraint_0"}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[SC]], %[[SC]] {
  // CHECK-SAME:   sdy.origin_sharding = {a = "constraint_0"},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  // CHECK-NEXT: return %[[ADD]]
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: multiple_axes
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.origin_sharding = {a = "self", b = "self"},
// CHECK-SAME:                            sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.origin_sharding = {a = "self", b = "input: 0"}
// CHECK-SAME:                         sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}) {
func.func @multiple_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {
  // CHECK-SAME:   sdy.origin_sharding = {a = "output: 0", b = "input: 0"},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b", ?}, {?}]>]>}
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: sub_axis_update
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.origin_sharding = {"c:(1)4" = "self"},
// CHECK-SAME:                            sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)4, ?}]>}
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.origin_sharding = {"c:(1)4" = "input: 0"},
// CHECK-SAME:                         sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)4, ?}]>})
func.func @sub_axis_update(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)4, ?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)2, ?}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {
  // CHECK-SAME:   sdy.origin_sharding = {"c:(1)4" = "input: 0"},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>}
  %1 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: two_sharding_constraint
// CHECK-SAME:    {sdy.origin_sharding = {a = "constraint_1", b = "constraint_2"},
// CHECK-SAME:     sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>})
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.origin_sharding = {a = "constraint_1", b = "constraint_2"},
// CHECK-SAME:                         sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>}) {
func.func @two_sharding_constraint(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[SC_1:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b", ?}]> {
  // CHECK-SAME:   sdy.origin_sharding = {a = "self", b = "constraint_2"}, sdy.origin_sharding_name = "constraint_1"}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[SC_1]], %[[SC_1]] {
  // CHECK-SAME:   sdy.origin_sharding = {a = "constraint_1", b = "constraint_2"},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: %[[SC_2:.*]] = sdy.sharding_constraint %[[ADD]] <@mesh, [{"a", ?}, {"b"}]> {
  // CHECK-SAME:   sdy.origin_sharding = {a = "constraint_1", b = "self"}, sdy.origin_sharding_name = "constraint_2"}
  // CHECK-NEXT: return %[[SC_2]]
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{?}, {"b"}]> : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// CHECK-LABEL: manual_computation_no_manual_axes
// CHECK-SAME:    %arg0: tensor<32x32x32xf32> {sdy.origin_sharding = {a = "mc_0_input: 0", b = "mc_0_output: 0"},
// CHECK-SAME:                                 sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}, {?}]>})
// CHECK-SAME:    -> (tensor<32x32x32xf32> {sdy.origin_sharding = {a = "mc_0_input: 0", b = "mc_0_output: 0"},
// CHECK-SAME:                              sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}, {?}]>}) {
func.func @manual_computation_no_manual_axes(%arg0: tensor<32x32x32xf32>) -> tensor<32x32x32xf32> {
  // CHECK-NEXT: %[[SUB:.*]] = stablehlo.subtract %arg0, %arg0 {sdy.origin_sharding = {a = "mc_0_input: 0", b = "mc_0_output: 0"},
  // CHECK-SAME:                                                sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]>}
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[SUB]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]
  // CHECK-SAME:   manual_axes={} (%arg1: tensor<32x32x32xf32>) {
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 {
  // CHECK-SAME:     sdy.origin_sharding = {a = "mc_0_input: 0", b = "mc_0_output: 0"},
  // CHECK-SAME:     sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]>}
  // CHECK-NEXT:   sdy.return %[[ADD]]
  // CHECK-NEXT: } {sdy.origin_sharding = {a = "mc_0_input: 0", b = "mc_0_output: 0"}, sdy.origin_sharding_name = "mc_0"}
  // CHECK-NEXT: %[[SUB_2:.*]] = stablehlo.subtract %[[MC]], %[[MC]] {
  // CHECK-SAME:   sdy.origin_sharding = {a = "mc_0_input: 0", b = "mc_0_output: 0"},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]>}
  // CHECK-NEXT: return %[[SUB_2]]
  %0 = stablehlo.subtract %arg0, %arg0 : tensor<32x32x32xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"a", ?}, {?}, {?}]>] out_shardings=[<@mesh, [{?}, {"b", ?}, {?}]>] manual_axes={} (%arg1: tensor<32x32x32xf32>) {
    %3 = stablehlo.add %arg1, %arg1 : tensor<32x32x32xf32>
    sdy.return %3 : tensor<32x32x32xf32>
  } : (tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  %2 = stablehlo.subtract %1, %1 : tensor<32x32x32xf32>
  return %2: tensor<32x32x32xf32>
}


// Show that when an axis came from multiple sources, we just use the first we
// see left to right, from operands to results.
// CHECK-LABEL: tie_across_operands_results
// CHECK-SAME:    %arg0: tensor<8xf32> {sdy.origin_sharding = {a = "self"},
// CHECK-SAME:                          sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
// CHECK-SAME:    %arg1: tensor<8xf32> {sdy.origin_sharding = {a = "self"},
// CHECK-SAME:                          sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}
// CHECK-SAME:    -> (tensor<8xf32> {sdy.origin_sharding = {a = "constraint_3"},
// CHECK-SAME:                       sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>}) {
func.func @tie_across_operands_results(
  %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
  %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}
  ) -> (tensor<8xf32>) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {
  // CHECK-SAME:   sdy.origin_sharding = {a = "input: 0"},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>]>} : tensor<8xf32>
  // CHECK-NEXT: %[[SC:.*]] = sdy.sharding_constraint %[[ADD]] <@mesh, [{"a"}]> {
  // CHECK-SAME:   sdy.origin_sharding = {a = "self"}, sdy.origin_sharding_name = "constraint_3"} : tensor<8xf32>
  // CHECK-NEXT: return %[[SC]]
  %0 = stablehlo.add %arg0, %arg1 : tensor<8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}]> : tensor<8xf32>
  return %1 :tensor<8xf32>
}

// CHECK-LABEL: push_sharding_constraints_to_func_results
// CHECK-SAME:   %arg0: tensor<8xf32> {sdy.origin_sharding = {a = "constraint_4"},
// CHECK-SAME:                         sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>},
// CHECK-SAME:   %arg1: tensor<8xf32> {sdy.origin_sharding = {a = "constraint_5"},
// CHECK-SAME:                         sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>})
// CHECK-SAME:   -> (tensor<8xf32> {sdy.origin_sharding = {a = "constraint_4"},
// CHECK-SAME:                      sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>},
// CHECK-SAME:       tensor<8xf32> {sdy.origin_sharding = {a = "constraint_5"},
// CHECK-SAME:                      sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>}) {
func.func @push_sharding_constraints_to_func_results(
  %arg0: tensor<8xf32>, %arg1: tensor<8xf32>
  ) -> (tensor<8xf32>, tensor<8xf32>) {
  // CHECK-NEXT: %[[SC_1:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}]> {
  // CHECK-SAME:   sdy.origin_sharding = {a = "self"}, sdy.origin_sharding_name = "constraint_4"} : tensor<8xf32>
  // CHECK-NEXT: %[[SC_2:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{"a"}]> {
  // CHECK-SAME:   sdy.origin_sharding = {a = "self"}, sdy.origin_sharding_name = "constraint_5"} : tensor<8xf32>
  // CHECK-NEXT: return %[[SC_1]], %[[SC_2]] : tensor<8xf32>, tensor<8xf32>
  %1 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}]> : tensor<8xf32>
  %2 = sdy.sharding_constraint %arg1 <@mesh, [{"a"}]> : tensor<8xf32>
  return %1, %2 : tensor<8xf32>, tensor<8xf32>
}

// TODO(b/379280210): The origin sharding on `a` should be `input: 0`.
// CHECK-LABEL: real_conflict_across_factors_diff_tensors_size
func.func @real_conflict_across_factors_diff_tensors_size(
    %arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}, {?}]>},
    %arg1: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", "c"}]>})
    -> tensor<8x16xf32> {
  // CHECK-NEXT: stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {
  // CHECK-SAME:   sdy.origin_sharding = {a = "input: 0", c = "input: 1"},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", "c", ?}]>]>} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] :
    (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// TODO(b/379279614): support ops with multiple results.

// TODO(b/379279617): After `ManualComputationOp` implements the
// `ShardableDataFlowOpInterface`, add a test for manual computation with
// manual axes.

// TODO(b/379279614): add test for while loop with multiple results, need to make
// sure the origin sharding info is preserved after sinking the dataflow edge
// ops.

// TODO(b/379631271): Add test doing a 8->4x4 reshape with axis size 8
// (introduces a new sub axis).
