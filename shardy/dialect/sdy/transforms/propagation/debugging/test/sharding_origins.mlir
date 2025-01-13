// RUN: sdy_opt %s -sdy-add-data-flow-edges -sdy-aggressive-propagate="debug-sharding-origins=true" -sdy-sink-data-flow-edges 2>&1 | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2, "c"=8]>

// CHECK-LABEL: input_output_source_sharding
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"c", ?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {a = "self", c = "input: 1"}}
// CHECK-SAME:    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"c", ?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {a = "input: 0", c = "self"}}
// CHECK-SAME:    %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c", ?}, {"b", ?}]>,
// CHECK-SAME:                             sdy.sharding_origins = {b = "output: 0", c = "input: 1"}}
// CHECK-SAME:    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>,
// CHECK-SAME:                          sdy.sharding_origins = {a = "input: 0", b = "self"}}) {
func.func @input_output_source_sharding(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
  %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c", ?}]>},
  %arg2: tensor<8x16xf32>
  ) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "input: 0", c = "input: 1"}]}
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[ADD]], %arg2, contracting_dims = [1] x [0] {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "input: 0", b = "output: 0"}]}
  // CHECK-NEXT: return %[[DOT]]
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: partial_axes_match
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {a = "self", b = "self"}},
// CHECK-SAME:    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {a = "input: 0", b = "self"}}
// CHECK-SAME:  ) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>,
// CHECK-SAME:                         sdy.sharding_origins = {a = "input: 0", b = "input: 0"}}) {
func.func @partial_axes_match(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>},
  %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>}
  ) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b", ?}, {?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "input: 0", b = "input: 0"}]}
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: larger_prefix_match
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {a = "self"}},
// CHECK-SAME:    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {a = "self", b = "self"}}
// CHECK-SAME:  ) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>,
// CHECK-SAME:                         sdy.sharding_origins = {a = "input: 1", b = "input: 1"}}) {
func.func @larger_prefix_match(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {?}]>},
  %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}
  ) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b", ?}, {?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "input: 1", b = "input: 1"}]}
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// TODO(b/379627411): This should say the output sharding on axis `a` came from
// self.
// CHECK-LABEL: direct_returned_arg_new_axis_input
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {a = "self", b = "self"}}
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>,
// CHECK-SAME:                         sdy.sharding_origins = {a = "input: 0", b = "input: 0"}}) {
func.func @direct_returned_arg_new_axis_input(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>})
  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) {
  return %arg0 : tensor<8x8xf32>
}

// TODO(b/379627411): This should say the output sharding on axis `a` came from
// self.
// CHECK-LABEL: direct_returned_arg_new_axis_output
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {a = "self", b = "output: 0"}}
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>,
// CHECK-SAME:                         sdy.sharding_origins = {a = "input: 0", b = "self"}}) {
func.func @direct_returned_arg_new_axis_output(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>})
  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}) {
  return %arg0 : tensor<8x8xf32>
}


// CHECK-LABEL: single_sharding_constraint
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {a = "constraint_0"}}
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>,
// CHECK-SAME:                         sdy.sharding_origins = {a = "constraint_0"}}) {
func.func @single_sharding_constraint(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[SC:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> {
  // CHECK-SAME:   sdy.sharding_origin_name = "constraint_0",
  // CHECK-SAME:   sdy.sharding_origins = {a = "self"}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[SC]], %[[SC]] {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "constraint_0"}]}
  // CHECK-NEXT: return %[[ADD]]
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: multiple_axes
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {a = "self", b = "self"}}
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>,
// CHECK-SAME:                         sdy.sharding_origins = {a = "self", b = "input: 0"}}) {
func.func @multiple_axes(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b", ?}, {?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "output: 0", b = "input: 0"}]}
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// TODO(b/381870366): pick something smarter for merging sub axis. Below we are
// saving intermediate shardings for the output, when it starts with c:(1)2 and
// then c:(1)4, but this may not be the best way to do it.
// CHECK-LABEL: sub_axis_update
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)4, ?}]>,
// CHECK-SAME:                            sdy.sharding_origins = {"c:(1)4" = "self"}}
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)4, ?}]>,
// CHECK-SAME:                         sdy.sharding_origins = {"c:(1)2" = "self", "c:(1)4" = "input: 0"}}) {
func.func @sub_axis_update(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)4, ?}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)2, ?}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{"c:(1)2" = "output: 0", "c:(1)4" = "input: 0"}]}
  %1 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: two_sharding_constraint
// CHECK-SAME:    {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>,
// CHECK-SAME:     sdy.sharding_origins = {a = "constraint_1", b = "constraint_2"}})
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>,
// CHECK-SAME:                         sdy.sharding_origins = {a = "constraint_1", b = "constraint_2"}}) {
func.func @two_sharding_constraint(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[SC_1:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b", ?}]> {
  // CHECK-SAME:   sdy.sharding_origin_name = "constraint_1",
  // CHECK-SAME:   sdy.sharding_origins = {a = "self", b = "constraint_2"}} : tensor<8x8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[SC_1]], %[[SC_1]] {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "constraint_1", b = "constraint_2"}]}
  // CHECK-NEXT: %[[SC_2:.*]] = sdy.sharding_constraint %[[ADD]] <@mesh, [{"a", ?}, {"b"}]> {
  // CHECK-SAME:   sdy.sharding_origin_name = "constraint_2",
  // CHECK-SAME:   sdy.sharding_origins = {a = "constraint_1", b = "self"}}
  // CHECK-NEXT: return %[[SC_2]]
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{?}, {"b"}]> : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// CHECK-LABEL: manual_computation_no_manual_axes
// CHECK-SAME:    %arg0: tensor<32x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}, {?}]>,
// CHECK-SAME:                                 sdy.sharding_origins = {a = "mc_0_input: 0", b = "mc_0_output: 0"}})
// CHECK-SAME:    -> (tensor<32x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}, {?}]>,
// CHECK-SAME:                              sdy.sharding_origins = {a = "mc_0_input: 0", b = "mc_0_output: 0"}}) {
func.func @manual_computation_no_manual_axes(%arg0: tensor<32x32x32xf32>) -> tensor<32x32x32xf32> {
  // CHECK-NEXT: %[[SUB:.*]] = stablehlo.subtract %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]>,
  // CHECK-SAME:                                                sdy.sharding_origins = [{a = "mc_0_input: 0", b = "mc_0_output: 0"}]}
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[SUB]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]
  // CHECK-SAME:   manual_axes={} (%arg1: tensor<32x32x32xf32>) {
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 {
  // CHECK-SAME:     sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]>,
  // CHECK-SAME:     sdy.sharding_origins = [{a = "mc_0_input: 0", b = "mc_0_output: 0"}]}
  // CHECK-NEXT:   sdy.return %[[ADD]]
  // CHECK-NEXT: } {
  // CHECK-SAME:   sdy.block_arg_sharding_origins = [{a = "self", b = "mc_0_output: 0"}],
  // CHECK-SAME:   sdy.result_sharding_origins = [{a = "mc_0_input: 0", b = "self"}],
  // CHECK-SAME:   sdy.sharding_origin_name = "mc_0"
  // CHECK-SAME: } : (tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  // CHECK-NEXT: %[[SUB_2:.*]] = stablehlo.subtract %[[MC]], %[[MC]] {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "mc_0_input: 0", b = "mc_0_output: 0"}]}
  // CHECK-NEXT: return %[[SUB_2]]
  %0 = stablehlo.subtract %arg0, %arg0 : tensor<32x32x32xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"a", ?}, {?}, {?}]>] out_shardings=[<@mesh, [{?}, {"b", ?}, {?}]>] manual_axes={} (%arg1: tensor<32x32x32xf32>) {
    %3 = stablehlo.add %arg1, %arg1 : tensor<32x32x32xf32>
    sdy.return %3 : tensor<32x32x32xf32>
  } : (tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  %2 = stablehlo.subtract %1, %1 : tensor<32x32x32xf32>
  return %2: tensor<32x32x32xf32>
}

// CHECK-LABEL: manual_computation_manual_axes
// CHECK-SAME:    %arg0: tensor<32x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}, {?}]>,
// CHECK-SAME:                                 sdy.sharding_origins = {a = "mc_1_input: 0", b = "mc_1_input: 0"}}
// CHECK-SAME:    -> (tensor<32x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}, {?}]>,
// CHECK-SAME:                              sdy.sharding_origins = {a = "mc_1_output: 0", b = "mc_1_input: 0"}}) {
func.func @manual_computation_manual_axes(%arg0: tensor<32x32x32xf32>) -> tensor<32x32x32xf32> {
  // CHECK-NEXT: %[[SUB:.*]] = stablehlo.subtract %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]>,
  // CHECK-SAME:                                                sdy.sharding_origins = [{a = "mc_1_input: 0", b = "mc_1_input: 0"}]}
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[SUB]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]
  // CHECK-SAME:   manual_axes={"a"} (%arg1: tensor<16x32x32xf32>) {
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 {
  // CHECK-SAME:     sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"b", ?}, {?}]>]>,
  // CHECK-SAME:     sdy.sharding_origins = [{b = "mc_1_input: 0"}]}
  // CHECK-NEXT:   sdy.return %[[ADD]]
  // CHECK-NEXT: } {
  // CHECK-SAME:   sdy.block_arg_sharding_origins = [{a = "self", b = "self"}],
  // CHECK-SAME:   sdy.result_sharding_origins = [{a = "self", b = "mc_1_input: 0"}],
  // CHECK-SAME:   sdy.sharding_origin_name = "mc_1"
  // CHECK-SAME: } : (tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  // CHECK-NEXT: %[[SUB_2:.*]] = stablehlo.subtract %[[MC]], %[[MC]] {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "mc_1_output: 0", b = "mc_1_input: 0"}]}
  // CHECK-NEXT: return %[[SUB_2]]
  %0 = stablehlo.subtract %arg0, %arg0 : tensor<32x32x32xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"a", ?}, {"b", ?}, {?}]>] out_shardings=[<@mesh, [{"a", ?}, {?}, {?}]>] manual_axes={"a"} (%arg1: tensor<16x32x32xf32>) {
    %3 = stablehlo.add %arg1, %arg1 : tensor<16x32x32xf32>
    sdy.return %3 : tensor<16x32x32xf32>
  } : (tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  %2 = stablehlo.subtract %1, %1 : tensor<32x32x32xf32>
  return %2: tensor<32x32x32xf32>
}

// CHECK-LABEL: manual_computation_multiple_results
// CHECK-SAME:    %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {"a", ?}]>,
// CHECK-SAME:                              sdy.sharding_origins = {a = "mc_2_output: 1", b = "mc_2_input: 0"}})
// CHECK-SAME:    -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>,
// CHECK-SAME:                           sdy.sharding_origins = {a = "mc_2_output: 1"}},
// CHECK-SAME:        tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {"a", ?}]>,
// CHECK-SAME:                           sdy.sharding_origins = {a = "mc_2_output: 1", b = "mc_2_output: 1"}}) {
func.func @manual_computation_multiple_results(%arg0: tensor<32x32xf32>) -> (tensor<16x32xf32>, tensor<32x32xf32>) {
  // CHECK-NEXT: %[[MC:.*]]:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {"a", ?}]>] out_shardings=[<@mesh, [{?}, {"a", ?}], replicated={"b"}>, <@mesh, [{"b", ?}, {"a", ?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 {
  // CHECK-SAME:     sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>,
  // CHECK-SAME:     sdy.sharding_origins = [{a = "mc_2_output: 1"}]} : tensor<16x32xf32>
  // CHECK-NEXT:   sdy.return %[[ADD]], %[[ADD]] : tensor<16x32xf32>, tensor<16x32xf32>
  // CHECK-NEXT: } {
  // CHECK-SAME:   sdy.block_arg_sharding_origins = [{a = "mc_2_output: 1", b = "self"}],
  // CHECK-SAME:   sdy.result_sharding_origins = [{a = "mc_2_output: 1"}, {a = "self", b = "self"}],
  // CHECK-SAME:   sdy.sharding_origin_name = "mc_2"
  // CHECK-SAME: } : (tensor<32x32xf32>) -> (tensor<16x32xf32>, tensor<32x32xf32>)
  // CHECK-NEXT: return %[[MC]]#0, %[[MC]]#1 : tensor<16x32xf32>, tensor<32x32xf32>
  %0:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}], replicated={"b"}>, <@mesh, [{"b", ?}, {"a", ?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1, %1 : tensor<16x32xf32>, tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> (tensor<16x32xf32>, tensor<32x32xf32>)
  return %0#0, %0#1 : tensor<16x32xf32>, tensor<32x32xf32>
}


// Show that when an axis came from multiple sources, we just use the first we
// see left to right, from operands to results.
// CHECK-LABEL: tie_across_operands_results
// CHECK-SAME:    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>,
// CHECK-SAME:                          sdy.sharding_origins = {a = "self"}},
// CHECK-SAME:    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>,
// CHECK-SAME:                          sdy.sharding_origins = {a = "self"}}
// CHECK-SAME:    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>,
// CHECK-SAME:                       sdy.sharding_origins = {a = "constraint_3"}}) {
func.func @tie_across_operands_results(
  %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
  %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}
  ) -> (tensor<8xf32>) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "input: 0"}]} : tensor<8xf32>
  // CHECK-NEXT: %[[SC:.*]] = sdy.sharding_constraint %[[ADD]] <@mesh, [{"a"}]> {
  // CHECK-SAME:   sdy.sharding_origin_name = "constraint_3",
  // CHECK-SAME:   sdy.sharding_origins = {a = "self"}}
  // CHECK-NEXT: return %[[SC]]
  %0 = stablehlo.add %arg0, %arg1 : tensor<8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}]> : tensor<8xf32>
  return %1 :tensor<8xf32>
}

// CHECK-LABEL: push_sharding_constraints_to_func_results
// CHECK-SAME:   %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>,
// CHECK-SAME:                         sdy.sharding_origins = {a = "constraint_4"}},
// CHECK-SAME:   %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>,
// CHECK-SAME:                         sdy.sharding_origins = {a = "constraint_5"}})
// CHECK-SAME:   -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>,
// CHECK-SAME:                      sdy.sharding_origins = {a = "constraint_4"}},
// CHECK-SAME:       tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>,
// CHECK-SAME:                      sdy.sharding_origins = {a = "constraint_5"}}) {
func.func @push_sharding_constraints_to_func_results(
  %arg0: tensor<8xf32>, %arg1: tensor<8xf32>
  ) -> (tensor<8xf32>, tensor<8xf32>) {
  // CHECK-NEXT: %[[SC_1:.*]] = sdy.sharding_constraint %arg0 <@mesh, [{"a"}]> {
  // CHECK-SAME:   sdy.sharding_origin_name = "constraint_4",
  // CHECK-SAME:   sdy.sharding_origins = {a = "self"}}
  // CHECK-NEXT: %[[SC_2:.*]] = sdy.sharding_constraint %arg1 <@mesh, [{"a"}]> {
  // CHECK-SAME:   sdy.sharding_origin_name = "constraint_5",
  // CHECK-SAME:   sdy.sharding_origins = {a = "self"}}
  // CHECK-NEXT: return %[[SC_1]], %[[SC_2]] : tensor<8xf32>, tensor<8xf32>
  %1 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}]> : tensor<8xf32>
  %2 = sdy.sharding_constraint %arg1 <@mesh, [{"a"}]> : tensor<8xf32>
  return %1, %2 : tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: real_conflict_across_factors_diff_tensors_size
func.func @real_conflict_across_factors_diff_tensors_size(
    %arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}, {?}]>},
    %arg1: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", "c"}]>})
    -> tensor<8x16xf32> {
  // CHECK-NEXT: stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", "c", ?}]>]>,
  // CHECK-SAME:   sdy.sharding_origins = [{a = "input: 1", c = "input: 1"}]} : (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] :
    (tensor<8x4xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: while_loop_with_multiple_results
// CHECK-SAME:      %arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>,
// CHECK-SAME:                              sdy.sharding_origins = {a = "self"}},
// CHECK-SAME:      %arg1: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>,
// CHECK-SAME:                              sdy.sharding_origins = {b = "output: 1"}})
// CHECK-SAME:      -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>,
// CHECK-SAME:                           sdy.sharding_origins = {a = "input: 0"}},
// CHECK-SAME           tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>,
// CHECK-SAME:                           sdy.sharding_origins = {b = "self"}}) {
func.func @while_loop_with_multiple_results(
    %arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>},
    %arg1: tensor<8x4xf32>)
    -> (tensor<8x4xf32>, tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}]>}) {
  // CHECK-NEXT: %[[C:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[C_0:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT: %[[C_1:.*]] = stablehlo.constant dense<32> : tensor<i32>
  // CHECK-NEXT: %[[WHILE:.*]]:3 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %arg1, %iterArg_3 = %c)
  // CHECK-SAME:     : tensor<8x4xf32>, tensor<8x4xf32>, tensor<i32> attributes {
  // CHECK-SAME:     sdy.result_sharding_origins = [{a = "input: 0"}, {b = "output: 1"}, {}],
  // CHECK-SAME:     sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>, <@mesh, [{?}, {"b", ?}]>, <@mesh, []>]>}
  // CHECK-NEXT: cond {
  // CHECK-NEXT:   %1 = stablehlo.compare  LT, %iterArg_3, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT:   stablehlo.return %1 : tensor<i1>
  // CHECK-NEXT: } do {
  // CHECK-NEXT:   %1 = stablehlo.add %iterArg_3, %c_0 : tensor<i32>
  // CHECK-NEXT:   %2 = stablehlo.add %iterArg, %iterArg {
  // CHECK-SAME:       sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>,
  // CHECK-SAME:       sdy.sharding_origins = [{a = "input: 0"}]} : tensor<8x4xf32>
  // CHECK-NEXT:   %3 = stablehlo.add %iterArg_2, %iterArg_2 {
  // CHECK-SAME:       sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"b", ?}]>]>,
  // CHECK-SAME:       sdy.sharding_origins = [{b = "output: 1"}]} : tensor<8x4xf32>
  // CHECK-NEXT:   stablehlo.return %2, %3, %1 : tensor<8x4xf32>, tensor<8x4xf32>, tensor<i32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[WHILE]]#0, %[[WHILE]]#1 : tensor<8x4xf32>, tensor<8x4xf32>
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<32> : tensor<i32>
  %0:3 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %arg1, %iterArg_3 = %c) : tensor<8x4xf32>, tensor<8x4xf32>, tensor<i32>
    cond {
    %1 = stablehlo.compare  LT, %iterArg_3, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %iterArg_3, %c_0 : tensor<i32>
    %2 = stablehlo.add %iterArg, %iterArg : tensor<8x4xf32>
    %3 = stablehlo.add %iterArg_2, %iterArg_2 : tensor<8x4xf32>
    stablehlo.return %2, %3, %1 : tensor<8x4xf32>, tensor<8x4xf32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<8x4xf32>, tensor<8x4xf32>
}

// TODO(b/379631271): Add test doing a 8->4x4 reshape with axis size 8
// (introduces a new sub axis).
