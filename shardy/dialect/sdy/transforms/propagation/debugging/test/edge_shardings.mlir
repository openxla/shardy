// RUN: sdy_opt %s -split-input-file -sdy-add-data-flow-edges -sdy-apply-sharding-constraints -sdy-aggressive-propagate=debug-propagation-edge-sharding=true -sdy-sink-data-flow-edges="sink-debug-propagation-edge-sharding=true" 2>&1 | FileCheck %s


sdy.mesh @mesh = <["a"=2, "b"=2, "c"=8]>

// CHECK-LABEL: input_output_source_sharding
// CHECK-SAME:    %arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}, {"c", ?}]>},
// CHECK-SAME:    %arg1: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}, {"c", ?}]>}
// CHECK-SAME:  ) -> (tensor<8x8x8xf32> {sdy.propagation_edges = {a = [{propagation_step = 2 : i64, source = "operand: 0", target = "result: 0"}],
// CHECK-SAME:                                                    b = [{propagation_step = 0 : i64, source = "result: 0", target = "operand: 0"}],
// CHECK-SAME:                                                    c = [{propagation_step = 2 : i64, source = "operand: 0", target = "result: 0"}]},
// CHECK-SAME:                           sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}, {"c", ?}]>}) {
func.func @input_output_source_sharding(
  %arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}, {?}]>},
  %arg1: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}, {"c", ?}]>}
) -> (tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", ?}, {?}]>}) {
  // CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {
  // CHECK-SAME:    sdy.propagation_edges = {a = [{propagation_step = 1 : i64, source = "operand: 0", target = "operand: 1"},
  // CHECK-SAME:                                  {propagation_step = 1 : i64, source = "operand: 0", target = "result: 0"}],
  // CHECK-SAME:                             b = [{propagation_step = 1 : i64, source = "result: 0", target = "operand: 0"},
  // CHECK-SAME:                                  {propagation_step = 1 : i64, source = "result: 0", target = "operand: 1"}],
  // CHECK-SAME:                             c = [{propagation_step = 1 : i64, source = "operand: 1", target = "operand: 0"},
  // CHECK-SAME:                                  {propagation_step = 1 : i64, source = "operand: 1", target = "result: 0"}]},
  // CHECK-SAME:    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}, {"c", ?}]>]>
  // CHECK-SAME:  } : tensor<8x8x8xf32>
  // CHECK-NEXT:  return %[[ADD]] : tensor<8x8x8xf32>
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8x8xf32>
  return %0 : tensor<8x8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// NOTE: Instead of saving `{source = "result: 0", target = "operand: 0"}` and
// `{source = "result: 0", target = "operand: 1"}` on the add due to the same
// value being used twice as an operand, we only save the edge once.
//
// CHECK-LABEL: duplicate_operands
// CHECK-SAME:    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>}
// CHECK-SAME:  ) -> (tensor<8xf32> {sdy.propagation_edges = {a = [{propagation_step = 0 : i64, source = "result: 0", target = "operand: 0"}]},
// CHECK-SAME:                       sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>}) {
func.func @duplicate_operands(
  %arg0: tensor<8xf32>
) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}]>}) {
  // CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {
  // CHECK-SAME:    sdy.propagation_edges = {a = [{propagation_step = 1 : i64, source = "result: 0", target = "operand: 0"}]},
  // CHECK-SAME:    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}]>]>
  // CHECK-SAME:  } : tensor<8xf32>
  // CHECK-NEXT:  return %[[ADD]] : tensor<8xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// NOTE: since the definition of an edge always contains an operand as a source
// or target, even though the result sharding added `a` before the arg sharding,
// then the edge where axis `a` was added to the add is stored on the func
// result sharding.
//
// CHECK-LABEL: multiple_axes
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}
// CHECK-SAME:  ) -> (tensor<8x8xf32> {sdy.propagation_edges = {a = [{propagation_step = 0 : i64, source = "result: 0", target = "operand: 0"}],
// CHECK-SAME:                                                  b = [{propagation_step = 2 : i64, source = "operand: 0", target = "result: 0"}]},
// CHECK-SAME:                         sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}) {
func.func @multiple_axes(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {?}]>}
) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {
  // CHECK-SAME:   sdy.propagation_edges = {b = [{propagation_step = 1 : i64, source = "operand: 0", target = "result: 0"}]},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b", ?}, {?}]>]>
  // CHECK-SAME: } : tensor<8x8xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["c"=8]>

// NOTE(b/385908435): note how we save the smaller and larger sub axes on the
// func result sharding. Maybe this behavior is good, or should change? To be
// seen.
//
// CHECK-LABEL: sub_axis_update
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)4, ?}]>}
// CHECK-SAME:  ) -> (tensor<8x8xf32> {sdy.propagation_edges = {"c:(1)2" = [{propagation_step = 0 : i64, source = "result: 0", target = "operand: 0"}],
// CHECK-SAME:                                                  "c:(1)4" = [{propagation_step = 2 : i64, source = "operand: 0", target = "result: 0"}]},
// CHECK-SAME:                         sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)4, ?}]>}) {
func.func @sub_axis_update(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)4, ?}]>}
) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)2, ?}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {
  // CHECK-SAME:   sdy.propagation_edges = {"c:(1)4" = [{propagation_step = 1 : i64, source = "operand: 0", target = "result: 0"}]},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>
  // CHECK-SAME: } : tensor<8x8xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<8x8xf32>
  %1 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: manual_computation_manual_axes
// CHECK-SAME:    %arg0: tensor<32x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}, {?}]>}
// CHECK-SAME:    -> (tensor<32x32x32xf32> {sdy.propagation_edges = {a = [{propagation_step = 5 : i64, source = "operand: 0", target = "result: 0"}],
// CHECK-SAME:                                                       b = [{propagation_step = 5 : i64, source = "operand: 0", target = "result: 0"}]},
// CHECK-SAME:                              sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}, {?}]>}) {
func.func @manual_computation_manual_axes(%arg0: tensor<32x32x32xf32>) -> tensor<32x32x32xf32> {
  // CHECK-NEXT: %[[SUB:.*]] = stablehlo.subtract %arg0, %arg0 {
  // CHECK-SAME:   sdy.propagation_edges = {a = [{propagation_step = 1 : i64, source = "result: 0", target = "operand: 0"}],
  // CHECK-SAME:                            b = [{propagation_step = 1 : i64, source = "result: 0", target = "operand: 0"}]},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]>
  // CHECK-SAME: } : tensor<32x32x32xf32>
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%[[SUB]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]
  // CHECK-SAME:   manual_axes={"a"} (%arg1: tensor<16x32x32xf32>) {
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 {
  // CHECK-SAME:     sdy.propagation_edges = {b = [{propagation_step = 2 : i64, source = "operand: 0", target = "result: 0"}]},
  // CHECK-SAME:     sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"b", ?}, {?}]>]>}
  // CHECK-NEXT:   sdy.return %[[ADD]]
  // CHECK-NEXT: } {
  // CHECK-SAME:   sdy.block_arg_propagation_edges = [{
  // CHECK-SAME:     a = [{propagation_step = 0 : i64, source = "result: 0", target = "operand: 0"}],
  // CHECK-SAME:     b = [{propagation_step = 0 : i64, source = "result: 0", target = "operand: 0"}]}],
  // CHECK-SAME:   sdy.result_propagation_edges = [{
  // CHECK-SAME:     b = [{propagation_step = 3 : i64, source = "operand: 0", target = "result: 0"}]}]
  // CHECK-SAME: } : (tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  // CHECK-NEXT: %[[SUB_2:.*]] = stablehlo.subtract %[[MC]], %[[MC]] {
  // CHECK-SAME:   sdy.propagation_edges = {a = [{propagation_step = 4 : i64, source = "operand: 0", target = "result: 0"}],
  // CHECK-SAME:                            b = [{propagation_step = 4 : i64, source = "operand: 0", target = "result: 0"}]},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}, {?}]>]>
  // CHECK-SAME: } : tensor<32x32x32xf32>
  // CHECK-NEXT: return %[[SUB_2]]
  %0 = stablehlo.subtract %arg0, %arg0 : tensor<32x32x32xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"a", ?}, {"b", ?}, {?}]>] out_shardings=[<@mesh, [{"a", ?}, {?}, {?}]>] manual_axes={"a"} (%arg1: tensor<16x32x32xf32>) {
    %3 = stablehlo.add %arg1, %arg1 : tensor<16x32x32xf32>
    sdy.return %3 : tensor<16x32x32xf32>
  } : (tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  %2 = stablehlo.subtract %1, %1 : tensor<32x32x32xf32>
  return %2: tensor<32x32x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// TODO(b/391840483): If the function has multiple results, then each
// `edge_source` on the func result attributes will have index 0. This is
// because of how propagation works with running propagation on each returned
// result. Reconsider this behavior.
//
// CHECK-LABEL: manual_computation_multiple_results
// CHECK-SAME:    %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {"a", ?}]>})
// CHECK-SAME:    -> (tensor<16x32xf32> {sdy.propagation_edges = {a = [{propagation_step = 0 : i64, source = "operand: 0", target = "result: 0"},
// CHECK-SAME:                                                         {propagation_step = 6 : i64, source = "operand: 0", target = "result: 0"}],
// CHECK-SAME:                                                    b = [{propagation_step = 0 : i64, source = "operand: 0", target = "result: 0"}]},
// CHECK-SAME:                           sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>},
// CHECK-SAME:        tensor<32x32xf32> {sdy.propagation_edges = {},
// CHECK-SAME:                           sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {"a", ?}]>}) {
func.func @manual_computation_multiple_results(%arg0: tensor<32x32xf32>) -> (tensor<16x32xf32>, tensor<32x32xf32>) {
  // CHECK-NEXT: %[[MC:.*]]:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {"a", ?}]>] out_shardings=[<@mesh, [{?}, {"a", ?}], replicated={"b"}>, <@mesh, [{"b", ?}, {"a", ?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 {
  // CHECK-SAME:     sdy.propagation_edges = {a = [{propagation_step = 4 : i64, source = "result: 0", target = "operand: 0"}]},
  // CHECK-SAME:     sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>
  // CHECK-SAME:   } : tensor<16x32xf32>
  // CHECK-NEXT:   sdy.return %[[ADD]], %[[ADD]] : tensor<16x32xf32>, tensor<16x32xf32>
  // CHECK-NEXT: } {
  // CHECK-SAME:   sdy.block_arg_propagation_edges = [{
  // CHECK-SAME:     a = [{propagation_step = 5 : i64, source = "result: 0", target = "operand: 0"}],
  // CHECK-SAME:     b = [{propagation_step = 1 : i64, source = "result: 0", target = "operand: 0"}]}],
  // CHECK-SAME:   sdy.result_propagation_edges = [
  // CHECK-SAME:     {a = [{propagation_step = 3 : i64, source = "operand: 0", target = "result: 0"}]},
  // CHECK-SAME:     {a = [{propagation_step = 2 : i64, source = "result: 0", target = "operand: 0"}]}]
  // CHECK-SAME: } : (tensor<32x32xf32>) -> (tensor<16x32xf32>, tensor<32x32xf32>)
  // CHECK-NEXT: return %[[MC]]#0, %[[MC]]#1 : tensor<16x32xf32>, tensor<32x32xf32>
  %0:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}], replicated={"b"}>, <@mesh, [{"b", ?}, {"a", ?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1, %1 : tensor<16x32xf32>, tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> (tensor<16x32xf32>, tensor<32x32xf32>)
  return %0#0, %0#1 : tensor<16x32xf32>, tensor<32x32xf32>
}

// -----

sdy.mesh @mesh = <["c"=8]>

// CHECK-LABEL: sub_axes_splitting_reshape
// CHECK-SAME:    %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c", ?}]>}
// CHECK-SAME:  ) -> (tensor<4x4xf32> {sdy.propagation_edges = {"c:(1)4" = [{propagation_step = 1 : i64, source = "operand: 0", target = "result: 0"}],
// CHECK-SAME:                                                  "c:(4)2" = [{propagation_step = 1 : i64, source = "operand: 0", target = "result: 0"}]},
// CHECK-SAME:                         sdy.sharding = #sdy.sharding<@mesh, [{"c":(1)4, ?}, {"c":(4)2, ?}]>}) {
func.func @sub_axes_splitting_reshape(
  %arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c", ?}]>}
) -> tensor<4x4xf32> {
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %arg0 {
  // CHECK-SAME:   sdy.propagation_edges = {"c:(1)4" = [{propagation_step = 0 : i64, source = "operand: 0", target = "result: 0"}],
  // CHECK-SAME:                         "c:(4)2" = [{propagation_step = 0 : i64, source = "operand: 0", target = "result: 0"}]},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c":(1)4, ?}, {"c":(4)2, ?}]>]>
  // CHECK-SAME: } : (tensor<16xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT: return %[[RESHAPE]]
  %0 = stablehlo.reshape %arg0 : (tensor<16xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

sdy.mesh @mesh = <["c"=8]>

// NOTE: since the reshape combines the two sub axes into one, we only save the
// merged axis on the reshape as an edge between the operand and result.
//
// CHECK-LABEL: sub_axes_merging_reshape
// CHECK-SAME:    %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c":(1)4, ?}, {"c":(4)2, ?}]>}
// CHECK-SAME:  ) -> (tensor<16xf32> {sdy.propagation_edges = {c = [{propagation_step = 1 : i64, source = "operand: 0", target = "result: 0"}]},
// CHECK-SAME:                        sdy.sharding = #sdy.sharding<@mesh, [{"c", ?}]>}) {
func.func @sub_axes_merging_reshape(
  %arg0: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c":(1)4, ?}, {"c":(4)2, ?}]>})
  -> tensor<16xf32> {
  // CHECK-NEXT: stablehlo.reshape %arg0 {
  // CHECK-SAME:   sdy.propagation_edges = {c = [{propagation_step = 0 : i64, source = "operand: 0", target = "result: 0"}]},
  // CHECK-SAME:   sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c", ?}]>]>
  // CHECK-SAME: } : (tensor<4x4xf32>) -> tensor<16xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<4x4xf32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}
