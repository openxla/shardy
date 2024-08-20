// RUN: sdy_opt %s -sdy-basic-propagate 2>&1 | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2, "d"=2, "e"=2, "f"=2, "g"=2]>

// -----------------------------------------------------------------------------
// Basic tests without manual axes
// -----------------------------------------------------------------------------


// Tests that a sharding on the out_shardings can enter the body, and propagate
// to a use of the manual_computation
// CHECK-LABEL: func @manual_computation_output_sharding_annotation(
// CHECK-SAME:      %arg0: tensor<32x32xf32>)
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) {
func.func @manual_computation_output_sharding_annotation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{"a", ?}, {?}]>] manual_axes={} (%arg1: tensor<32x32xf32>) {
    %1 = stablehlo.custom_call @sdy_testonly(%arg1) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<32x32xf32>
    %2 = stablehlo.add %1, %1 : tensor<32x32xf32>
    sdy.return %2 : tensor<32x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// Tests that a sharding in the body can exit the body forward to out_shardings.
// CHECK-LABEL: func @manual_computation_output_inside_body_forward_propagation(
// CHECK-SAME:      %arg0: tensor<32x32xf32>)
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) {
func.func @manual_computation_output_inside_body_forward_propagation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={} (%arg1: tensor<32x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%arg1: tensor<32x32xf32>) {
    %1 = stablehlo.custom_call @sdy_testonly(%arg1) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    %2 = stablehlo.add %1, %1  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<32x32xf32>
    sdy.return %2 : tensor<32x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// CHECK-LABEL: func @manual_computation_directly_returned_body_arg(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) {
func.func @manual_computation_directly_returned_body_arg(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={} (%arg1: tensor<32x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a", ?}, {?}]>] out_shardings=[<@mesh, [{"a", ?}, {?}]>] manual_axes={} (%arg1: tensor<32x32xf32>) {
    sdy.return %arg1 : tensor<32x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// Same as `manual_computation_directly_returned_body_arg` except the
// input/output shardings conflict.
// CHECK-LABEL: func @manual_computation_directly_returned_body_arg_conflict(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {"c", ?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", ?}, {"d", ?}]>}) {
func.func @manual_computation_directly_returned_body_arg_conflict(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"a", "b", ?}, {"c", ?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"a", "b", ?}, {"d", ?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={} (%arg1: tensor<32x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a", ?}, {"c", ?}]>] out_shardings=[<@mesh, [{"a", "b", ?}, {"d", ?}]>] manual_axes={} (%arg1: tensor<32x32xf32>) {
    sdy.return %arg1 : tensor<32x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// CHECK-LABEL: func @manual_computation_sharding_inside_body_propagation(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) {
func.func @manual_computation_sharding_inside_body_propagation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={} (%arg1: tensor<32x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}]>] manual_axes={} (%arg1: tensor<32x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1  {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<32x32xf32>
    sdy.return %1 : tensor<32x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// -----------------------------------------------------------------------------
// Tests with manual axes
// -----------------------------------------------------------------------------

// The free axis lives inside the body. Make sure it propagates backwards out of
// the body.
// CHECK-LABEL: func @append_in_sharding_from_inside(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>}) {
func.func @append_in_sharding_from_inside(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", "a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    %2 = stablehlo.custom_call @sdy_testonly(%1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
    sdy.return %2 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// The free axis lives inside the body. Make sure it propagates forwards out of
// the body.
// CHECK-LABEL: func @append_out_sharding_from_inside(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>}) {
func.func @append_out_sharding_from_inside(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.custom_call @sdy_testonly(%arg1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
    %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    sdy.return %2 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// The free axis lives outside the body on an operand of the ManualComputation.
// Make sure it propagates forwards into the body.
// CHECK-LABEL: func @append_in_sharding_from_outside(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>}) {
func.func @append_in_sharding_from_outside(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  // CHECK:      sdy.manual_computation(%[[ADD]])
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", "a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    // CHECK: stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    %2 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    %3 = stablehlo.custom_call @sdy_testonly(%2) : (tensor<16x32xf32>) -> tensor<16x32xf32>
    sdy.return %3 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %1: tensor<32x32xf32>
}

// Make sure we can handle multiple values propagating between the
// operands and block arguments. Here we propagate to the `in_sharding` at index
// 1.
// CHECK-LABEL: func @append_in_sharding_from_outside_multiple_operands(
// CHECK-SAME:      %arg0: tensor<32x32xf32>,
// CHECK-SAME:      %arg1: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>}) {
func.func @append_in_sharding_from_outside_multiple_operands(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  %0 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  // CHECK:      sdy.manual_computation(%arg0, %[[ADD]])
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{?}, {?}], replicated={"b"}>, <@mesh, [{"b", "a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg2: tensor<32x32xf32>, %arg3: tensor<16x32xf32>) {
  %1 = sdy.manual_computation(%arg0, %0) in_shardings=[<@mesh, [{?}, {?}], replicated={"b"}>, <@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg2: tensor<32x32xf32>, %arg3: tensor<16x32xf32>) {
    // CHECK:      %[[INNER_ADD:.*]] = stablehlo.add %arg3, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    // CHECK-NEXT: stablehlo.custom_call @sdy_testonly(%arg2) : (tensor<32x32xf32>) -> ()
    // CHECK-NEXT: sdy.return %[[INNER_ADD]] : tensor<16x32xf32>
    %2 = stablehlo.add %arg3, %arg3 : tensor<16x32xf32>
    stablehlo.custom_call @sdy_testonly(%arg2) : (tensor<32x32xf32>) -> ()
    sdy.return %2 : tensor<16x32xf32>
  } : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %1: tensor<32x32xf32>
}

// The free axis lives outside the body on an op using the result of the
// ManualComputation. Make sure it propagates backwards into the body.
// CHECK-LABEL: func @append_out_sharding_from_outside(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>}) {
func.func @append_out_sharding_from_outside(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      %[[MC:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    // CHECK: %[[CUSTOM_CALL:.*]] = stablehlo.custom_call @sdy_testonly(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
    %1 = stablehlo.custom_call @sdy_testonly(%arg1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
    // CHECK: stablehlo.add %[[CUSTOM_CALL]], %[[CUSTOM_CALL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    %2 = stablehlo.add %1, %1 : tensor<16x32xf32>
    sdy.return %2 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[MC]], %[[MC]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  %3 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  func.return %3: tensor<32x32xf32>
}

// Make sure we can handle multiple values propagating between the
// the terminator of the body and `out_shardings`. Here we propagate to the
// `out_sharding` at index 1 from the body.
// CHECK-LABEL: func @append_out_sharding_from_inside_multiple_results(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32>,
// CHECK-SAME:          tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>}) {
func.func @append_out_sharding_from_inside_multiple_results(%arg0: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  // CHECK:      %[[MC:.*]]:2 = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{?}, {?}], replicated={"b"}>,
  // CHECK-SAME{LITERAL}:                  <@mesh, [{"b", "a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %0:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{?}, {?}], replicated={"b"}>, <@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    // CHECK:      %[[CUSTOM_CALL_ARG_1:.*]] = stablehlo.custom_call @sdy_testonly(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
    // CHECK-NEXT: %[[RET_1:.*]] = stablehlo.add %[[CUSTOM_CALL_ARG_1]], %[[CUSTOM_CALL_ARG_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    // CHECK:      %[[CUSTOM_CALL_NOT_PROPAGATED:.*]] = stablehlo.custom_call @sdy_testonly() : () -> tensor<32x32xf32>
    // CHECK-NEXT: %[[RET_0:.*]] = stablehlo.add %[[CUSTOM_CALL_NOT_PROPAGATED]], %[[CUSTOM_CALL_NOT_PROPAGATED]] : tensor<32x32xf32>
    // CHECK-NEXT: sdy.return %[[RET_0]], %[[RET_1]] : tensor<32x32xf32>, tensor<16x32xf32>
    %1 = stablehlo.custom_call @sdy_testonly(%arg1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
    %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    %3 = stablehlo.custom_call @sdy_testonly() : () -> tensor<32x32xf32>
    %4 = stablehlo.add %3, %3 : tensor<32x32xf32>
    sdy.return %4, %2 : tensor<32x32xf32>, tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>)
  // CHECK: return %[[MC]]#0, %[[MC]]#1 : tensor<32x32xf32>, tensor<32x32xf32>
  return %0#0, %0#1 : tensor<32x32xf32>, tensor<32x32xf32>
}


// Make sure we remove any existing free axes on in_shardings before updating
// it.
// CHECK-LABEL: func @remove_existing_free_axes_in_shardings(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", "c", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>}) {
func.func @remove_existing_free_axes_in_shardings(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", "a", "c", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", "a", ?}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "c", ?}, {?}]>]>} : tensor<16x32xf32>
    %2 = stablehlo.custom_call @sdy_testonly(%1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
    sdy.return %2 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// Make sure we remove any existing free axes on out_shardings before updating
// it.
// CHECK-LABEL: func @remove_existing_free_axes_out_shardings(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", "c", ?}, {?}]>}) {
func.func @remove_existing_free_axes_out_shardings(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", "c", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{"b", "a", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.custom_call @sdy_testonly(%arg1) : (tensor<16x32xf32>) -> tensor<16x32xf32>
    %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "c", ?}, {?}]>]>} : tensor<16x32xf32>
    sdy.return %2 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// CHECK-LABEL: func @multiple_manual_axes_same_dim(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", "c", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", "c", ?}, {?}]>}) {
func.func @multiple_manual_axes_same_dim(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"a", "b", "c", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", "c", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"a", "b"} (%arg1: tensor<8x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a", "b", ?}, {?}]>] out_shardings=[<@mesh, [{"b", "a", ?}, {?}]>] manual_axes={"a", "b"} (%arg1: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c", ?}, {?}]>]>} : tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// Now there are free and manual axes on multiple dimensions. An overall complex
// test that makes sure everything works well together.
// CHECK-LABEL: func @multiple_manual_axes_different_dim
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b", "e", ?}, {"d", "f", "g", ?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", "e", ?}, {"c", "f", "g", ?}]>}) {
func.func @multiple_manual_axes_different_dim(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"a", "b", "e", ?}, {"d", "f", "g", ?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", "e", ?}, {"c", "f", "g", ?}], replicated={"d"}>]
  // CHECK-SAME{LITERAL}:   manual_axes={"a", "b", "c", "d"} (%arg1: tensor<8x16xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a", "b", ?}, {"d", ?}], replicated={"c"}>] out_shardings=[<@mesh, [{"b", "a", ?}, {"c", ?}], replicated={"d"}>] manual_axes={"a", "b", "c", "d"} (%arg1: tensor<8x16xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"e", ?}, {"f", "g", ?}]>]>} : tensor<8x16xf32>
    sdy.return %1 : tensor<8x16xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// Check we remove/add axes correctly when there are manual axes but a free axis
// is in its own dimension.
// This specifically checks when propagating from the inside of the body out.
// CHECK-LABEL: func @free_axis_on_own_dim_from_inside(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {"a", ?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {"a", ?}]>}) {
func.func @free_axis_on_own_dim_from_inside(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", ?}, {"a", ?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", ?}, {"a", ?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a", ?}]>]>} : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// Check we remove/add axes correctly when there are manual axes but a free axis
// is in its own dimension.
// This specifically checks when propagating from the outside of the body in.
// CHECK-LABEL: func @free_axis_on_own_dim_from_outside(
// CHECK-SAME:      %arg0: tensor<32x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {"a", ?}, {"c", ?}]>})
// CHECK-SAME:      -> (tensor<32x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {"a", ?}, {"c", ?}]>}) {
func.func @free_axis_on_own_dim_from_outside(%arg0: tensor<32x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}, {?}]>}) -> (tensor<32x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}, {"c", ?}]>}) {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", ?}, {"a", ?}, {"c", ?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", ?}, {"a", ?}, {"c", ?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32x32xf32>
    sdy.return %1 : tensor<16x32x32xf32>
  } : (tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  func.return %0: tensor<32x32x32xf32>
}

// -----------------------------------------------------------------------------
// Tests with replicated axes (manual and free)
// -----------------------------------------------------------------------------

// Make sure any replicated axes that are manual are preserved, and not
// propagated to any `Value` inside of the `sdy.manual_computation` op.
// CHECK-LABEL: func @replicated_manual_axes(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>}) {
func.func @replicated_manual_axes(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<32x32xf32>
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[ADD]])
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", "a", ?}, {?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", ?}, {?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b", "c"} (%arg1: tensor<16x32xf32>) {
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"b", ?}, {?}], replicated={"c"}>] out_shardings=[<@mesh, [{"b", ?}, {?}], replicated={"c"}>] manual_axes={"b", "c"} (%arg1: tensor<16x32xf32>) {
    %2 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    sdy.return %2 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK:  stablehlo.add %[[MANUAL]], %[[MANUAL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  %3 = stablehlo.add %1, %1 : tensor<32x32xf32>
  func.return %3: tensor<32x32xf32>
}

// For results of a ManualComputation that are directly returned, the replicated
// axes are kept. Not for directly used func arguments.
// CHECK-LABEL: func @replicated_manual_axes_directly_used_returned_from_func(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>}) {
func.func @replicated_manual_axes_directly_used_returned_from_func(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", "a", ?}, {?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", ?}, {?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b", "c"} (%arg1: tensor<16x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}], replicated={"c"}>] out_shardings=[<@mesh, [{"b", ?}, {?}], replicated={"c"}>] manual_axes={"b", "c"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// Make sure any replicated axes that are free are preserved, and not
// propagated to an intermediate.
// CHECK-LABEL: func @replicated_free_axes(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>}) {
func.func @replicated_free_axes(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<32x32xf32>
  // CHECK:      %[[MANUAL:.*]] = sdy.manual_computation(%[[ADD]])
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", "a", ?}, {?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", ?}, {?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"b", ?}, {?}], replicated={"c"}>] out_shardings=[<@mesh, [{"b", ?}, {?}], replicated={"c"}>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %2 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    sdy.return %2 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK:  stablehlo.add %[[MANUAL]], %[[MANUAL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  %3 = stablehlo.add %1, %1 : tensor<32x32xf32>
  func.return %3: tensor<32x32xf32>
}

// For results of a ManualComputation that are directly returned, the replicated
// axes are kept. Not for directly used func arguments.
// CHECK-LABEL: func @replicated_free_axes_directly_used_returned_from_func(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>}) {
func.func @replicated_free_axes_directly_used_returned_from_func(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", "a", ?}, {?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", ?}, {?}], replicated={"c"}>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}], replicated={"c"}>] out_shardings=[<@mesh, [{"b", ?}, {?}], replicated={"c"}>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// Replicated axes inside the body shouldn't propagate to in/out shardings.
// CHECK-LABEL: func @replicated_inside_body(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>}) {
func.func @replicated_inside_body(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", "a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}], replicated={"c"}>]>} : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// -----------------------------------------------------------------------------
// Tests with closed dimensions (manual and free)
// -----------------------------------------------------------------------------

// CHECK-LABEL: func @preserve_untouched_closed_dim(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>}) {
func.func @preserve_untouched_closed_dim(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", "a", ?}, {}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", "a", ?}, {}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {}]>] out_shardings=[<@mesh, [{"b", ?}, {}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// CHECK-LABEL: func @dont_propagate_into_closed_dim_from_inside(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>}) {
func.func @dont_propagate_into_closed_dim_from_inside(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b"}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b"}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b"}, {?}]>] out_shardings=[<@mesh, [{"b"}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// CHECK-LABEL: func @dont_propagate_into_out_sharding_closed_dim_from_outside(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>}) {
func.func @dont_propagate_into_out_sharding_closed_dim_from_outside(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK:      sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b"}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"b", ?}, {?}]>] out_shardings=[<@mesh, [{"b"}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  %2 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  func.return %2: tensor<32x32xf32>
}

// CHECK-LABEL: func @dont_propagate_into_in_sharding_closed_dim_from_outside(
// CHECK-SAME:      %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "a", ?}, {?}]>})
// CHECK-SAME:      -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>}) {
func.func @dont_propagate_into_in_sharding_closed_dim_from_outside(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a", ?}, {?}]>]>} : tensor<32x32xf32>
  // CHECK:      sdy.manual_computation(%[[ADD]])
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"b"}, {?}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"b", ?}, {?}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"b"}, {?}]>] out_shardings=[<@mesh, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %2 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %2 : tensor<16x32xf32>
  } : (tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %1: tensor<32x32xf32>
}
