// RUN: sdy_opt %s 2>&1 | FileCheck %s

// CHECK: sdy.mesh
sdy.mesh @meshA = <["a"=2, "b"=2]>
sdy.mesh @meshB = <["a"=4]>

// CHECK-LABEL: func @manual_computation_no_inputs_or_outputs
func.func @manual_computation_no_inputs_or_outputs() {
  // CHECK-NEXT: sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={} () {
  // CHECK-NEXT:   sdy.return
  // CHECK-NEXT: } : () -> ()
  sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={} () {
    sdy.return
  } : () -> ()
  func.return
}

// CHECK-LABEL: func @manual_computation_single_replicated_input_output
func.func @manual_computation_single_replicated_input_output(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK{LITERAL}: sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{?}, {?}]>] out_shardings=[<@meshA, [{?}, {?}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:       %[[BODY_RET:.*]] = stablehlo.add
  // CHECK-NEXT:       sdy.return %[[BODY_RET]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{?}, {?}]>] out_shardings=[<@meshA, [{?}, {?}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// CHECK-LABEL: func @manual_computation_single_sharded_input_output
func.func @manual_computation_single_sharded_input_output(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK{LITERAL}: sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", ?}, {?}]>] out_shardings=[<@meshA, [{"a", ?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
  // CHECK-NEXT:       %[[BODY_RET:.*]] = stablehlo.add
  // CHECK-NEXT:       sdy.return %[[BODY_RET]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", ?}, {?}]>] out_shardings=[<@meshA, [{"a", ?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// CHECK-LABEL: func @manual_computation_single_sharded_input
func.func @manual_computation_single_sharded_input(%arg0: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK{LITERAL}: sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", ?}, {?}]>] out_shardings=[<@meshA, [{?}, {?}], replicated={"a"}>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
  // CHECK-NEXT:       %[[BODY_RET:.*]] = stablehlo.add
  // CHECK-NEXT:       sdy.return %[[BODY_RET]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<8x32xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", ?}, {?}]>] out_shardings=[<@meshA, [{?}, {?}], replicated={"a"}>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<8x32xf32>
  func.return %0: tensor<8x32xf32>
}

// CHECK-LABEL: func @manual_computation_single_sharded_output
func.func @manual_computation_single_sharded_output(%arg0: tensor<16x32xf32>) -> tensor<32x32xf32> {
  // CHECK{LITERAL}: sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{?}, {?}], replicated={"a"}>] out_shardings=[<@meshA, [{"a", ?}, {?}]>] manual_axes={"a"} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:       %[[BODY_RET:.*]] = stablehlo.add
  // CHECK-NEXT:       sdy.return %[[BODY_RET]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<32x32xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{?}, {?}], replicated={"a"}>] out_shardings=[<@meshA, [{"a", ?}, {?}]>] manual_axes={"a"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<32x32xf32>
  func.return %0: tensor<32x32xf32>
}

// CHECK-LABEL: func @manual_computation_multiple_args_sharded
func.func @manual_computation_multiple_args_sharded(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK{LITERAL}: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@meshA, [{"a", ?}, {?}]>, <@meshA, [{"a", ?}, {?}]>] out_shardings=[<@meshA, [{"a", ?}, {?}]>] manual_axes={"a"}  (%arg2: tensor<8x32xf32>, %arg3: tensor<8x32xf32>) {
  // CHECK-NEXT:       %[[BODY_RET:.*]] = stablehlo.add
  // CHECK-NEXT:       sdy.return %[[BODY_RET]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@meshA, [{"a", ?}, {?}]>, <@meshA, [{"a", ?}, {?}]>] out_shardings=[<@meshA, [{"a", ?}, {?}]>] manual_axes={"a"}  (%arg2: tensor<8x32xf32>, %arg3: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// CHECK-LABEL: func @manual_computation_multiple_results_sharded
func.func @manual_computation_multiple_results_sharded(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK{LITERAL}: sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", ?}, {?}], replicated={"b"}>] out_shardings=[<@meshA, [{"a", ?}, {?}], replicated={"b"}>, <@meshA, [{"b", ?}, {?}], replicated={"a"}>] manual_axes={"a", "b"}  (%arg1: tensor<8x32xf32>) {
  // CHECK-NEXT:       %[[BODY_RET:.*]] = stablehlo.add
  // CHECK-NEXT:       sdy.return %[[BODY_RET]], %[[BODY_RET]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>)
  %0:2 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", ?}, {?}], replicated={"b"}>] out_shardings=[<@meshA, [{"a", ?}, {?}], replicated={"b"}>, <@meshA, [{"b", ?}, {?}], replicated={"a"}>] manual_axes={"a", "b"}  (%arg1: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x32xf32>
    sdy.return %1, %1 : tensor<8x32xf32>, tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>)
  %2 = stablehlo.add %0#0, %0#1 : tensor<16x32xf32>
  func.return %2: tensor<16x32xf32>
}

// CHECK-LABEL: func @manual_computation_scalar_output
func.func @manual_computation_scalar_output(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK{LITERAL}: sdy.manual_computation(%arg0) in_shardings=[<@meshA, []>] out_shardings=[<@meshA, []>] manual_axes={} (%arg1: tensor<f32>) {
  // CHECK-NEXT:      %[[BODY_RET:.*]] = stablehlo.add
  // CHECK-NEXT:      sdy.return %[[BODY_RET]]
  // CHECK-NEXT:    } : (tensor<f32>) -> tensor<f32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, []>] out_shardings=[<@meshA, []>] manual_axes={} (%arg1: tensor<f32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<f32>
    sdy.return %1 : tensor<f32>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0: tensor<f32>
}

// It's legal to have an in/out sharding with an axis that isn't manual, and
// isn't propagated through the body.
// CHECK-LABEL: func @manual_computation_non_propagated_axis
func.func @manual_computation_non_propagated_axis(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK{LITERAL}: sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", "b"}, {?}]>] out_shardings=[<@meshA, [{"a", ?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
  // CHECK-NEXT:       %[[BODY_RET:.*]] = stablehlo.add
  // CHECK-NEXT:       sdy.return %[[BODY_RET]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", "b"}, {?}]>] out_shardings=[<@meshA, [{"a", ?}, {?}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

func.func @man_comp_split_axes_sharding(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK{LITERAL}: sdy.manual_computation(%arg0) in_shardings=[<@meshB, [{"a":(1)2}, {}]>] out_shardings=[<@meshB, [{"a":(1)2}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
  // CHECK-NEXT:       %[[BODY_RET:.*]] = stablehlo.add
  // CHECK-NEXT:       sdy.return %[[BODY_RET]]
  // CHECK-NEXT:     } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshB, [{"a":(1)2}, {}]>] out_shardings=[<@meshB, [{"a":(1)2}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// CHECK-LABEL: func @manual_computation_nested
func.func @manual_computation_nested(%arg0: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // CHECK{LITERAL}:      sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", ?}, {?}]>] out_shardings=[<@meshA, [{?}, {?}], replicated={"a"}>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
  // CHECK-NEXT:            %[[INNER:.*]] = sdy.manual_computation(%arg1)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@meshA, [{"b", ?}, {?}]>] out_shardings=[<@meshA, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg2: tensor<4x32xf32>) {
  // CHECK-NEXT:              %[[NESTED_RET:.*]] = stablehlo.add
  // CHECK-NEXT:              sdy.return %[[NESTED_RET]]
  // CHECK-NEXT:            } : (tensor<8x32xf32>) -> tensor<8x32xf32>
  // CHECK-NEXT:            sdy.return %[[INNER]]
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{"a", ?}, {?}]>] out_shardings=[<@meshA, [{?}, {?}], replicated={"a"}>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@meshA, [{"b", ?}, {?}]>] out_shardings=[<@meshA, [{"b", ?}, {?}]>] manual_axes={"b"} (%arg2: tensor<4x32xf32>) {
      %2 = stablehlo.add %arg2, %arg2 : tensor<4x32xf32>
      sdy.return %2 : tensor<4x32xf32>
    } : (tensor<8x32xf32>) -> tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<8x32xf32>
  func.return %0: tensor<8x32xf32>
}
