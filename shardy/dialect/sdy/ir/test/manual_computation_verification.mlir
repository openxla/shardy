// RUN: sdy_opt %s -split-input-file -verify-diagnostics

sdy.mesh @meshA = <["a"=2]>
sdy.mesh @meshB = <["a"=2]>

func.func @man_comp_different_meshes(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op result shardings can only refer to one mesh: @meshB vs @meshA}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{}, {}]>] out_shardings=[<@meshB, [{}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

func.func @unknown_mesh(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op operand - unknown mesh: @meshA}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@meshA, [{?}, {?}]>] out_shardings=[<@meshA, [{?}, {?}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

func.func @manual_computation_no_inputs_or_outputs_with_manual_axes() {
  // expected-error @+1 {{op cannot have manual_axes when there are no input/output shardings.}}
  sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={"a"} () {
    sdy.return
  } : () -> ()
  func.return
}

// -----

sdy.mesh @mesh = <["a"=4]>

func.func @man_comp_split_axes_sharding(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op operand sharding at index 0 cannot use a manual axis as a sub/split axis. Saw manual axes {a} and sharding #sdy.sharding<@mesh, [{"a":(1)2}, {}]>.}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a":(1)2}, {}]>] out_shardings=[<@mesh, [{"a":(1)2}, {}]>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=4]>

func.func @man_comp_split_axes_sharding_two_axes_sharding(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op operand sharding at index 0 cannot use a manual axis as a sub/split axis. Saw manual axes {a, b} and sharding #sdy.sharding<@mesh, [{"a", "b":(1)2}, {}]>.}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a", "b":(1)2}, {}]>] out_shardings=[<@mesh, [{"a", "b":(1)2}, {}]>] manual_axes={"a", "b"} (%arg1: tensor<4x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<4x32xf32>
    sdy.return %1 : tensor<4x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=4]>

func.func @man_comp_split_axes_sharding_two_axes_replicated(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op operand sharding at index 0 cannot use a manual axis as a sub/split axis. Saw manual axes {a, b} and sharding #sdy.sharding<@mesh, [{}, {}], replicated={"a", "b":(1)2}>.}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}], replicated={"a", "b":(1)2}>] out_shardings=[<@mesh, [{}, {}], replicated={"a", "b":(1)2}>] manual_axes={"a", "b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @man_comp_more_arg_specs(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op operand shardings don't match number of values: 2 shardings vs 1 values}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @man_comp_less_arg_specs(%arg0: tensor<16x32xf32>, %arg1: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op operand shardings don't match number of values: 1 shardings vs 2 values}}
  %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>] manual_axes={} (%arg2: tensor<16x32xf32>, %arg3: tensor<16x32xf32>){
    %1 = stablehlo.add %arg2, %arg3 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>, tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @man_comp_more_result_specs(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op result shardings don't match number of values: 2 shardings vs 1 values}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>, <@mesh, [{}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @man_comp_less_result_specs(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op result shardings don't match number of values: 1 shardings vs 2 values}}
  %0:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1, %1 : tensor<16x32xf32>, tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>)
  func.return %0#1: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @man_comp_operand_rank_mistmatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op operand - sharding doesn't match tensor rank: 3 != 2}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @man_comp_result_rank_mistmatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{sharding doesn't match tensor rank: 1 != 2}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @man_comp_operand_shape_mismatch_replicated(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op operand shape, corresponding sharding, and region operand shape at index 0 must match. Expected local shape 'tensor<16x32xf32>', actual local shape 'tensor<8x32xf32>'}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>] manual_axes={} (%arg1: tensor<8x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @man_comp_operand_shape_mismatch_sharded(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op operand shape, corresponding sharding, and region operand shape at index 0 must match. Expected local shape 'tensor<8x32xf32>', actual local shape 'tensor<16x32xf32>'}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"a"}, {}]>] out_shardings=[<@mesh, [{}, {}]>] manual_axes={"a"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @man_comp_result_shape_mismatch_sharded(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{op result shape, corresponding sharding, and region result shape at index 0 must match. Expected local shape 'tensor<8x32xf32>', actual local shape 'tensor<16x32xf32>'}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}], replicated={"a"}>] out_shardings=[<@mesh, [{"a"}, {}]>] manual_axes={"a"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @man_comp_results_number_mismatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{number of op results and region results must match. Op has 1 op results and 2 region results}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x32xf32>
    sdy.return %1, %1 : tensor<16x32xf32>, tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @man_comp_operands_number_mismatch(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error @+1 {{number of op operands and region operands must match. Op has 1 op operands and 2 region operands}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>, %arg2: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg2 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

func.func @man_comp_free_variables(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-note @+1 {{required by region isolation constraints}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}]>] manual_axes={} (%arg1: tensor<16x32xf32>) {
    // expected-error @+1 {{'stablehlo.add' op using value defined outside the region}}
    %1 = stablehlo.add %arg0, %arg1 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

sdy.mesh @foo = <["a"=2, "b"=2]>

func.func @manual_computation_nested_same_manual_axis(%arg0: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // expected-note @+1  {{parent bounding this axis as manual}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@foo, [{"a"}, {}]>] out_shardings=[<@foo, [{}, {}], replicated={"a"}>] manual_axes={"a"} (%arg1: tensor<8x32xf32>) {
    %1 = sdy.manual_computation(%arg1) in_shardings=[<@foo, [{"b"}, {}]>] out_shardings=[<@foo, [{"b"}, {}]>] manual_axes={"b"} (%arg2: tensor<4x32xf32>) {
      // expected-error @+1 {{op operates on axis "a" which is already bound by a parent sdy.manual_computation op}}
      %2 = sdy.manual_computation(%arg2) in_shardings=[<@foo, [{}, {}]>] out_shardings=[<@foo, [{}, {}]>] manual_axes={"a"} (%arg3: tensor<4x32xf32>) {
        %3 = stablehlo.add %arg3, %arg3 : tensor<4x32xf32>
        sdy.return %3 : tensor<4x32xf32>
      } : (tensor<4x32xf32>) -> tensor<4x32xf32>
      sdy.return %2 : tensor<4x32xf32>
    } : (tensor<8x32xf32>) -> tensor<8x32xf32>
    sdy.return %1 : tensor<8x32xf32>
  } : (tensor<16x32xf32>) -> tensor<8x32xf32>
  func.return %0: tensor<8x32xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @free_axes_before_manual_dim_sharding(%arg0: tensor<16x32xf32>) -> tensor<16x16xf32> {
  // expected-error @+1 {{op operand sharding at index 0 must have all manual axes come before free axes in its dimension sharding at index 1. Saw manual axis "b" after free axis "a"}}
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"a", "b"}]>] out_shardings=[<@mesh, [{}, {}], replicated={"b"}>] manual_axes={"b"} (%arg1: tensor<16x16xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<16x16xf32>
    sdy.return %1 : tensor<16x16xf32>
  } : (tensor<16x32xf32>) -> tensor<16x16xf32>
  func.return %0: tensor<16x16xf32>
}
