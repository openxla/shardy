// RUN: sdy_opt %s -split-input-file -verify-diagnostics

// test: valid operand is block argument of a func op.
func.func @valid_block_arg(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// test: valid operand is call result.
func.func @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  return %arg0 : tensor<8xf32>
}

func.func @valid_call_result(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = func.call @bar(%arg0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %1 = sdy.func_data_flow_edge %0 : tensor<8xf32>
  return %1 : tensor<8xf32>
}

// -----

func.func @dynamic_shaped_type(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // expected-error @+1 {{expected sdy.func_data_flow_edge to have a static-shaped result}}
  %0 = sdy.func_data_flow_edge %arg0 : tensor<?xf32>
  return %0 : tensor<?xf32>
}

// -----

func.func @input_has_multiple_users(%arg0: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  // expected-error @+1 {{expected operand of sdy.func_data_flow_edge to have a single user}}
  %0 = sdy.func_data_flow_edge %arg0 : tensor<8xf32>
  return %arg0, %0 : tensor<8xf32>, tensor<8xf32>
}

// -----

func.func @operand_not_block_arg_of_func() -> tensor<8xf32> {
  %cst = arith.constant dense<0.0> : tensor<8xf32>
  // expected-error @+1 {{expected operand of sdy.func_data_flow_edge to be a result of call op.}}
  %0 = sdy.func_data_flow_edge %cst : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @operand_block_arg_not_of_func(%arg0: tensor<16x16xf32>, %cst: tensor<f32>) -> tensor<16xf32> {
  %0 = "stablehlo.reduce"(%arg0, %cst) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    // expected-error @+1 {{expected operand of sdy.func_data_flow_edge to be a block argument of a func op.}}
    %1 = sdy.func_data_flow_edge %arg1 : tensor<f32>
    %2 = stablehlo.add %1, %arg2 : tensor<f32>
    "stablehlo.return"(%2) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 0>} : (tensor<16x16xf32>, tensor<f32>) -> tensor<16xf32>
  return %0 : tensor<16xf32>
}
