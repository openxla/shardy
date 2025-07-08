// RUN: sdy_opt %s -split-input-file -verify-diagnostics

module {
  sdy.mesh @mesh = <["c"=8, "d"=8, "e"=8]>
  func.func @simple_edge_sharding(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{propagation edges have duplicate step index: 1}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-1 = [{"c":(1)4 = OPERAND-0 -> [RESULT-0]}, {"e" = OPERAND-0 -> [RESULT-0]}]},{step-1 = [{"d" = OPERAND-1 -> [RESULT-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["c"=8, "d"=8, "e"=8]>
  func.func @negative_step_index(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{propagation edges have negative step index: -3}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step--3 = [{"c":(1)4 = OPERAND-0 -> [RESULT-0]}, {"e" = OPERAND-0 -> [RESULT-0]}]},{step-1 = [{"d" = OPERAND-1 -> [RESULT-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["a"=8]>
  func.func @source_same_as_target(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{propagation edges have a source that is the same as a target}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-1 = [{"a" = OPERAND-0 -> [OPERAND-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a":(1)4, ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["x"=4]>
  func.func @duplicate_targets(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{propagation edges have duplicate targets for step index: 123 and axis: x}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-123 = [{"x" = RESULT-0 -> [OPERAND-0, OPERAND-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x":(1)4, ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["z"=4]>
  func.func @operand_index_out_of_range(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{'stablehlo.add' op expected a value ref to have an operand index in range [0, 2), got: 3}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-22 = [{"z" = RESULT-0 -> [OPERAND-3]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"z", ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["z"=4]>
  func.func @result_index_out_of_range(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{'stablehlo.add' op expected a value ref to have a result index in range [0, 1), got: 1}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-22 = [{"z" = OPERAND-1-> [OPERAND-0,RESULT-1]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"z":(1)4, ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["z"=4]>
  func.func @missing_sharding(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{expected sharding attrs for propagation edges attr}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-93 = [{"z" = OPERAND-1-> [OPERAND-0,RESULT-1]}]}]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["axis"=4]>
  func.func @axis_not_in_mesh(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{expected axis ref to be in one of the meshes}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-93 = [{"z" = OPERAND-1-> [OPERAND-0,RESULT-1]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"z":(1)4, ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["shardA"=4]>
  func.func @sdy_propagation_edges_not_correct_attr_type(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{should have a propagation edges attribute of type PropagationEdgesAttr for attr named 'sdy.propagation_edges'}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = 64} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["shardA"=4]>
  func.func @sdy_result_propagation_edges_not_correct_attr_type(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{should have a propagation edges attribute of type ArrayAttr<PropagationEdgesAttr> for attr named 'sdy.result_propagation_edges' or 'sdy.block_arg_propagation_edges'}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.result_propagation_edges = 64} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["shardA"=4]>
  func.func @sdy_result_propagation_edges_not_correct_attr_type_in_array(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{should have a propagation edges attribute of type ArrayAttr<PropagationEdgesAttr> for attr named 'sdy.result_propagation_edges' or 'sdy.block_arg_propagation_edges'}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.result_propagation_edges = [1,2,3]} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["shardA"=4]>
  func.func @sdy_block_arg_propagation_edges_not_correct_attr_type(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{should have a propagation edges attribute of type ArrayAttr<PropagationEdgesAttr> for attr named 'sdy.result_propagation_edges' or 'sdy.block_arg_propagation_edges'}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.block_arg_propagation_edges = 64} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}


// -----

module {
  sdy.mesh @mesh = <["shardA"=4]>
  func.func @sdy_block_arg_propagation_edges_not_correct_attr_type_in_array(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{should have a propagation edges attribute of type ArrayAttr<PropagationEdgesAttr> for attr named 'sdy.result_propagation_edges' or 'sdy.block_arg_propagation_edges'}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.block_arg_propagation_edges = [0,2,4]} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

