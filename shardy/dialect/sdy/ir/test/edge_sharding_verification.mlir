// RUN: sdy_opt %s -split-input-file -verify-diagnostics

module {
  sdy.mesh @mesh = <["c"=8, "d"=8, "e"=8]>
  func.func @simple_edge_sharding(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>}) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{propagation edges have duplicate step index: 1}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-1 = [{"c":(1)4 = operand-0 -> [result-0]}, {"e" = operand-0 -> [result-0]}]},{step-1 = [{"d" = operand-1 -> [result-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["c"=8, "d"=8, "e"=8]>
  func.func @negative_step_index(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{propagation edges have negative step index: -3}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step--3 = [{"c":(1)4 = operand-0 -> [result-0]}, {"e" = operand-0 -> [result-0]}]},{step-1 = [{"d" = operand-1 -> [result-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["a"=8]>
  func.func @source_same_as_target(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{propagation edges have a source that is the same as a target}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-1 = [{"a" = operand-0 -> [operand-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"a":(1)4, ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["x"=4]>
  func.func @duplicate_targets(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{propagation edges have duplicate targets for step index: 123 and axis: x}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-123 = [{"x" = result-0 -> [operand-0, operand-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x":(1)4, ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["z"=4]>
  func.func @operand_index_out_of_range(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{'stablehlo.add' op expected a value ref to have an operand index in range [0, 2), got: 3}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-22 = [{"z" = result-0 -> [operand-3]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"z", ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["z"=4]>
  func.func @result_index_out_of_range(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"z"}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"z"}]>}) {
    // expected-error @+1 {{'stablehlo.add' op expected a value ref to have a result index in range [0, 1), got: 1}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-22 = [{"z" = operand-1-> [operand-0,result-1]}]}]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["z"=4]>
  func.func @missing_sharding(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>) {
    // expected-error @+1 {{expected propagation edges attr to reference a sharding}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-93 = [{"z" = operand-1-> [operand-0,result-0]}]}]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

module {
  sdy.mesh @mesh = <["axis"=4]>
  func.func @axis_not_in_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"axis"}, {}]>}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"axis"}]>}) {
    // expected-error @+1 {{expected axis ref to be in one of the meshes}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-93 = [{"z" = operand-1-> [operand-0,result-1]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"z":(1)4, ?}]>]>} : tensor<8x8xf32>
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

// -----

module {
  sdy.mesh @mesh = <["dfe"=4]>
  func.func @sdy_propagation_edges_works_for_dfe(%arg0: tensor<32x96xf32>, %arg1: tensor<?x?xf32>) -> (tensor<32x96xf32>, tensor<?x?xf32>) {
    %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<32x96xf32>, tensor<?x?xf32>
    // expected-error @+1 {{should have a propagation edges attribute of type PropagationEdgesAttr for attr named 'sdy.propagation_edges'}}
    %1 = sdy.data_flow_edge %0#0  {sdy.propagation_edges = 64} : tensor<32x96xf32>
    return %1, %0#1 : tensor<32x96xf32>, tensor<?x?xf32>
  }
}
