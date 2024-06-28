// RUN: sdy_opt %s -split-input-file -verify-diagnostics

sdy.mesh @mesh = <"a"=2>

// Since DataFlowEdgeOp::verify has the same verification as any
// TensorShardingAttr, there is no need to check different types of failures.
func.func @invalid_sharding(%arg0 : tensor<8xf32>) -> tensor<8xf32> {
  // expected-error @+1 {{sharding doesn't match tensor rank: 2 != 1}}
  %0 = sdy.data_flow_edge %arg0 sharding=<@mesh, [{}, {"a"}]> : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @input_has_multiple_users(%arg0: tensor<32x96xf32>)
    -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  // expected-error @+1 {{expected input of sdy.data_flow_edge to have a single user}}
  %0 = sdy.data_flow_edge %arg0 : tensor<32x96xf32>
  return %arg0, %0 : tensor<32x96xf32>, tensor<32x96xf32>
}

// -----

sdy.mesh @mesh = <"a"=2>

func.func @input_defined_by_sdy_op(%arg0: tensor<32x96xf32>)
    -> tensor<32x96xf32> {
  // expected-note @+1  {{sdy op defining the input of the sdy.data_flow_edge}}
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {}]> : tensor<32x96xf32>
  // expected-error @+1 {{expected input of sdy.data_flow_edge to not be defined by an SdyDialect op}}
  %1 = sdy.data_flow_edge %0 : tensor<32x96xf32>
  return %1 : tensor<32x96xf32>
}
