// RUN: sdy_opt %s -split-input-file -verify-diagnostics

module {
  sdy.mesh @mesh = <["c"=8, "d"=8, "e"=8]>
  func.func @simple_edge_sharding(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)4, ?}]>}) -> (tensor<8x8xf32> {sdy.propagation_edges = #sdy.propagation_edges<[{step-0 = [{"c":(1)2 = RESULT-0 -> [OPERAND-0]}]}, {step-2 = [{"c":(1)4 = OPERAND-0 -> [RESULT-0]}]}]>, sdy.sharding = #sdy.sharding<@mesh, [{?}, {"c":(1)4, ?}]>}) {
    // expected-error @+1 {{propagation edges have duplicate step index: 1}}
    %0 = stablehlo.add %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-1 = [{"c":(1)4 = OPERAND-0 -> [RESULT-0]}, {"e" = OPERAND-0 -> [RESULT-0]}]},{step-1 = [{"d" = OPERAND-1 -> [RESULT-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"c":(1)4, ?}]>]>} : tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}
