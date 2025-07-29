// RUN: sdy_opt %s -sdy-remove-propagation-debug-info | FileCheck %s

sdy.mesh @mesh_a = <["a"=2]>
sdy.mesh @mesh_c = <["c"=8]>
sdy.mesh @mesh_ab = <["a"=2, "b"=2]>
sdy.mesh @mesh_abc = <["a"=2, "b"=2, "c"=8]>

// CHECK-NOT: sdy.propagation_edges
// CHECK-NOT: sdy.block_arg_propagation_edges
// CHECK-NOT: sdy.result_propagation_edges
// CHECK-NOT: sdy.sharding_origins
// CHECK-NOT: sdy.block_arg_sharding_origins
// CHECK-NOT: sdy.result_sharding_origins

func.func @remove_propagation_edges(
    %arg0: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"a", ?}, {"b", ?}, {"c", ?}]>},
    %arg1: tensor<8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"a", ?}, {"b", ?}, {"c", ?}]>}) -> (
      tensor<8x8x8xf32> {sdy.propagation_edges = #sdy.propagation_edges<[{step-0 = [{"b" = result-0 -> [operand-0]}]}, {step-2 = [{"a" = operand-0 -> [result-0]}, {"c" = operand-0 -> [result-0]}]}]>, sdy.sharding = #sdy.sharding<@mesh_abc, [{"a", ?}, {"b", ?}, {"c", ?}]>}) {
  %0 = stablehlo.add %arg0, %arg1 {sdy.propagation_edges = #sdy.propagation_edges<[{step-1 = [{"a" = operand-0 -> [operand-1, result-0]}, {"b" = result-0 -> [operand-0, operand-1]}, {"c" = operand-1 -> [operand-0, result-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"a", ?}, {"b", ?}, {"c", ?}]>]>} : tensor<8x8x8xf32>
  return %0 : tensor<8x8x8xf32>
}

func.func @remove_propagation_edges_for_manual_computation(%arg0: tensor<32x32x32xf32> {sdy.sharding = #sdy.sharding<@mesh_ab, [{"a", ?}, {"b", ?}, {?}]>}) -> (tensor<32x32x32xf32> {sdy.propagation_edges = #sdy.propagation_edges<[{step-5 = [{"a" = operand-0 -> [result-0]}, {"b" = operand-0 -> [result-0]}]}]>, sdy.sharding = #sdy.sharding<@mesh_ab, [{"a", ?}, {"b", ?}, {?}]>}) {
  %0 = stablehlo.subtract %arg0, %arg0 {sdy.propagation_edges = #sdy.propagation_edges<[{step-1 = [{"a" = result-0 -> [operand-0, operand-1]}, {"b" = result-0 -> [operand-0, operand-1]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh_ab, [{"a", ?}, {"b", ?}, {?}]>]>} : tensor<32x32x32xf32>
  %1 = sdy.manual_computation(%0) in_shardings=[<@mesh_ab, [{"a", ?}, {"b", ?}, {?}]>] out_shardings=[<@mesh_ab, [{"a", ?}, {"b", ?}, {?}]>] manual_axes={"a"} (%arg1: tensor<16x32x32xf32>) {
    %3 = stablehlo.add %arg1, %arg1 {sdy.propagation_edges = #sdy.propagation_edges<[{step-2 = [{"b" = operand-0 -> [result-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh_ab, [{?}, {"b", ?}, {?}]>]>} : tensor<16x32x32xf32>
    sdy.return %3 : tensor<16x32x32xf32>
  } {sdy.block_arg_propagation_edges = [#sdy.propagation_edges<[{step-0 = [{"a" = result-0 -> [operand-0]}, {"b" = result-0 -> [operand-0]}]}]>], sdy.result_propagation_edges = [#sdy.propagation_edges<[{step-3 = [{"b" = operand-0 -> [result-0]}]}]>]} : (tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  %2 = stablehlo.subtract %1, %1 {sdy.propagation_edges = #sdy.propagation_edges<[{step-4 = [{"a" = operand-0 -> [result-0]}, {"b" = operand-0 -> [result-0]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh_ab, [{"a", ?}, {"b", ?}, {?}]>]>} : tensor<32x32x32xf32>
  return %2 : tensor<32x32x32xf32>
}

func.func @remove_propagation_edges_multiple_results(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh_ab, [{"b", ?}, {"a", ?}]>}) -> (tensor<16x32xf32> {sdy.propagation_edges = #sdy.propagation_edges<[{step-0 = [{"a" = operand-0 -> [result-0]}, {"b" = operand-0 -> [result-0]}]}, {step-6 = [{"a" = operand-0 -> [result-0]}]}]>, sdy.sharding = #sdy.sharding<@mesh_ab, [{?}, {"a", ?}]>}, tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh_ab, [{"b", ?}, {"a", ?}]>}) {
  %0:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_ab, [{"b", ?}, {"a", ?}]>] out_shardings=[<@mesh_ab, [{?}, {"a", ?}], replicated={"b"}>, <@mesh_ab, [{"b", ?}, {"a", ?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.propagation_edges = #sdy.propagation_edges<[{step-4 = [{"a" = result-0 -> [operand-0, operand-1]}]}]>, sdy.sharding = #sdy.sharding_per_value<[<@mesh_ab, [{?}, {"a", ?}]>]>} : tensor<16x32xf32>
    sdy.return %1, %1 : tensor<16x32xf32>, tensor<16x32xf32>
  } {sdy.block_arg_propagation_edges = [#sdy.propagation_edges<[{step-1 = [{"b" = result-0 -> [operand-0]}]}, {step-5 = [{"a" = result-0 -> [operand-0]}]}]>], sdy.result_propagation_edges = [#sdy.propagation_edges<[{step-3 = [{"a" = operand-0 -> [result-0]}]}]>, #sdy.propagation_edges<[{step-2 = [{"a" = result-0 -> [operand-0]}]}]>]} : (tensor<32x32xf32>) -> (tensor<16x32xf32>, tensor<32x32xf32>)
  return %0#0, %0#1 : tensor<16x32xf32>, tensor<32x32xf32>
}

func.func @remove_origin_shardings(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"a", ?}, {"c", ?}]>, sdy.sharding_origins = {a = "self", c = "input: 1"}}, %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"a", ?}, {"c", ?}]>, sdy.sharding_origins = {a = "input: 0", c = "self"}}, %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"c", ?}, {"b", ?}]>, sdy.sharding_origins = {b = "output: 0", c = "input: 1"}}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"a", ?}, {"b", ?}]>, sdy.sharding_origins = {a = "input: 0", b = "self"}}) {
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"a", ?}, {"c", ?}]>]>, sdy.sharding_origins = [{a = "input: 0", c = "input: 1"}]} : tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"a", ?}, {"b", ?}]>]>, sdy.sharding_origins = [{a = "input: 0", b = "output: 0"}]} : (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

func.func @remove_origin_shardings_manual_computation_multiple_results(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b", ?}, {"a", ?}]>, sdy.sharding_origins = {a = "mc_0_output: 1", b = "mc_0_input: 0"}}) -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{?}, {"a", ?}]>, sdy.sharding_origins = {a = "mc_0_output: 1"}}, tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b", ?}, {"a", ?}]>, sdy.sharding_origins = {a = "mc_0_output: 1", b = "mc_0_output: 1"}}) {
  %0:2 = sdy.manual_computation(%arg0) in_shardings=[<@mesh_abc, [{"b", ?}, {"a", ?}]>] out_shardings=[<@mesh_abc, [{?}, {"a", ?}], replicated={"b"}>, <@mesh_abc, [{"b", ?}, {"a", ?}]>] manual_axes={"b"} (%arg1: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{?}, {"a", ?}]>]>, sdy.sharding_origins = [{a = "mc_0_output: 1"}]} : tensor<16x32xf32>
    sdy.return %1, %1 : tensor<16x32xf32>, tensor<16x32xf32>
  } {sdy.block_arg_sharding_origins = [{a = "mc_0_output: 1", b = "self"}], sdy.result_sharding_origins = [{a = "mc_0_output: 1"}, {a = "self", b = "self"}], sdy.sharding_origin_name = "mc_0"} : (tensor<32x32xf32>) -> (tensor<16x32xf32>, tensor<32x32xf32>)
  return %0#0, %0#1 : tensor<16x32xf32>, tensor<32x32xf32>
}

func.func @remove_origin_shardings_while_loop_with_multiple_results(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"a", ?}, {?}]>, sdy.sharding_origins = {a = "self"}}, %arg1: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{?}, {"b", ?}]>, sdy.sharding_origins = {b = "output: 1"}}) -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"a", ?}, {?}]>, sdy.sharding_origins = {a = "input: 0"}}, tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{?}, {"b", ?}]>, sdy.sharding_origins = {b = "self"}}) {
  %c = stablehlo.constant dense<0> : tensor<i32>
  %c_0 = stablehlo.constant dense<1> : tensor<i32>
  %c_1 = stablehlo.constant dense<32> : tensor<i32>
  %0:3 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %arg1, %iterArg_3 = %c) : tensor<8x4xf32>, tensor<8x4xf32>, tensor<i32> attributes {sdy.block_arg_sharding_origins = [], sdy.result_sharding_origins = [{a = "input: 0"}, {b = "output: 1"}], sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"a", ?}, {?}]>, <@mesh_abc, [{?}, {"b", ?}]>, <@mesh_abc, []>]>}
  cond {
    %1 = stablehlo.compare  LT, %iterArg_3, %c_1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %1 : tensor<i1>
  } do {
    %1 = stablehlo.add %iterArg_3, %c_0 : tensor<i32>
    %2 = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"a", ?}, {?}]>]>, sdy.sharding_origins = [{a = "input: 0"}]} : tensor<8x4xf32>
    %3 = stablehlo.add %iterArg_2, %iterArg_2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{?}, {"b", ?}]>]>, sdy.sharding_origins = [{b = "output: 1"}]} : tensor<8x4xf32>
    stablehlo.return %2, %3, %1 : tensor<8x4xf32>, tensor<8x4xf32>, tensor<i32>
  }
  return %0#0, %0#1 : tensor<8x4xf32>, tensor<8x4xf32>
}

func.func @remove_origin_shardings_manual_computation_with_sharding_constraints(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"a", "b", ?}, {?}]>, sdy.sharding_origins = {a = "mc_1_input: 0", b = "mc_1_input: 0"}}, %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"b", "a", ?}, {?}]>, sdy.sharding_origins = {a = "mc_1_input: 1", b = "mc_1_input: 1"}}) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"a", "b", ?}, {?}]>, sdy.sharding_origins = {a = "mc_1_input: 0", b = "mc_1_input: 0"}}, tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_abc, [{"a", "b", ?}, {?}]>, sdy.sharding_origins = {a = "mc_1_output: 0", b = "mc_1_output: 0"}}) {
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"a", "b", ?}, {?}]>]>, sdy.sharding_origins = [{a = "mc_1_input: 0", b = "mc_1_input: 0"}]} : tensor<8x8xf32>
  %1 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_abc, [{"b", "a"}, {}]>]>, sdy.sharding_origins = [{a = "mc_1_input: 1", b = "mc_1_input: 1"}]} : tensor<8x8xf32>
  %2 = sdy.manual_computation(%0, %1) in_shardings=[<@mesh_abc, [{"a", "b"}, {?}]>, <@mesh_abc, [{"b", "a"}, {}]>] out_shardings=[<@mesh_abc, [{"a", "b"}, {}]>] manual_axes={"a", "b"} (%arg2: tensor<2x8xf32>, %arg3: tensor<2x8xf32>) {
    %3 = stablehlo.add %arg2, %arg3 : tensor<2x8xf32>
    sdy.return %3 : tensor<2x8xf32>
  } {sdy.block_arg_sharding_origins = [{a = "self", b = "self"}, {a = "self", b = "self"}], sdy.result_sharding_origins = [{a = "self", b = "self"}], sdy.sharding_origin_name = "mc_1"} : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0, %2 : tensor<8x8xf32>, tensor<8x8xf32>
}
