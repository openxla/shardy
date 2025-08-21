// RUN: sdy_opt %s -split-input-file -verify-diagnostics

func.func @trailing_comma_custom_rule(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error@+1 {{expected 'custom'}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=32},>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

func.func @custom_rule_wrong_keyword(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // expected-error@+1 {{expected 'custom'}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=32}, custom_rule>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}

// -----

func.func @space_between_factors(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+2 {{failed to parse Sdy_OpShardingRule parameter 'result_mappings'}}
  // expected-error@+1 {{expected ']'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i j]) {i=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @factors_reversed(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{expecting factor indices to be ordered like an iota ([0,1,2,...], e.g. {i=#, j=#, ...}). Expecting factor index symbol 'i', received: 'j'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([ij]) {j=4, i=2}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @factor_skipped(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{expecting factor indices to be ordered like an iota ([0,1,2,...], e.g. {i=#, j=#, ...}). Expecting factor index symbol 'j', received: 'k'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, k])->([ik]) {i=2, k=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @factors_not_starting_at_0(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{expecting factor indices to be ordered like an iota ([0,1,2,...], e.g. {i=#, j=#, ...}). Expecting factor index symbol 'i', received: 'j'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([j, k])->([jk]) {j=2, k=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @factor_symbol_z_(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+3 {{failed to parse Sdy_TensorMapping parameter 'dim_mappings'}}
  // expected-error@+2 {{failed to parse Sdy_OpShardingRule parameter 'operand_mappings'}}
  // expected-error@+1 {{expecting integer after 'z_'. Received: 'z_'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([z_, j])->([i]) {z_=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @factor_symbol_i_1(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+3 {{failed to parse Sdy_TensorMapping parameter 'dim_mappings'}}
  // expected-error@+2 {{failed to parse Sdy_OpShardingRule parameter 'operand_mappings'}}
  // expected-error@+1 {{expecting symbol from 'i' to 'z'. Received: '_'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i_1, j])->([i]) {i_1=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @factor_symbol_z_z_(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+3 {{failed to parse Sdy_TensorMapping parameter 'dim_mappings'}}
  // expected-error@+2 {{failed to parse Sdy_OpShardingRule parameter 'operand_mappings'}}
  // expected-error@+1 {{expecting integer after 'z_'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([z_z_1, j])->([i]) {z_1=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @factor_symbol_z_0z_(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+3 {{failed to parse Sdy_TensorMapping parameter 'dim_mappings'}}
  // expected-error@+2 {{failed to parse Sdy_OpShardingRule parameter 'operand_mappings'}}
  // expected-error@+1 {{expecting positive integer without leading zeros. Received: '0'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([z_0z_, j])->([i]) {z_0=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @factor_symbol_out_of_range(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+3 {{failed to parse Sdy_TensorMapping parameter 'dim_mappings'}}
  // expected-error@+2 {{failed to parse Sdy_OpShardingRule parameter 'operand_mappings'}}
  // expected-error@+1 {{expecting symbol from 'i' to 'z'. Received: 'a'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([a, j])->([j]) {a=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @factor_symbol_negative(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+3 {{failed to parse Sdy_TensorMapping parameter 'dim_mappings'}}
  // expected-error@+2 {{failed to parse Sdy_OpShardingRule parameter 'operand_mappings'}}
  // expected-error@+1 {{expecting integer after 'z_'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([z_-1, j])->([i]) {i=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @factor_symbol_whitespace(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+3 {{failed to parse Sdy_TensorMapping parameter 'dim_mappings'}}
  // expected-error@+2 {{failed to parse Sdy_OpShardingRule parameter 'operand_mappings'}}
  // expected-error@+1 {{expecting integer after 'z_'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([z_ 1, j])->([i]) {i=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @z_0(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+3 {{failed to parse Sdy_TensorMapping parameter 'dim_mappings'}}
  // expected-error@+2 {{failed to parse Sdy_OpShardingRule parameter 'operand_mappings'}}
  // expected-error@+1 {{expecting positive integer}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([j, z_0])->([jz_0]) {i=4, z_0=2}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @int_overflow(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+2 {{expecting factor indices to be ordered like an iota ([0,1,2,...], e.g. {i=#, j=#, ...})}}
  // expected-error@+1 {{expecting symbol index <=2^63-1. Received: '9223372036854775808'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([j, z_9223372036854775808])->([jz_9223372036854775808]) {i=4, z_9223372036854775808=2}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// Only issue should be the ordering, but int64 max should be fine.
func.func @int_max(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{expecting factor indices to be ordered like an iota ([0,1,2,...], e.g. {i=#, j=#, ...})}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([j, z_9223372036854775807])->([jz_9223372036854775807]) {i=4, z_9223372036854775807=2}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// Only issue should be the ordering, but parsing z_2i should be fine.
func.func @num_then_ijk(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{expecting factor indices to be ordered like an iota ([0,1,2,...], e.g. {i=#, j=#, ...})}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([j, z_2i])->([jz_9223372036854775807]) {i=4, z_9223372036854775807=2}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @unkown_symbol(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // expected-error@+3 {{failed to parse Sdy_TensorMapping parameter 'dim_mappings'}}
  // expected-error@+2 {{failed to parse Sdy_OpShardingRule parameter 'operand_mappings'}}
  // expected-error@+1 {{expecting symbol from 'i' to 'z'. Received: '_'}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([j, z_2_i])->([j]) {i=4, z_9223372036854775807=2}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @no_operands(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // expected-error@+2 {{failed to parse Sdy_OpShardingRule parameter 'operand_mappings'}}
  // expected-error@+1 {{expected '['}}
  %0 = stablehlo.custom_call @foo() {sdy.sharding_rule = #sdy.op_sharding_rule<()->([i, j]) {i=2, j=8}, custom>} : () -> tensor<2x8xf32>
  func.return %0 : tensor<2x8xf32>
}

// -----

func.func @no_results(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // expected-error@+2 {{failed to parse Sdy_OpShardingRule parameter 'result_mappings'}}
  // expected-error@+1 {{expected '['}}
  stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->() {i=2, j=8}, custom>} : (tensor<2x8xf32>) -> ()
  func.return %arg0 : tensor<2x8xf32>
}

// -----

func.func @equality_sign_after_reduction(%arg0: tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32> {
  // expected-error@+1 {{expected '='}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, k, l]) {i=2, j=3, k=5, l=7} reduction: {j}>} : (tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32>
  func.return %0: tensor<2x5x7xf32>
}

// -----

func.func @reduce_is_an_unknown_keyword(%arg0: tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32> {
  // expected-error@+1 {{expected '>'}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, k, l]) {i=2, j=3, k=5, l=7} reduce={j}>} : (tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32>
  func.return %0: tensor<2x5x7xf32>
}

// -----

func.func @reduction_should_be_before_need_replication(%arg0: tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32> {
  // expected-error@+1 {{expected '>'}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, k, l]) {i=2, j=3, k=5, l=7} need_replication={i} reduction={j}>} : (tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32>
  func.return %0: tensor<2x5x7xf32>
}

// -----

func.func @invalid_dimension_symbol(%arg0: tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32> {
  // expected-error@+1 {{expecting symbol from 'i' to 'z'. Received: 'a'}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, k, l]) {i=2, j=3, k=5, l=7} reduction={a}>} : (tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32>
  func.return %0: tensor<2x5x7xf32>
}

// -----

func.func @reduction_factor_in_result(%arg0: tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32> {
  // expected-error@+1 {{reduction factor cannot be in result mappings}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l])->([i, k, l]) {i=2, j=3, k=5, l=7} reduction={k}>} : (tensor<2x3x5x7xf32>) -> tensor<2x5x7xf32>
  func.return %0: tensor<2x5x7xf32>
}
