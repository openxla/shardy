// RUN: sdy_opt %s -split-input-file -verify-diagnostics

func.func @custom_rule_not_custom_call(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{can only define custom sharding rules on stablehlo.custom_call}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i], [j])->([j]) {i=8, j=8}, custom>} : tensor<8xf32>
  func.return %0 : tensor<8xf32>
}

// -----

func.func @sharding_rule_wrong_attr_type(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{should have a sharding rule attribute of type OpShardingRuleAttr}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = 1 : i64} : (tensor<8xf32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
}

// -----

func.func @unranked_tensor_type(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // expected-error@+1 {{operand 0 - expected a ranked tensor with a static shape}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=2, j=4}>} : tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func.func @dynamic_shaped_tensor_type(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error@+1 {{operand 0 - expected a ranked tensor with a static shape}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=2, j=4}>} : tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

func.func @operand_mappings_wrong_rank(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{operand 1 - mapping rank must match: 1 != 2}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i])->([i, j]) {i=2, j=4}>} : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @result_mappings_wrong_rank(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{result - mapping rank must match: 1 != 2}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i]) {i=2, j=4}>} : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @result_mappings_wrong_num(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{number of results and mappings must match: 1 != 2}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j], [i, j]) {i=2, j=4}>} : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @operand_mappings_wrong_num(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // expected-error@+1 {{number of operands and mappings must match: 2 != 1}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=2, j=4}>} : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @dim_mapping_multiple_factors_with_size_one_factor(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // expected-error@+1 {{operand 0 - dim mapping can't have a factor of size 1 if there are multiple factors}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, jk], [i, j])->([i, j]) {i=2, j=8, k=1}>} : tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// -----

func.func @dim_mapping_with_no_factors(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // expected-error@+1 {{operand 0 - dim mapping must have at least one factor}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i,], [i,])->([i, j]) {i=2, j=8}>} : tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// -----

func.func @unknown_factor_mapping(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // expected-error@+1 {{operand 0 - expecting factor indices to be within 0<=...<num_factors; received: 12, num_factors: 2}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, u], [i, j])->([i, j]) {i=2, j=8}>} : tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// -----

func.func @unused_factor(%arg0: tensor<2x8xf32>) -> tensor<2x8xf32> {
  // expected-error@+1 {{has factor k=2 that isn't used in operand and result mappings}}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=2, j=8, k=2}>} : tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// -----

func.func @duplicate_factor_same_dim(%arg0: tensor<4x2xf32>) -> tensor<8xf32> {
  // expected-error@+1 {{op result - cannot reuse factors for the same tensor value}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([ij, k])->([iik]) {i=2, j=2, k=2}>} : (tensor<4x2xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

func.func @duplicate_factor_different_dim(%arg0: tensor<2x2xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{op operand - cannot reuse factors for the same tensor value}}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, i])->([ij]) {i=2, j=2}>} : (tensor<2x2xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @unsorted_special_factors(%arg0: tensor<2x4x8xf32>) -> tensor<2x8xf32> {
  // expected-error@+1 {{indices of special factors must be sorted}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, k]) {i=2, j=4, k=8} need_replication={k, i}>} : (tensor<2x4x8xf32>) -> tensor<2x8xf32>
  func.return %0: tensor<2x8xf32>
}

// -----

func.func @repeated_special_factors(%arg0: tensor<2x4x8xf32>) -> tensor<2x8xf32> {
  // expected-error@+1 {{indices of special factors must be unique}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, k]) {i=2, j=4, k=8} need_replication={i, i}>} : (tensor<2x4x8xf32>) -> tensor<2x8xf32>
  func.return %0: tensor<2x8xf32>
}

// -----

func.func @invalid_special_factor_index(%arg0: tensor<2x4x8xf32>) -> tensor<2x8xf32> {
  // expected-error@+1 {{index must be less than 3, got: 17}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, k]) {i=2, j=4, k=8} need_replication={z}>} : (tensor<2x4x8xf32>) -> tensor<2x8xf32>
  func.return %0: tensor<2x8xf32>
}

// -----

func.func @a_factor_in_two_special_factor_sets(%arg0: tensor<2x4x8xf32>) -> tensor<2x8xf32> {
  // expected-error@+1 {{a factor can only be in one of the special factor sets}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, k]) {i=2, j=4, k=8} reduction={j} need_replication={j}>} : (tensor<2x4x8xf32>) -> tensor<2x8xf32>
  func.return %0: tensor<2x8xf32>
}

// -----

func.func @a_factor_in_three_special_factor_sets(%arg0: tensor<2x4x8xf32>) -> tensor<2x8xf32> {
  // expected-error@+1 {{a factor can only be in one of the special factor sets}}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, k]) {i=2, j=4, k=8} reduction={j} need_replication={j} permutation={j}>} : (tensor<2x4x8xf32>) -> tensor<2x8xf32>
  func.return %0: tensor<2x8xf32>
}
