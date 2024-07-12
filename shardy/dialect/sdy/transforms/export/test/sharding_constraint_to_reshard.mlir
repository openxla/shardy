// RUN: sdy_opt %s -sdy-sharding-constraint-to-reshard | FileCheck %s

sdy.mesh @mesh = <"a"=2, "b"=2>

// CHECK-LABEL: func @remove_redundant_sharding_constraint
func.func @remove_redundant_sharding_constraint(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) -> tensor<8x8xf32> {
  // CHECK: return %arg0 : tensor<8x8xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a", ?}, {?}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @sharding_constraint_to_reshard
func.func @sharding_constraint_to_reshard(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) -> tensor<8x8xf32> {
  // CHECK: %0 = sdy.reshard %arg0 <@mesh, [{?}, {?}]> : tensor<8x8xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{?}, {?}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK: func.func @get_sharding_from_sharding_constraint(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) -> tensor<8x8xf32>
func.func @get_sharding_from_sharding_constraint(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: return %arg0 : tensor<8x8xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a", ?}, {?}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
