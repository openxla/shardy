// RUN: sdy_opt %s -canonicalize | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @empty_mesh = <[]>

// CHECK-LABEL: func @reshard_of_reshard_no_other_uses
// CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]>
// CHECK-NEXT: return %0
func.func @reshard_of_reshard_no_other_uses(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  %1 = sdy.reshard %0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @reshard_of_reshard_with_other_uses
// CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]>
// CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{"a", ?}, {?}]>
// CHECK-NEXT: return %0, %1
func.func @reshard_of_reshard_with_other_uses(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>)  {
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  %1 = sdy.reshard %0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @reshard_chain_of_three_no_other_uses
// CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{?}, {"a", ?}]>
// CHECK-NEXT: return %0
func.func @reshard_chain_of_three_no_other_uses(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  %1 = sdy.reshard %0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  %2 = sdy.reshard %1 <@mesh, [{?}, {"a", ?}]> : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// CHECK-LABEL: func @reshard_input_output_have_same_shardings
func.func @reshard_input_output_have_same_shardings(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: return %arg0
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @reshard_input_output_both_fully_replicated
func.func @reshard_input_output_both_fully_replicated(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: return %arg0
  %0 = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @reshard_input_has_no_sharding_output_fully_replicated
func.func @reshard_input_has_no_sharding_output_fully_replicated(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: return %arg0
  %0 = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @reshard_input_is_fully_replicated_output_with_empty_mesh
func.func @reshard_input_is_fully_replicated_output_with_empty_mesh(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: return %arg0
  %0 = sdy.reshard %arg0 <@empty_mesh, [{}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @reshard_input_output_have_different_shardings
func.func @reshard_input_output_have_different_shardings(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"b"}, {"a"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh, [{"b"}, {"a"}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

