// RUN: sdy_opt %s -canonicalize | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

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

// CHECK-LABEL: func @reshard_cse_two_same_reshards
// CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]>
// CHECK-NEXT: return %0, %0
func.func @reshard_cse_two_same_reshards(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  %1 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @reshard_cse_two_different_reshards
// CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]>
// CHECK-NEXT: %1 = sdy.reshard %arg0 <@mesh, [{?}, {"a", ?}]>
// CHECK-NEXT: return %0, %1
func.func @reshard_cse_two_different_reshards(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  %1 = sdy.reshard %arg0 <@mesh, [{?}, {"a", ?}]> : tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @reshard_cse_three_same_reshards
// CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]>
// CHECK-NEXT: %1 = stablehlo.sine %0 : tensor<8x8xf32>
// CHECK-NEXT: %2 = stablehlo.cosine %0 : tensor<8x8xf32>
// CHECK-NEXT: %3 = stablehlo.abs %0 : tensor<8x8xf32>
// CHECK-NEXT: return %1, %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
func.func @reshard_cse_three_same_reshards(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  %0 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  %1 = stablehlo.sine %0 : tensor<8x8xf32>
  %2 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  %3 = stablehlo.cosine %2 : tensor<8x8xf32>
  %4 = sdy.reshard %arg0 <@mesh, [{"a", ?}, {?}]> : tensor<8x8xf32>
  %5 = stablehlo.abs %4 : tensor<8x8xf32>
  return %1, %3, %5 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @reshard_chains_and_cse
// CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"b"}, {"a"}]> : tensor<8x8xf32>
// CHECK-NEXT: return %0, %0 : tensor<8x8xf32>, tensor<8x8xf32>
func.func @reshard_chains_and_cse(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {}]> : tensor<8x8xf32>
  %1 = sdy.reshard %arg0 <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  %2 = sdy.reshard %0 <@mesh, [{"b"}, {"a"}]> : tensor<8x8xf32>
  %3 = sdy.reshard %1 <@mesh, [{"b"}, {"a"}]> : tensor<8x8xf32>
  return %2, %3 : tensor<8x8xf32>, tensor<8x8xf32>
}
