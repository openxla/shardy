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

