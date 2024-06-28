// RUN: sdy_opt %s 2>&1 | FileCheck %s

// CHECK-LABEL: func @scalar
func.func @scalar(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: {sdy.sharding_rule = #sdy.op_sharding_rule<([], [])->([])>}
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([], [])->([])>} : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @nested
func.func @nested(%arg0: tensor<2x4xf32>) -> tensor<8xf32> {
  // CHECK: {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([ij]) {i=2, j=4}>}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([ij]) {i=2, j=4}>} : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// Verify that we can parse `z_#` and `i`-`z` symbols together, specifically
// `zz_1` for the 2nd last dimension of the result tensor. This only happens for
// tensors of rank > 17 since for indices 0-17 we use symbols `i` through `z`.
// This is why the tensor here is massive.
// CHECK-LABEL: func @zz_1
func.func @zz_1(%arg0: tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2xf32>) -> tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x4x2xf32> {
  // CHECK:      #sdy.op_sharding_rule<([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2])
  // CHECK-SAME: ->([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, zz_1, z_2])
  // CHECK-SAME: {i=2, j=2, k=2, l=2, m=2, n=2, o=2, p=2, q=2, r=2, s=2, t=2, u=2, v=2, w=2, x=2, y=2, z=2, z_1=2, z_2=2}>}
  %0 = stablehlo.reshape %arg0 {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2])->([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, zz_1, z_2]) {i=2, j=2, k=2, l=2, m=2, n=2, o=2, p=2, q=2, r=2, s=2, t=2, u=2, v=2, w=2, x=2, y=2, z=2, z_1=2, z_2=2}>} : (tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2xf32>) -> tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x4x2xf32>
  return %0 : tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x4x2xf32>
}

// Verify that we can parse multiple `z_#` symbols together, specifically
// `z_8z_9z_10` for the last dimension of the result tensor. This only happens
// for tensors of rank > 17 since for indices 0-17 we use symbols `i` through
// `z`. This is why the tensor here is massive.
// CHECK-LABEL: func @z_8z_9z_10
func.func @z_8z_9z_10(%arg0: tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2xf32>) -> tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x8xf32> {
  // CHECK:      #sdy.op_sharding_rule<([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8, z_9, z_10])
  // CHECK-SAME: ->([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8z_9z_10])
  // CHECK-SAME: {i=2, j=2, k=2, l=2, m=2, n=2, o=2, p=2, q=2, r=2, s=2, t=2, u=2, v=2, w=2, x=2, y=2, z=2, z_1=2, z_2=2, z_3=2, z_4=2, z_5=2, z_6=2, z_7=2, z_8=2, z_9=2, z_10=2}>} :
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8, z_9, z_10])->([i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8z_9z_10]) {i=2, j=2, k=2, l=2, m=2, n=2, o=2, p=2, q=2, r=2, s=2, t=2, u=2, v=2, w=2, x=2, y=2, z=2, z_1=2, z_2=2, z_3=2, z_4=2, z_5=2, z_6=2, z_7=2, z_8=2, z_9=2, z_10=2}>} : (tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2xf32>) -> tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x8xf32>
  return %0 : tensor<2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x2x8xf32>
}

// CHECK-LABEL: func @custom_call_custom_rule
func.func @custom_call_custom_rule(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
  // CHECK: {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=32}, custom>}
  %0 = stablehlo.custom_call @foo(%arg0) {sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([i, j]) {i=16, j=32}, custom>} : (tensor<16x32xf32>) -> tensor<16x32xf32>
  func.return %0: tensor<16x32xf32>
}
