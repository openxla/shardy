// RUN: sdy_opt %s -sdy-populate-op-sharding-rules="conservative-propagation=true" 2>&1 | FileCheck %s

// CHECK-LABEL: func @concat
func.func @concat(%arg0: tensor<4x3x256xf32>, %arg1: tensor<4x5x256xf32>) -> tensor<4x8x256xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, k, j], [i, l, j])->([i, m, j]) {i=4, j=256, k=1, l=1, m=1}>
 %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x3x256xf32>, tensor<4x5x256xf32>) -> tensor<4x8x256xf32>
 return %0 : tensor<4x8x256xf32>
}

// CHECK-LABEL: func @conv
func.func @conv(%arg0 : tensor<2x224x224x192xf32>, %arg1 : tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, l, m, j], [n, o, j, k])->([i, p, q, k]) {i=2, j=192, k=64, l=1, m=1, n=1, o=1, p=1, q=1} reduction={j}>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<2x224x224x192xf32>, tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32>
  return %0 : tensor<2x112x112x64xf32>
}

// CHECK-LABEL: func @pad
func.func @pad(%arg0: tensor<28x28x16xf32>, %arg1: tensor<f32>) -> tensor<30x26x16xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [])->([i, j, k]) {i=28, j=28, k=16} permutation={i, j} blocked_propagation={i, j}>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @reduce_window
func.func @reduce_window(%arg0: tensor<48x48x3xf32>, %arg1: tensor<48x48x3xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>)
    -> (tensor<16x48x1xf32>, tensor<16x48x1xi32>) {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k], [i, j, k], [], [])->([i, j, k], [i, j, k]) {i=16, j=48, k=1} permutation={i, k} blocked_propagation={i, k}>
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%arg4: tensor<f32>, %arg5 : tensor<i32>, %arg6: tensor<f32>, %arg7 : tensor<i32>):
    %1 = stablehlo.maximum %arg4, %arg6 : tensor<f32>
    %2 = stablehlo.maximum %arg5, %arg7 : tensor<i32>
    stablehlo.return %1, %2 : tensor<f32>, tensor<i32>
  }) {window_dimensions = array<i64: 3, 1, 3>,
      window_strides = array<i64: 3, 1, 3>,
      padding = dense<[[0, 0], [2, -2], [0, 0]]> : tensor<3x2xi64>}
      : (tensor<48x48x3xf32>, tensor<48x48x3xi32>, tensor<f32>, tensor<i32>) -> (tensor<16x48x1xf32>, tensor<16x48x1xi32>)
  func.return %0#0, %0#1 : tensor<16x48x1xf32>, tensor<16x48x1xi32>
}

// CHECK-LABEL: func @select_and_scatter
func.func @select_and_scatter(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<10x12x12x64xf32>, %arg2: tensor<f32>)
   -> tensor<10x24x24x64xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k, l], [i, j, k, l], [])->([i, j, k, l]) {i=10, j=12, k=12, l=64} permutation={j, k} blocked_propagation={j, k}>
  %1 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.compare GT, %arg3, %arg4 :(tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %2 = stablehlo.add %arg3, %arg4 : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>,
    window_strides = array<i64: 1, 2, 2, 1>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  return %1 : tensor<10x24x24x64xf32>
}

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<32x4x8xf32>) -> tensor<32x1x2xf32> {
  // CHECK: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j, k])->([i, j, k]) {i=32, j=4, k=8} permutation={j, k} blocked_propagation={j, k}>
  %0 = stablehlo.slice %arg0 [0:32, 1:2, 4:8:2] : (tensor<32x4x8xf32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}
