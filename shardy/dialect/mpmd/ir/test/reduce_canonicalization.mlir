// RUN: mpmd_opt %s -canonicalize 2>&1 | FileCheck %s

#topology =#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["y"=2]>>>

// CHECK-LABEL: func @reduce_of_reduce_chain_of_none_type
func.func @reduce_of_reduce_chain_of_none_type(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {"topology"=#topology} {
  // CHECK-NEXT: %[[REDUCE:.*]] = mpmd.reduce<none> %arg0 : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4x8xf32>
  %0 = mpmd.reduce<none> %arg0 : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %1 = mpmd.reduce<none> %0 : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %2 = mpmd.reduce<none> %1 : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %2 : tensor<4x8xf32>
}

// CHECK-LABEL: func @flatten_reduce_chain_if_types_match
func.func @flatten_reduce_chain_if_types_match(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {"topology"=#topology} {
  // CHECK-NEXT: %[[REDUCE:.*]] = mpmd.reduce<add> %arg0, %arg0 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4x8xf32>
  %0 = mpmd.reduce<add> %arg0, %arg0 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %1 = mpmd.reduce<add> %0 : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @flatten_reduce_chain_if_outer_reduce_is_none
func.func @flatten_reduce_chain_if_outer_reduce_is_none(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {"topology"=#topology} {
  // CHECK-NEXT: %[[REDUCE:.*]] = mpmd.reduce<add> %arg0, %arg0 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4x8xf32>
  %0 = mpmd.reduce<add> %arg0, %arg0 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %1 = mpmd.reduce<none> %0 : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @do_not_flatten_reduce_chain_if_types_dont_match
func.func @do_not_flatten_reduce_chain_if_types_dont_match(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {"topology"=#topology} {
  // CHECK-NEXT: %[[REDUCE_MUL:.*]] = mpmd.reduce<mul> %arg0, %arg0 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: %[[REDUCE_ADD:.*]] = mpmd.reduce<add> %[[REDUCE_MUL]] : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[REDUCE_ADD]] : tensor<4x8xf32>
  %0 = mpmd.reduce<mul> %arg0, %arg0 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %1 = mpmd.reduce<add> %0 : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @single_reduce_with_block_arg
func.func @single_reduce_with_block_arg(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {"topology"=#topology} {
  // CHECK-NEXT: %[[REDUCE:.*]] = mpmd.reduce<none> %arg0 : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4x8xf32>
  %0 = mpmd.reduce<none> %arg0 : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
