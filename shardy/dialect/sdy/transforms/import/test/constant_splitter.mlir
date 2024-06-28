// RUN: sdy_opt %s -sdy-constant-splitter 2>&1 | FileCheck %s

// CHECK-LABEL: func @constant_with_unregistered_attr
func.func @constant_with_unregistered_attr() -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[CONST:.*]] = sdy.constant {foo} dense<1.000000e+00>
  // CHECK-NEXT: return %[[CONST]]
  %0 = stablehlo.constant {foo} dense<1.000000e+00> : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @func_arg_is_not_constant
func.func @func_arg_is_not_constant(%arg0: tensor<8x16xf32>, %arg1: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %[[CONST]]
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[ADD]], %arg1
  // CHECK-NEXT: return %[[ADD]], %[[DOT_GENERAL]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.add %arg0, %0 : tensor<8x16xf32>
  %2 = stablehlo.dot_general %1, %arg1, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  return %1, %2 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @constant_multiple_users
func.func @constant_multiple_users(%arg0: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST_1]], %[[DOT_GENERAL]]
  // CHECK-NEXT: return %[[CONST_2]], %[[ADD]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %0, %2 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @constant_sub_computation_multiple_users
func.func @constant_sub_computation_multiple_users(%arg0: tensor<5x8xi32>) -> (tensor<4x5xi32>, tensor<4x8xi32>) {
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<2>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<2>
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[IOTA_0]], %[[IOTA_0]]
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[IOTA_1]], %[[IOTA_1]]
  // CHECK-NEXT: %[[MAX_0:.*]] = stablehlo.maximum %[[ADD_0]], %[[CONST_0]]
  // CHECK-NEXT: %[[MAX_1:.*]] = stablehlo.maximum %[[ADD_1]], %[[CONST_1]]
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[MAX_0]], %arg0
  // CHECK-NEXT: return %[[MAX_1]], %[[DOT_GENERAL]]
  %0 = stablehlo.iota dim = 0 : tensor<4x5xi32>
  %1 = stablehlo.constant dense<2> : tensor<4x5xi32>
  %2 = stablehlo.add %0, %0 : tensor<4x5xi32>
  %3 = stablehlo.maximum %2, %1 : tensor<4x5xi32>
  %4 = stablehlo.dot_general %3, %arg0, contracting_dims = [1] x [0] : (tensor<4x5xi32>, tensor<5x8xi32>) -> tensor<4x8xi32>
  return %3, %4 : tensor<4x5xi32>, tensor<4x8xi32>
}

// CHECK-LABEL: func @constant_multiple_uses_by_same_op
func.func @constant_multiple_uses_by_same_op() -> (tensor<8x16xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %[[CONST_1]]
  // CHECK-NEXT: return %[[CONST_2]], %[[DOT_GENERAL]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %0, contracting_dims = [1] x [1] : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x8xf32>
  return %0, %1 : tensor<8x16xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @non_constant_broadcast_multiple_users
func.func @non_constant_broadcast_multiple_users(%arg0: tensor<4x5xf32>, %arg1: tensor<5xf32>) -> (tensor<5x8xf32>, tensor<4x8xf32>) {
  // CHECK-NEXT: %[[BROADCAST:.*]] = stablehlo.broadcast_in_dim %arg1
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %[[BROADCAST]]
  // CHECK-NEXT: return %[[BROADCAST]], %[[DOT_GENERAL]]
  %0 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<5xf32>) -> tensor<5x8xf32>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [1] x [0] : (tensor<4x5xf32>, tensor<5x8xf32>) -> tensor<4x8xf32>
  return %0, %1 : tensor<5x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @constant_broadcast_multiple_users
func.func @constant_broadcast_multiple_users(%arg0: tensor<5x8xi32>) -> (tensor<4x5xi32>, tensor<4x8xi32>) {
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[IOTA_0]], %[[IOTA_0]]
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[IOTA_1]], %[[IOTA_1]]
  // CHECK-NEXT: %[[BROADCAST_0:.*]] = stablehlo.broadcast_in_dim %[[ADD_0]]
  // CHECK-NEXT: %[[BROADCAST_1:.*]] = stablehlo.broadcast_in_dim %[[ADD_1]]
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[BROADCAST_0]], %arg0
  // CHECK-NEXT: return %[[BROADCAST_1]], %[[DOT_GENERAL]]
  %0 = stablehlo.iota dim = 0 : tensor<5xi32>
  %1 = stablehlo.add %0, %0 : tensor<5xi32>
  %2 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<5xi32>) -> tensor<4x5xi32>
  %3 = stablehlo.dot_general %2, %arg0, contracting_dims = [1] x [0] : (tensor<4x5xi32>, tensor<5x8xi32>) -> tensor<4x8xi32>
  return %2, %3 : tensor<4x5xi32>, tensor<4x8xi32>
}

// CHECK-LABEL: func @constant_slice_multiple_users
func.func @constant_slice_multiple_users(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[SLICE_0:.*]] = stablehlo.slice %[[IOTA_0]]
  // CHECK-NEXT: %[[SLICE_1:.*]] = stablehlo.slice %[[IOTA_1]]
  // CHECK-NEXT: %[[SLICE_2:.*]] = stablehlo.slice %[[IOTA_0]]
  // CHECK-NEXT: %[[SLICE_3:.*]] = stablehlo.slice %[[IOTA_1]]
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[SLICE_0]], %[[SLICE_2]]
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[SLICE_1]], %[[SLICE_3]]
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[ADD_0]], %arg0
  // CHECK-NEXT: return %[[ADD_1]], %[[ADD_2]]
  %0 = stablehlo.iota dim = 0 : tensor<10xi32>
  %1 = stablehlo.slice %0 [0:8]: (tensor<10xi32>) -> tensor<8xi32>
  %2 = stablehlo.slice %0 [2:10]: (tensor<10xi32>) -> tensor<8xi32>
  %3 = stablehlo.add %1, %2 : tensor<8xi32>
  %4 = stablehlo.add %3, %arg0 : tensor<8xi32>
  return %3, %4 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func @splits_parts_of_const_sub_computation
func.func @splits_parts_of_const_sub_computation(%arg0: tensor<5x8xi32>) -> (tensor<4x5xi32>, tensor<4x5xi32>, tensor<4x8xi32>) {
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_2:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[CONST:.*]] = sdy.constant dense<2>
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[IOTA_1]], %[[IOTA_1]]
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[IOTA_2]], %[[IOTA_2]]
  // CHECK-NEXT: %[[MAX:.*]] = stablehlo.maximum %[[ADD_1]], %[[CONST]]
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[MAX]], %arg0
  // CHECK-NEXT: return %[[IOTA_0]], %[[ADD_0]], %[[DOT_GENERAL]]
  %0 = stablehlo.iota dim = 0 : tensor<4x5xi32>
  %1 = stablehlo.constant dense<2> : tensor<4x5xi32>
  %2 = stablehlo.add %0, %0 : tensor<4x5xi32>
  %3 = stablehlo.maximum %2, %1 : tensor<4x5xi32>
  %4 = stablehlo.dot_general %3, %arg0, contracting_dims = [1] x [0] : (tensor<4x5xi32>, tensor<5x8xi32>) -> tensor<4x8xi32>
  return %0, %2, %4 : tensor<4x5xi32>, tensor<4x5xi32>, tensor<4x8xi32>
}

// CHECK-LABEL: func @splits_sdy_constants
func.func @splits_sdy_constants(%arg0: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST_1]], %[[DOT_GENERAL]]
  // CHECK-NEXT: return %[[CONST_2]], %[[ADD]]
  %0 = sdy.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %0, %2 : tensor<8x16xf32>, tensor<8x16xf32>
}
