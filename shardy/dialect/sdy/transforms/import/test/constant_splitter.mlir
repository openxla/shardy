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
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_2]], %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST_0]], %[[DOT_GENERAL]]
  // CHECK-NEXT: return %[[CONST_1]], %[[ADD]]
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
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[MAX_1]], %arg0
  // CHECK-NEXT: return %[[MAX_0]], %[[DOT_GENERAL]]
  %0 = stablehlo.iota dim = 0 : tensor<4x5xi32>
  %1 = stablehlo.constant dense<2> : tensor<4x5xi32>
  %2 = stablehlo.add %0, %0 : tensor<4x5xi32>
  %3 = stablehlo.maximum %2, %1 : tensor<4x5xi32>
  %4 = stablehlo.dot_general %3, %arg0, contracting_dims = [1] x [0] : (tensor<4x5xi32>, tensor<5x8xi32>) -> tensor<4x8xi32>
  return %3, %4 : tensor<4x5xi32>, tensor<4x8xi32>
}

// We need to duplicate %0 and %1 because they are used by two different
// non-constant expressions. %2, %3, %4, %5 do not need to be duplicated.
// CHECK-LABEL: func @duplicate_the_necessary_subcomputation_only
func.func @duplicate_the_necessary_subcomputation_only() -> (tensor<4x5xi32>, tensor<4x5xi32>) {
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<2>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<2>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[IOTA_1]], %[[CONST_1]]
  // CHECK-NEXT: %[[ABS_ADD:.*]] = stablehlo.abs %[[ADD]]
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %[[IOTA_0]], %[[CONST_0]]
  // CHECK-NEXT: %[[ABS_MUL:.*]] = stablehlo.abs %[[MUL]]
  // CHECK-NEXT: return %[[ABS_ADD]], %[[ABS_MUL]]
  %0 = stablehlo.iota dim = 0 : tensor<4x5xi32>
  %1 = stablehlo.constant dense<2> : tensor<4x5xi32>
  %2 = stablehlo.add %0, %1 : tensor<4x5xi32>
  %3 = stablehlo.abs %2 : tensor<4x5xi32>
  %4 = stablehlo.multiply %0, %1 : tensor<4x5xi32>
  %5 = stablehlo.abs %4 : tensor<4x5xi32>
  return %3, %5 : tensor<4x5xi32>, tensor<4x5xi32>
}

// CHECK-LABEL: func @constant_multiple_uses_by_same_op
func.func @constant_multiple_uses_by_same_op() -> (tensor<8x16xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_2]], %[[CONST_0]]
  // CHECK-NEXT: return %[[CONST_1]], %[[DOT_GENERAL]]
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
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[BROADCAST_1]], %arg0
  // CHECK-NEXT: return %[[BROADCAST_0]], %[[DOT_GENERAL]]
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
  // CHECK-NEXT: %[[ADD_2:.*]] = stablehlo.add %[[ADD_1]], %arg0
  // CHECK-NEXT: return %[[ADD_0]], %[[ADD_2]]
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
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_2]], %arg0
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST_0]], %[[DOT_GENERAL]]
  // CHECK-NEXT: return %[[CONST_1]], %[[ADD]]
  %0 = sdy.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %0, %2 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @splits_sharding_groups
func.func @splits_sharding_groups(%arg0: tensor<16x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  // Each intermediate result is included in a distinct sharding group. Post
  // splitting, we verify that all subcomputation results corresponding to an
  // original unsplit result are included in the same sharding group.
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: sdy.sharding_group %[[CONST_0]] group_id=0
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: sdy.sharding_group %[[CONST_1]] group_id=0
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: sdy.sharding_group %[[CONST_2]] group_id=0
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_2]], %arg0
  // CHECK-NEXT: sdy.sharding_group %[[DOT_GENERAL]] group_id=1
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CONST_0]], %[[DOT_GENERAL]]
  // CHECK-NEXT: sdy.sharding_group %[[ADD]] group_id=2
  // CHECK-NEXT: return %[[CONST_1]], %[[ADD]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  sdy.sharding_group %0 group_id=0 : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %arg0, contracting_dims = [1] x [0] : (tensor<8x16xf32>, tensor<16x16xf32>) -> tensor<8x16xf32>
  sdy.sharding_group %1 group_id=1 : tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  sdy.sharding_group %2 group_id=2 : tensor<8x16xf32>
  return %0, %2 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @splits_const_subexpr_with_sharding_group
func.func @splits_const_subexpr_with_sharding_group(%arg0: tensor<4x8xi32>) -> (tensor<4x8xi32>, tensor<4x8xi32>) {
  // CHECK-NEXT: sdy.sharding_group %arg0 group_id=0
  // CHECK-NEXT: %[[IOTA_0:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: sdy.sharding_group %[[IOTA_0]] group_id=0
  // CHECK-NEXT: %[[IOTA_1:.*]] = stablehlo.iota dim = 0
  // CHECK-NEXT: sdy.sharding_group %[[IOTA_1]] group_id=0
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant dense<2>
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %[[IOTA_0]], %[[IOTA_0]]
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[IOTA_1]], %[[IOTA_1]]
  // CHECK-NEXT: %[[MAX_0:.*]] = stablehlo.maximum %[[ADD_0]], %[[CONST_0]]
  // CHECK-NEXT: return %[[ADD_1]], %[[MAX_0]]
  sdy.sharding_group %arg0 group_id=0 : tensor<4x8xi32>
  %0 = stablehlo.iota dim = 0 : tensor<4x8xi32>
  sdy.sharding_group %0 group_id=0 : tensor<4x8xi32>
  %1 = stablehlo.constant dense<2> : tensor<4x8xi32>
  %2 = stablehlo.add %0, %0 : tensor<4x8xi32>
  %3 = stablehlo.maximum %2, %1 : tensor<4x8xi32>
  return %2, %3 : tensor<4x8xi32>, tensor<4x8xi32>
}
