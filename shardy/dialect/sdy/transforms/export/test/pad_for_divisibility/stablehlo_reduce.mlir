// RUN: sdy_opt %s -sdy-pad-for-divisibility -split-input-file -verify-diagnostics | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Reuses zero padding directly for addition (identity 0).
// CHECK-LABEL: func @reduce_add_padded_reduction_dim
func.func @reduce_add_padded_reduction_dim(%arg0: tensor<4x7xf32>) -> tensor<4xf32> {
  // CHECK-DAG:  %[[CST_ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST_ZERO]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xf32>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SLICE]] init: %{{.*}}) applies stablehlo.add across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xf32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xf32>, tensor<f32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Reuses zero padding directly for bitwise OR (identity 0).
// CHECK-LABEL: func @reduce_or_padded_reduction_dim
func.func @reduce_or_padded_reduction_dim(%arg0: tensor<4x7xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %[[CST_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST_ZERO]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi32>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SLICE]] init: %{{.*}}) applies stablehlo.or across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xi32>, tensor<i32>) -> tensor<4xi32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xi32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>
  %cst = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.or across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xi32>, tensor<i32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Reuses zero padding directly for bitwise XOR (identity 0).
// CHECK-LABEL: func @reduce_xor_padded_reduction_dim
func.func @reduce_xor_padded_reduction_dim(%arg0: tensor<4x7xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %[[CST_ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST_ZERO]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi32>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SLICE]] init: %{{.*}}) applies stablehlo.xor across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xi32>, tensor<i32>) -> tensor<4xi32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xi32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>
  %cst = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.xor across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xi32>, tensor<i32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Reuses zero padding directly for logical OR (identity false).
// CHECK-LABEL: func @reduce_or_bool_padded_reduction_dim
func.func @reduce_or_bool_padded_reduction_dim(%arg0: tensor<4x7xi1>) -> tensor<4xi1> {
  // CHECK-DAG:  %[[CST_FALSE:.*]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST_FALSE]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi1>, tensor<i1>) -> tensor<4x8xi1>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi1>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SLICE]] init: %{{.*}}) applies stablehlo.or across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xi1>, tensor<i1>) -> tensor<4xi1>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xi1>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi1>
  %cst = stablehlo.constant dense<false> : tensor<i1>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.or across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xi1>, tensor<i1>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Reuses zero padding directly for logical XOR (identity false).
// CHECK-LABEL: func @reduce_xor_bool_padded_reduction_dim
func.func @reduce_xor_bool_padded_reduction_dim(%arg0: tensor<4x7xi1>) -> tensor<4xi1> {
  // CHECK-DAG:  %[[CST_FALSE:.*]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST_FALSE]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi1>, tensor<i1>) -> tensor<4x8xi1>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi1>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SLICE]] init: %{{.*}}) applies stablehlo.xor across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xi1>, tensor<i1>) -> tensor<4xi1>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xi1>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi1>
  %cst = stablehlo.constant dense<false> : tensor<i1>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.xor across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xi1>, tensor<i1>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Masks padded reduction elements with float multiplication identity (1.0).
// CHECK-LABEL: func @reduce_mul_padded_reduction_dim
func.func @reduce_mul_padded_reduction_dim(%arg0: tensor<4x7xf32>) -> tensor<4xf32> {
  // CHECK-DAG:  %[[PAD_CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[PAD_CST]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xf32>
  // CHECK-DAG:  %[[IOTA:.*]] = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi32>
  // CHECK-DAG:  %[[LIMIT_CST:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-DAG:  %[[BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LIMIT_CST]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[MASK:.*]] = stablehlo.compare  LT, %[[IOTA]], %[[BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi1>
  // CHECK-DAG:  %[[BCAST_ONE:.*]] = stablehlo.broadcast_in_dim %{{.*}}, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<f32>) -> tensor<4x8xf32>
  // CHECK-DAG:  %[[SELECT:.*]] = stablehlo.select %[[MASK]], %[[SLICE]], %[[BCAST_ONE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi1>, tensor<4x8xf32>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SELECT]] init: %{{.*}}) applies stablehlo.multiply across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xf32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xf32>
  %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.multiply across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xf32>, tensor<f32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Masks padded reduction elements with integer multiplication identity (1).
// CHECK-LABEL: func @reduce_mul_int_padded_reduction_dim
func.func @reduce_mul_int_padded_reduction_dim(%arg0: tensor<4x7xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %[[PAD_CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[PAD_CST]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi32>
  // CHECK-DAG:  %[[IOTA:.*]] = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi32>
  // CHECK-DAG:  %[[LIMIT_CST:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-DAG:  %[[BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LIMIT_CST]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[MASK:.*]] = stablehlo.compare  LT, %[[IOTA]], %[[BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi1>
  // CHECK-DAG:  %[[BCAST_ONE:.*]] = stablehlo.broadcast_in_dim %{{.*}}, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[SELECT:.*]] = stablehlo.select %[[MASK]], %[[SLICE]], %[[BCAST_ONE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi1>, tensor<4x8xi32>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SELECT]] init: %{{.*}}) applies stablehlo.multiply across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xi32>, tensor<i32>) -> tensor<4xi32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xi32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>
  %cst = stablehlo.constant dense<1> : tensor<i32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.multiply across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xi32>, tensor<i32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Masks padded reduction elements with float max identity (-inf).
// CHECK-LABEL: func @reduce_max_padded_reduction_dim
func.func @reduce_max_padded_reduction_dim(%arg0: tensor<4x7xf32>) -> tensor<4xf32> {
  // CHECK-DAG:  %[[PAD_CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[PAD_CST]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xf32>
  // CHECK-DAG:  %[[IOTA:.*]] = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi32>
  // CHECK-DAG:  %[[LIMIT_CST:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-DAG:  %[[BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LIMIT_CST]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[MASK:.*]] = stablehlo.compare  LT, %[[IOTA]], %[[BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi1>
  // CHECK-DAG:  %[[BCAST_NEG_INF:.*]] = stablehlo.broadcast_in_dim %{{.*}}, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<f32>) -> tensor<4x8xf32>
  // CHECK-DAG:  %[[SELECT:.*]] = stablehlo.select %[[MASK]], %[[SLICE]], %[[BCAST_NEG_INF]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi1>, tensor<4x8xf32>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SELECT]] init: %{{.*}}) applies stablehlo.maximum across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xf32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xf32>
  %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.maximum across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xf32>, tensor<f32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Masks padded reduction elements with float min identity (+inf).
// CHECK-LABEL: func @reduce_min_padded_reduction_dim
func.func @reduce_min_padded_reduction_dim(%arg0: tensor<4x7xf32>) -> tensor<4xf32> {
  // CHECK-DAG:  %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[ZERO]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xf32>
  // CHECK-DAG:  %[[IOTA:.*]] = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi32>
  // CHECK-DAG:  %[[LIMIT_CST:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-DAG:  %[[BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LIMIT_CST]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[MASK:.*]] = stablehlo.compare  LT, %[[IOTA]], %[[BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi1>
  // CHECK-DAG:  %[[BCAST_POS_INF:.*]] = stablehlo.broadcast_in_dim %{{.*}}, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<f32>) -> tensor<4x8xf32>
  // CHECK-DAG:  %[[SELECT:.*]] = stablehlo.select %[[MASK]], %[[SLICE]], %[[BCAST_POS_INF]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi1>, tensor<4x8xf32>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SELECT]] init: %{{.*}}) applies stablehlo.minimum across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xf32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xf32>
  %cst = stablehlo.constant dense<0x7F800000> : tensor<f32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.minimum across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xf32>, tensor<f32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Masks padded reduction elements with logical AND identity (true).
// CHECK-LABEL: func @reduce_and_padded_reduction_dim
func.func @reduce_and_padded_reduction_dim(%arg0: tensor<4x7xi1>) -> tensor<4xi1> {
  // CHECK-DAG:  %[[FALSE:.*]] = stablehlo.constant dense<false> : tensor<i1>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[FALSE]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi1>, tensor<i1>) -> tensor<4x8xi1>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi1>
  // CHECK-DAG:  %[[IOTA:.*]] = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi32>
  // CHECK-DAG:  %[[LIMIT_CST:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-DAG:  %[[BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LIMIT_CST]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[MASK:.*]] = stablehlo.compare  LT, %[[IOTA]], %[[BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi1>
  // CHECK-DAG:  %[[BCAST_TRUE:.*]] = stablehlo.broadcast_in_dim %{{.*}}, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i1>) -> tensor<4x8xi1>
  // CHECK-DAG:  %[[SELECT:.*]] = stablehlo.select %[[MASK]], %[[SLICE]], %[[BCAST_TRUE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi1>, tensor<4x8xi1>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SELECT]] init: %{{.*}}) applies stablehlo.and across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xi1>, tensor<i1>) -> tensor<4xi1>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xi1>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi1>
  %cst = stablehlo.constant dense<true> : tensor<i1>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.and across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xi1>, tensor<i1>) -> tensor<4xi1>
  return %1 : tensor<4xi1>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Pads all inputs to divisible shapes along the reduction dimension.
// CHECK-LABEL: func @reduce_variadic
func.func @reduce_variadic(%arg0: tensor<4x7xf32>, %arg1: tensor<4x7xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // CHECK-DAG:  %[[PAD0:.*]] = stablehlo.pad %arg0, %{{.*}}, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-DAG:  %[[SLICE0:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD0]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xf32>
  // CHECK-DAG:  %[[PAD1:.*]] = stablehlo.pad %arg1, %{{.*}}, low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-DAG:  %[[SLICE1:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD1]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xf32>
  // CHECK:      %[[REDUCE:.*]]:2 = stablehlo.reduce(%[[SLICE0]] init: %{{.*}}), (%[[SLICE1]] init: %{{.*}}) across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
  // CHECK:      return %[[REDUCE]]#0, %[[REDUCE]]#1 : tensor<4xf32>, tensor<4xf32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xf32>
  %1 = sdy.all_slice [{}, {"y"}] %arg1 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %2:2 = "stablehlo.reduce"(%0, %1, %cst, %cst) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<f32>, %arg5: tensor<f32>):
    %res0 = stablehlo.add %arg2, %arg4 : tensor<f32>
    %res1 = stablehlo.add %arg3, %arg5 : tensor<f32>
    "stablehlo.return"(%res0, %res1) : (tensor<f32>, tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>, sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>, #sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xf32>, tensor<4x7xf32>, tensor<f32>, tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
  return %2#0, %2#1 : tensor<4xf32>, tensor<4xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Propagates padded shape through non-reduction dimension and trims downstream.
// CHECK-LABEL: func @reduce_pass_through_dim_padded
func.func @reduce_pass_through_dim_padded(%arg0: tensor<3x8xf32>) -> tensor<3xf32> {
  // CHECK-DAG:  %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[ZERO]], low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<3x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{"x"}, {}] %[[PAD]] out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SLICE]] init: %{{.*}}) applies stablehlo.add across dimensions = [1]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}]>]>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  // CHECK-NEXT: %[[AG:.*]] = sdy.all_gather [{"x"}] %[[REDUCE]] out_sharding=<@mesh_4_2, [{}]> : tensor<4xf32>
  // CHECK-NEXT: %[[TRIM:.*]] = stablehlo.slice %[[AG]] [0:3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}]>]>} : (tensor<4xf32>) -> tensor<3xf32>
  // CHECK-NEXT: return %[[TRIM]] : tensor<3xf32>
  %0 = sdy.all_slice [{"x"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<3x8xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"x"}]>]>} : (tensor<3x8xf32>, tensor<f32>) -> tensor<3xf32>
  %2 = sdy.all_gather [{"x"}] %1 out_sharding=<@mesh_4_2, [{}]> : tensor<3xf32>
  return %2 : tensor<3xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Masks padded reduction elements with signed integer max identity (signed min).
// CHECK-LABEL: func @reduce_max_int_padded_reduction_dim
func.func @reduce_max_int_padded_reduction_dim(%arg0: tensor<4x7xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %[[PAD_CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[PAD_CST]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi32>
  // CHECK-DAG:  %[[IOTA:.*]] = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi32>
  // CHECK-DAG:  %[[LIMIT_CST:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-DAG:  %[[BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LIMIT_CST]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[MASK:.*]] = stablehlo.compare  LT, %[[IOTA]], %[[BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi1>
  // CHECK-DAG:  %[[BCAST_MIN_INT:.*]] = stablehlo.broadcast_in_dim %{{.*}}, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[SELECT:.*]] = stablehlo.select %[[MASK]], %[[SLICE]], %[[BCAST_MIN_INT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi1>, tensor<4x8xi32>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SELECT]] init: %{{.*}}) applies stablehlo.maximum across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xi32>, tensor<i32>) -> tensor<4xi32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xi32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>
  %cst = stablehlo.constant dense<-2147483648> : tensor<i32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.maximum across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xi32>, tensor<i32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Masks padded reduction elements with signed integer min identity (signed max).
// CHECK-LABEL: func @reduce_min_int_padded_reduction_dim
func.func @reduce_min_int_padded_reduction_dim(%arg0: tensor<4x7xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %[[PAD_CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[PAD_CST]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi32>
  // CHECK-DAG:  %[[IOTA:.*]] = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi32>
  // CHECK-DAG:  %[[LIMIT_CST:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-DAG:  %[[BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LIMIT_CST]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[MASK:.*]] = stablehlo.compare  LT, %[[IOTA]], %[[BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi1>
  // CHECK-DAG:  %[[BCAST_MAX_INT:.*]] = stablehlo.broadcast_in_dim %{{.*}}, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[SELECT:.*]] = stablehlo.select %[[MASK]], %[[SLICE]], %[[BCAST_MAX_INT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi1>, tensor<4x8xi32>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SELECT]] init: %{{.*}}) applies stablehlo.minimum across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xi32>, tensor<i32>) -> tensor<4xi32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xi32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>
  %cst = stablehlo.constant dense<2147483647> : tensor<i32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.minimum across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xi32>, tensor<i32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Masks padded reduction elements with bitwise AND identity (all-ones / -1).
// CHECK-LABEL: func @reduce_and_int_padded_reduction_dim
func.func @reduce_and_int_padded_reduction_dim(%arg0: tensor<4x7xi32>) -> tensor<4xi32> {
  // CHECK-DAG:  %[[PAD_CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[PAD_CST]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xi32>, tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xi32>
  // CHECK-DAG:  %[[IOTA:.*]] = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi32>
  // CHECK-DAG:  %[[LIMIT_CST:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK-DAG:  %[[BCAST_LIMIT:.*]] = stablehlo.broadcast_in_dim %[[LIMIT_CST]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[MASK:.*]] = stablehlo.compare  LT, %[[IOTA]], %[[BCAST_LIMIT]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<4x8xi32>, tensor<4x8xi32>) -> tensor<4x8xi1>
  // CHECK-DAG:  %[[BCAST_ALL_ONES:.*]] = stablehlo.broadcast_in_dim %{{.*}}, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK-DAG:  %[[SELECT:.*]] = stablehlo.select %[[MASK]], %[[SLICE]], %[[BCAST_ALL_ONES]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x8xi1>, tensor<4x8xi32>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SELECT]] init: %{{.*}}) applies stablehlo.and across dimensions = [1]
  // CHECK-SAME: : (tensor<4x8xi32>, tensor<i32>) -> tensor<4xi32>
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xi32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xi32>
  %cst = stablehlo.constant dense<-1> : tensor<i32>
  %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.and across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xi32>, tensor<i32>) -> tensor<4xi32>
  return %1 : tensor<4xi32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Reuses zero padding directly for subtraction (identity 0).
// CHECK-LABEL: func @reduce_sub_padded_reduction_dim
func.func @reduce_sub_padded_reduction_dim(%arg0: tensor<4x7xf32>) -> tensor<4xf32> {
  // CHECK-DAG:  %[[CST_ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK-DAG:  %[[PAD:.*]] = stablehlo.pad %arg0, %[[CST_ZERO]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-DAG:  %[[SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xf32>
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%[[SLICE]] init: %{{.*}}) across dimensions = [1]
  // CHECK-NEXT: reducer(%[[ACC:.*]]: tensor<f32>, %[[ELEM:.*]]: tensor<f32>) {
  // CHECK-NEXT:   %[[SUB:.*]] = stablehlo.subtract %[[ACC]], %[[ELEM]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[SUB]] : tensor<f32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[REDUCE]] : tensor<4xf32>
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "stablehlo.reduce"(%0, %cst) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %res = stablehlo.subtract %arg1, %arg2 : tensor<f32>
    "stablehlo.return"(%res) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>, sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xf32>, tensor<f32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// Rejects unsupported reduction operations in the reduce body.
func.func @unsupported_reduce_body(%arg0: tensor<4x7xf32>) -> tensor<4xf32> {
  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xf32>
  %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // expected-error @+2 {{unsupported reduction operation in reduce body}}
  // expected-error @+1 {{failed to legalize operation 'stablehlo.reduce'}}
  %1 = "stablehlo.reduce"(%0, %cst) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %diff = stablehlo.subtract %arg1, %arg2 : tensor<f32>
    %res = stablehlo.multiply %diff, %arg1 : tensor<f32>
    "stablehlo.return"(%res) : (tensor<f32>) -> ()
  }) {dimensions = array<i64: 1>, sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{}]>]>} : (tensor<4x7xf32>, tensor<f32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}
