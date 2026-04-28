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

// CHECK-LABEL: func @reshard_cse_with_shape
// CHECK-NEXT: %[[RS1:.*]] = stablehlo.reshape %arg0 : (tensor<2x4x4xf32>) -> tensor<8x4xf32>
// CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[RS1]] <@mesh, [{"a"}, {}]> : tensor<8x4xf32>
// CHECK-NEXT: %[[RS2:.*]] = stablehlo.reshape %[[R1]] : (tensor<8x4xf32>) -> tensor<2x4x4xf32>
// CHECK-NEXT: %[[RS3:.*]] = stablehlo.reshape %[[RS2]] : (tensor<2x4x4xf32>) -> tensor<2x16xf32>
// CHECK-NEXT: %[[RS4:.*]] = stablehlo.reshape %[[RS3]] : (tensor<2x16xf32>) -> tensor<32xf32>
// CHECK-NEXT: return %[[R1]], %[[RS4]] : tensor<8x4xf32>, tensor<32xf32>
func.func @reshard_cse_with_shape(%arg0: tensor<2x4x4xf32>)
  -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
      tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  %0 = stablehlo.reshape %arg0 : (tensor<2x4x4xf32>) -> tensor<8x4xf32>
  %1 = sdy.reshard %0 <@mesh, [{"a"}, {}]> : tensor<8x4xf32>
  %2 = stablehlo.reshape %arg0 : (tensor<2x4x4xf32>) -> tensor<2x16xf32>
  %3 = stablehlo.reshape %2 : (tensor<2x16xf32>) -> tensor<32xf32>
  %4 = sdy.reshard %3 <@mesh, [{"a"}]> : tensor<32xf32>
  return %1, %4 : tensor<8x4xf32>, tensor<32xf32>
}

// CHECK-LABEL: func @reshard_cse_with_shape_2
// CHECK-NEXT: %[[R0:.*]] = sdy.reshard %arg0 <@mesh, [{"a"}, {}]> : tensor<8x4xf32>
// CHECK-NEXT: %[[RS1:.*]] = stablehlo.reshape %[[R0]] : (tensor<8x4xf32>) -> tensor<8x2x2xf32>
// CHECK-NEXT: %[[RS2:.*]] = stablehlo.reshape %[[RS1]] : (tensor<8x2x2xf32>) -> tensor<16x2xf32>
// CHECK-NEXT: return %[[R0]], %[[RS2]] : tensor<8x4xf32>, tensor<16x2xf32>
func.func @reshard_cse_with_shape_2(%arg0: tensor<8x4xf32>)
  -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
      tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {}]> : tensor<8x4xf32>
  %1 = stablehlo.reshape %arg0 : (tensor<8x4xf32>) -> tensor<8x2x2xf32>
  %2 = stablehlo.reshape %1 : (tensor<8x2x2xf32>) -> tensor<16x2xf32>
  %3 = sdy.reshard %2 <@mesh, [{"a"}, {}]> : tensor<16x2xf32>
  return %0, %3 : tensor<8x4xf32>, tensor<16x2xf32>
}

// CHECK-LABEL: func @reshard_cse_with_reshape_not_redundant_reshard
// CHECK-NEXT: %[[R0:.*]] = sdy.reshard %arg0 <@mesh, [{"a"}]> : tensor<16xf32>
// CHECK-NEXT: %[[RS:.*]] = stablehlo.reshape %arg0 : (tensor<16xf32>) -> tensor<4x4xf32>
// CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[RS]] <@mesh, [{}, {"a"}]> : tensor<4x4xf32>
// CHECK-NEXT: return %[[R0]], %[[R1]] : tensor<16xf32>, tensor<4x4xf32>
func.func @reshard_cse_with_reshape_not_redundant_reshard(%arg0: tensor<16xf32>)
  -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
      tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}) {
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}]> : tensor<16xf32>
  %1 = stablehlo.reshape %arg0 : (tensor<16xf32>) -> tensor<4x4xf32>
  %2 = sdy.reshard %1 <@mesh, [{}, {"a"}]> : tensor<4x4xf32>
  return %0, %2 : tensor<16xf32>, tensor<4x4xf32>
}

// CHECK-LABEL: func @reshard_cse_transpose_non_sharded_dims
// CHECK-NEXT: %[[R0:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}]> : tensor<8x4x16xf32>
// CHECK-NEXT: %[[T:.*]] = stablehlo.transpose %[[R0]], dims = [1, 0, 2] : (tensor<8x4x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: return %[[R0]], %[[T]] : tensor<8x4x16xf32>, tensor<4x8x16xf32>
func.func @reshard_cse_transpose_non_sharded_dims(
  %arg0: tensor<8x4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {"b"}]>})
  -> (tensor<8x4x16xf32>, tensor<4x8x16xf32>) {
  %0 = sdy.reshard %arg0 <@mesh, [{}, {}, {}]> : tensor<8x4x16xf32>
  %1 = stablehlo.transpose %arg0, dims = [1, 0, 2] : (tensor<8x4x16xf32>) -> tensor<4x8x16xf32>
  %2 = sdy.reshard %1 <@mesh, [{}, {}, {}]> : tensor<4x8x16xf32>
  return %0, %2 : tensor<8x4x16xf32>, tensor<4x8x16xf32>
}

// CHECK-LABEL: func @reshard_cse_with_transpose_sharded_dims(
// CHECK-SAME: %[[ARG0:.*]]: tensor<8x4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {"b"}]>})
// CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{"a"}, {}, {}]> : tensor<8x4x16xf32>
// CHECK-NEXT: %[[T1:.*]] = stablehlo.transpose %[[R0]], dims = [1, 0, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}, {"b"}]>]>} : (tensor<8x4x16xf32>) -> tensor<4x8x16xf32>
// CHECK-NEXT: %[[T2:.*]] = stablehlo.transpose %[[T1]], dims = [0, 2, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}, {"a"}]>]>} : (tensor<4x8x16xf32>) -> tensor<4x16x8xf32>
// CHECK-NEXT: return %[[R0]], %[[T2]] : tensor<8x4x16xf32>, tensor<4x16x8xf32>
func.func @reshard_cse_with_transpose_sharded_dims(
  %arg0: tensor<8x4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {"b"}]>})
  -> (tensor<8x4x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}, {}]>},
      tensor<4x16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"a"}]>}) {
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {}, {}]> : tensor<8x4x16xf32>
  %1 = stablehlo.transpose %arg0, dims = [1, 0, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}, {"b"}]>]>}
    : (tensor<8x4x16xf32>) -> tensor<4x8x16xf32>
  %2 = stablehlo.transpose %1, dims = [0, 2, 1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}, {"a"}]>]>}
    : (tensor<4x8x16xf32>) -> tensor<4x16x8xf32>
  %3 = sdy.reshard %2 <@mesh, [{}, {}, {"a"}]> : tensor<4x16x8xf32>
  return %0, %3 : tensor<8x4x16xf32>, tensor<4x16x8xf32>
}

// CHECK-LABEL: func @reshard_cse_with_shape_and_transpose
// CHECK-NEXT: %[[RS1:.*]] = stablehlo.reshape %arg0 : (tensor<2x4x4xf32>) -> tensor<8x4xf32>
// CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[RS1]] <@mesh, [{"a"}, {}]> : tensor<8x4xf32>
// CHECK-NEXT: %[[RS2:.*]] = stablehlo.reshape %[[R1]] : (tensor<8x4xf32>) -> tensor<2x4x4xf32>
// CHECK-NEXT: %[[RS3:.*]] = stablehlo.reshape %[[RS2]] : (tensor<2x4x4xf32>) -> tensor<2x4x2x2xf32>
// CHECK-NEXT: %[[T:.*]] = stablehlo.transpose %[[RS3]], dims = [0, 1, 3, 2] : (tensor<2x4x2x2xf32>) -> tensor<2x4x2x2xf32>
// CHECK-NEXT: %[[RS4:.*]] = stablehlo.reshape %[[T]] : (tensor<2x4x2x2xf32>) -> tensor<2x4x4xf32>
// CHECK-NEXT: %[[RS5:.*]] = stablehlo.reshape %[[RS4]] : (tensor<2x4x4xf32>) -> tensor<32xf32>
// CHECK-NEXT: return %[[R1]], %[[RS5]] : tensor<8x4xf32>, tensor<32xf32>
func.func @reshard_cse_with_shape_and_transpose(%arg0: tensor<2x4x4xf32>)
  -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
      tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  %0 = stablehlo.reshape %arg0 : (tensor<2x4x4xf32>) -> tensor<8x4xf32>
  %1 = sdy.reshard %0 <@mesh, [{"a"}, {}]> : tensor<8x4xf32>
  %2 = stablehlo.reshape %arg0 : (tensor<2x4x4xf32>) -> tensor<2x4x2x2xf32>
  %3 = stablehlo.transpose %2, dims = [0, 1, 3, 2] : (tensor<2x4x2x2xf32>) -> tensor<2x4x2x2xf32>
  %4 = stablehlo.reshape %3 : (tensor<2x4x2x2xf32>) -> tensor<2x4x4xf32>
  %5 = stablehlo.reshape %4 : (tensor<2x4x4xf32>) -> tensor<32xf32>
  %6 = sdy.reshard %5 <@mesh, [{"a"}]> : tensor<32xf32>
  return %1, %6 : tensor<8x4xf32>, tensor<32xf32>
}

// This test swap the order of the two reshard in test
// @reshard_cse_with_shape_and_transpose, but the current implementation
// only accepts a reshard as a cache hit if the path from the root to that
// reshard contains no transposes.
//
// CHECK-LABEL: func @reshard_cse_with_shape_and_transpose_reshard_not_reused
// CHECK-NEXT: %[[R0:.*]] = stablehlo.reshape %arg0 : (tensor<2x4x4xf32>) -> tensor<2x4x2x2xf32>
// CHECK-NEXT: %[[T:.*]] = stablehlo.transpose %[[R0]], dims = [0, 1, 3, 2] : (tensor<2x4x2x2xf32>) -> tensor<2x4x2x2xf32>
// CHECK-NEXT: %[[R1:.*]] = stablehlo.reshape %[[T]] : (tensor<2x4x2x2xf32>) -> tensor<2x4x4xf32>
// CHECK-NEXT: %[[R2:.*]] = stablehlo.reshape %[[R1]] : (tensor<2x4x4xf32>) -> tensor<32xf32>
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[R2]] <@mesh, [{"a"}]> : tensor<32xf32>
// CHECK-NEXT: %[[R3:.*]] = stablehlo.reshape %arg0 : (tensor<2x4x4xf32>) -> tensor<8x4xf32>
// CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[R3]] <@mesh, [{"a"}, {}]> : tensor<8x4xf32>
// CHECK-NEXT: return %[[RESHARD1]], %[[RESHARD2]] : tensor<32xf32>, tensor<8x4xf32>
func.func @reshard_cse_with_shape_and_transpose_reshard_not_reused(%arg0: tensor<2x4x4xf32>)
  -> (tensor<32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>},
      tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  %0 = stablehlo.reshape %arg0 : (tensor<2x4x4xf32>) -> tensor<2x4x2x2xf32>
  %1 = stablehlo.transpose %0, dims = [0, 1, 3, 2] : (tensor<2x4x2x2xf32>) -> tensor<2x4x2x2xf32>
  %2 = stablehlo.reshape %1 : (tensor<2x4x2x2xf32>) -> tensor<2x4x4xf32>
  %3 = stablehlo.reshape %2 : (tensor<2x4x4xf32>) -> tensor<32xf32>
  %4 = sdy.reshard %3 <@mesh, [{"a"}]> : tensor<32xf32>

  %5 = stablehlo.reshape %arg0 : (tensor<2x4x4xf32>) -> tensor<8x4xf32>
  %6 = sdy.reshard %5 <@mesh, [{"a"}, {}]> : tensor<8x4xf32>

  return %4, %6 : tensor<32xf32>, tensor<8x4xf32>
}

// CHECK-LABEL: func @reshard_cse_with_reshape_and_transpose_not_redundant_reshard
// CHECK-NEXT: %[[RS1:.*]] = stablehlo.reshape %arg0 : (tensor<2x4x4xf32>) -> tensor<8x4xf32>
// CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[RS1]] <@mesh, [{"a"}, {}]> : tensor<8x4xf32>
// CHECK-NEXT: %[[T:.*]] = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<2x4x4xf32>) -> tensor<4x4x2xf32>
// CHECK-NEXT: %[[RS2:.*]] = stablehlo.reshape %[[T]] : (tensor<4x4x2xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[R2:.*]] = sdy.reshard %[[RS2]] <@mesh, [{}, {"a"}]> : tensor<4x8xf32>
// CHECK-NEXT: return %[[R1]], %[[R2]] : tensor<8x4xf32>, tensor<4x8xf32>
func.func @reshard_cse_with_reshape_and_transpose_not_redundant_reshard(%arg0: tensor<2x4x4xf32>)
  -> (tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
      tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}) {
  %0 = stablehlo.reshape %arg0 : (tensor<2x4x4xf32>) -> tensor<8x4xf32>
  %1 = sdy.reshard %0 <@mesh, [{"a"}, {}]> : tensor<8x4xf32>
  %2 = stablehlo.transpose %arg0, dims = [2, 1, 0] : (tensor<2x4x4xf32>) -> tensor<4x4x2xf32>
  %3 = stablehlo.reshape %2 : (tensor<4x4x2xf32>) -> tensor<4x8xf32>
  %4 = sdy.reshard %3 <@mesh, [{}, {"a"}]> : tensor<4x8xf32>
  return %1, %4 : tensor<8x4xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @reshard_cse_different_blocks
func.func @reshard_cse_different_blocks(%arg0: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  // CHECK: %[[R0:.*]] = sdy.reshard %arg0 <@mesh, [{"a"}]>
  %0 = sdy.reshard %arg0 <@mesh, [{"a"}]> : tensor<8xf32>
  %1 = sdy.named_computation<"my_comp">(%arg0) in_shardings=[<@mesh, [{"a"}]>]
      out_shardings=[<@mesh, [{"a"}]>] (%arg1: tensor<8xf32>) {
    // CHECK: %[[R_INNER:.*]] = sdy.reshard %arg1 <@mesh, [{"a"}]>
    // CHECK: sdy.return %[[R_INNER]]
    %2 = sdy.reshard %arg1 <@mesh, [{"a"}]> : tensor<8xf32>
    sdy.return %2 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  return %0, %1 : tensor<8xf32>, tensor<8xf32>
}
