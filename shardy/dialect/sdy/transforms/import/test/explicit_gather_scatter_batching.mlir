// RUN: sdy_opt %s -sdy-explicit-gather-scatter-batching | FileCheck %s

// ===== Test 1: Basic row-wise gather with iota in concatenated indices =====
// This pattern typically arises from row-wise indexing in JAX using a batch iota,
// such as: arr.at[jnp.arange(B), offset] or arr.at[jax.lax.iota(jnp.int32, B), offset]
// Both emit stablehlo.iota concatenated with the column index.

// CHECK-LABEL: func @gather_iota_concat_batch_dim
func.func @gather_iota_concat_batch_dim(
    %operand: tensor<4x8xf32>, %offset: tensor<4x1xi32>)
    -> tensor<4x1xf32> {
  // The iota generates [0, 1, 2, 3] for the batch dimension.
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  // Concatenate iota with the actual offset to form start_indices.
  %indices = stablehlo.concatenate %iota, %offset, dim = 1
      : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
  // CHECK: "stablehlo.gather"
  // CHECK-SAME: offset_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: index_vector_dim = 1
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x8xf32>, tensor<4x2xi32>) -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// ===== Test 2: Iota with reshape through concat =====
// Pattern: iota -> reshape -> concat with other indices.

// CHECK-LABEL: func @gather_iota_reshaped_concat
func.func @gather_iota_reshaped_concat(
    %operand: tensor<4x8x16xf32>, %other: tensor<4x1xi32>)
    -> tensor<4x1xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<4xi32>
  %reshaped = stablehlo.reshape %iota : (tensor<4xi32>) -> tensor<4x1xi32>
  %indices = stablehlo.concatenate %reshaped, %other, dim = 1
      : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
  // CHECK: "stablehlo.gather"
  // CHECK-SAME: offset_dims = [1]
  // CHECK-SAME: collapsed_slice_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: index_vector_dim = 1
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x8x16xf32>, tensor<4x2xi32>) -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// ===== Test 3: Iota at non-first position in concat =====
// The iota component is at the end of the concatenated indices.

// CHECK-LABEL: func @gather_iota_at_end_of_concat
func.func @gather_iota_at_end_of_concat(
    %operand: tensor<4x8xf32>, %col_idx: tensor<4x1xi32>)
    -> tensor<4x1xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  // Iota is second in concat (maps to start_index_map[1] = 0).
  %indices = stablehlo.concatenate %col_idx, %iota, dim = 1
      : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
  // CHECK: "stablehlo.gather"
  // CHECK-SAME: offset_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: index_vector_dim = 1
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [1, 0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x8xf32>, tensor<4x2xi32>) -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// ===== Test 4: Iota in middle of three-element concat =====
// The iota component is in the middle position of the concatenated indices.

// CHECK-LABEL: func @gather_iota_in_middle_of_concat
func.func @gather_iota_in_middle_of_concat(
    %operand: tensor<4x8x16xf32>,
    %idx0: tensor<4x1xi32>, %idx1: tensor<4x1xi32>)
    -> tensor<4x1xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  // Iota is in the middle (position 1), mapping to start_index_map[1] = 0.
  %indices = stablehlo.concatenate %idx0, %iota, %idx1, dim = 1
      : (tensor<4x1xi32>, tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x3xi32>
  // CHECK: "stablehlo.gather"
  // CHECK-SAME: offset_dims = [1]
  // CHECK-SAME: collapsed_slice_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1, 2]
  // CHECK-SAME: index_vector_dim = 1
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0, 1],
      start_index_map = [1, 0, 2],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x8x16xf32>, tensor<4x3xi32>) -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// ===== Test 5: Iota with broadcast through concat =====
// Pattern: iota -> broadcast_in_dim -> concat

// CHECK-LABEL: func @gather_iota_broadcast_concat
func.func @gather_iota_broadcast_concat(
    %operand: tensor<4x8xf32>, %other: tensor<4x1xi32>)
    -> tensor<4x1xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<4xi32>
  %broadcast = "stablehlo.broadcast_in_dim"(%iota) {
    broadcast_dimensions = array<i64: 0>
  } : (tensor<4xi32>) -> tensor<4x1xi32>
  %indices = stablehlo.concatenate %broadcast, %other, dim = 1
      : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
  // CHECK: "stablehlo.gather"
  // CHECK-SAME: offset_dims = [1]
  // CHECK-SAME: operand_batching_dims = [0]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-SAME: start_index_map = [1]
  // CHECK-SAME: index_vector_dim = 1
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x8xf32>, tensor<4x2xi32>) -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// ===== Test 6: No iota (no transformation) =====
// When start_indices are not from an iota, the pass should not fire.

// CHECK-LABEL: func @gather_no_iota_no_transform
func.func @gather_no_iota_no_transform(
    %operand: tensor<4x8xf32>, %indices: tensor<4x2xi32>)
    -> tensor<4x1xf32> {
  // CHECK: "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [0]
  // CHECK-SAME: start_index_map = [0, 1]
  // CHECK-NOT: operand_batching_dims
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x8xf32>, tensor<4x2xi32>) -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// ===== Test 7: Mismatched batch dim sizes (no transformation) =====
// operand dim 0 has size 4 but indices dim 0 has size 3.

// CHECK-LABEL: func @gather_mismatched_batch_size_no_transform
func.func @gather_mismatched_batch_size_no_transform(
    %operand: tensor<4x8xf32>, %offset: tensor<3x1xi32>)
    -> tensor<3x1xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<3x1xi32>
  %indices = stablehlo.concatenate %iota, %offset, dim = 1
      : (tensor<3x1xi32>, tensor<3x1xi32>) -> tensor<3x2xi32>
  // CHECK: "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [0]
  // CHECK-SAME: start_index_map = [0, 1]
  // CHECK-NOT: operand_batching_dims
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x8xf32>, tensor<3x2xi32>) -> tensor<3x1xf32>
  return %result : tensor<3x1xf32>
}

// ===== Test 8: Already has explicit batching dims (no-op) =====

// CHECK-LABEL: func @gather_already_explicit_batching
func.func @gather_already_explicit_batching(
    %operand: tensor<4x4x8xf32>, %offset: tensor<4x1xi32>)
    -> tensor<4x1xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %indices = stablehlo.concatenate %iota, %offset, dim = 1
      : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
  // CHECK: "stablehlo.gather"
  // CHECK-SAME: operand_batching_dims = [1]
  // CHECK-SAME: start_indices_batching_dims = [0]
  // CHECK-NOT: operand_batching_dims = [0, 1]
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      operand_batching_dims = [1],
      start_indices_batching_dims = [0],
      start_index_map = [0, 2],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x4x8xf32>, tensor<4x2xi32>) -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// ===== Test 9: Index vector dim size 1 with iota (degenerate gather -> broadcast) =====
// Single index component originating from iota is rewritten to broadcast_in_dim.

// CHECK-LABEL: func @gather_single_index_dim_iota_to_broadcast
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<4x8xf32>) -> tensor<4x8xf32>
func.func @gather_single_index_dim_iota_to_broadcast(
    %operand: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  // The canonicalizer would replace this broadcast_in_dim with its operand.
  //
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OPERAND]], dims = [0, 1] : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[BCAST]] : tensor<4x8xf32>
  %result = "stablehlo.gather"(%operand, %iota) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 8>,
    indices_are_sorted = false
  } : (tensor<4x8xf32>, tensor<4x1xi32>) -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// ===== Test 9b: Index vector dim size 1 with non-iota (no transformation) =====
// If the single index component is not an iota, the pass should bail out.

// CHECK-LABEL: func @gather_single_index_dim_no_iota_no_transform
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<4x8xf32>, %[[OFFSET:.*]]: tensor<4x1xi32>) -> tensor<4x8xf32>
func.func @gather_single_index_dim_no_iota_no_transform(
    %operand: tensor<4x8xf32>, %offset: tensor<4x1xi32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[RES:.*]] = "stablehlo.gather"(%[[OPERAND]], %[[OFFSET]])
  // CHECK-SAME: collapsed_slice_dims = [0]
  // CHECK-SAME: start_index_map = [0]
  // CHECK-NOT: operand_batching_dims
  %result = "stablehlo.gather"(%operand, %offset) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 8>,
    indices_are_sorted = false
  } : (tensor<4x8xf32>, tensor<4x1xi32>) -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// ===== Test 10: dim 0 not in collapsed_slice_dims (no transformation) =====
// Explicit batch dims require slice size 1 along the batch dimension, which
// means the dimension must be in collapsed_slice_dims. If it isn't, bail out.

// CHECK-LABEL: func @gather_dim0_not_collapsed_no_transform
func.func @gather_dim0_not_collapsed_no_transform(
    %operand: tensor<4x8xf32>, %offset: tensor<4x1xi32>)
    -> tensor<4x1xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %indices = stablehlo.concatenate %iota, %offset, dim = 1
      : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>
  // CHECK: "stablehlo.gather"
  // CHECK-SAME: collapsed_slice_dims = [1]
  // CHECK-SAME: start_index_map = [0, 1]
  // CHECK-NOT: operand_batching_dims
  %result = "stablehlo.gather"(%operand, %indices) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [1],
      start_index_map = [0, 1],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 1>,
    indices_are_sorted = false
  } : (tensor<4x8xf32>, tensor<4x2xi32>) -> tensor<4x1xf32>
  return %result : tensor<4x1xf32>
}

// ===== Test 11: Scatter explicit batching extraction =====
// Validates that stablehlo.scatter extracts batch iotas correctly.

// CHECK-LABEL: func @scatter_batch_dim_test
func.func @scatter_batch_dim_test(
    %operand: tensor<32x8xi32>, %offset: tensor<32x1xi32>, %updates: tensor<32xi32>) -> tensor<32x8xi32> {
  %iota = stablehlo.iota dim = 0 : tensor<32x1xi32>
  %indices = stablehlo.concatenate %iota, %offset, dim = 1 : (tensor<32x1xi32>, tensor<32x1xi32>) -> tensor<32x2xi32>

  // CHECK: "stablehlo.scatter"
  // CHECK-SAME: inserted_window_dims = [1]
  // CHECK-SAME: input_batching_dims = [0]
  // CHECK-SAME: scatter_indices_batching_dims = [0]
  // CHECK-SAME: scatter_dims_to_operand_dims = [1]
  // CHECK-SAME: index_vector_dim = 1
  %result = "stablehlo.scatter"(%operand, %indices, %updates) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1>,
    unique_indices = false
  }> ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    stablehlo.return %arg1 : tensor<i32>
  }) : (tensor<32x8xi32>, tensor<32x2xi32>, tensor<32xi32>) -> tensor<32x8xi32>

  return %result : tensor<32x8xi32>
}

// ===== Test 12: dim 0 not in inserted_window_dims (scatter, no transformation) =====
// If 0 is not in inserted_window_dims, we can't promote it to a batch dim.

// CHECK-LABEL: func @scatter_dim0_not_inserted_no_transform
func.func @scatter_dim0_not_inserted_no_transform(
    %operand: tensor<4x8xi32>, %offset: tensor<4x1xi32>,
    %updates: tensor<4x4xi32>) -> tensor<4x8xi32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %indices = stablehlo.concatenate %iota, %offset, dim = 1 : (tensor<4x1xi32>, tensor<4x1xi32>) -> tensor<4x2xi32>

  // CHECK: "stablehlo.scatter"
  // CHECK-SAME: inserted_window_dims = [1]
  // CHECK-SAME: scatter_dims_to_operand_dims = [0, 1]
  // CHECK-NOT: input_batching_dims
  %result = "stablehlo.scatter"(%operand, %indices, %updates) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1>,
    unique_indices = false
  }> ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    stablehlo.return %arg1 : tensor<i32>
  }) : (tensor<4x8xi32>, tensor<4x2xi32>, tensor<4x4xi32>) -> tensor<4x8xi32>

  return %result : tensor<4x8xi32>
}

// ===== Test 13: Degenerate gather to broadcast_in_dim =====
// All mapped indices originate from an identity iota. The gather is rewritten directly
// to stablehlo.broadcast_in_dim.

// CHECK-LABEL: func @gather_degenerate_iota_to_broadcast
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<4xf32>) -> tensor<4x8xf32>
func.func @gather_degenerate_iota_to_broadcast(%operand: tensor<4xf32>) -> tensor<4x8xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x8xi32>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OPERAND]], dims = [0] : (tensor<4xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[BCAST]] : tensor<4x8xf32>
  %result = "stablehlo.gather"(%operand, %iota) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 2>,
    slice_sizes = array<i64: 1>,
    indices_are_sorted = false
  } : (tensor<4xf32>, tensor<4x8xi32>) -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// ===== Test 14: Degenerate scatter to reduce + add =====
// All mapped scatter indices originate from identity iota, with extra non-batch dimension.
// The update is reduced across the extra dimension and added to the operand.

// CHECK-LABEL: func @scatter_degenerate_iota_to_reduce_add
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<8x2x2xi32>, %[[UPDATE:.*]]: tensor<8x4x2x2xi32>) -> tensor<8x2x2xi32>
func.func @scatter_degenerate_iota_to_reduce_add(
    %operand: tensor<8x2x2xi32>, %update: tensor<8x4x2x2xi32>) -> tensor<8x2x2xi32> {
  %iota = stablehlo.iota dim = 1 : tensor<1x8x4xi32>
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[REDUCE:.*]] = stablehlo.reduce(%[[UPDATE]] init: %[[CST]]) applies stablehlo.add across dimensions = [1] : (tensor<8x4x2x2xi32>, tensor<i32>) -> tensor<8x2x2xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[OPERAND]], %[[REDUCE]] : tensor<8x2x2xi32>
  // CHECK-NEXT: return %[[ADD]] : tensor<8x2x2xi32>
  %result = "stablehlo.scatter"(%operand, %iota, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 0>,
    unique_indices = false
  }> ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %sum = stablehlo.add %arg0, %arg1 : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) : (tensor<8x2x2xi32>, tensor<1x8x4xi32>, tensor<8x4x2x2xi32>) -> tensor<8x2x2xi32>
  return %result : tensor<8x2x2xi32>
}

// ===== Test 15: Degenerate scatter to add directly (no extra non-batch dim) =====
// All mapped scatter indices originate from identity iota, with no extra non-batch dimension.
// The update is combined with the operand directly.

// CHECK-LABEL: func @scatter_degenerate_iota_to_add_direct
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<8x2x2xi32>, %[[UPDATE:.*]]: tensor<8x2x2xi32>) -> tensor<8x2x2xi32>
func.func @scatter_degenerate_iota_to_add_direct(
    %operand: tensor<8x2x2xi32>, %update: tensor<8x2x2xi32>) -> tensor<8x2x2xi32> {
  %iota = stablehlo.iota dim = 0 : tensor<8x1xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[OPERAND]], %[[UPDATE]] : tensor<8x2x2xi32>
  // CHECK-NEXT: return %[[ADD]] : tensor<8x2x2xi32>
  %result = "stablehlo.scatter"(%operand, %iota, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1, 2],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    unique_indices = false
  }> ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %sum = stablehlo.add %arg0, %arg1 : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) : (tensor<8x2x2xi32>, tensor<8x1xi32>, tensor<8x2x2xi32>) -> tensor<8x2x2xi32>
  return %result : tensor<8x2x2xi32>
}

// ===== Test 16: Degenerate scatter with float max reduction =====
// Verifies reduction constant for float max is -infinity.

// CHECK-LABEL: func @scatter_degenerate_float_max
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<8x2xf32>, %[[UPDATE:.*]]: tensor<8x4x2xf32>) -> tensor<8x2xf32>
func.func @scatter_degenerate_float_max(
    %operand: tensor<8x2xf32>, %update: tensor<8x4x2xf32>) -> tensor<8x2xf32> {
  %iota = stablehlo.iota dim = 1 : tensor<1x8x4xi32>
  // CHECK-NEXT: %[[CST:.*]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
  // CHECK-NEXT: %[[REDUCE:.*]] = stablehlo.reduce(%[[UPDATE]] init: %[[CST]]) applies stablehlo.maximum across dimensions = [1] : (tensor<8x4x2xf32>, tensor<f32>) -> tensor<8x2xf32>
  // CHECK-NEXT: %[[MAX:.*]] = stablehlo.maximum %[[OPERAND]], %[[REDUCE]] : tensor<8x2xf32>
  // CHECK-NEXT: return %[[MAX]] : tensor<8x2xf32>
  %result = "stablehlo.scatter"(%operand, %iota, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 0>,
    unique_indices = false
  }> ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = stablehlo.maximum %arg0, %arg1 : tensor<f32>
    stablehlo.return %max : tensor<f32>
  }) : (tensor<8x2xf32>, tensor<1x8x4xi32>, tensor<8x4x2xf32>) -> tensor<8x2xf32>
  return %result : tensor<8x2xf32>
}

// ===== Test 17: Degenerate scatter with operandDim = 1 (transposition) =====
// Verifies that when operandDim != 0, dimensions are transposed to align with operand.

// CHECK-LABEL: func @scatter_degenerate_dim1_transpose
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<2x4xi32>, %[[UPDATE:.*]]: tensor<4x2xi32>) -> tensor<2x4xi32>
func.func @scatter_degenerate_dim1_transpose(
    %operand: tensor<2x4xi32>, %update: tensor<4x2xi32>) -> tensor<2x4xi32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  // CHECK-NEXT: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[UPDATE]], dims = [1, 0] : (tensor<4x2xi32>) -> tensor<2x4xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[OPERAND]], %[[TRANSPOSE]] : tensor<2x4xi32>
  // CHECK-NEXT: return %[[ADD]] : tensor<2x4xi32>
  %result = "stablehlo.scatter"(%operand, %iota, %update) <{
    indices_are_sorted = false,
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [1],
      scatter_dims_to_operand_dims = [1],
      index_vector_dim = 1>,
    unique_indices = false
  }> ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %sum = stablehlo.add %arg0, %arg1 : tensor<i32>
    stablehlo.return %sum : tensor<i32>
  }) : (tensor<2x4xi32>, tensor<4x1xi32>, tensor<4x2xi32>) -> tensor<2x4xi32>
  return %result : tensor<2x4xi32>
}

// ===== Test 18: Degenerate gather with iota wrapped in convert =====
// Verifies that stablehlo.convert (e.g. i32 -> i64) in index chain is traced to iota.

// CHECK-LABEL: func @gather_iota_with_convert
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<4x8xf32>) -> tensor<4x8xf32>
func.func @gather_iota_with_convert(
    %operand: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %conv = stablehlo.convert %iota : (tensor<4x1xi32>) -> tensor<4x1xi64>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OPERAND]], dims = [0, 1] : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[BCAST]] : tensor<4x8xf32>
  %result = "stablehlo.gather"(%operand, %conv) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 8>,
    indices_are_sorted = false
  } : (tensor<4x8xf32>, tensor<4x1xi64>) -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// ===== Test 19: Degenerate gather with nested single-input concatenate =====
// Verifies that a single-input concatenate nested before a convert is traced to iota.

// CHECK-LABEL: func @gather_iota_nested_single_concat
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<4x8xf32>) -> tensor<4x8xf32>
func.func @gather_iota_nested_single_concat(
    %operand: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %iota = stablehlo.iota dim = 0 : tensor<4x1xi32>
  %concat = stablehlo.concatenate %iota, dim = 1 : (tensor<4x1xi32>) -> tensor<4x1xi32>
  %conv = stablehlo.convert %concat : (tensor<4x1xi32>) -> tensor<4x1xi64>
  // CHECK-NEXT: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OPERAND]], dims = [0, 1] : (tensor<4x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[BCAST]] : tensor<4x8xf32>
  %result = "stablehlo.gather"(%operand, %conv) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1, 8>,
    indices_are_sorted = false
  } : (tensor<4x8xf32>, tensor<4x1xi64>) -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// ===== Test 20: Scatter with partial window size is not rewritten to combiner =====
// Verifies that a scatter whose update window size does not match the operand
// dimension size is not rewritten into a combiner reduction.

// CHECK-LABEL: func @scatter_iota_partial_window_not_rewritten
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<4x8xf32>, %[[UPDATES:.*]]: tensor<4x4xf32>) -> tensor<4x8xf32>
func.func @scatter_iota_partial_window_not_rewritten(
    %operand: tensor<4x8xf32>, %updates: tensor<4x4xf32>) -> tensor<4x8xf32> {
  %indices = stablehlo.iota dim = 0 : tensor<4x1xi32>
  // CHECK: %[[SCATTER:.*]] = "stablehlo.scatter"(%[[OPERAND]],
  // CHECK: return %[[SCATTER]] : tensor<4x8xf32>
  %result = "stablehlo.scatter"(%operand, %indices, %updates) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %add = stablehlo.add %arg0, %arg1 : tensor<f32>
    stablehlo.return %add : tensor<f32>
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    indices_are_sorted = false,
    unique_indices = false
  } : (tensor<4x8xf32>, tensor<4x1xi32>, tensor<4x4xf32>) -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}
