// RUN: sdy_opt %s -sdy-convert-global-to-local -split-input-file -verify-diagnostics | FileCheck %s

// Tests communication-free export lowering for stablehlo.dynamic_update_slice.

// CHECK: sdy.mesh @mesh_2 = <["x"=2]>
sdy.mesh @mesh_2 = <["x"=2]>

// Sliced dim 0 is partitioned over "x"=2 with update size 1 and a dynamic
// index. Verifies that the start index is adjusted to shard-local coordinates
// and the update is applied only on the shard that owns the slice.
// CHECK-LABEL: func @single_element_slice_dynamic_index
// CHECK-SAME:    (%[[OPERAND:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
// CHECK-SAME:     %[[UPDATE:.*]]: tensor<1x8xf32>, %[[IDX0:.*]]: tensor<i32>, %[[IDX1:.*]]: tensor<i32>)
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})

// Clamps global start index to valid range.
// CHECK-NEXT:  %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:  %[[MAX_START:.*]] = stablehlo.constant dense<15> : tensor<i32>
// CHECK-NEXT:  %[[CLAMPED:.*]] = stablehlo.clamp %[[ZERO]], %[[IDX0]], %[[MAX_START]] : tensor<i32>

// Looks up shard start offset from partition_id.
// CHECK-NEXT:  %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:  %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
// CHECK-NEXT:  %[[OFFSET_TABLE:.*]] = stablehlo.constant dense<[0, 8]> : tensor<2xi64>
// CHECK-NEXT:  %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[OFFSET_TABLE]], %[[PID_I64]], sizes = [1] : (tensor<2xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:  %[[OFFSET_I64:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:  %[[OFFSET:.*]] = stablehlo.convert %[[OFFSET_I64]] : (tensor<i64>) -> tensor<i32>

// Checks if this shard owns the slice.
// CHECK-NEXT:  %[[SLICE_SIZE:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:  %[[SLICE_END:.*]] = stablehlo.add %[[CLAMPED]], %[[SLICE_SIZE]] : tensor<i32>
// CHECK-NEXT:  %[[SHARD_SIZE:.*]] = stablehlo.constant dense<8> : tensor<i32>
// CHECK-NEXT:  %[[SHARD_END:.*]] = stablehlo.add %[[OFFSET]], %[[SHARD_SIZE]] : tensor<i32>
// CHECK-NEXT:  %[[GE:.*]] = stablehlo.compare GE, %[[CLAMPED]], %[[OFFSET]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:  %[[LE:.*]] = stablehlo.compare LE, %[[SLICE_END]], %[[SHARD_END]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:  %[[IS_OWNER:.*]] = stablehlo.and %[[GE]], %[[LE]] : tensor<i1>

// Converts global start index to shard-local index.
// CHECK-NEXT:  %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %[[OFFSET]] : tensor<i32>
// CHECK-NEXT:  %[[SAFE_IDX:.*]] = stablehlo.select %[[IS_OWNER]], %[[LOCAL_IDX]], %[[ZERO]] : tensor<i1>, tensor<i32>

// Performs local dynamic-update-slice.
// CHECK-NEXT:  %[[LOCAL_DUS:.*]] = stablehlo.dynamic_update_slice %[[OPERAND]], %[[UPDATE]], %[[SAFE_IDX]], %[[IDX1]] : (tensor<8x8xf32>, tensor<1x8xf32>, tensor<i32>, tensor<i32>) -> tensor<8x8xf32>

// Applies the update only if this shard owns the slice.
// CHECK-NEXT:  %[[MASK:.*]] = stablehlo.broadcast_in_dim %[[IS_OWNER]], dims = [] : (tensor<i1>) -> tensor<8x8xi1>
// CHECK-NEXT:  %[[RES:.*]] = stablehlo.select %[[MASK]], %[[LOCAL_DUS]], %[[OPERAND]] : tensor<8x8xi1>, tensor<8x8xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<8x8xf32>
func.func @single_element_slice_dynamic_index(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
                                              %arg1: tensor<1x8xf32>,
                                              %arg2: tensor<i32>,
                                              %arg3: tensor<i32>) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<16x8xf32>, tensor<1x8xf32>, tensor<i32>, tensor<i32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// -----

sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// Both sliced dims 0 and 1 are partitioned over "x"=2 and "y"=2 with update
// size 1. Verifies that each dimension adjusts its start index to shard-local
// coordinates, and the update is applied only when the shard owns the slice
// along both dimensions.
// CHECK-LABEL: func @two_partitioned_slice_dims
// CHECK-SAME:    (%[[OPERAND:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>},
// CHECK-SAME:     %[[UPDATE:.*]]: tensor<1x1xf32>, %[[IDX0:.*]]: tensor<i32>, %[[IDX1:.*]]: tensor<i32>)
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>})

// Clamps global start index for dim 0 to valid range.
// CHECK-NEXT:  %[[ZERO_0:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:  %[[MAX_START_0:.*]] = stablehlo.constant dense<15> : tensor<i32>
// CHECK-NEXT:  %[[CLAMPED_0:.*]] = stablehlo.clamp %[[ZERO_0]], %[[IDX0]], %[[MAX_START_0]] : tensor<i32>

// Looks up shard start offset for dim 0 from partition_id.
// CHECK-NEXT:  %[[PID_0:.*]] = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:  %[[PID_I64_0:.*]] = stablehlo.convert %[[PID_0]] : (tensor<ui32>) -> tensor<i64>
// CHECK-NEXT:  %[[OFFSET_TABLE_0:.*]] = stablehlo.constant dense<[0, 0, 8, 8]> : tensor<4xi64>
// CHECK-NEXT:  %[[OFFSET_SLICE_0:.*]] = stablehlo.dynamic_slice %[[OFFSET_TABLE_0]], %[[PID_I64_0]], sizes = [1] : (tensor<4xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:  %[[OFFSET_I64_0:.*]] = stablehlo.reshape %[[OFFSET_SLICE_0]] : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:  %[[OFFSET_0:.*]] = stablehlo.convert %[[OFFSET_I64_0]] : (tensor<i64>) -> tensor<i32>

// Checks if this shard owns the slice along dim 0.
// CHECK-NEXT:  %[[SLICE_SIZE_0:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:  %[[SLICE_END_0:.*]] = stablehlo.add %[[CLAMPED_0]], %[[SLICE_SIZE_0]] : tensor<i32>
// CHECK-NEXT:  %[[SHARD_SIZE_0:.*]] = stablehlo.constant dense<8> : tensor<i32>
// CHECK-NEXT:  %[[SHARD_END_0:.*]] = stablehlo.add %[[OFFSET_0]], %[[SHARD_SIZE_0]] : tensor<i32>
// CHECK-NEXT:  %[[GE_0:.*]] = stablehlo.compare GE, %[[CLAMPED_0]], %[[OFFSET_0]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:  %[[LE_0:.*]] = stablehlo.compare LE, %[[SLICE_END_0]], %[[SHARD_END_0]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:  %[[IS_OWNER_0:.*]] = stablehlo.and %[[GE_0]], %[[LE_0]] : tensor<i1>

// Converts global start index to shard-local index for dim 0.
// CHECK-NEXT:  %[[LOCAL_IDX_0:.*]] = stablehlo.subtract %[[CLAMPED_0]], %[[OFFSET_0]] : tensor<i32>
// CHECK-NEXT:  %[[SAFE_IDX_0:.*]] = stablehlo.select %[[IS_OWNER_0]], %[[LOCAL_IDX_0]], %[[ZERO_0]] : tensor<i1>, tensor<i32>

// Clamps global start index for dim 1 to valid range.
// CHECK-NEXT:  %[[ZERO_1:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:  %[[MAX_START_1:.*]] = stablehlo.constant dense<15> : tensor<i32>
// CHECK-NEXT:  %[[CLAMPED_1:.*]] = stablehlo.clamp %[[ZERO_1]], %[[IDX1]], %[[MAX_START_1]] : tensor<i32>

// Looks up shard start offset for dim 1 from partition_id.
// CHECK-NEXT:  %[[PID_1:.*]] = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:  %[[PID_I64_1:.*]] = stablehlo.convert %[[PID_1]] : (tensor<ui32>) -> tensor<i64>
// CHECK-NEXT:  %[[OFFSET_TABLE_1:.*]] = stablehlo.constant dense<[0, 8, 0, 8]> : tensor<4xi64>
// CHECK-NEXT:  %[[OFFSET_SLICE_1:.*]] = stablehlo.dynamic_slice %[[OFFSET_TABLE_1]], %[[PID_I64_1]], sizes = [1] : (tensor<4xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:  %[[OFFSET_I64_1:.*]] = stablehlo.reshape %[[OFFSET_SLICE_1]] : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:  %[[OFFSET_1:.*]] = stablehlo.convert %[[OFFSET_I64_1]] : (tensor<i64>) -> tensor<i32>

// Checks if this shard owns the slice along dim 1.
// CHECK-NEXT:  %[[SLICE_SIZE_1:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:  %[[SLICE_END_1:.*]] = stablehlo.add %[[CLAMPED_1]], %[[SLICE_SIZE_1]] : tensor<i32>
// CHECK-NEXT:  %[[SHARD_SIZE_1:.*]] = stablehlo.constant dense<8> : tensor<i32>
// CHECK-NEXT:  %[[SHARD_END_1:.*]] = stablehlo.add %[[OFFSET_1]], %[[SHARD_SIZE_1]] : tensor<i32>
// CHECK-NEXT:  %[[GE_1:.*]] = stablehlo.compare GE, %[[CLAMPED_1]], %[[OFFSET_1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:  %[[LE_1:.*]] = stablehlo.compare LE, %[[SLICE_END_1]], %[[SHARD_END_1]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:  %[[IS_OWNER_1:.*]] = stablehlo.and %[[GE_1]], %[[LE_1]] : tensor<i1>

// Converts global start index to shard-local index for dim 1.
// CHECK-NEXT:  %[[LOCAL_IDX_1:.*]] = stablehlo.subtract %[[CLAMPED_1]], %[[OFFSET_1]] : tensor<i32>
// CHECK-NEXT:  %[[SAFE_IDX_1:.*]] = stablehlo.select %[[IS_OWNER_1]], %[[LOCAL_IDX_1]], %[[ZERO_1]] : tensor<i1>, tensor<i32>

// Performs local dynamic-update-slice.
// CHECK-NEXT:  %[[LOCAL_DUS:.*]] = stablehlo.dynamic_update_slice %[[OPERAND]], %[[UPDATE]], %[[SAFE_IDX_0]], %[[SAFE_IDX_1]] : (tensor<8x8xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<8x8xf32>

// Applies the update only if this shard owns the slice along both dimensions.
// CHECK-NEXT:  %[[ALL_OWNER:.*]] = stablehlo.and %[[IS_OWNER_0]], %[[IS_OWNER_1]] : tensor<i1>
// CHECK-NEXT:  %[[MASK:.*]] = stablehlo.broadcast_in_dim %[[ALL_OWNER]], dims = [] : (tensor<i1>) -> tensor<8x8xi1>
// CHECK-NEXT:  %[[RES:.*]] = stablehlo.select %[[MASK]], %[[LOCAL_DUS]], %[[OPERAND]] : tensor<8x8xi1>, tensor<8x8xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<8x8xf32>
func.func @two_partitioned_slice_dims(%arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>},
                                      %arg1: tensor<1x1xf32>,
                                      %arg2: tensor<i32>,
                                      %arg3: tensor<i32>) -> (tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {"y"}]>]>} : (tensor<16x16xf32>, tensor<1x1xf32>, tensor<i32>, tensor<i32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// -----

sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// Non-sliced dim 0 is partitioned over "x"=2 and sliced dim 1 is partitioned
// over "y"=2. Verifies that non-sliced dim 0 keeps its start index unchanged
// while sliced dim 1 adjusts its index and checks shard ownership.
// CHECK-LABEL: func @partitioned_non_sliced_dim
// CHECK-SAME:    (%[[OPERAND:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>},
// CHECK-SAME:     %[[UPDATE:.*]]: tensor<8x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}]>},
// CHECK-SAME:     %[[IDX0:.*]]: tensor<i32>, %[[IDX1:.*]]: tensor<i32>)
// CHECK-SAME:    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>})

// Clamps global start index for dim 1 to valid range.
// CHECK-NEXT:  %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<i32>
// CHECK-NEXT:  %[[MAX_START:.*]] = stablehlo.constant dense<31> : tensor<i32>
// CHECK-NEXT:  %[[CLAMPED:.*]] = stablehlo.clamp %[[ZERO]], %[[IDX1]], %[[MAX_START]] : tensor<i32>

// Looks up shard start offset for dim 1 from partition_id.
// CHECK-NEXT:  %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
// CHECK-NEXT:  %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
// CHECK-NEXT:  %[[OFFSET_TABLE:.*]] = stablehlo.constant dense<[0, 16, 0, 16]> : tensor<4xi64>
// CHECK-NEXT:  %[[OFFSET_SLICE:.*]] = stablehlo.dynamic_slice %[[OFFSET_TABLE]], %[[PID_I64]], sizes = [1] : (tensor<4xi64>, tensor<i64>) -> tensor<1xi64>
// CHECK-NEXT:  %[[OFFSET_I64:.*]] = stablehlo.reshape %[[OFFSET_SLICE]] : (tensor<1xi64>) -> tensor<i64>
// CHECK-NEXT:  %[[OFFSET:.*]] = stablehlo.convert %[[OFFSET_I64]] : (tensor<i64>) -> tensor<i32>

// Checks if this shard owns the slice along dim 1.
// CHECK-NEXT:  %[[SLICE_SIZE:.*]] = stablehlo.constant dense<1> : tensor<i32>
// CHECK-NEXT:  %[[SLICE_END:.*]] = stablehlo.add %[[CLAMPED]], %[[SLICE_SIZE]] : tensor<i32>
// CHECK-NEXT:  %[[SHARD_SIZE:.*]] = stablehlo.constant dense<16> : tensor<i32>
// CHECK-NEXT:  %[[SHARD_END:.*]] = stablehlo.add %[[OFFSET]], %[[SHARD_SIZE]] : tensor<i32>
// CHECK-NEXT:  %[[GE:.*]] = stablehlo.compare GE, %[[CLAMPED]], %[[OFFSET]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:  %[[LE:.*]] = stablehlo.compare LE, %[[SLICE_END]], %[[SHARD_END]] : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:  %[[IS_OWNER:.*]] = stablehlo.and %[[GE]], %[[LE]] : tensor<i1>

// Converts global start index to shard-local index for dim 1.
// CHECK-NEXT:  %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %[[OFFSET]] : tensor<i32>
// CHECK-NEXT:  %[[SAFE_IDX:.*]] = stablehlo.select %[[IS_OWNER]], %[[LOCAL_IDX]], %[[ZERO]] : tensor<i1>, tensor<i32>

// Performs local dynamic-update-slice (dim 0 index unchanged).
// CHECK-NEXT:  %[[LOCAL_DUS:.*]] = stablehlo.dynamic_update_slice %[[OPERAND]], %[[UPDATE]], %[[IDX0]], %[[SAFE_IDX]] : (tensor<8x16xf32>, tensor<8x1xf32>, tensor<i32>, tensor<i32>) -> tensor<8x16xf32>

// Applies the update only if this shard owns the slice.
// CHECK-NEXT:  %[[MASK:.*]] = stablehlo.broadcast_in_dim %[[IS_OWNER]], dims = [] : (tensor<i1>) -> tensor<8x16xi1>
// CHECK-NEXT:  %[[RES:.*]] = stablehlo.select %[[MASK]], %[[LOCAL_DUS]], %[[OPERAND]] : tensor<8x16xi1>, tensor<8x16xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<8x16xf32>
func.func @partitioned_non_sliced_dim(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>},
                                      %arg1: tensor<16x1xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {}]>},
                                      %arg2: tensor<i32>,
                                      %arg3: tensor<i32>) -> (tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x"}, {"y"}]>}) {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{"x"}, {"y"}]>]>} : (tensor<16x32xf32>, tensor<16x1xf32>, tensor<i32>, tensor<i32>) -> tensor<16x32xf32>
  return %0 : tensor<16x32xf32>
}

// -----

sdy.mesh @mesh_2 = <["x"=2]>

// Operand is partitioned on non-sliced dim 0, while sliced dim 1 is
// unpartitioned. Verifies that the op lowers directly without shard ownership
// checks.
// CHECK-LABEL: func @no_partitioned_slice_dim
// CHECK-SAME:    (%[[OPERAND:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
// CHECK-SAME:     %[[UPDATE:.*]]: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
// CHECK-SAME:     %[[IDX0:.*]]: tensor<i32>, %[[IDX1:.*]]: tensor<i32>)
// CHECK-SAME:    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>})
// CHECK-NEXT:  %[[RES:.*]] = stablehlo.dynamic_update_slice %[[OPERAND]], %[[UPDATE]], %[[IDX0]], %[[IDX1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<8x16xf32>, tensor<8x4xf32>, tensor<i32>, tensor<i32>) -> tensor<8x16xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<8x16xf32>
func.func @no_partitioned_slice_dim(%arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
                                    %arg1: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
                                    %arg2: tensor<i32>,
                                    %arg3: tensor<i32>) -> (tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<16x16xf32>, tensor<16x4xf32>, tensor<i32>, tensor<i32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// -----

// Unsharded operand and update with dynamic start indices. Verifies that the
// op lowers unchanged.
// CHECK-LABEL: func @unsharded_operand
// CHECK-SAME:    (%[[OPERAND:.*]]: tensor<16x16xf32>, %[[UPDATE:.*]]: tensor<4x4xf32>,
// CHECK-SAME:     %[[IDX0:.*]]: tensor<i32>, %[[IDX1:.*]]: tensor<i32>) -> tensor<16x16xf32>
// CHECK-NEXT:  %[[RES:.*]] = stablehlo.dynamic_update_slice %[[OPERAND]], %[[UPDATE]], %[[IDX0]], %[[IDX1]] : (tensor<16x16xf32>, tensor<4x4xf32>, tensor<i32>, tensor<i32>) -> tensor<16x16xf32>
// CHECK-NEXT:  return %[[RES]] : tensor<16x16xf32>
func.func @unsharded_operand(%arg0: tensor<16x16xf32>,
                             %arg1: tensor<4x4xf32>,
                             %arg2: tensor<i32>,
                             %arg3: tensor<i32>) -> tensor<16x16xf32> {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 : (tensor<16x16xf32>, tensor<4x4xf32>, tensor<i32>, tensor<i32>) -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// -----

sdy.mesh @mesh_2 = <["x"=2]>

// Sliced dim 0 is partitioned over "x"=2 with update size 2 (> 1) and a
// dynamic index. Verifies that cross-shard updates fail legalization when not
// replicated.
func.func @cross_shard_slice_dim_not_replicated(%arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>},
                                                %arg1: tensor<2x8xf32>,
                                                %arg2: tensor<i32>,
                                                %arg3: tensor<i32>) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2, [{"x"}, {}]>}) {
  // expected-error @+2 {{partitioned dynamic-update-slice crossing shard boundaries requires upstream resharding/replication}}
  // expected-error @+1 {{failed to legalize operation 'stablehlo.dynamic_update_slice' that was explicitly marked illegal}}
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2, [{"x"}, {}]>]>} : (tensor<16x8xf32>, tensor<2x8xf32>, tensor<i32>, tensor<i32>) -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
