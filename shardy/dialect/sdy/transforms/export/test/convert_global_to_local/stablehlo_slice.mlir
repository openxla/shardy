// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>
sdy.mesh @mesh_1_2 = <["x"=1, "y"=2]>

// CHECK-LABEL: func @replicated_after_all_gather
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"y"}, {}]>}) -> tensor<4x8xf32>
func.func @replicated_after_all_gather(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"y"}, {}]>}) -> tensor<4x8xf32> {
  // CHECK: %[[GATHER:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{all_gather_dim = 0 : i64,
  // CHECK-SAME: channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // CHECK-SAME: replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_4_2, axes = [#stablehlo.axis_ref<name = "y">]>
  // CHECK-SAME: (tensor<4x16xf32>) -> tensor<8x16xf32>
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<8x16xf32>
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %[[GATHER]] [0:4, 0:8] : (tensor<8x16xf32>) -> tensor<4x8xf32>
  %1 = stablehlo.slice %0 [0:4, 0:8] : (tensor<8x16xf32>) -> tensor<4x8xf32>
  // CHECK: return %[[SLICE]]
  return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @slicing_dim_not_sharded(
// CHECK-SAME:    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>})
// CHECK-SAME:    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>}) {
func.func @slicing_dim_not_sharded(%arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>})
    -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>}) {
  // CHECK:         %[[RES:.*]] = stablehlo.slice %arg0 [4:12:2, 0:8] : (tensor<16x8xf32>) -> tensor<4x8xf32>
  %0 = stablehlo.slice %arg0 [4:12:2, 0:32] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"x"}]>]>} : (tensor<16x32xf32>) -> tensor<4x32xf32>
  // CHECK:         return %[[RES]] : tensor<4x8xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @slicing_dim_sharded(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
// CHECK-SAME:  -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>}) {
func.func @slicing_dim_sharded(%arg0: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
  -> (tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT:    %[[RES:.*]] = stablehlo.slice %arg0 [0:8:2, 0:8] : (tensor<8x8xf32>) -> tensor<4x8xf32>
  %0 = stablehlo.slice %arg0 [0:32:2, 0:16] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>]>} : (tensor<32x16xf32>) -> tensor<16x16xf32>
  // CHECK-NEXT:    return %[[RES]] : tensor<4x8xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK-LABEL: func @communication_free_slice_sharded_dim(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"x"}, {"y"}]>})
// CHECK-SAME:  -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"x"}, {"y"}]>}) {
func.func @communication_free_slice_sharded_dim(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"x"}, {"y"}]>})
    -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT:    %[[RES:.*]] = stablehlo.slice %arg0 [0:4, 0:8] : (tensor<8x8xf32>) -> tensor<4x8xf32>
  %0 = stablehlo.slice %arg0 [0:4, 0:16] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_1_2, [{"x"}, {"y"}]>]>} : (tensor<8x16xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT:    return %[[RES]] : tensor<4x8xf32>
  return %0 : tensor<4x16xf32>
}

// The original slice is [0:1] and pad-for-divisibility expanded it to [0:4].
// CHECK-LABEL: func @communication_free_slice_to_1_sharded_dim
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
// CHECK-SAME:  -> (tensor<1x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>}) {
func.func @communication_free_slice_to_1_sharded_dim(%arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
    -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT:    %[[RES:.*]] = stablehlo.slice %[[ARG0]] [0:1, 0:8] : (tensor<4x8xf32>) -> tensor<1x8xf32>
  %0 = stablehlo.slice %arg0 [0:4, 0:16] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>]>} : (tensor<16x16xf32>) -> tensor<4x16xf32>
  // CHECK-NEXT:    return %[[RES]] : tensor<1x8xf32>
  return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @all_gather_slice_to_1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"y"}]>}) -> tensor<1xf32>
func.func @all_gather_slice_to_1(%arg0: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"y"}]>}) -> tensor<1xf32> {
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[NON_LEADER:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i1>
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
  // CHECK-NEXT: %[[MASKED:.*]] = stablehlo.select %[[NON_LEADER]], %[[ZERO]], %[[ARG0]] : tensor<i1>, tensor<1xf32>
  // CHECK-NEXT: %[[AR:.*]] = "stablehlo.all_reduce"(%[[MASKED]]) <{
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // CHECK-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_1_2, axes = [#stablehlo.axis_ref<name = "y">]>,
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> ({
  // CHECK-NEXT: ^bb0(%[[LHS:.*]]: tensor<f32>, %[[RHS:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[LHS]], %[[RHS]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT: }) : (tensor<1xf32>) -> tensor<1xf32>
  // CHECK-NEXT: return %[[AR]] : tensor<1xf32>
  %0 = sdy.all_gather [{"y"}] %arg0 out_sharding=<@mesh_1_2, [{}]> : tensor<2xf32>
  %1 = stablehlo.slice %0 [0:1] : (tensor<2xf32>) -> tensor<1xf32>
  return %1 : tensor<1xf32>
}

// CHECK-LABEL: func @all_gather_slice_to_local_shard_size_2d
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
// CHECK-SAME: -> (tensor<4x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
func.func @all_gather_slice_to_local_shard_size_2d(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
    -> (tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[TABLE:.*]] = stablehlo.constant dense<[true, false, true, false, true, false, true, false]> : tensor<8xi1>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<8xi1>, tensor<ui32>) -> tensor<1xi1>
  // CHECK-NEXT: %[[IS_LEADER:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi1>) -> tensor<i1>
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<4x4xf32>
  // CHECK-NEXT: %[[MASKED:.*]] = stablehlo.select %[[IS_LEADER]], %[[ARG0]], %[[ZERO]] : tensor<i1>, tensor<4x4xf32>
  // CHECK-NEXT: %[[AR:.*]] = "stablehlo.all_reduce"(%[[MASKED]]) <{
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // CHECK-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_4_2, axes = [#stablehlo.axis_ref<name = "y">]>,
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> ({
  // CHECK-NEXT: ^bb0(%[[LHS:.*]]: tensor<f32>, %[[RHS:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[LHS]], %[[RHS]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT: }) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT: return %[[AR]] : tensor<4x4xf32>
  %0 = sdy.all_gather [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<16x8xf32>
  %1 = stablehlo.slice %0 [0:16, 0:4] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<16x8xf32>) -> tensor<16x4xf32>
  return %1 : tensor<16x4xf32>
}

// Positive test: multi-axis all_gather on a single dimension with i32 element type.
// CHECK-LABEL: func @all_gather_slice_multi_axis_i32
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xi32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x", "y"}]>}) -> tensor<2xi32>
func.func @all_gather_slice_multi_axis_i32(%arg0: tensor<16xi32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x", "y"}]>}) -> tensor<2xi32> {
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[NON_LEADER:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i1>
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<2xi32>
  // CHECK-NEXT: %[[MASKED:.*]] = stablehlo.select %[[NON_LEADER]], %[[ZERO]], %[[ARG0]] : tensor<i1>, tensor<2xi32>
  // CHECK-NEXT: %[[AR:.*]] = "stablehlo.all_reduce"(%[[MASKED]]) <{
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // CHECK-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_4_2, axes = [#stablehlo.axis_ref<name = "x">, #stablehlo.axis_ref<name = "y">]>,
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> ({
  // CHECK-NEXT: ^bb0(%[[LHS:.*]]: tensor<i32>, %[[RHS:.*]]: tensor<i32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[LHS]], %[[RHS]] : tensor<i32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<i32>
  // CHECK-NEXT: }) : (tensor<2xi32>) -> tensor<2xi32>
  // CHECK-NEXT: return %[[AR]] : tensor<2xi32>
  %0 = sdy.all_gather [{"x", "y"}] %arg0 out_sharding=<@mesh_4_2, [{}]> : tensor<16xi32>
  %1 = stablehlo.slice %0 [0:2] : (tensor<16xi32>) -> tensor<2xi32>
  return %1 : tensor<2xi32>
}

// Negative test: all_gather has multiple users.
// CHECK-LABEL: func @all_gather_multi_use_slice_not_converted
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"y"}]>}) -> (tensor<1xf32>, tensor<2xf32>)
func.func @all_gather_multi_use_slice_not_converted(%arg0: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"y"}]>}) -> (tensor<1xf32>, tensor<2xf32>) {
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{all_gather_dim = 0 : i64
  // CHECK-SAME: (tensor<1xf32>) -> tensor<2xf32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[AG]] [0:1] : (tensor<2xf32>) -> tensor<1xf32>
  // CHECK-NEXT: return %[[SLICE]], %[[AG]] : tensor<1xf32>, tensor<2xf32>
  %0 = sdy.all_gather [{"y"}] %arg0 out_sharding=<@mesh_1_2, [{}]> : tensor<2xf32>
  %1 = stablehlo.slice %0 [0:1] : (tensor<2xf32>) -> tensor<1xf32>
  return %1, %0 : tensor<1xf32>, tensor<2xf32>
}

// Negative test: i1 element type is not converted to all_reduce(SUM).
// CHECK-LABEL: func @all_gather_slice_i1_not_converted
// CHECK-SAME: (%[[ARG0:.*]]: tensor<1xi1> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"y"}]>}) -> tensor<1xi1>
func.func @all_gather_slice_i1_not_converted(%arg0: tensor<2xi1> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"y"}]>}) -> tensor<1xi1> {
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{all_gather_dim = 0 : i64
  // CHECK-SAME: (tensor<1xi1>) -> tensor<2xi1>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[AG]] [0:1] : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: return %[[SLICE]] : tensor<1xi1>
  %0 = sdy.all_gather [{"y"}] %arg0 out_sharding=<@mesh_1_2, [{}]> : tensor<2xi1>
  %1 = stablehlo.slice %0 [0:1] : (tensor<2xi1>) -> tensor<1xi1>
  return %1 : tensor<1xi1>
}

// Negative test: all_gather is on dim 1, but slice is on dim 0.
// CHECK-LABEL: func @all_gather_slice_different_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{}, {"y"}]>}) -> tensor<4x8xf32>
func.func @all_gather_slice_different_dim(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{}, {"y"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{all_gather_dim = 1 : i64
  // CHECK-SAME: (tensor<8x4xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[AG]] [0:4, 0:8] : (tensor<8x8xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[SLICE]] : tensor<4x8xf32>
  %0 = sdy.all_gather [{}, {"y"}] %arg0 out_sharding=<@mesh_1_2, [{}, {}]> : tensor<8x8xf32>
  %1 = stablehlo.slice %0 [0:4, 0:8] : (tensor<8x8xf32>) -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}

// Negative test: slice size (2) does not match local all_gather operand size (4).
// CHECK-LABEL: func @all_gather_slice_size_mismatch
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"y"}]>}) -> tensor<2xf32>
func.func @all_gather_slice_size_mismatch(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"y"}]>}) -> tensor<2xf32> {
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{all_gather_dim = 0 : i64
  // CHECK-SAME: (tensor<4xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[AG]] [0:2] : (tensor<8xf32>) -> tensor<2xf32>
  // CHECK-NEXT: return %[[SLICE]] : tensor<2xf32>
  %0 = sdy.all_gather [{"y"}] %arg0 out_sharding=<@mesh_1_2, [{}]> : tensor<8xf32>
  %1 = stablehlo.slice %0 [0:2] : (tensor<8xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// Negative test: slice size matches local all_gather operand size (4), but start index is non-zero.
// CHECK-LABEL: func @all_gather_slice_nonzero_start
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"y"}]>}) -> tensor<4xf32>
func.func @all_gather_slice_nonzero_start(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"y"}]>}) -> tensor<4xf32> {
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{all_gather_dim = 0 : i64
  // CHECK-SAME: (tensor<4xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[AG]] [4:8] : (tensor<8xf32>) -> tensor<4xf32>
  // CHECK-NEXT: return %[[SLICE]] : tensor<4xf32>
  %0 = sdy.all_gather [{"y"}] %arg0 out_sharding=<@mesh_1_2, [{}]> : tensor<8xf32>
  %1 = stablehlo.slice %0 [4:8] : (tensor<8xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}

// Negative test: partial all_gather leaves gatherDim still sharded on "x".
// CHECK-LABEL: func @partial_all_gather_slice_not_converted
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x", "y"}, {}]>})
// CHECK-SAME: -> (tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
func.func @partial_all_gather_slice_not_converted(
    %arg0: tensor<16x4xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x", "y"}, {}]>})
    -> (tensor<16x2xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{all_gather_dim = 0 : i64,
  // CHECK-SAME: (tensor<2x4xf32>) -> tensor<4x4xf32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[AG]] [0:4, 0:2] : (tensor<4x4xf32>) -> tensor<4x2xf32>
  // CHECK-NEXT: return %[[SLICE]] : tensor<4x2xf32>
  %0 = sdy.all_gather [{"y"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"x"}, {}]> : tensor<16x4xf32>
  %1 = stablehlo.slice %0 [0:16, 0:2] {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<16x4xf32>) -> tensor<16x2xf32>
  return %1 : tensor<16x2xf32>
}

// Negative test: slice stride is not 1.
// CHECK-LABEL: func @all_gather_slice_stride_not_one
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"y"}]>}) -> tensor<2xf32>
func.func @all_gather_slice_stride_not_one(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"y"}]>}) -> tensor<2xf32> {
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[ARG0]]) <{all_gather_dim = 0 : i64,
  // CHECK-SAME: (tensor<4xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[AG]] [0:4:2] : (tensor<8xf32>) -> tensor<2xf32>
  // CHECK-NEXT: return %[[SLICE]] : tensor<2xf32>
  %0 = sdy.all_gather [{"y"}] %arg0 out_sharding=<@mesh_1_2, [{}]> : tensor<8xf32>
  %1 = "stablehlo.slice"(%0) <{limit_indices = array<i64: 4>, start_indices = array<i64: 0>, strides = array<i64: 2>}> : (tensor<8xf32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// Negative test: multi-dim all_gather (numGatheringDims > 1) is not converted to all_reduce.
// CHECK-LABEL: func @all_gather_slice_multi_dim_gather_not_converted
// CHECK-SAME: (%[[ARG0:.*]]: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>}) -> tensor<4x8xf32>
func.func @all_gather_slice_multi_dim_gather_not_converted(%arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[RESHAPE0:.*]] = stablehlo.reshape %[[ARG0]] : (tensor<4x8xf32>) -> tensor<1x4x8xf32>
  // CHECK-NEXT: %[[AG:.*]] = "stablehlo.all_gather"(%[[RESHAPE0]]) <{all_gather_dim = 0 : i64,
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // CHECK-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_4_2, axes = [#stablehlo.axis_ref<name = "x">, #stablehlo.axis_ref<name = "y">]>,
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> : (tensor<1x4x8xf32>) -> tensor<8x4x8xf32>
  // CHECK-NEXT: %[[RESHAPE1:.*]] = stablehlo.reshape %[[AG]] : (tensor<8x4x8xf32>) -> tensor<4x2x4x8xf32>
  // CHECK-NEXT: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[RESHAPE1]], dims = [0, 2, 1, 3] : (tensor<4x2x4x8xf32>) -> tensor<4x4x2x8xf32>
  // CHECK-NEXT: %[[RESHAPE2:.*]] = stablehlo.reshape %[[TRANSPOSE]] : (tensor<4x4x2x8xf32>) -> tensor<16x16xf32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHAPE2]] [0:4, 0:8] : (tensor<16x16xf32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[SLICE]] : tensor<4x8xf32>
  %0 = sdy.all_gather [{"x"}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {}]> : tensor<16x16xf32>
  %1 = stablehlo.slice %0 [0:4, 0:8] : (tensor<16x16xf32>) -> tensor<4x8xf32>
  return %1 : tensor<4x8xf32>
}
