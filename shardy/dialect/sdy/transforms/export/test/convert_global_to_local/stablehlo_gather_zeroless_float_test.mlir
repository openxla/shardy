// RUN: sdy_opt %s --sdy-convert-global-to-local | FileCheck %s
// RUN: sdy_opt %s --sdy-insert-explicit-reshards="enable-full-version=true mark-partial-result-with-unreduced-axes=true" --sdy-reshard-to-collectives --sdy-convert-global-to-local | FileCheck %s

sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// CHECK-LABEL: func @gather_f8e8m0fnu_bitcasts_to_ui8_for_zero_mask_and_sum
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x", "y"}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<3x1xi32>)
// CHECK-SAME: -> (tensor<3xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}]>}) {
func.func @gather_f8e8m0fnu_bitcasts_to_ui8_for_zero_mask_and_sum(
  %arg0: tensor<8xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x", "y"}]>},
  %arg1: tensor<3x1xi32>)
  -> (tensor<3xf8E8M0FNU> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}]>}) {
  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<3x1xi32>
  // CHECK-NEXT: %[[C7:.*]] = stablehlo.constant dense<7> : tensor<3x1xi32>
  // CHECK-NEXT: %[[CLAMPED:.*]] = stablehlo.clamp %[[C0]], %[[ARG1]], %[[C7]] : tensor<3x1xi32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFFSETS:.*]] = stablehlo.constant dense<[0, 2, 4, 6]> : tensor<4xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[OFFSETS]], %[[PID]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFFSET:.*]] = stablehlo.convert %[[RESHAPE]] : tensor<i32>
  // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT: %[[LIMIT:.*]] = stablehlo.add %[[OFFSET]], %[[C1]] : tensor<i32>
  // CHECK-NEXT: %[[BCAST_OFF:.*]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{}, {}]>]>} : (tensor<i32>) -> tensor<3x1xi32>
  // CHECK-NEXT: %[[BCAST_LIM:.*]] = stablehlo.broadcast_in_dim %[[LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{}, {}]>]>} : (tensor<i32>) -> tensor<3x1xi32>
  // CHECK-NEXT: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %[[BCAST_OFF]] : tensor<3x1xi32>
  // CHECK-NEXT: %[[GE:.*]] = stablehlo.compare GE, %[[CLAMPED]], %[[BCAST_OFF]] : (tensor<3x1xi32>, tensor<3x1xi32>) -> tensor<3x1xi1>
  // CHECK-NEXT: %[[LE:.*]] = stablehlo.compare LE, %[[CLAMPED]], %[[BCAST_LIM]] : (tensor<3x1xi32>, tensor<3x1xi32>) -> tensor<3x1xi1>
  // CHECK-NEXT: %[[IN_BOUNDS:.*]] = stablehlo.and %[[GE]], %[[LE]] : tensor<3x1xi1>
  // CHECK-NEXT: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[LOCAL_IDX]])
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{}]>]>}
  // CHECK-SAME:   (tensor<2xf8E8M0FNU>, tensor<3x1xi32>) -> tensor<3xf8E8M0FNU>
  // CHECK-NEXT: %[[TRUE:.*]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK-NEXT: %[[REDUCED_MASK:.*]] = stablehlo.reduce(%[[IN_BOUNDS]] init: %[[TRUE]]) applies stablehlo.and across dimensions = [1] : (tensor<3x1xi1>, tensor<i1>) -> tensor<3xi1>
  // CHECK-NEXT: %[[MASK:.*]] = stablehlo.broadcast_in_dim %[[REDUCED_MASK]], dims = [0] : (tensor<3xi1>) -> tensor<3xi1>
  // CHECK-NEXT: %[[TO_UI8:.*]] = stablehlo.bitcast_convert %[[GATHER]] : (tensor<3xf8E8M0FNU>) -> tensor<3xui8>
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0> : tensor<3xui8>
  // CHECK-NEXT: %[[MASKED:.*]] = stablehlo.select %[[MASK]], %[[TO_UI8]], %[[ZERO]] : tensor<3xi1>, tensor<3xui8>
  // CHECK-NEXT: %[[RED:.*]] = "stablehlo.all_reduce"(%[[MASKED]]) <{
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // CHECK-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_2, axes = [#stablehlo.axis_ref<name = "x">, #stablehlo.axis_ref<name = "y">]>,
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> ({
  // CHECK-NEXT: ^bb0(%[[LHS:.*]]: tensor<ui8>, %[[RHS:.*]]: tensor<ui8>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[LHS]], %[[RHS]] : tensor<ui8>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<ui8>
  // CHECK-NEXT: }) : (tensor<3xui8>) -> tensor<3xui8>
  // CHECK-NEXT: %[[FROM_UI8:.*]] = stablehlo.bitcast_convert %[[RED]] : (tensor<3xui8>) -> tensor<3xf8E8M0FNU>
  // CHECK-NEXT: return %[[FROM_UI8]] : tensor<3xf8E8M0FNU>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_2, [{}]>]>
  } : (tensor<8xf8E8M0FNU>, tensor<3x1xi32>) -> tensor<3xf8E8M0FNU>
  return %0 : tensor<3xf8E8M0FNU>
}

// CHECK-LABEL: func @gather_f32_zero_mask_and_sum_without_bitcast
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x", "y"}]>},
// CHECK-SAME: %[[ARG1:.*]]: tensor<3x1xi32>)
// CHECK-SAME: -> (tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}]>}) {
func.func @gather_f32_zero_mask_and_sum_without_bitcast(
  %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{"x", "y"}]>},
  %arg1: tensor<3x1xi32>)
  -> (tensor<3xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}]>}) {
  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<3x1xi32>
  // CHECK-NEXT: %[[C7:.*]] = stablehlo.constant dense<7> : tensor<3x1xi32>
  // CHECK-NEXT: %[[CLAMPED:.*]] = stablehlo.clamp %[[C0]], %[[ARG1]], %[[C7]] : tensor<3x1xi32>
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[OFFSETS:.*]] = stablehlo.constant dense<[0, 2, 4, 6]> : tensor<4xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[OFFSETS]], %[[PID]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[OFFSET:.*]] = stablehlo.convert %[[RESHAPE]] : tensor<i32>
  // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT: %[[LIMIT:.*]] = stablehlo.add %[[OFFSET]], %[[C1]] : tensor<i32>
  // CHECK-NEXT: %[[BCAST_OFF:.*]] = stablehlo.broadcast_in_dim %[[OFFSET]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{}, {}]>]>} : (tensor<i32>) -> tensor<3x1xi32>
  // CHECK-NEXT: %[[BCAST_LIM:.*]] = stablehlo.broadcast_in_dim %[[LIMIT]], dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{}, {}]>]>} : (tensor<i32>) -> tensor<3x1xi32>
  // CHECK-NEXT: %[[LOCAL_IDX:.*]] = stablehlo.subtract %[[CLAMPED]], %[[BCAST_OFF]] : tensor<3x1xi32>
  // CHECK-NEXT: %[[GE:.*]] = stablehlo.compare GE, %[[CLAMPED]], %[[BCAST_OFF]] : (tensor<3x1xi32>, tensor<3x1xi32>) -> tensor<3x1xi1>
  // CHECK-NEXT: %[[LE:.*]] = stablehlo.compare LE, %[[CLAMPED]], %[[BCAST_LIM]] : (tensor<3x1xi32>, tensor<3x1xi32>) -> tensor<3x1xi1>
  // CHECK-NEXT: %[[IN_BOUNDS:.*]] = stablehlo.and %[[GE]], %[[LE]] : tensor<3x1xi1>
  // CHECK-NEXT: %[[GATHER:.*]] = "stablehlo.gather"(%[[ARG0]], %[[LOCAL_IDX]])
  // CHECK-SAME:   (tensor<2xf32>, tensor<3x1xi32>) -> tensor<3xf32>
  // CHECK-NEXT: %[[TRUE:.*]] = stablehlo.constant dense<true> : tensor<i1>
  // CHECK-NEXT: %[[REDUCED_MASK:.*]] = stablehlo.reduce(%[[IN_BOUNDS]] init: %[[TRUE]]) applies stablehlo.and across dimensions = [1] : (tensor<3x1xi1>, tensor<i1>) -> tensor<3xi1>
  // CHECK-NEXT: %[[MASK:.*]] = stablehlo.broadcast_in_dim %[[REDUCED_MASK]], dims = [0] : (tensor<3xi1>) -> tensor<3xi1>
  // CHECK-NEXT: %[[ZERO:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<3xf32>
  // CHECK-NEXT: %[[MASKED:.*]] = stablehlo.select %[[MASK]], %[[GATHER]], %[[ZERO]] : tensor<3xi1>, tensor<3xf32>
  // CHECK-NEXT: %[[RED:.*]] = "stablehlo.all_reduce"(%[[MASKED]]) <{
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = {{.*}}, type = 1>,
  // CHECK-SAME:   replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_2, axes = [#stablehlo.axis_ref<name = "x">, #stablehlo.axis_ref<name = "y">]>,
  // CHECK-SAME:   use_global_device_ids
  // CHECK-SAME: }> ({
  // CHECK-NEXT: ^bb0(%[[LHS:.*]]: tensor<f32>, %[[RHS:.*]]: tensor<f32>):
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[LHS]], %[[RHS]] : tensor<f32>
  // CHECK-NEXT:   stablehlo.return %[[ADD]] : tensor<f32>
  // CHECK-NEXT: }) : (tensor<3xf32>) -> tensor<3xf32>
  // CHECK-NEXT: return %[[RED]] : tensor<3xf32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1>,
    slice_sizes = array<i64: 1>,
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh_2_2, [{}]>]>
  } : (tensor<8xf32>, tensor<3x1xi32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}
