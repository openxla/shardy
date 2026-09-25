// RUN: sdy_opt %s -sdy-resolve-permutation-factors="rng-bit-generator-unsafe=true" -sdy-reshard-to-collectives -sdy-convert-global-to-local | FileCheck %s --check-prefixes=CHECK,UNSAFE
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="rng-bit-generator-unsafe=false" -sdy-reshard-to-collectives -sdy-convert-global-to-local | FileCheck %s --check-prefixes=CHECK,SAFE

sdy.mesh @mesh_4 = <["x"=4]>
sdy.mesh @mesh_2_2 = <["x"=2, "y"=2]>

// CHECK-LABEL: func @replicated_output
// CHECK-SAME: (%[[STATE:.*]]: tensor<2xui64>) -> (tensor<2xui64>, tensor<8x16xf32>) {
func.func @replicated_output(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<8x16xf32>) {
  // CHECK-NEXT: %[[OUT_STATE:.*]], %[[OUT:.*]] = stablehlo.rng_bit_generator %[[STATE]], algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<8x16xf32>)
  %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<8x16xf32>)
  // CHECK-NEXT: return %[[OUT_STATE]], %[[OUT]] : tensor<2xui64>, tensor<8x16xf32>
  return %output_state, %output : tensor<2xui64>, tensor<8x16xf32>
}

// CHECK-LABEL: func @sharded_1d_ui64_state
// CHECK-SAME: (%[[STATE:.*]]: tensor<2xui64> {sdy.sharding = #sdy.sharding<@mesh_4, [{}]>})
// CHECK-SAME: -> (tensor<2xui64> {sdy.sharding = #sdy.sharding<@mesh_4, [{}]>},
// CHECK-SAME:     tensor<2x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) {
func.func @sharded_1d_ui64_state(
    %arg0: tensor<2xui64> {sdy.sharding = #sdy.sharding<@mesh_4, [{}]>}
) -> (
    tensor<2xui64> {sdy.sharding = #sdy.sharding<@mesh_4, [{}]>},
    tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}
) {
  // UNSAFE-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // UNSAFE-NEXT: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // UNSAFE-NEXT: %[[HASH_TABLE:.*]] = stablehlo.constant dense<[0, 3801824426178300716, 7279650877546595303, 718562520007760078]> : tensor<4xui64>
  // UNSAFE-NEXT: %[[HASH_SLICE:.*]] = stablehlo.dynamic_slice %[[HASH_TABLE]], %[[PID_I64]], sizes = [1] : (tensor<4xui64>, tensor<i64>) -> tensor<1xui64>
  // UNSAFE-NEXT: %[[SCALAR_HASH:.*]] = stablehlo.reshape %[[HASH_SLICE]] : (tensor<1xui64>) -> tensor<ui64>
  // UNSAFE-NEXT: %[[BCAST_HASH:.*]] = stablehlo.broadcast_in_dim %[[SCALAR_HASH]], dims = [] : (tensor<ui64>) -> tensor<2xui64>
  // UNSAFE-NEXT: %[[ADJ_STATE:.*]] = stablehlo.add %[[STATE]], %[[BCAST_HASH]] : tensor<2xui64>
  // UNSAFE-NEXT: %[[LOCAL_STATE:.*]], %[[LOCAL_DATA:.*]] = stablehlo.rng_bit_generator %[[ADJ_STATE]], algorithm = DEFAULT : (tensor<2xui64>) -> (tensor<2xui64>, tensor<2x16xf32>)
  // UNSAFE-NEXT: %[[LEADER_TABLE:.*]] = stablehlo.constant dense<[true, false, false, false]> : tensor<4xi1>
  // UNSAFE-NEXT: %[[IS_LEADER_SLICE:.*]] = stablehlo.dynamic_slice %[[LEADER_TABLE]], %[[PID_I64]], sizes = [1] : (tensor<4xi1>, tensor<i64>) -> tensor<1xi1>
  // UNSAFE-NEXT: %[[IS_LEADER_SCALAR:.*]] = stablehlo.reshape %[[IS_LEADER_SLICE]] : (tensor<1xi1>) -> tensor<i1>
  // UNSAFE-NEXT: %[[IS_LEADER:.*]] = stablehlo.broadcast_in_dim %[[IS_LEADER_SCALAR]], dims = [] : (tensor<i1>) -> tensor<2xi1>
  // UNSAFE-NEXT: %[[ZERO_STATE:.*]] = stablehlo.constant dense<0> : tensor<2xui64>
  // UNSAFE-NEXT: %[[MASKED_STATE:.*]] = stablehlo.select %[[IS_LEADER]], %[[LOCAL_STATE]], %[[ZERO_STATE]] : tensor<2xi1>, tensor<2xui64>
  // UNSAFE-NEXT: %[[FINAL_STATE:.*]] = "stablehlo.all_reduce"(%[[MASKED_STATE]])
  // UNSAFE-SAME{LITERAL}: <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_4, axes = [#stablehlo.axis_ref<name = "x">]>, use_global_device_ids}> ({
  // UNSAFE-NEXT: ^bb0(%[[ARG0:.*]]: tensor<ui64>, %[[ARG1:.*]]: tensor<ui64>):
  // UNSAFE-NEXT:   %[[SUM:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<ui64>
  // UNSAFE-NEXT:   stablehlo.return %[[SUM]] : tensor<ui64>
  // UNSAFE-NEXT: }) : (tensor<2xui64>) -> tensor<2xui64>

  // SAFE-NEXT: %[[FINAL_STATE:.*]], %[[GLOBAL_DATA:.*]] = stablehlo.rng_bit_generator %[[STATE]], algorithm = DEFAULT {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}]>, <@mesh_4, [{}, {}]>]>} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<8x16xf32>)
  // SAFE-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // SAFE-NEXT: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // SAFE-NEXT: %[[OFFSET_TABLE:.*]] = stablehlo.constant dense<[0, 2, 4, 6]> : tensor<4xi64>
  // SAFE-NEXT: %[[SLICE_IDX:.*]] = stablehlo.dynamic_slice %[[OFFSET_TABLE]], %[[PID_I64]], sizes = [1] : (tensor<4xi64>, tensor<i64>) -> tensor<1xi64>
  // SAFE-NEXT: %[[OFFSET_0:.*]] = stablehlo.reshape %[[SLICE_IDX]] : (tensor<1xi64>) -> tensor<i64>
  // SAFE-NEXT: %[[OFFSET_1:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // SAFE-NEXT: %[[LOCAL_DATA:.*]] = stablehlo.dynamic_slice %[[GLOBAL_DATA]], %[[OFFSET_0]], %[[OFFSET_1]], sizes = [2, 16] : (tensor<8x16xf32>, tensor<i64>, tensor<i64>) -> tensor<2x16xf32>
  %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = DEFAULT {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}]>, <@mesh_4, [{"x"}, {}]>]>
  } : (tensor<2xui64>) -> (tensor<2xui64>, tensor<8x16xf32>)
  // CHECK-NEXT: return %[[FINAL_STATE]], %[[LOCAL_DATA]] : tensor<2xui64>, tensor<2x16xf32>
  return %output_state, %output : tensor<2xui64>, tensor<8x16xf32>
}

// CHECK-LABEL: func @sharded_partial_mesh_ui32_state
// CHECK-SAME: (%[[STATE:.*]]: tensor<4xui32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}]>})
// CHECK-SAME: -> (tensor<4xui32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}]>},
// CHECK-SAME:     tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {"y"}]>}) {
func.func @sharded_partial_mesh_ui32_state(
    %arg0: tensor<4xui32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}]>}
) -> (
    tensor<4xui32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}]>},
    tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_2_2, [{}, {"y"}]>}
) {
  // UNSAFE-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // UNSAFE-NEXT: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // UNSAFE-NEXT: %[[HASH_TABLE:.*]] = stablehlo.constant dense<[0, 885181228, 0, 885181228]> : tensor<4xui32>
  // UNSAFE-NEXT: %[[HASH_SLICE:.*]] = stablehlo.dynamic_slice %[[HASH_TABLE]], %[[PID_I64]], sizes = [1] : (tensor<4xui32>, tensor<i64>) -> tensor<1xui32>
  // UNSAFE-NEXT: %[[SCALAR_HASH:.*]] = stablehlo.reshape %[[HASH_SLICE]] : (tensor<1xui32>) -> tensor<ui32>
  // UNSAFE-NEXT: %[[BCAST_HASH:.*]] = stablehlo.broadcast_in_dim %[[SCALAR_HASH]], dims = [] : (tensor<ui32>) -> tensor<4xui32>
  // UNSAFE-NEXT: %[[ADJ_STATE:.*]] = stablehlo.add %[[STATE]], %[[BCAST_HASH]] : tensor<4xui32>
  // UNSAFE-NEXT: %[[LOCAL_STATE:.*]], %[[LOCAL_DATA:.*]] = stablehlo.rng_bit_generator %[[ADJ_STATE]], algorithm = THREE_FRY : (tensor<4xui32>) -> (tensor<4xui32>, tensor<8x8xf32>)
  // UNSAFE-NEXT: %[[LEADER_TABLE:.*]] = stablehlo.constant dense<[true, false, true, false]> : tensor<4xi1>
  // UNSAFE-NEXT: %[[IS_LEADER_SLICE:.*]] = stablehlo.dynamic_slice %[[LEADER_TABLE]], %[[PID_I64]], sizes = [1] : (tensor<4xi1>, tensor<i64>) -> tensor<1xi1>
  // UNSAFE-NEXT: %[[IS_LEADER_SCALAR:.*]] = stablehlo.reshape %[[IS_LEADER_SLICE]] : (tensor<1xi1>) -> tensor<i1>
  // UNSAFE-NEXT: %[[IS_LEADER:.*]] = stablehlo.broadcast_in_dim %[[IS_LEADER_SCALAR]], dims = [] : (tensor<i1>) -> tensor<4xi1>
  // UNSAFE-NEXT: %[[ZERO_STATE:.*]] = stablehlo.constant dense<0> : tensor<4xui32>
  // UNSAFE-NEXT: %[[MASKED_STATE:.*]] = stablehlo.select %[[IS_LEADER]], %[[LOCAL_STATE]], %[[ZERO_STATE]] : tensor<4xi1>, tensor<4xui32>
  // UNSAFE-NEXT: %[[FINAL_STATE:.*]] = "stablehlo.all_reduce"(%[[MASKED_STATE]])
  // UNSAFE-SAME{LITERAL}: <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = #stablehlo.replica_group_mesh_axes<mesh = @mesh_2_2, axes = [#stablehlo.axis_ref<name = "y">]>, use_global_device_ids}> ({
  // UNSAFE-NEXT: ^bb0(%[[ARG0:.*]]: tensor<ui32>, %[[ARG1:.*]]: tensor<ui32>):
  // UNSAFE-NEXT:   %[[SUM:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<ui32>
  // UNSAFE-NEXT:   stablehlo.return %[[SUM]] : tensor<ui32>
  // UNSAFE-NEXT: }) : (tensor<4xui32>) -> tensor<4xui32>

  // SAFE-NEXT: %[[FINAL_STATE:.*]], %[[GLOBAL_DATA:.*]] = stablehlo.rng_bit_generator %[[STATE]], algorithm = THREE_FRY {sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{}]>, <@mesh_2_2, [{}, {}]>]>} : (tensor<4xui32>) -> (tensor<4xui32>, tensor<8x16xf32>)
  // SAFE-NEXT: %[[OFFSET_0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // SAFE-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // SAFE-NEXT: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // SAFE-NEXT: %[[OFFSET_TABLE:.*]] = stablehlo.constant dense<[0, 8, 0, 8]> : tensor<4xi64>
  // SAFE-NEXT: %[[SLICE_IDX:.*]] = stablehlo.dynamic_slice %[[OFFSET_TABLE]], %[[PID_I64]], sizes = [1] : (tensor<4xi64>, tensor<i64>) -> tensor<1xi64>
  // SAFE-NEXT: %[[OFFSET_1:.*]] = stablehlo.reshape %[[SLICE_IDX]] : (tensor<1xi64>) -> tensor<i64>
  // SAFE-NEXT: %[[LOCAL_DATA:.*]] = stablehlo.dynamic_slice %[[GLOBAL_DATA]], %[[OFFSET_0]], %[[OFFSET_1]], sizes = [8, 8] : (tensor<8x16xf32>, tensor<i64>, tensor<i64>) -> tensor<8x8xf32>
  %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = THREE_FRY {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_2_2, [{}]>, <@mesh_2_2, [{}, {"y"}]>]>
  } : (tensor<4xui32>) -> (tensor<4xui32>, tensor<8x16xf32>)
  // CHECK-NEXT: return %[[FINAL_STATE]], %[[LOCAL_DATA]] : tensor<4xui32>, tensor<8x8xf32>
  return %output_state, %output : tensor<4xui32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @sharded_1d_ui16_state_replicated
// CHECK-SAME: (%[[STATE:.*]]: tensor<2xui16> {sdy.sharding = #sdy.sharding<@mesh_4, [{}]>})
// CHECK-SAME: -> (tensor<2xui16> {sdy.sharding = #sdy.sharding<@mesh_4, [{}]>},
// CHECK-SAME:     tensor<2x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}) {
func.func @sharded_1d_ui16_state_replicated(
    %arg0: tensor<2xui16> {sdy.sharding = #sdy.sharding<@mesh_4, [{}]>}
) -> (
    tensor<2xui16> {sdy.sharding = #sdy.sharding<@mesh_4, [{}]>},
    tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_4, [{"x"}, {}]>}
) {
  // CHECK-NEXT: %[[FINAL_STATE:.*]], %[[GLOBAL_DATA:.*]] = stablehlo.rng_bit_generator %[[STATE]], algorithm = DEFAULT {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}]>, <@mesh_4, [{}, {}]>]>} : (tensor<2xui16>) -> (tensor<2xui16>, tensor<8x16xf32>)
  // CHECK-NEXT: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[PID_I64:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-NEXT: %[[OFFSET_TABLE:.*]] = stablehlo.constant dense<[0, 2, 4, 6]> : tensor<4xi64>
  // CHECK-NEXT: %[[SLICE_IDX:.*]] = stablehlo.dynamic_slice %[[OFFSET_TABLE]], %[[PID_I64]], sizes = [1] : (tensor<4xi64>, tensor<i64>) -> tensor<1xi64>
  // CHECK-NEXT: %[[OFFSET_0:.*]] = stablehlo.reshape %[[SLICE_IDX]] : (tensor<1xi64>) -> tensor<i64>
  // CHECK-NEXT: %[[OFFSET_1:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT: %[[LOCAL_DATA:.*]] = stablehlo.dynamic_slice %[[GLOBAL_DATA]], %[[OFFSET_0]], %[[OFFSET_1]], sizes = [2, 16] : (tensor<8x16xf32>, tensor<i64>, tensor<i64>) -> tensor<2x16xf32>
  %output_state, %output = stablehlo.rng_bit_generator %arg0, algorithm = DEFAULT {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}]>, <@mesh_4, [{"x"}, {}]>]>
  } : (tensor<2xui16>) -> (tensor<2xui16>, tensor<8x16xf32>)
  // CHECK-NEXT: return %[[FINAL_STATE]], %[[LOCAL_DATA]] : tensor<2xui16>, tensor<2x16xf32>
  return %output_state, %output : tensor<2xui16>, tensor<8x16xf32>
}
