// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=false rng-bit-generator-unsafe=false" | FileCheck %s
// RUN: sdy_opt %s -sdy-resolve-permutation-factors="enable-halo-exchange=true rng-bit-generator-unsafe=true" | FileCheck %s

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>
sdy.mesh @mesh = <["a"=2, "b"=2]>

// An update of size 1 along dim 0 with input sharding `[{"a"}, {}]` cannot cross
// shard boundaries. Factor resolution skips replication reshard.
// CHECK-LABEL: func @dynamic_update_slice_comm_free_single_element
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, %[[ARG1:.*]]: tensor<1x8xf32>, %[[ARG2:.*]]: tensor<i32>, %[[ARG3:.*]]: tensor<i32>)
// CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
// CHECK-NEXT: return %[[DUS]]
func.func @dynamic_update_slice_comm_free_single_element(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<1x8xf32>,
    %arg2: tensor<i32>,
    %arg3: tensor<i32>)
    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>
  } : (tensor<16x8xf32>, tensor<1x8xf32>, tensor<i32>, tensor<i32>)
      -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// Even when a sliced dimension has update size 1, replicate it if all users of
// the dynamic_update_slice are sdy.reshard ops that replicate that dimension.
// CHECK-LABEL: func @dynamic_update_slice_replicated_when_all_users_replicate_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}, %[[ARG1:.*]]: tensor<2x1xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %[[ARG2:.*]]: tensor<i32>, %[[ARG3:.*]]: tensor<i32>)
// CHECK-SAME: -> (tensor<8x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
func.func @dynamic_update_slice_replicated_when_all_users_replicate_dim(
    %arg0: tensor<8x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<2x1xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %arg2: tensor<i32>,
    %arg3: tensor<i32>)
    -> (tensor<8x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK:      %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x2xi64>
  // CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[RESHARD0]], %[[ARG1]], %[[ARG2]], %[[ARG3]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD_OUT:.*]] = sdy.reshard %[[DUS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x2xi64>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x2xi64>, tensor<2x1xi64>, tensor<i32>, tensor<i32>)
      -> tensor<8x2xi64>
  // CHECK-NEXT: %[[USER_RESHARD:.*]] = sdy.reshard %[[RESHARD_OUT]] <@mesh, [{}, {}]> : tensor<8x2xi64>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<8x2xi64>
  // CHECK-NEXT: return %[[USER_RESHARD]]
  return %1 : tensor<8x2xi64>
}

// Even when a sliced dimension has constant start indices and the update slice
// is owned by a single shard, replicate it if all users of the
// dynamic_update_slice are sdy.reshard ops that replicate that dimension.
// CHECK-LABEL: func @dynamic_update_slice_replicated_when_all_users_replicate_constant_single_shard_dim
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}, %[[ARG1:.*]]: tensor<4x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %[[ARG2:.*]]: tensor<i32>)
// CHECK-SAME: -> (tensor<8x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
func.func @dynamic_update_slice_replicated_when_all_users_replicate_constant_single_shard_dim(
    %arg0: tensor<8x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<4x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %arg2: tensor<i32>)
    -> (tensor<8x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK:      %[[C:.*]] = stablehlo.constant dense<0> : tensor<i32>
  %c = stablehlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x4xi64>
  // CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[RESHARD0]], %[[ARG1]], %[[ARG2]], %[[C]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD_OUT:.*]] = sdy.reshard %[[DUS]] <@mesh, [{"a"}, {"b"}]> : tensor<8x4xi64>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %c {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<8x4xi64>, tensor<4x2xi64>, tensor<i32>, tensor<i32>)
      -> tensor<8x4xi64>
  // CHECK-NEXT: %[[USER_RESHARD:.*]] = sdy.reshard %[[RESHARD_OUT]] <@mesh, [{}, {}]> : tensor<8x4xi64>
  %1 = sdy.reshard %0 <@mesh, [{}, {}]> : tensor<8x4xi64>
  // CHECK-NEXT: return %[[USER_RESHARD]]
  return %1 : tensor<8x4xi64>
}

// Constant update slice [2, 4) on dim 0 with input sharding `[{"a"}, {}]` lies
// entirely within shard [0, 8). Factor resolution skips replication reshard for
// the operand `%arg0` while replicating the sharded update `%arg1`.
// CHECK-LABEL: func @dynamic_update_slice_comm_free_constant_index
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, %[[ARG1:.*]]: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
// CHECK-NEXT: %[[C2:.*]] = sdy.constant dense<2> : tensor<i32>
// CHECK-NEXT: %[[C0:.*]] = sdy.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[RESHARD_UPDATE:.*]] = sdy.reshard %[[ARG1]] <@mesh, [{}, {}]> : tensor<2x8xf32>
// CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[ARG0]], %[[RESHARD_UPDATE]], %[[C2]], %[[C0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
// CHECK-NEXT: return %[[DUS]]
func.func @dynamic_update_slice_comm_free_constant_index(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  %c2 = sdy.constant dense<2> : tensor<i32>
  %c0 = sdy.constant dense<0> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c2, %c0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>
  } : (tensor<16x8xf32>, tensor<2x8xf32>, tensor<i32>, tensor<i32>)
      -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// Constant start index 20 on dim 0 (size 16) with slice size 2 is clamped to
// 14 (`min(max(20, 0), 16 - 2)`), so the slice [14, 16) lies entirely within
// shard 1 [8, 16). Factor resolution skips replication reshard for `%arg0`.
// CHECK-LABEL: func @dynamic_update_slice_clamped_high_constant_index
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, %[[ARG1:.*]]: tensor<2x8xf32>)
// CHECK-NEXT: %[[C20:.*]] = sdy.constant dense<20> : tensor<i32>
// CHECK-NEXT: %[[C0:.*]] = sdy.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[ARG0]], %[[ARG1]], %[[C20]], %[[C0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
// CHECK-NEXT: return %[[DUS]]
func.func @dynamic_update_slice_clamped_high_constant_index(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<2x8xf32>)
    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  %c20 = sdy.constant dense<20> : tensor<i32>
  %c0 = sdy.constant dense<0> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c20, %c0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>
  } : (tensor<16x8xf32>, tensor<2x8xf32>, tensor<i32>, tensor<i32>)
      -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// Constant start index -5 on dim 0 (size 16) with slice size 2 is clamped to
// 0 (`min(max(-5, 0), 16 - 2)`), so the slice [0, 2) lies entirely within
// shard 0 [0, 8). Factor resolution skips replication reshard for `%arg0`.
// CHECK-LABEL: func @dynamic_update_slice_clamped_negative_constant_index
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, %[[ARG1:.*]]: tensor<2x8xf32>)
// CHECK-NEXT: %[[CM5:.*]] = sdy.constant dense<-5> : tensor<i32>
// CHECK-NEXT: %[[C0:.*]] = sdy.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[ARG0]], %[[ARG1]], %[[CM5]], %[[C0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
// CHECK-NEXT: return %[[DUS]]
func.func @dynamic_update_slice_clamped_negative_constant_index(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<2x8xf32>)
    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  %c_minus5 = sdy.constant dense<-5> : tensor<i32>
  %c0 = sdy.constant dense<0> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c_minus5, %c0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>
  } : (tensor<16x8xf32>, tensor<2x8xf32>, tensor<i32>, tensor<i32>)
      -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// Constant update slice [7, 9) on dim 0 with input sharding `[{"a"}, {}]`
// crosses the shard boundary at 8. Triggers replication to `[{}, {}]` and
// reshards back.
// CHECK-LABEL: func @dynamic_update_slice_cross_shard_constant_index
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, %[[ARG1:.*]]: tensor<2x8xf32>)
// CHECK-NEXT: %[[C7:.*]] = sdy.constant dense<7> : tensor<i32>
// CHECK-NEXT: %[[C0:.*]] = sdy.constant dense<0> : tensor<i32>
// CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<16x8xf32>
// CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[RESHARD0]], %[[ARG1]], %[[C7]], %[[C0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
// CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[DUS]] <@mesh, [{"a"}, {}]> : tensor<16x8xf32>
// CHECK-NEXT: return %[[RESHARD1]]
func.func @dynamic_update_slice_cross_shard_constant_index(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<2x8xf32>)
    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  %c7 = sdy.constant dense<7> : tensor<i32>
  %c0 = sdy.constant dense<0> : tensor<i32>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %c7, %c0 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>
  } : (tensor<16x8xf32>, tensor<2x8xf32>, tensor<i32>, tensor<i32>)
      -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// Dynamic update of size 2 along dim 0 with input sharding `[{"a"}, {}]` may
// cross shard boundaries. Triggers replication to `[{}, {}]` and reshards back.
// CHECK-LABEL: func @dynamic_update_slice_cross_shard_dynamic_index
func.func @dynamic_update_slice_cross_shard_dynamic_index(
    %arg0: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<2x8xf32>,
    %arg2: tensor<i32>,
    %arg3: tensor<i32>)
    -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  // CHECK:      %[[RESHARD_IN:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}]>
  // CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[RESHARD_IN]], %arg1, %arg2, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD_OUT:.*]] = sdy.reshard %[[DUS]] <@mesh, [{"a"}, {}]>
  // CHECK-NEXT: return %[[RESHARD_OUT]]
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>
  } : (tensor<16x8xf32>, tensor<2x8xf32>, tensor<i32>, tensor<i32>)
      -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}

// With input sharding `[{"a"}, {"b"}]` and a 1x2 update slice, dim 0 is
// communication-free while dim 1 may cross shard boundaries. Performs partial
// factor-level replication to `[{"a"}, {}]` and reshards back.
// CHECK-LABEL: func @dynamic_update_slice_multidim_mixed
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}, %[[ARG1:.*]]: tensor<1x2xf32>, %[[ARG2:.*]]: tensor<i32>, %[[ARG3:.*]]: tensor<i32>)
// CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{"a"}, {}]>
// CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[RESHARD_IN]], %[[ARG1]], %[[ARG2]], %[[ARG3]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
// CHECK-NEXT: %[[RESHARD_OUT:.*]] = sdy.reshard %[[DUS]] <@mesh, [{"a"}, {"b"}]>
// CHECK-NEXT: return %[[RESHARD_OUT]]
func.func @dynamic_update_slice_multidim_mixed(
    %arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<1x2xf32>,
    %arg2: tensor<i32>,
    %arg3: tensor<i32>)
    -> (tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>
  } : (tensor<16x16xf32>, tensor<1x2xf32>, tensor<i32>, tensor<i32>)
      -> tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// Shard boundaries are unknown at compile time for dynamic dim 0 with input
// sharding `[{"a"}, {}]` and update size 1. Dynamic shape guard priority
// triggers replication to `[{}, {}]` and reshards back.
// CHECK-LABEL: func @dynamic_update_slice_dynamic_operand_shape
// CHECK-SAME: (%[[ARG0:.*]]: tensor<?x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, %[[ARG1:.*]]: tensor<1x8xf32>, %[[ARG2:.*]]: tensor<i32>, %[[ARG3:.*]]: tensor<i32>)
// CHECK-NEXT: %[[RESHARD_IN:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]>
// CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[RESHARD_IN]], %[[ARG1]], %[[ARG2]], %[[ARG3]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
// CHECK-NEXT: %[[RESHARD_OUT:.*]] = sdy.reshard %[[DUS]] <@mesh, [{"a"}, {}]>
// CHECK-NEXT: return %[[RESHARD_OUT]]
func.func @dynamic_update_slice_dynamic_operand_shape(
    %arg0: tensor<?x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
    %arg1: tensor<1x8xf32>,
    %arg2: tensor<i32>,
    %arg3: tensor<i32>)
    -> (tensor<?x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 {
    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>
  } : (tensor<?x8xf32>, tensor<1x8xf32>, tensor<i32>, tensor<i32>)
      -> tensor<?x8xf32>
  return %0 : tensor<?x8xf32>
}

// CHECK-LABEL: func @dynamic_update_slice_missing_operand_sharding
// CHECK-SAME: (%[[ARG0:.*]]: tensor<16x8xf32>, %[[ARG1:.*]]: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %[[ARG2:.*]]: tensor<i32>, %[[ARG3:.*]]: tensor<i32>)
// CHECK-NEXT: %[[DUS:.*]] = stablehlo.dynamic_update_slice %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]]
// CHECK-NEXT: return %[[DUS]]
func.func @dynamic_update_slice_missing_operand_sharding(
    %arg0: tensor<16x8xf32>,
    %arg1: tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %arg2: tensor<i32>,
    %arg3: tensor<i32>)
    -> tensor<16x8xf32> {
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3
      : (tensor<16x8xf32>, tensor<2x8xf32>, tensor<i32>, tensor<i32>)
      -> tensor<16x8xf32>
  return %0 : tensor<16x8xf32>
}
