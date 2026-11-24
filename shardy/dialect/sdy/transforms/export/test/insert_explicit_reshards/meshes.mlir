// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xt = <["x"=2, "t"=4]>
sdy.mesh @mesh_xy = <["x"=2, "y"=3]>
sdy.mesh @mesh_iota = <["x"=3, "y"=2]>
sdy.mesh @mesh_iota_dup = <["x"=3, "y"=2]>
sdy.mesh @mesh_non_iota = <["x"=3, "y"=2], device_ids=[5, 4, 3, 2, 1, 0]>
sdy.mesh @empty_mesh = <[]>


// CHECK-LABEL: func @optimization_barrier_different_meshes
func.func @optimization_barrier_different_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"x"}]>}) {
  // CHECK: %[[RESHARD_1:.*]] = sdy.reshard %arg0 <@mesh, [{}]> : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD_2:.*]] = sdy.reshard %[[RESHARD_1]] <@mesh_xt, [{}]> : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD_3:.*]] = sdy.reshard %[[RESHARD_2]] <@mesh_xt, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{"x"}]>]>} %[[RESHARD_3]]
  %1 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{"x"}]>]>} %arg0 : tensor<210xf32>
  %2 = stablehlo.negate %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{"x"}]>]>} : tensor<210xf32>
  return %2 : tensor<210xf32>
}

// CHECK-LABEL: func @optimization_barrier_meshes_different_device_order
func.func @optimization_barrier_meshes_different_device_order(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_non_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} %[[RESHARD]]
  %1 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} %arg0 : tensor<210xf32>
  %2 = stablehlo.negate %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} : tensor<210xf32>
  return %2 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_empty_sharding_to_iota_sharded
func.func @negate_from_empty_sharding_to_iota_sharded(%arg0: tensor<210xf32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD]]
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_empty_sharding_to_iota_unsharded
func.func @negate_from_empty_sharding_to_iota_unsharded(%arg0: tensor<210xf32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @binary_op_from_empty_sharding_to_iota_unsharded
func.func @binary_op_from_empty_sharding_to_iota_unsharded(%arg0: tensor<210xf32>) -> tensor<210xf32> {
  // CHECK-NEXT: %[[MULTIPLY:.*]] = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}]>]>} : tensor<210xf32>
  // CHECK-NEXT: return %[[MULTIPLY]]
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_empty_sharding_to_non_iota_sharded
func.func @negate_from_empty_sharding_to_non_iota_sharded(%arg0: tensor<210xf32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_non_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD]]
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_empty_sharding_to_non_iota_unsharded
func.func @negate_from_empty_sharding_to_non_iota_unsharded(%arg0: tensor<210xf32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_iota_unsharded_to_non_iota_unsharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}]>})
func.func @negate_from_iota_unsharded_to_non_iota_unsharded(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}]>}) {
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}]>]>} : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh_non_iota, [{}]> : tensor<210xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<210xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_non_iota_sharded_to_empty_sharding
func.func @negate_from_non_iota_sharded_to_empty_sharding(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}]>}) -> (tensor<210xf32>) {
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh_non_iota, [{}]> : tensor<210xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.negate %arg0 : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_axes_compatible_different_device_orders
func.func @negate_axes_compatible_different_device_orders(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}]>}) {
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} : tensor<210xf32>
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}]>]>} : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh_non_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_axes_incompatible_different_device_orders
func.func @negate_axes_incompatible_different_device_orders(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"y"}]>}) {
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"y"}]>]>} : tensor<210xf32>
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}]>]>} : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh_non_iota, [{"y"}]> : tensor<210xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_axes_incompatible_different_device_orders_output_sharding_is_larger
func.func @negate_axes_incompatible_different_device_orders_output_sharding_is_larger(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}]>}) {
  // TODO(enver): In fact, this can go from <@mesh_non_iota, [{"x"}]> to <@mesh_iota, [{"y"}]> in one reshard, and consequently, in one collective permute.
  // CHECK: %[[RESHARD_1:.*]] = sdy.reshard %arg0 <@mesh_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[RESHARD_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}]>]>} : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD_2:.*]] = sdy.reshard %[[NEGATE]] <@mesh_non_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: return %[[RESHARD_2]]
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_same_axes_different_meshes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>})
func.func @negate_same_axes_different_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}]>}) {
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}]>]>} : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[NEGATE]] <@mesh_iota, [{}]> : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[RESHARD0]] <@mesh_xy, [{}]> : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[RESHARD1]] <@mesh_xy, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<210xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_different_axes_different_meshes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>})
func.func @negate_different_axes_different_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>}) {
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}]>]>} : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[NEGATE]] <@mesh_iota, [{}]> : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[RESHARD0]] <@mesh_xy, [{}]> : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[RESHARD1]] <@mesh_xy, [{"y"}]> : tensor<210xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<210xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"y"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}


// CHECK-LABEL: func @dot_same_axes_different_meshes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}]>})
func.func @dot_same_axes_different_meshes(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG1]] <@mesh_xy, [{}, {}]> : tensor<24x12xf32>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[RESHARD0]] <@mesh_iota, [{}, {}]> : tensor<24x12xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[RESHARD1]] <@mesh_iota, [{}, {"y"}]> : tensor<24x12xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[ARG0]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {"y"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  // CHECK-NEXT: return %[[DOT]] : tensor<6x12xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {"y"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @dot_same_axes_different_device_orders_lhs_and_result_majority
func.func @dot_same_axes_different_device_orders_lhs_and_result_majority(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"y"}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh_iota, [{}, {"y"}]>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD]]
  // CHECK-NEXT: return %[[DOT]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {"y"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @dot_same_axes_different_device_orders_lhs_and_rhs_majority
func.func @dot_same_axes_different_device_orders_lhs_and_rhs_majority(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"y"}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {"y"}]>}) {
  // CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh_non_iota, [{"x"}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}, {"y"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @dot_different_axes_different_device_orders_lhs_and_result_majority
func.func @dot_different_axes_different_device_orders_lhs_and_result_majority(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"y"}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {"x"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_iota, [{}, {"y"}]> : tensor<24x12xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {"y"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DOT]] <@mesh_iota, [{"y"}, {"x"}]> : tensor<6x12xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<6x12xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"y"}, {"x"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @dot_different_axes_different_device_orders_lhs_and_rhs_majority
func.func @dot_different_axes_different_device_orders_lhs_and_rhs_majority(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"y"}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"y"}, {"x"}]>}) {
  // CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {"y"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh_non_iota, [{"y"}, {"x"}]> : tensor<6x12xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<6x12xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"y"}, {"x"}]>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @dot_unreduced_result_different_device_orders_lhs_and_rhs_majority
func.func @dot_unreduced_result_different_device_orders_lhs_and_rhs_majority(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"x"}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {}], unreduced={"x"}>}) {
  // CHECK: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {}], unreduced={"x"}>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh_non_iota, [{}, {}], unreduced={"x"}> : tensor<6x12xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<6x12xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{}, {}], unreduced={"x"}>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @dot_unreduced_result_different_device_orders_rhs_and_result_majority
func.func @dot_unreduced_result_different_device_orders_rhs_and_result_majority(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"x"}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {}], unreduced={"x"}>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_non_iota, [{}, {"x"}]> : tensor<6x24xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[RESHARD]], %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{}, {}], unreduced={"x"}>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  // CHECK-NEXT: return %[[DOT]] : tensor<6x12xf32>
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{}, {}], unreduced={"x"}>]>} : (tensor<6x24xf32>, tensor<24x12xf32>) -> tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @reshard_different_meshes_in_unreduced_axes
func.func @reshard_different_meshes_in_unreduced_axes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}], unreduced={"y"}>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"x"}]>}) {
  // CHECK:      %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD_1:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{}]> : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD_2:.*]] = sdy.reshard %[[RESHARD_1]] <@mesh_xt, [{}]> : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD_3:.*]] = sdy.reshard %[[RESHARD_2]] <@mesh_xt, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: return %[[RESHARD_3]] : tensor<210xf32>
  %0 = sdy.reshard %arg0 <@mesh_xt, [{"x"}]> : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @reshard_different_meshes_out_unreduced_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
func.func @reshard_different_meshes_out_unreduced_axes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}], unreduced={"t"}>}) {
  // CHECK-NEXT: %[[RESHARD_1:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]> : tensor<210xf32>
  // CHECK-NEXT: %[[RESHARD_2:.*]] = sdy.reshard %[[RESHARD_1]] <@mesh_xt, [{}]> : tensor<210xf32>
  // CHECK-NEXT: %[[R2U:.*]] = sdy.replicated_to_unreduced {"t"} %[[RESHARD_2]] out_sharding=<@mesh_xt, [{}], unreduced={"t"}> : tensor<210xf32>
  // CHECK-NEXT: return %[[R2U]] : tensor<210xf32>
  %0 = sdy.reshard %arg0 <@mesh_xt, [{}], unreduced={"t"}> : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @add_all_replicated_different_meshes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}]>})
func.func @add_all_replicated_different_meshes(
    %arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
    %arg1: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}]>})
    -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}]>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh_xt, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[RESHARD]], %[[ARG1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{}]>]>} : tensor<24xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<24xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{}]>]>} : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @add_three_meshes_voting
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"t"}]>})
func.func @add_three_meshes_voting(
    %arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>},
    %arg1: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"t"}]>})
    -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}]>}) {
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[ARG0]] <@mesh_xt, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <@mesh_xt, [{"t"}]> : tensor<24xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[R1]], %[[ARG1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{"t"}]>]>} : tensor<24xf32>
  // CHECK-NEXT: %[[R2:.*]] = sdy.reshard %[[ADD]] <@mesh_xt, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R3:.*]] = sdy.reshard %[[R2]] <@mesh_xy, [{}]> : tensor<24xf32>
  // CHECK-NEXT: return %[[R3]] : tensor<24xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{"t"}]>]>} : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @negate_empty_mesh_to_mesh
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>})
func.func @negate_empty_mesh_to_mesh(%arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <@mesh, [{"x"}]> : tensor<24xf32>
  // CHECK-NEXT: %[[NEG:.*]] = stablehlo.negate %[[R1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<24xf32>
  // CHECK-NEXT: return %[[NEG]] : tensor<24xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @negate_mesh_to_empty_mesh
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>})
func.func @negate_mesh_to_empty_mesh(%arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) {
  // CHECK-NEXT: %[[NEG:.*]] = stablehlo.negate %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<24xf32>
  // CHECK-NEXT: %[[R:.*]] = sdy.reshard %[[NEG]] <@empty_mesh, [{}]> : tensor<24xf32>
  // CHECK-NEXT: return %[[R]] : tensor<24xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@empty_mesh, [{}]>]>} : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @reshape_cross_mesh
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @reshape_cross_mesh(%arg0: tensor<8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<2x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"t"}, {}]>}) {
  // CHECK-NEXT: %[[RESHAPE:.*]] = stablehlo.reshape %[[ARG0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"x":(2)2}]>]>} : (tensor<8x2xf32>) -> tensor<2x8xf32>
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[RESHAPE]] <@mesh, [{}, {}]> : tensor<2x8xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <@mesh_xt, [{}, {}]> : tensor<2x8xf32>
  // CHECK-NEXT: %[[R2:.*]] = sdy.reshard %[[R1]] <@mesh_xt, [{"t"}, {}]> : tensor<2x8xf32>
  // CHECK-NEXT: return %[[R2]] : tensor<2x8xf32>
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{"t"}, {}]>]>} : (tensor<8x2xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// CHECK-LABEL: func @reshard_same_axes_diff_device_ids_out_unreduced
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>})
func.func @reshard_same_axes_diff_device_ids_out_unreduced(%arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}], unreduced={"x"}>}) {
  // CHECK-NEXT: %[[MOVED:.*]] = sdy.reshard %[[ARG0]] <@mesh_non_iota, [{"x"}]> : tensor<24xf32>
  // CHECK-NEXT: %[[S2U:.*]] = sdy.sharded_to_unreduced [{"x"}] %[[MOVED]] out_sharding=<@mesh_non_iota, [{}], unreduced={"x"}> : tensor<24xf32>
  // CHECK-NEXT: return %[[S2U]] : tensor<24xf32>
  %0 = sdy.reshard %arg0 <@mesh_non_iota, [{}], unreduced={"x"}> : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @reshard_cross_mesh_unreduced_to_unreduced
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}], unreduced={"x"}>})
func.func @reshard_cross_mesh_unreduced_to_unreduced(%arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}], unreduced={"x"}>}) -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}], unreduced={"x"}>}) {
  // CHECK-NEXT: %[[AR:.*]] = sdy.all_reduce {"x"} %[[ARG0]] out_sharding=<@mesh_iota, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R:.*]] = sdy.reshard %[[AR]] <@mesh_xy, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R2U:.*]] = sdy.replicated_to_unreduced {"x"} %[[R]] out_sharding=<@mesh_xy, [{}], unreduced={"x"}> : tensor<24xf32>
  // CHECK-NEXT: return %[[R2U]] : tensor<24xf32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{}], unreduced={"x"}> : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @reshard_cross_mesh_replicated_preserved
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}]>})
func.func @reshard_cross_mesh_replicated_preserved(%arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}]>}) -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}]>}) {
  // CHECK-NEXT: %[[R:.*]] = sdy.reshard %[[ARG0]] <@mesh_xy, [{}]> : tensor<24xf32>
  // CHECK-NEXT: return %[[R]] : tensor<24xf32>
  %0 = sdy.reshard %arg0 <@mesh_xy, [{}]> : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @reshard_cross_mesh_inlined_mesh_attrs
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}]>})
func.func @reshard_cross_mesh_inlined_mesh_attrs(
    %arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}]>})
    -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=2]>, [{"a"}]>}) {
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[ARG0]] <mesh<["x"=2, "y"=2]>, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <mesh<["a"=2, "b"=2]>, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R2:.*]] = sdy.reshard %[[R1]] <mesh<["a"=2, "b"=2]>, [{"a"}]> : tensor<24xf32>
  // CHECK-NEXT: return %[[R2]] : tensor<24xf32>
  %0 = sdy.reshard %arg0 <mesh<["a"=2, "b"=2]>, [{"a"}]> : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @reshard_cross_mesh_inlined_to_symbol_unreduced
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}]>})
func.func @reshard_cross_mesh_inlined_to_symbol_unreduced(
    %arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}]>})
    -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{}], unreduced={"t"}>}) {
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[ARG0]] <mesh<["x"=2, "y"=2]>, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <@mesh_xt, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R2U:.*]] = sdy.replicated_to_unreduced {"t"} %[[R1]] out_sharding=<@mesh_xt, [{}], unreduced={"t"}> : tensor<24xf32>
  // CHECK-NEXT: return %[[R2U]] : tensor<24xf32>
  %0 = sdy.reshard %arg0 <@mesh_xt, [{}], unreduced={"t"}> : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @add_inlined_operand_to_symbol_mesh
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=4]>, [{"a"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @add_inlined_operand_to_symbol_mesh(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=4]>, [{"a"}, {}]>},
    %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[ARG0]] <mesh<["a"=2, "b"=4]>, [{}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <@mesh, [{}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[R2:.*]] = sdy.reshard %[[R1]] <@mesh, [{"x"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[R2]], %[[ARG1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @func_return_cross_mesh_inlined_attrs
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}]>})
func.func @func_return_cross_mesh_inlined_attrs(
    %arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=2, "y"=2]>, [{"x"}]>})
    -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=2]>, [{"a"}]>}) {
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[ARG0]] <mesh<["x"=2, "y"=2]>, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <mesh<["a"=2, "b"=2]>, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R2:.*]] = sdy.reshard %[[R1]] <mesh<["a"=2, "b"=2]>, [{"a"}]> : tensor<24xf32>
  // CHECK-NEXT: return %[[R2]] : tensor<24xf32>
  return %arg0 : tensor<24xf32>
}

// CHECK-LABEL: func @add_empty_mesh_replicated_and_mesh_replicated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>})
func.func @add_empty_mesh_replicated_and_mesh_replicated(
    %arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>},
    %arg1: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>})
    -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) {
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[R0]], %[[ARG1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : tensor<24xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[ADD]] <@empty_mesh, [{}]> : tensor<24xf32>
  // CHECK-NEXT: return %[[R1]] : tensor<24xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@empty_mesh, [{}]>]>} : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// --- stablehlo.dot_general: Group 1 (same axes, different device_ids) ---

// Group 1a: batch on @mesh_iota, contracting on @mesh_non_iota, disjoint axes,
// @mesh_iota wins majority. Both factor shardings are preserved; %arg1 gets a
// 1-step cross-device_ids reshard onto @mesh_iota.
// CHECK-LABEL: func @dot_general_cross_device_ids_disjoint_axes_iota_majority
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"y"}, {}]>})
func.func @dot_general_cross_device_ids_disjoint_axes_iota_majority(
    %arg0: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}, {}]>},
    %arg1: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"y"}, {}]>})
    -> (tensor<6x12x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh_iota, [{"x"}, {}, {"y"}]> : tensor<6x12x18xf32>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG1]] <@mesh_iota, [{"x"}, {"y"}, {}]> : tensor<6x18x12xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[RESHARD0]], %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {}, {}]>]>} : (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh_iota, [{"x"}, {}, {}]> : tensor<6x12x12xf32>
  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<6x12x12xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {}, {}]>]>} :
    (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  return %0 : tensor<6x12x12xf32>
}

// Group 1b: @mesh_non_iota wins majority, so dot_general + all_reduce run on
// @mesh_non_iota and only the result reshards onto @mesh_iota.
// CHECK-LABEL: func @dot_general_cross_device_ids_disjoint_axes_non_iota_majority
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"y"}, {}]>})
func.func @dot_general_cross_device_ids_disjoint_axes_non_iota_majority(
    %arg0: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}, {}]>},
    %arg1: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"y"}, {}]>})
    -> (tensor<6x12x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh_non_iota, [{"x"}, {}, {"y"}]> : tensor<6x12x18xf32>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG1]] <@mesh_non_iota, [{"x"}, {"y"}, {}]> : tensor<6x18x12xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[RESHARD0]], %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}, {}, {}]>]>} : (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[DOT]] out_sharding=<@mesh_non_iota, [{"x"}, {}, {}]> : tensor<6x12x12xf32>
  // CHECK-NEXT: %[[RESHARD_OUT:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh_iota, [{"x"}, {}, {}]> : tensor<6x12x12xf32>
  // CHECK-NEXT: return %[[RESHARD_OUT]] : tensor<6x12x12xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {}, {}]>]>} :
    (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  return %0 : tensor<6x12x12xf32>
}

// Group 1c (batch wins): batch and contracting both use "x" across meshes;
// batch wins by larger total tensor size and precedence (no all_reduce).
// CHECK-LABEL: func @dot_general_cross_device_ids_conflicting_axis_batch_wins
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"x"}, {}]>})
func.func @dot_general_cross_device_ids_conflicting_axis_batch_wins(
    %arg0: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}, {}]>},
    %arg1: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"x"}, {}]>})
    -> (tensor<6x12x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG1]] <@mesh_iota, [{"x"}, {}, {}]> : tensor<6x18x12xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[ARG0]], %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {}, {}]>]>} : (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  // CHECK-NEXT: return %[[DOT]] : tensor<6x12x12xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {}, {}]>]>} :
    (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  return %0 : tensor<6x12x12xf32>
}

// Group 1c (contracting wins): %arg1 is larger than (%arg0 + %dot), so
// contracting "x" wins over batch "x" despite @mesh_iota winning mesh majority.
// CHECK-LABEL: func @dot_general_cross_device_ids_conflicting_axis_contracting_wins
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<6x12x48xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"x"}, {}]>})
func.func @dot_general_cross_device_ids_conflicting_axis_contracting_wins(
    %arg0: tensor<6x6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}, {}]>},
    %arg1: tensor<6x12x48xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"x"}, {}]>})
    -> (tensor<6x6x48xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ARG0]] <@mesh_iota, [{}, {}, {"x"}]> : tensor<6x6x12xf32>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG1]] <@mesh_iota, [{}, {"x"}, {}]> : tensor<6x12x48xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[RESHARD0]], %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {}, {}]>]>} : (tensor<6x6x12xf32>, tensor<6x12x48xf32>) -> tensor<6x6x48xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[DOT]] out_sharding=<@mesh_iota, [{}, {}, {}]> : tensor<6x6x48xf32>
  // CHECK-NEXT: %[[RESHARD_OUT:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh_iota, [{"x"}, {}, {}]> : tensor<6x6x48xf32>
  // CHECK-NEXT: return %[[RESHARD_OUT]] : tensor<6x6x48xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {}, {}]>]>} :
    (tensor<6x6x12xf32>, tensor<6x12x48xf32>) -> tensor<6x6x48xf32>
  return %0 : tensor<6x6x48xf32>
}

// Group 1d: non-contracting M on @mesh_iota conflicts on "x" with contracting K
// on @mesh_non_iota, dropping K's sharding to replicated on @mesh_iota.
// CHECK-LABEL: func @dot_general_cross_device_ids_conflict_to_replicated_rhs
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"x"}, {}]>})
func.func @dot_general_cross_device_ids_conflict_to_replicated_rhs(
    %arg0: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"x"}, {}]>},
    %arg1: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"x"}, {}]>})
    -> (tensor<6x12x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"x"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[ARG1]] <@mesh_iota, [{}, {}, {}]> : tensor<6x18x12xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[ARG0]], %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {"x"}, {}]>]>} : (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  // CHECK-NEXT: return %[[DOT]] : tensor<6x12x12xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {"x"}, {}]>]>} :
    (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  return %0 : tensor<6x12x12xf32>
}

// --- stablehlo.dot_general: Group 2 (different axis structures, needsPrePivot) ---

// Group 2: @mesh_xy (["x"=2, "y"=3]) and @mesh_non_iota (["x"=3, "y"=2]) have
// equal total device count (6) but incompatible axis structures, so
// prePivotForeignMeshes wipes %arg1's contracting sharding via a 2-step
// replicated pivot before slicing %arg1 to batch "x".
// CHECK-LABEL: func @dot_general_different_axes_pre_pivot
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"x"}, {}]>})
func.func @dot_general_different_axes_pre_pivot(
    %arg0: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}, {}]>},
    %arg1: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {"x"}, {}]>})
    -> (tensor<6x12x12xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}, {}, {}]>}) {
  // CHECK-NEXT: %[[PIVOT0:.*]] = sdy.reshard %[[ARG1]] <@mesh_non_iota, [{}, {}, {}]> : tensor<6x18x12xf32>
  // CHECK-NEXT: %[[PIVOT1:.*]] = sdy.reshard %[[PIVOT0]] <@mesh_xy, [{}, {}, {}]> : tensor<6x18x12xf32>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[PIVOT1]] <@mesh_xy, [{"x"}, {}, {}]> : tensor<6x18x12xf32>
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[ARG0]], %[[RESHARD1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}]>]>} : (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  // CHECK-NEXT: return %[[DOT]] : tensor<6x12x12xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {}, {}]>]>} :
    (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  return %0 : tensor<6x12x12xf32>
}

// --- stablehlo.dot_general: Group 3 (unreduced axes across device_ids) ---

// Group 3a: dot_general runs on @mesh_iota producing dim-replicated unreduced
// output, then reshards across device_ids to @mesh_non_iota with unreduced.
// CHECK-LABEL: func @dot_general_unreduced_cross_device_ids_replicated_dims
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {}, {"x"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"x"}, {}]>})
func.func @dot_general_unreduced_cross_device_ids_replicated_dims(
    %arg0: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {}, {"x"}]>},
    %arg1: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}, {"x"}, {}]>})
    -> (tensor<6x12x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}, {}, {}], unreduced={"x"}>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}, {}, {}], unreduced={"x"}>]>} : (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh_non_iota, [{}, {}, {}], unreduced={"x"}> : tensor<6x12x12xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<6x12x12xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{}, {}, {}], unreduced={"x"}>]>} :
    (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  return %0 : tensor<6x12x12xf32>
}

// Group 3b: dot_general with sharded batch and unreduced contracting axes
// across @mesh_iota -> @mesh_non_iota.
// CHECK-LABEL: func @dot_general_unreduced_cross_device_ids_sharded_batch
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}, {"x"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {"x"}, {}]>})
func.func @dot_general_unreduced_cross_device_ids_sharded_batch(
    %arg0: tensor<6x12x18xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}, {"x"}]>},
    %arg1: tensor<6x18x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {"x"}, {}]>})
    -> (tensor<6x12x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"y"}, {}, {}], unreduced={"x"}>}) {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot_general %[[ARG0]], %[[ARG1]], batching_dims = [0] x [0], contracting_dims = [2] x [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"y"}, {}, {}], unreduced={"x"}>]>} : (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh_non_iota, [{"y"}, {}, {}], unreduced={"x"}> : tensor<6x12x12xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<6x12x12xf32>
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"y"}, {}, {}], unreduced={"x"}>]>} :
    (tensor<6x12x18xf32>, tensor<6x18x12xf32>) -> tensor<6x12x12xf32>
  return %0 : tensor<6x12x12xf32>
}

// Group 3c: same-axes different-device-ids reshard and return boundary with
// matching unreduced=max axes preserve 1-step permute without all_reduce + r2u.
// CHECK-LABEL: func @reshard_same_axes_diff_device_ids_matching_unreduced_max
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}], unreduced=max{"x"}>})
func.func @reshard_same_axes_diff_device_ids_matching_unreduced_max(
    %arg0: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}], unreduced=max{"x"}>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"y"}, {}], unreduced=max{"x"}>}) {
  // CHECK-NOT: sdy.all_reduce
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh_non_iota, [{"y"}, {}], unreduced=max{"x"}> : tensor<6x12xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<6x12xf32>
  %0 = sdy.reshard %arg0 <@mesh_non_iota, [{"y"}, {}], unreduced=max{"x"}> : tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// CHECK-LABEL: func @func_return_same_axes_diff_device_ids_matching_unreduced_max
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}], unreduced=max{"x"}>})
func.func @func_return_same_axes_diff_device_ids_matching_unreduced_max(
    %arg0: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"y"}, {}], unreduced=max{"x"}>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"y"}, {}], unreduced=max{"x"}>}) {
  // CHECK-NOT: sdy.all_reduce
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ARG0]] <@mesh_non_iota, [{"y"}, {}], unreduced=max{"x"}> : tensor<6x12xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<6x12xf32>
  return %arg0 : tensor<6x12xf32>
}

// Mixing a non-iota symbol mesh and an iota inlined mesh with the same axes
// falls back cleanly to the symbol mesh without crashing in getMesh.
// CHECK-LABEL: func @add_non_iota_and_inlined_iota_same_axes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=3, "y"=2]>, [{"x"}, {}]>})
func.func @add_non_iota_and_inlined_iota_same_axes(
    %arg0: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>},
    %arg1: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<mesh<["x"=3, "y"=2]>, [{"x"}, {}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ARG1]] <@mesh_non_iota, [{"x"}, {}]> : tensor<6x12xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}, {}]>]>} : tensor<6x12xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<6x12xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}, {}]>]>} : tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}

// Distinct mesh symbols with identical axes and identical device_ids must
// pivot through replicated instead of emitting a non-replicated cross-symbol
// reshard across identical device_ids.
// CHECK-LABEL: func @reshard_identical_mesh_symbols_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>})
func.func @reshard_identical_mesh_symbols_sharded(
    %arg0: tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>})
    -> (tensor<24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota_dup, [{"x"}]>}) {
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[ARG0]] <@mesh_iota, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <@mesh_iota_dup, [{}]> : tensor<24xf32>
  // CHECK-NEXT: %[[R2:.*]] = sdy.reshard %[[R1]] <@mesh_iota_dup, [{"x"}]> : tensor<24xf32>
  // CHECK-NEXT: return %[[R2]] : tensor<24xf32>
  %0 = sdy.reshard %arg0 <@mesh_iota_dup, [{"x"}]> : tensor<24xf32>
  return %0 : tensor<24xf32>
}

// CHECK-LABEL: func @add_identical_mesh_symbols_sharded
// CHECK-SAME: (%[[ARG0:.*]]: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota_dup, [{"x"}, {}]>})
func.func @add_identical_mesh_symbols_sharded(
    %arg0: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
    %arg1: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota_dup, [{"x"}, {}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[P0:.*]] = sdy.reshard %[[ARG1]] <@mesh_iota_dup, [{}, {}]> : tensor<6x12xf32>
  // CHECK-NEXT: %[[P1:.*]] = sdy.reshard %[[P0]] <@mesh_iota, [{}, {}]> : tensor<6x12xf32>
  // CHECK-NEXT: %[[P2:.*]] = sdy.reshard %[[P1]] <@mesh_iota, [{"x"}, {}]> : tensor<6x12xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ARG0]], %[[P2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {}]>]>} : tensor<6x12xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<6x12xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{"x"}, {}]>]>} : tensor<6x12xf32>
  return %0 : tensor<6x12xf32>
}
