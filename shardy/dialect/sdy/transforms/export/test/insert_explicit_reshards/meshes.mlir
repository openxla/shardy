// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xt = <["x"=2, "t"=4]>
sdy.mesh @mesh_xy = <["x"=2, "y"=3]>
sdy.mesh @mesh_iota = <["x"=3, "y"=2]>
sdy.mesh @mesh_non_iota = <["x"=3, "y"=2], device_ids=[5, 4, 3, 2, 1, 0]>
sdy.mesh @mesh_maximal = #sdy.mesh<[], device_ids=[0]>
sdy.mesh @mesh_maximal_another = #sdy.mesh<[], device_ids=[1]>
sdy.mesh @mesh_maximal_copy = #sdy.mesh<[], device_ids=[0]>
sdy.mesh @empty_mesh = <[]>

// CHECK-LABEL: func @optimization_barrier_different_meshes
func.func @optimization_barrier_different_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xt, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xt, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{"x"}]>]>} %[[RESHARD]]
  %1 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xt, [{"x"}]>]>} %arg0 : tensor<210xf32>
  %2 = stablehlo.negate %1 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_xt, [{"x"}]>]>} : tensor<210xf32>
  return %2 : tensor<210xf32>
}

// CHECK-LABEL: func @optimization_barrier_meshes_different_device_order
func.func @optimization_barrier_meshes_different_device_order(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_non_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} %[[RESHARD]]
  %1 = stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} %arg0 : tensor<210xf32>
  %2 = stablehlo.negate %1 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} : tensor<210xf32>
  return %2 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_empty_sharding_to_iota_sharded
func.func @negate_from_empty_sharding_to_iota_sharded(%arg0: tensor<210xf32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD]]
  %0 = stablehlo.negate %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_iota, [{"x"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_empty_sharding_to_iota_unsharded
func.func @negate_from_empty_sharding_to_iota_unsharded(%arg0: tensor<210xf32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.negate %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_iota, [{}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @binary_op_from_empty_sharding_to_iota_unsharded
func.func @binary_op_from_empty_sharding_to_iota_unsharded(%arg0: tensor<210xf32>) -> tensor<210xf32> {
  // CHECK-NEXT: %[[MULTIPLY:.*]] = stablehlo.multiply %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_iota, [{}]>]>} : tensor<210xf32>
  // CHECK-NEXT: return %[[MULTIPLY]]
  %0 = stablehlo.multiply %arg0, %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_iota, [{}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_empty_sharding_to_non_iota_sharded
func.func @negate_from_empty_sharding_to_non_iota_sharded(%arg0: tensor<210xf32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_non_iota, [{"x"}]> : tensor<210xf32>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD]]
  %0 = stablehlo.negate %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_empty_sharding_to_non_iota_unsharded
func.func @negate_from_empty_sharding_to_non_iota_unsharded(%arg0: tensor<210xf32>) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.negate %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_non_iota, [{}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_from_iota_unsharded_to_non_iota_unsharded
func.func @negate_from_iota_unsharded_to_non_iota_unsharded(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.negate %arg0 {sdy.sharding= #sdy.sharding_per_value<[<@mesh_non_iota, [{}]>]>} : tensor<210xf32>
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
func.func @negate_same_axes_different_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"x"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_different_axes_different_meshes
func.func @negate_different_axes_different_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{"y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"y"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_identical_maximal_meshes
func.func @negate_identical_maximal_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_maximal, []>}) -> tensor<210xf32> {
  // CHECK-NOT: sdy.reshard
  // TODO(enver): Reshard to output mesh.
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_maximal_copy, []>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @negate_different_maximal_meshes
func.func @negate_different_maximal_meshes(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh_maximal, []>}) -> tensor<210xf32> {
  // CHECK-NOT: sdy.reshard
  // TODO(enver): Reshard to output mesh.
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_maximal_another, []>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func @dot_same_axes_different_meshes
func.func @dot_same_axes_different_meshes(
    %arg0: tensor<6x24xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {}]>},
    %arg1: tensor<24x12xf32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}]>})
    -> (tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh_iota, [{"x"}, {"y"}]>}) {
  // CHECK-NOT: sdy.reshard
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

// CHECK-LABEL: reshard_to_empty_mesh_from_no_sharding
func.func @reshard_to_empty_mesh_from_no_sharding(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@empty_mesh, [{}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@empty_mesh, [{}]> : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: reshard_to_empty_mesh_from_empty_mesh
func.func @reshard_to_empty_mesh_from_empty_mesh(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> tensor<8xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@empty_mesh, [{}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@empty_mesh, [{}]> : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: reshard_to_empty_mesh_from_fully_replicated
func.func @reshard_to_empty_mesh_from_fully_replicated(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) -> tensor<8xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] =  sdy.reshard %arg0 <@mesh, [{}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@empty_mesh, [{}]> : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: reshard_to_empty_mesh_from_sharded
func.func @reshard_to_empty_mesh_from_sharded(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<8xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@empty_mesh, [{}]> : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: reshard_to_fully_replicated_from_empty_mesh
func.func @reshard_to_fully_replicated_from_empty_mesh(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> tensor<8xf32> {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh, [{}]> : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: reshard_to_fully_replicated_from_empty_mesh_which_is_op_also_from_empty_mesh_twice
func.func @reshard_to_fully_replicated_from_empty_mesh_which_is_op_also_from_empty_mesh_twice(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@empty_mesh, [{}]>]>}
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[ABS]] {sdy.sharding = #sdy.sharding_per_value<[<@empty_mesh, [{}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh, [{}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@empty_mesh, [{}]>]>}: tensor<8xf32>
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@empty_mesh, [{}]>]>}: tensor<8xf32>
  %2 = sdy.reshard %1 <@mesh, [{}]> : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: reshard_to_sharded_from_empty_mesh
func.func @reshard_to_sharded_from_empty_mesh(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = sdy.reshard %arg0 <@mesh, [{"x"}]> : tensor<8xf32>
  return %0 : tensor<8xf32>
}
