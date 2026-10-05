// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

sdy.mesh @mesh_4 = <["x"=4]>
sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @non_sharded
func.func @non_sharded() -> tensor<16xi32> {
  // CHECK: %[[RES:.*]] = stablehlo.iota dim = 0 : tensor<16xi32>
  // CHECK: return %[[RES]]
  %0 = stablehlo.iota dim = 0 : tensor<16xi32>
  return %0 : tensor<16xi32>
}


// CHECK-LABEL: func @iota_on_non_sharded_dim
// CHECK-SAME:   -> (tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) {
func.func @iota_on_non_sharded_dim() -> (tensor<8x16xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) {
  // CHECK:  %[[RES:.*]] = stablehlo.iota dim = 0 : tensor<8x4xi32>
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}]>]>} : tensor<8x16xi32>
  // CHECK:  return %[[RES]] : tensor<8x4xi32>
  return %0 : tensor<8x16xi32>
}

// CHECK-LABEL: func @iota_on_sharded_dim
// CHECK-SAME:   -> (tensor<8x4xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) {
func.func @iota_on_sharded_dim() -> (tensor<8x16xi32> {sdy.sharding = #sdy.sharding<@mesh_4, [{}, {"x"}]>}) {
  // CHECK: %[[LOCAL_IOTA:.*]] = stablehlo.iota dim = 1 : tensor<8x4xi32>
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 4, 8, 12]> : tensor<4xi32>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK: %[[OFFSET_I32:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK: %[[OFFSET_CONV:.*]] = stablehlo.convert %[[OFFSET_I32]] : tensor<i32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFFSET_CONV]], dims = []  : (tensor<i32>) -> tensor<8x4xi32>
  // CHECK: %[[RES:.*]] = stablehlo.add %[[LOCAL_IOTA]], %[[BCAST]] : tensor<8x4xi32>
  %0 = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4, [{}, {"x"}]>]>} : tensor<8x16xi32>
  // CHECK: return %[[RES]] : tensor<8x4xi32>
  return %0 : tensor<8x16xi32>
}

// CHECK-LABEL: func @sharded_2d
// CHECK-SAME:   -> (tensor<4x4xi32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>})
func.func @sharded_2d() -> (tensor<16x8xi32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {"y"}]>}) {
  // CHECK: %[[LOCAL_IOTA:.*]] = stablehlo.iota dim = 1 : tensor<4x4xi32>
  // CHECK: %[[PID:.*]] = stablehlo.partition_id

  // Offset table for 8 devices (4x2 mesh). Dimension 1 is sharded by "y".
  // Coordinates for "y": [0, 1, 0, 1, 0, 1, 0, 1]. Local size 4 -> Offsets: [0, 4, 0, 4, ...]
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[0, 4, 0, 4, 0, 4, 0, 4]> : tensor<8xi32>
  // CHECK: %[[OFFSET_S:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]]
  // CHECK: %[[OFFSET_C:.*]] = stablehlo.convert %{{.*}} : tensor<i32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFFSET_C]], dims = [] : (tensor<i32>) -> tensor<4x4xi32>
  // CHECK:  %[[RES:.*]] = stablehlo.add %[[LOCAL_IOTA]], %[[BCAST]]
  %0 = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {"y"}]>]>} : tensor<16x8xi32>
  // CHECK: return %[[RES]] : tensor<4x4xi32>
  return %0 : tensor<16x8xi32>
}
