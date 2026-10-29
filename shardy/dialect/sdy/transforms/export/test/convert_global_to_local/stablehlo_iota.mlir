// RUN: sdy_opt %s -sdy-convert-global-to-local | FileCheck %s

sdy.mesh @mesh_4 = <["x"=4]>
sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>
sdy.mesh @mesh_non_iota = <["x"=2, "y"=2], device_ids=[3, 1, 2, 0]>

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
  // CHECK: %[[IDX:.*]] = stablehlo.convert %[[PID]] : (tensor<ui32>) -> tensor<i32>
  // CHECK: %[[C4:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK: %[[OFFSET_I32:.*]] = stablehlo.multiply %[[IDX]], %[[C4]] : tensor<i32>
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
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[C2:.*]] = stablehlo.constant dense<2> : tensor<ui32>
  // CHECK: %[[REM:.*]] = stablehlo.remainder %[[PID]], %[[C2]] : tensor<ui32>
  // CHECK: %[[IDX:.*]] = stablehlo.convert %[[REM]] : (tensor<ui32>) -> tensor<i32>
  // CHECK: %[[C4:.*]] = stablehlo.constant dense<4> : tensor<i32>
  // CHECK: %[[OFFSET_I32:.*]] = stablehlo.multiply %[[IDX]], %[[C4]] : tensor<i32>
  // CHECK: %[[OFFSET_C:.*]] = stablehlo.convert %[[OFFSET_I32]] : tensor<i32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFFSET_C]], dims = [] : (tensor<i32>) -> tensor<4x4xi32>
  // CHECK:  %[[RES:.*]] = stablehlo.add %[[LOCAL_IOTA]], %[[BCAST]]
  %0 = stablehlo.iota dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {"y"}]>]>} : tensor<16x8xi32>
  // CHECK: return %[[RES]] : tensor<4x4xi32>
  return %0 : tensor<16x8xi32>
}

// CHECK-LABEL: func @sharded_non_iota_device_ids
// CHECK-SAME:   -> (tensor<4x8xi32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>})
func.func @sharded_non_iota_device_ids() -> (tensor<8x8xi32> {sdy.sharding = #sdy.sharding<@mesh_non_iota, [{"x"}, {}]>}) {
  // CHECK: %[[LOCAL_IOTA:.*]] = stablehlo.iota dim = 0 : tensor<4x8xi32>
  // CHECK: %[[PID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK: %[[TABLE:.*]] = stablehlo.constant dense<[4, 0, 4, 0]> : tensor<4xi32>
  // CHECK: %[[SLICE:.*]] = stablehlo.dynamic_slice %[[TABLE]], %[[PID]], sizes = [1] : (tensor<4xi32>, tensor<ui32>) -> tensor<1xi32>
  // CHECK: %[[OFFSET_I32:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK: %[[OFFSET_C:.*]] = stablehlo.convert %[[OFFSET_I32]] : tensor<i32>
  // CHECK: %[[BCAST:.*]] = stablehlo.broadcast_in_dim %[[OFFSET_C]], dims = [] : (tensor<i32>) -> tensor<4x8xi32>
  // CHECK: %[[RES:.*]] = stablehlo.add %[[LOCAL_IOTA]], %[[BCAST]] : tensor<4x8xi32>
  %0 = stablehlo.iota dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_non_iota, [{"x"}, {}]>]>} : tensor<8x8xi32>
  // CHECK: return %[[RES]] : tensor<4x8xi32>
  return %0 : tensor<8x8xi32>
}
