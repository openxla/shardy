// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @dynamic_slice
func.func @dynamic_slice(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<32x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"y"}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {}, {}]> : tensor<32x4x8xf32>
  // CHECK-NEXT: %[[DYNAMIC_SLICE:.*]] = stablehlo.dynamic_slice %[[RESHARD1]], %arg1, %arg2, %arg3, sizes = [32, 1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DYNAMIC_SLICE]] <@mesh, [{}, {}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD2]]
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [32, 1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {"y"}]>]>}: (tensor<32x4x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}

// CHECK-LABEL: func @dynamic_slice_input_output_same_sharding
func.func @dynamic_slice_input_output_same_sharding(%arg0: tensor<32x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> (tensor<32x4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x", "y"}, {}, {}]> : tensor<32x8x8xf32>
  // CHECK-NEXT: %[[DYNAMIC_SLICE:.*]] = stablehlo.dynamic_slice %[[RESHARD]], %arg1, %arg2, %arg3, sizes = [32, 4, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}, {}]>]>} : (tensor<32x8x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x2xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DYNAMIC_SLICE:.*]] <@mesh, [{}, {"x"}, {"y"}]> : tensor<32x4x2xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<32x4x2xf32>
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [32, 4, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {"y"}]>]>}: (tensor<32x8x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x2xf32>
  return %0 : tensor<32x4x2xf32>
}

// CHECK-LABEL: func @dynamic_slice_batching_dim_is_sharded_on_input
func.func @dynamic_slice_batching_dim_is_sharded_on_input(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<32x1x2xf32> {
  // CHECK: %[[DYNAMIC_SLICE:.*]] = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [32, 1, 2]
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DYNAMIC_SLICE]] <@mesh, [{}, {}, {}]> : tensor<32x1x2xf32>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, %arg3, sizes = [32, 1, 2] : (tensor<32x4x8xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}

// CHECK-LABEL: func @dynamic_update_slice
func.func @dynamic_update_slice(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}, %arg1: tensor<32x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"y"}]>}, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> (tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{}, {}, {}]> : tensor<32x1x2xf32>
  // CHECK-NEXT: %[[DYNAMIC_UPDATE_SLICE:.*]] = stablehlo.dynamic_update_slice %arg0, %[[RESHARD1]], %arg2, %arg3, %arg4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {"y"}]>]>}
  // CHECK-NEXT: return %[[DYNAMIC_UPDATE_SLICE]] : tensor<32x4x8xf32>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {"y"}]>]>} : (tensor<32x4x8xf32>, tensor<32x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x8xf32>
  return %0 : tensor<32x4x8xf32>
}

// CHECK-LABEL: func @dynamic_update_slice_different_input_and_output_sharding
func.func @dynamic_update_slice_different_input_and_output_sharding(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}, %arg1: tensor<32x1x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"y"}]>}, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> (tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {"x"}]>}){
  // CHECK-NEXT: %0 = sdy.reshard %arg1 <@mesh, [{}, {}, {}]> : tensor<32x1x2xf32>
  // CHECK-NEXT: %1 = stablehlo.dynamic_update_slice %arg0, %0, %arg2, %arg3, %arg4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {"y"}]>]>} : (tensor<32x4x8xf32>, tensor<32x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x8xf32>
  // CHECK-NEXT: %2 = sdy.reshard %1 <@mesh, [{}, {"y"}, {"x"}]> : tensor<32x4x8xf32>
  // CHECK-NEXT: return %2 : tensor<32x4x8xf32>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {"x"}]>]>} : (tensor<32x4x8xf32>, tensor<32x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x8xf32>
  return %0 : tensor<32x4x8xf32>
}

// CHECK-LABEL: func @dynamic_update_slice_batching_dim_is_sharded_on_input
func.func @dynamic_update_slice_batching_dim_is_sharded_on_input(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<32x1x2xf32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<i32>) -> tensor<32x4x8xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<32x1x2xf32>
  // CHECK-NEXT: %[[DYNAMIC_UPDATE_SLICE:.*]] = stablehlo.dynamic_update_slice %arg0, %[[RESHARD1]], %arg2, %arg3, %arg4 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[DYNAMIC_UPDATE_SLICE]] <@mesh, [{}, {}, {}]> : tensor<32x4x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]]
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4 : (tensor<32x4x8xf32>, tensor<32x1x2xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<32x4x8xf32>
  return %0 : tensor<32x4x8xf32>
}
