// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>

// CHECK-LABEL: func @clamp
func.func @clamp(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg2 <@mesh, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[CLAMP:.*]] = stablehlo.clamp %[[RESHARD1]], %arg1, %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CLAMP]] <@mesh, [{}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<4x8xf32>
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @clamp_scalar_min_max
func.func @clamp_scalar_min_max(%arg0: tensor<f32>, %arg1: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}]>}, %arg2: tensor<f32>) -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"z"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{"y"}, {"z"}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[CLAMP:.*]] = stablehlo.clamp %arg0, %0, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {"z"}]>]>} : (tensor<f32>, tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %1 : tensor<4x8xf32>
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {"z"}]>]>}: (tensor<f32>, tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @select
func.func @select(%arg0: tensor<4x8xi1>, %arg1: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x8xi1>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg2 <@mesh, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[SELECT:.*]] = stablehlo.select %[[RESHARD1]], %arg1, %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x8xi1>, tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[SELECT]] <@mesh, [{}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<4x8xf32>
  %0 = stablehlo.select %arg0, %arg1, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @select_scalar_pred
func.func @select_scalar_pred(%arg0: tensor<i1>, %arg1: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg2 <@mesh, [{"x"}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: %[[SELECT:.*]] = stablehlo.select %arg0, %arg1, %[[RESHARD1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<i1>, tensor<4x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[SELECT]] <@mesh, [{}, {}]> : tensor<4x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x8xf32>
  %0 = stablehlo.select %arg0, %arg1, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<i1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
