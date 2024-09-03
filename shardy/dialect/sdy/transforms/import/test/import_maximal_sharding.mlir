// RUN: sdy_opt  -split-input-file %s -sdy-import-maximal-sharding | FileCheck %s

// CHECK: sdy.mesh @maximal_mesh_3 = <device_ids=[3]>
// CHECK-LABEL: func @maximal_sharding_static_shaped_type_result
func.func @maximal_sharding_static_shaped_type_result(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME: sdy.sharding_per_value<[<@maximal_mesh_3, [{}, {}]>]
  %0 = stablehlo.add %arg0, %arg0
    {sdy.sharding = 3} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK: sdy.mesh @maximal_mesh_3 = <device_ids=[3]>
// CHECK-LABEL: func @maximal_sharding_non_shaped_type_result
func.func @maximal_sharding_non_shaped_type_result(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.custom_call
  // CHECK-SAME: sdy.sharding_per_value<[<@maximal_mesh_3, []>]
  %0 = stablehlo.custom_call @foo(%arg0)
    {sdy.sharding = 3} : (tensor<8x8xf32>) -> tuple<>
  return %arg0 : tensor<8x8xf32>
}

// -----

// CHECK: sdy.mesh @maximal_mesh_3 = <device_ids=[3]>
// CHECK-LABEL: func @maximal_sharding_multiple_results
func.func @maximal_sharding_multiple_results(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.custom_call
  // CHECK-SAME: sdy.sharding_per_value<[<@maximal_mesh_3, [{}, {}]>, <@maximal_mesh_3, []>]
  %0, %1 = stablehlo.custom_call @bar(%arg0)
    {sdy.sharding = 3} : (tensor<8x8xf32>) -> (tensor<8x8xf32>, tuple<>)
  return %arg0 : tensor<8x8xf32>
}

// -----

sdy.mesh @maximal_mesh_3 = <device_ids=[3]>

// CHECK: sdy.mesh @maximal_mesh_3 = <device_ids=[3]>
// CHECK-LABEL: func @maximal_sharding_maximal_mesh_exists
func.func @maximal_sharding_maximal_mesh_exists(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME: sdy.sharding_per_value<[<@maximal_mesh_3, [{}, {}]>]
  %0 = stablehlo.add %arg0, %arg0
    {sdy.sharding = 3} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

// CHECK: sdy.mesh @maximal_mesh_0 = <device_ids=[0]>
// CHECK-LABEL: func @maximal_sharding_repeated_mesh
func.func @maximal_sharding_repeated_mesh(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: stablehlo.custom_call
  // CHECK-SAME: sdy.sharding_per_value<[<@maximal_mesh_0, []>]
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME: sdy.sharding_per_value<[<@maximal_mesh_0, [{}, {}]>]
  %0 = stablehlo.custom_call @foo(%arg0)
    {sdy.sharding = 0}: (tensor<8x8xf32>) -> tuple<>
  %1 = stablehlo.add %arg0, %arg0 {sdy.sharding = 0} : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----

// CHECK: sdy.mesh @maximal_mesh_5 = <device_ids=[5]>
// CHECK-NEXT: sdy.mesh @maximal_mesh_0 = <device_ids=[0]>
// CHECK-LABEL: func @maximal_sharding_two_different_meshes
func.func @maximal_sharding_two_different_meshes(%arg0 : tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: stablehlo.custom_call
  // CHECK-SAME: sdy.sharding_per_value<[<@maximal_mesh_5, []>]
  // CHECK-NEXT: stablehlo.add
  // CHECK-SAME: sdy.sharding_per_value<[<@maximal_mesh_0, [{}, {}]>]
  %0 = stablehlo.custom_call @bar(%arg0)
    {sdy.sharding = 5}: (tensor<8x8xf32>) -> tuple<>
  %1 = stablehlo.add %arg0, %arg0 {sdy.sharding = 0} : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}
