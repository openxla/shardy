// RUN: sdy_opt %s -sdy-convert-global-to-local -split-input-file -verify-diagnostics | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>
sdy.mesh @mesh_1_2 = <["x"=1, "y"=2]>

// CHECK-LABEL: func @replicated
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<16x8xf32>, %[[IDX0:.*]]: tensor<i32>, %[[IDX1:.*]]: tensor<i32>) -> tensor<4x8xf32>
func.func @replicated(%arg0: tensor<16x8xf32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[IDX0]], %[[IDX1]], sizes = [4, 8] : (tensor<16x8xf32>, tensor<i32>, tensor<i32>) -> tensor<4x8xf32>
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [4, 8] : (tensor<16x8xf32>, tensor<i32>, tensor<i32>) -> tensor<4x8xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @slicing_dim_not_sharded
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>},
// CHECK-SAME:     %[[IDX0:.*]]: tensor<i32>, %[[IDX1:.*]]: tensor<i32>)
// CHECK-SAME:    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>})
func.func @slicing_dim_not_sharded(
    %arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>},
    %arg1: tensor<i32>, %arg2: tensor<i32>)
    -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>}) {
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[IDX0]], %[[IDX1]], sizes = [4, 8] : (tensor<16x8xf32>, tensor<i32>, tensor<i32>) -> tensor<4x8xf32>
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [4, 32] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"x"}]>]>} : (tensor<16x32xf32>, tensor<i32>, tensor<i32>) -> tensor<4x32xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<4x8xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @slicing_dim_sharded_on_size_one_axis
// CHECK-SAME:    (%[[ARG0:.*]]: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"x"}, {"y"}]>},
// CHECK-SAME:     %[[IDX0:.*]]: tensor<i32>, %[[IDX1:.*]]: tensor<i32>)
// CHECK-SAME:    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"x"}, {"y"}]>})
func.func @slicing_dim_sharded_on_size_one_axis(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"x"}, {"y"}]>},
    %arg1: tensor<i32>, %arg2: tensor<i32>)
    -> (tensor<4x16xf32> {sdy.sharding = #sdy.sharding<@mesh_1_2, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[RES:.*]] = stablehlo.dynamic_slice %[[ARG0]], %[[IDX0]], %[[IDX1]], sizes = [4, 8] : (tensor<8x8xf32>, tensor<i32>, tensor<i32>) -> tensor<4x8xf32>
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [4, 16] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_1_2, [{"x"}, {"y"}]>]>} : (tensor<8x16xf32>, tensor<i32>, tensor<i32>) -> tensor<4x16xf32>
  // CHECK-NEXT: return %[[RES]] : tensor<4x8xf32>
  return %0 : tensor<4x16xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

func.func @mismatched_sharding_axes(
    %arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"x"}]>},
    %arg1: tensor<i32>, %arg2: tensor<i32>)
    -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{}, {"y"}]>}) {
  // expected-error @+2 {{'stablehlo.dynamic_slice' op dimension 1 has mismatched sharding axes ([axis_ref"x"] vs [axis_ref"y"]). This requires a reshard which should have been handled by previous passes.}}
  // expected-error @+1 {{failed to legalize operation 'stablehlo.dynamic_slice' that was explicitly marked illegal}}
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [4, 32] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<16x32xf32>, tensor<i32>, tensor<i32>) -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// -----

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

func.func @slicing_dim_sharded_non_full_slice(
    %arg0: tensor<16x32xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>},
    %arg1: tensor<i32>, %arg2: tensor<i32>)
    -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh_4_2, [{"x"}, {}]>}) {
  // expected-error @+2 {{'stablehlo.dynamic_slice' op dimension 0 is sharded but the slice is not a full slice and requires device communication.}}
  // expected-error @+1 {{failed to legalize operation 'stablehlo.dynamic_slice' that was explicitly marked illegal}}
  %0 = stablehlo.dynamic_slice %arg0, %arg1, %arg2, sizes = [4, 32] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"x"}, {}]>]>} : (tensor<16x32xf32>, tensor<i32>, tensor<i32>) -> tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}
