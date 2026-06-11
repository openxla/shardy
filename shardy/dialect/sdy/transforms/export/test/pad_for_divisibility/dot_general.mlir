// RUN: sdy_opt %s -sdy-pad-for-divisibility | FileCheck %s

sdy.mesh @mesh_4_2 = <["x"=4, "y"=2]>

// CHECK-LABEL: func @padded_contracting_dims_reuse
func.func @padded_contracting_dims_reuse(%arg0: tensor<4x7xf32>, %arg1: tensor<7x5xf32>) -> tensor<4x5xf32> {
  // Pad LHS with zero (contracting dimension).
  // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %arg0, %[[CST0]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK: %[[SLICE0:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD0]] out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x8xf32>

  // Pad RHS with zero (both contracting and non-contracting dimensions).
  // CHECK: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %arg1, %[[CST1]], low = [0, 0], high = [1, 3], interior = [0, 0] : (tensor<7x5xf32>, tensor<f32>) -> tensor<8x8xf32>
  // CHECK: %[[SLICE1:.*]] = sdy.all_slice [{"y"}, {"x"}] %[[PAD1]] out_sharding=<@mesh_4_2, [{"y"}, {"x"}]> : tensor<8x8xf32>

  // Perform dot_general (result is padded on non-contracting dimension).
  // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[SLICE0]], %[[SLICE1]], contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"x"}]>]>} : (tensor<4x8xf32>, tensor<8x8xf32>) -> tensor<4x8xf32>

  // Trim the padded result back to original shape.
  // CHECK: %[[TRIM:.*]] = stablehlo.slice %[[DOT]] [0:4, 0:5] : (tensor<4x8xf32>) -> tensor<4x5xf32>
  // CHECK: return %[[TRIM]] : tensor<4x5xf32>

  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xf32>
  %1 = sdy.all_slice [{"y"}, {"x"}] %arg1 out_sharding=<@mesh_4_2, [{"y"}, {"x"}]> : tensor<7x5xf32>
  %2 = stablehlo.dot_general %0, %1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"x"}]>]>} : (tensor<4x7xf32>, tensor<7x5xf32>) -> tensor<4x5xf32>
  %3 = stablehlo.slice %2 [0:4, 0:5] : (tensor<4x5xf32>) -> tensor<4x5xf32>
  return %3 : tensor<4x5xf32>
}

// CHECK-LABEL: func @padded_contracting_dims_not_reuse
func.func @padded_contracting_dims_not_reuse(%arg0: tensor<4x7xf32>, %arg1: tensor<7x5xf32>) -> tensor<4x5xf32> {
  // Prepare padded LHS and RHS with unknown padding (via abs).
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %arg0, {{.*}}
  // CHECK: %[[LHS_SLICE:.*]] = sdy.all_slice [{}, {"y"}] %[[PAD0]]
  // CHECK: %[[LHS_ABS:.*]] = stablehlo.abs %[[LHS_SLICE]] {{.*}}
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %arg1, {{.*}}
  // CHECK: %[[RHS_SLICE:.*]] = sdy.all_slice [{"y"}, {"x"}] %[[PAD1]]
  // CHECK: %[[RHS_ABS:.*]] = stablehlo.abs %[[RHS_SLICE]] {{.*}}

  // Enforce zero-padding on LHS contracting dim (dim 1).
  // CHECK: %[[LHS_IOTA:.*]] = stablehlo.iota{{.*}}dim = 1
  // CHECK: %[[LHS_LIMIT:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK: %[[LHS_LIMIT_BCAST:.*]] = stablehlo.broadcast_in_dim %[[LHS_LIMIT]], dims = []
  // CHECK: %[[LHS_MASK:.*]] = stablehlo.compare{{.*}}LT, %[[LHS_IOTA]], %[[LHS_LIMIT_BCAST]]
  // CHECK: %[[LHS_CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[LHS_BCAST:.*]] = stablehlo.broadcast_in_dim %[[LHS_CST]], dims = []
  // CHECK: %[[LHS_SELECT:.*]] = stablehlo.select %[[LHS_MASK]], %[[LHS_ABS]], %[[LHS_BCAST]]

  // Enforce zero-padding on RHS contracting dim (dim 0).
  // CHECK: %[[RHS_IOTA:.*]] = stablehlo.iota{{.*}}dim = 0
  // CHECK: %[[RHS_LIMIT:.*]] = stablehlo.constant dense<7> : tensor<i32>
  // CHECK: %[[RHS_LIMIT_BCAST:.*]] = stablehlo.broadcast_in_dim %[[RHS_LIMIT]], dims = []
  // CHECK: %[[RHS_MASK:.*]] = stablehlo.compare{{.*}}LT, %[[RHS_IOTA]], %[[RHS_LIMIT_BCAST]]
  // CHECK: %[[RHS_CST:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RHS_BCAST:.*]] = stablehlo.broadcast_in_dim %[[RHS_CST]], dims = []
  // CHECK: %[[RHS_SELECT:.*]] = stablehlo.select %[[RHS_MASK]], %[[RHS_ABS]], %[[RHS_BCAST]]
  // CHECK-NOT: stablehlo.iota {{.*}} dim = 1

  // Perform dot_general and trim result.
  // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[LHS_SELECT]], %[[RHS_SELECT]], contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"x"}]>]>} : (tensor<4x8xf32>, tensor<8x8xf32>) -> tensor<4x8xf32>
  // CHECK: %[[TRIM:.*]] = stablehlo.slice %[[DOT]] [0:4, 0:5]
  // CHECK: return %[[TRIM]]

  %0 = sdy.all_slice [{}, {"y"}] %arg0 out_sharding=<@mesh_4_2, [{}, {"y"}]> : tensor<4x7xf32>
  %1 = stablehlo.abs %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : tensor<4x7xf32>
  %2 = sdy.all_slice [{"y"}, {"x"}] %arg1 out_sharding=<@mesh_4_2, [{"y"}, {"x"}]> : tensor<7x5xf32>
  %3 = stablehlo.abs %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"y"}, {"x"}]>]>} : tensor<7x5xf32>
  %4 = stablehlo.dot_general %1, %3, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"x"}]>]>} : (tensor<4x7xf32>, tensor<7x5xf32>) -> tensor<4x5xf32>
  %5 = stablehlo.slice %4 [0:4, 0:5] : (tensor<4x5xf32>) -> tensor<4x5xf32>
  return %5 : tensor<4x5xf32>
}

// CHECK-LABEL: func @padded_non_contracting_dims_any
func.func @padded_non_contracting_dims_any(%arg0: tensor<3x8xf32>, %arg1: tensor<8x5xf32>) -> tensor<3x5xf32> {
  // Prepare padded LHS with unknown padding.
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %arg0, {{.*}}
  // CHECK: %[[LHS_SLICE:.*]] = sdy.all_slice [{"y"}, {}] %[[PAD0]]
  // CHECK: %[[LHS_ABS:.*]] = stablehlo.abs %[[LHS_SLICE]] {{.*}}

  // Verify no select is generated for non-contracting dim.
  // CHECK-NOT: stablehlo.select
  // CHECK-NOT: stablehlo.compare

  // Prepare padded RHS.
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %arg1, {{.*}}
  // CHECK: %[[RHS_SLICE:.*]] = sdy.all_slice [{}, {"x"}] %[[PAD1]]

  // Perform dot_general and trim result.
  // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[LHS_ABS]], %[[RHS_SLICE]], contracting_dims = [1] x [0] {{.*}}
  // CHECK: %[[TRIM:.*]] = stablehlo.slice %[[DOT]] [0:3, 0:5]
  // CHECK: return %[[TRIM]]

  %0 = sdy.all_slice [{"y"}, {}] %arg0 out_sharding=<@mesh_4_2, [{"y"}, {}]> : tensor<3x8xf32>
  %1 = stablehlo.abs %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"y"}, {}]>]>} : tensor<3x8xf32>
  %2 = sdy.all_slice [{}, {"x"}] %arg1 out_sharding=<@mesh_4_2, [{}, {"x"}]> : tensor<8x5xf32>
  %3 = stablehlo.dot_general %1, %2, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{"y"}, {"x"}]>]>} : (tensor<3x8xf32>, tensor<8x5xf32>) -> tensor<3x5xf32>
  %4 = stablehlo.slice %3 [0:3, 0:5] : (tensor<3x5xf32>) -> tensor<3x5xf32>
  return %4 : tensor<3x5xf32>
}

// CHECK-LABEL: func @padded_contracting_dims_reuse_with_all_gather
func.func @padded_contracting_dims_reuse_with_all_gather(%arg0: tensor<4x7xf32>, %arg1: tensor<7x5xf32>) -> tensor<4x5xf32> {
  // Pad LHS and do all_slice then all_gather on the non-contracting dim (dim 0).
  // The contracting dim (dim 1) remains padded and its padding should be reused.
  // CHECK: %[[CST0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD0:.*]] = stablehlo.pad %arg0, %[[CST0]], low = [0, 0], high = [0, 1], interior = [0, 0] : (tensor<4x7xf32>, tensor<f32>) -> tensor<4x8xf32>
  // CHECK: %[[SLICE0:.*]] = sdy.all_slice [{"y"}, {"x"}] %[[PAD0]] out_sharding=<@mesh_4_2, [{"y"}, {"x"}]> : tensor<4x8xf32>
  // CHECK: %[[AG0:.*]] = sdy.all_gather [{"y"}, {}] %[[SLICE0]] out_sharding=<@mesh_4_2, [{}, {"x"}]> : tensor<4x8xf32>

  // CHECK: %[[CST1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[PAD1:.*]] = stablehlo.pad %arg1, %[[CST1]], low = [0, 0], high = [1, 1], interior = [0, 0] : (tensor<7x5xf32>, tensor<f32>) -> tensor<8x6xf32>
  // CHECK: %[[SLICE1:.*]] = sdy.all_slice [{"x"}, {"y"}] %[[PAD1]] out_sharding=<@mesh_4_2, [{"x"}, {"y"}]> : tensor<8x6xf32>

  // Verify NO select/iota is generated for LHS contracting dim!
  // CHECK-NOT: stablehlo.select
  // CHECK-NOT: stablehlo.compare

  // CHECK: %[[DOT:.*]] = stablehlo.dot_general %[[AG0]], %[[SLICE1]], contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<4x8xf32>, tensor<8x6xf32>) -> tensor<4x6xf32>
  // CHECK: %[[TRIM:.*]] = stablehlo.slice %[[DOT]] [0:4, 0:5] : (tensor<4x6xf32>) -> tensor<4x5xf32>
  // CHECK: return %[[TRIM]] : tensor<4x5xf32>

  %0 = sdy.all_slice [{"y"}, {"x"}] %arg0 out_sharding=<@mesh_4_2, [{"y"}, {"x"}]> : tensor<4x7xf32>
  %1 = sdy.all_gather [{"y"}, {}] %0 out_sharding=<@mesh_4_2, [{}, {"x"}]> : tensor<4x7xf32>
  %2 = sdy.all_slice [{"x"}, {"y"}] %arg1 out_sharding=<@mesh_4_2, [{"x"}, {"y"}]> : tensor<7x5xf32>
  %3 = stablehlo.dot_general %1, %2, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_4_2, [{}, {"y"}]>]>} : (tensor<4x7xf32>, tensor<7x5xf32>) -> tensor<4x5xf32>
  %4 = stablehlo.slice %3 [0:4, 0:5] : (tensor<4x5xf32>) -> tensor<4x5xf32>
  return %4 : tensor<4x5xf32>
}

