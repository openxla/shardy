// RUN: sdy_opt %s -sdy-temp-explicit-reshards-for-optimizations | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2, "z"=4]>
sdy.mesh @other_mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @reshard_dot_result_to_match_lhs
func.func @reshard_dot_result_to_match_lhs(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT]] <@mesh, [{"x"}, {}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.dot %arg0, %arg1
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @dot_general_no_conflict
func.func @dot_general_no_conflict(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {?}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {?}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @reshard_dot_general_result_to_match_lhs
func.func @reshard_dot_general_result_to_match_lhs(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {?}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT_GENERAL]] <@mesh, [{"x"}, {?}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @reshard_dot_general_result_to_match_rhs
func.func @reshard_dot_general_result_to_match_rhs(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT_GENERAL]] <@mesh, [{?}, {"x"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"x"}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @reshard_dot_general_result_with_multiple_axes
func.func @reshard_dot_general_result_with_multiple_axes(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x", "z"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x", "z"}, {}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT_GENERAL]] <@mesh, [{"x", "z"}, {}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "z"}, {}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @reshard_dot_general_result_multiple_uses
func.func @reshard_dot_general_result_multiple_uses(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {?}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT_GENERAL]] <@mesh, [{"x"}, {?}]>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[RESHARD]], %[[RESHARD]]
  // CHECK-NEXT: return %[[RESHARD]], %[[ADD]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {?}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<4x8xf32>
  return %0, %1 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @reshard_dot_general_batching_dim
func.func @reshard_dot_general_batching_dim(
    %arg0: tensor<2x4x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"z"}, {}, {"x"}]>},
    %arg1: tensor<2x8x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"z"}, {"x"}, {"y"}]>}) -> tensor<2x4x32xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT_GENERAL]] <@mesh, [{"x"}, {}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0],
      contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {"y"}]>]>} :
      (tensor<2x4x8xf32>, tensor<2x8x32xf32>) -> tensor<2x4x32xf32>
  return %0 : tensor<2x4x32xf32>
}

// CHECK-LABEL: func @reshard_dot_general_with_multiple_sharded_contracting_dims
func.func @reshard_dot_general_with_multiple_sharded_contracting_dims(
    %arg0: tensor<4x2x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"z"}, {"x"}]>},
    %arg1: tensor<2x32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"z"}, {"x"}, {"y"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT_GENERAL]] <@mesh, [{}, {"z", "x"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [1, 2] x [0, 1], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"z", "x"}]>]>} :
      (tensor<4x2x32xf32>, tensor<2x32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @reshard_dot_general_with_multiple_non_contracting_dims
func.func @reshard_dot_general_with_multiple_non_contracting_dims(
    %arg0: tensor<4x2x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"z"}, {"x"}]>},
    %arg1: tensor<32x16x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {}, {"y"}]>}) -> tensor<4x2x16x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"z"}, {}, {"y"}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[DOT_GENERAL]] <@mesh, [{}, {"z"}, {}, {"x"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"z"}, {}, {"x"}]>]>} :
      (tensor<4x2x32xf32>, tensor<32x16x8xf32>) -> tensor<4x2x16x8xf32>
  return %0 : tensor<4x2x16x8xf32>
}

// CHECK-LABEL: func @dot_result_missing_sharding
func.func @dot_result_missing_sharding(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @dot_different_meshes
func.func @dot_different_meshes(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@other_mesh, [{"x"}, {?}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@other_mesh, [{"x"}, {?}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @dot_lhs_and_rhs_conflicting_contracting_dim
func.func @dot_lhs_and_rhs_conflicting_contracting_dim(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @dot_lhs_and_rhs_conflicting_non_contracting_dim
func.func @dot_lhs_and_rhs_conflicting_non_contracting_dim(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @dot_lhs_and_rhs_conflicting_non_contracting_dim_sub_axis
func.func @dot_lhs_and_rhs_conflicting_non_contracting_dim_sub_axis(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"z"}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {"z":(2)2}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @dot_lhs_and_rhs_conflicting_batching_dim
func.func @dot_lhs_and_rhs_conflicting_batching_dim(
    %arg0: tensor<2x4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {}, {"x"}]>},
    %arg1: tensor<2x32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"z"}, {"x"}, {"y"}]>}) -> tensor<2x4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {}, {"x"}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0],
      contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}, {}, {"x"}]>]>} :
      (tensor<2x4x32xf32>, tensor<2x32x8xf32>) -> tensor<2x4x8xf32>
  return %0 : tensor<2x4x8xf32>
}

// This is a reduce-scatter pattern, and shouldn't trigger this optimization.
// CHECK-LABEL: func @dot_result_conflict_with_lhs_empty_lhs_sharding
func.func @dot_result_conflict_with_lhs_empty_lhs_sharding(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @dot_multiple_conflicts_with_result
func.func @dot_multiple_conflicts_with_result(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"x", "y"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x", "y"}, {"z"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @dot_result_bigger_than_conflicting_lhs
func.func @dot_result_bigger_than_conflicting_lhs(
    %arg0: tensor<2x4xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>},
    %arg1: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<2x32xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} :
      (tensor<2x4xf32>, tensor<4x32xf32>) -> tensor<2x32xf32>
  return %0 : tensor<2x32xf32>
}

// CHECK-LABEL: func @dot_result_bigger_than_conflicting_rhs
func.func @dot_result_bigger_than_conflicting_rhs(
    %arg0: tensor<32x4xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"x"}]>},
    %arg1: tensor<4x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<32x2xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} :
      (tensor<32x4xf32>, tensor<4x2xf32>) -> tensor<32x2xf32>
  return %0 : tensor<32x2xf32>
}

// CHECK-LABEL: func @dot_result_conflicting_sharding_empty
func.func @dot_result_conflicting_sharding_empty(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @dot_result_conflicting_sharding_mismatch_with_reduction_axes
func.func @dot_result_conflicting_sharding_mismatch_with_reduction_axes(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"z"}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"z"}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @dot_result_conflicting_sharding_mismatch_with_reduction_axes_2
func.func @dot_result_conflicting_sharding_mismatch_with_reduction_axes_2(
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"x", "z"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x", "z"}, {"y"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} :
      (tensor<4x32xf32>, tensor<32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @dot_result_conflicting_sharding_mismatch_with_reduction_axes_3
func.func @dot_result_conflicting_sharding_mismatch_with_reduction_axes_3(
    %arg0: tensor<4x2x32xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"z"}, {"x"}]>},
    %arg1: tensor<2x32x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"z"}, {"x"}, {"y"}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [1, 2] x [0, 1], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} :
      (tensor<4x2x32xf32>, tensor<2x32x8xf32>) -> tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}
