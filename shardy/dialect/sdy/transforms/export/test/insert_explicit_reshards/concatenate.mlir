// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @concatenate_single_input
func.func @concatenate_single_input(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>}) -> (tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}]> : tensor<4x32x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %[[RESHARD]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>) -> tensor<4x32x256xf32>
  // CHECK-NEXT: return %[[CONCATENATE]] : tensor<4x32x256xf32>
  %0 = stablehlo.concatenate %arg0, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>) -> tensor<4x32x256xf32>
  return %0 : tensor<4x32x256xf32>
}

// CHECK-LABEL: func @concatenate
func.func @concatenate(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x48x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {}]>}) -> tensor<4x80x256xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<4x48x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %arg0, %[[RESHARD1]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CONCATENATE]] <@mesh, [{}, {}, {}]> : tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x80x256xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %0 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concatenate_concat_dim_is_sharded
func.func @concatenate_concat_dim_is_sharded(%arg0: tensor<8x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}, %arg1: tensor<8x48x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) -> (tensor<8x80x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x", "y"}, {}, {}]> : tensor<8x32x256xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"x", "y"}, {}, {}]> : tensor<8x48x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %[[RESHARD1]], %[[RESHARD2]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CONCATENATE]] <@mesh, [{"x"}, {"y"}, {}]> : tensor<8x80x256xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<8x80x256xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<8x32x256xf32>, tensor<8x48x256xf32>) -> tensor<8x80x256xf32>
  return %0 : tensor<8x80x256xf32>
}

// TODO(b/435070275). A better solution is to use the compatible sharding [{"x", "y"}, {}].
// CHECK-LABEL: func @concatenate_an_axis_is_used_in_both_batch_and_concat_dims
func.func @concatenate_an_axis_is_used_in_both_batch_and_concat_dims(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<8x48xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x80xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}]>}) {
  // CHECK: %[[CONCATENATE:.*]] = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x32xf32>, tensor<8x48xf32>) -> tensor<8x80xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CONCATENATE]] <@mesh, [{}, {"x", "y"}]> : tensor<8x80xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x80xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x", "y"}]>]>} : (tensor<8x32xf32>, tensor<8x48xf32>) -> tensor<8x80xf32>
  return %0 : tensor<8x80xf32>
}

// CHECK-LABEL: func @concatenate_operands_are_results_of_slices
func.func @concatenate_operands_are_results_of_slices(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}, %arg1: tensor<4x60x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) -> (tensor<4x80x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  // CHECK-NOT: sdy.reshard
  %2 = stablehlo.concatenate %0, %1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concatenate_operands_are_from_slices_of_the_same_tensor
func.func @concatenate_operands_are_from_slices_of_the_same_tensor(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) -> (tensor<4x96x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg0 [0:4, 0:24, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x40x256xf32>) -> tensor<4x24x256xf32>
  // CHECK-NOT: sdy.reshard
  %2 = stablehlo.concatenate %0, %arg0, %1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x40x256xf32>, tensor<4x24x256xf32>) -> tensor<4x96x256xf32>
  return %2 : tensor<4x96x256xf32>
}

// CHECK-LABEL: func @concatenate_operands_are_results_of_slices_different_shardings_on_permutation_dim_with_equal_counts
func.func @concatenate_operands_are_results_of_slices_different_shardings_on_permutation_dim_with_equal_counts(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x60x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) -> (tensor<4x80x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %0 <@mesh, [{"x"}, {"y"}, {}]> : tensor<4x32x256xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %1 <@mesh, [{"x"}, {"y"}, {}]> : tensor<4x48x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %[[RESHARD1]], %[[RESHARD2]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CONCATENATE]] <@mesh, [{}, {"x"}, {}]> : tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<4x80x256xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [
{}, {"x"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concatenate_operands_are_results_of_slices_different_shardings_on_permutation_dim_with_equal_counts_but_conflicting_on_batching_dim
func.func @concatenate_operands_are_results_of_slices_different_shardings_on_permutation_dim_with_equal_counts_but_conflicting_on_batching_dim(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(2)2}, {}, {}]>}, %arg1: tensor<4x60x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) -> (tensor<4x80x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(2)2}, {}, {}]>]>} : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %0 <@mesh, [{"x":(2)2}, {"y"}, {}]> : tensor<4x32x256xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %1 <@mesh, [{"x":(2)2}, {"y"}, {}]> : tensor<4x48x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %[[RESHARD1]], %[[RESHARD2]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(2)2}, {"y"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CONCATENATE]] <@mesh, [{}, {"x"}, {}]> : tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<4x80x256xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concatenate_operands_are_results_of_slices_conflicting_shardings
func.func @concatenate_operands_are_results_of_slices_conflicting_shardings(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}, %arg1: tensor<4x60x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}, {}]>}) -> (tensor<4x80x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}, {}]>}) {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}, {}]>]>} : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %1 <@mesh, [{}, {"y"}, {}]> : tensor<4x48x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %0, %[[RESHARD]], dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[CONCATENATE]] : tensor<4x80x256xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
}
