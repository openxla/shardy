// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>
sdy.mesh @mesh_xyzt = <["x"=4, "y"=4, "z"=4, "t"=8]>
sdy.mesh @mesh_xyzp = <["x"=4, "y"=2, "z"=4, "p"=3]>

// CHECK-LABEL: func @pad
func.func @pad(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"y"}]>}){
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[PAD]] <@mesh_xyz, [{}, {}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_only_input_permutation_dim_is_sharded_non_permutation_dim_compatible
func.func @pad_only_input_permutation_dim_is_sharded_non_permutation_dim_compatible(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {}, {"x"}]>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]]
  // CHECK-NEXT: return %[[PAD]]
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {"x"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_only_input_permutation_dim_is_sharded
func.func @pad_only_input_permutation_dim_is_sharded(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {}, {"y"}]>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]]
  // CHECK-NEXT: return %[[PAD]]
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_only_output_permutation_dim_is_sharded
func.func @pad_only_output_permutation_dim_is_sharded(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"y"}]>}) {
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[PAD]] <@mesh_xyz, [{"z"}, {}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_both_input_and_output_permutation_dims_are_sharded_input_is_larger
// TODO(enver): Consider to prefer t over x along the batch dimension, even if it means two reshards.
func.func @pad_both_input_and_output_permutation_dims_are_sharded_input_is_larger(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y"}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"y"}, {"t"}]>}) {
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{"y"}, {}, {"x"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[PAD]] <@mesh_xyzt, [{}, {"y"}, {"t"}]> : tensor<30x26x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<30x26x16xf32>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"y"}, {"t"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_both_input_and_output_permutation_dims_are_sharded_output_is_larger
func.func @pad_both_input_and_output_permutation_dims_are_sharded_output_is_larger(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{"y"}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x30x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzt, [{}, {"y"}, {"t"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyzt, [{}, {"y"}, {"t"}]> : tensor<28x28x16xf32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]], %arg1, low = [1, 1, 0], high = [1, 1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"y"}, {"t"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x30x16xf32>
  // CHECK-NEXT: return %[[PAD]] : tensor<30x30x16xf32>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, 1, 0], high = [1, 1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzt, [{}, {"y"}, {"t"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x30x16xf32>
  return %0 : tensor<30x30x16xf32>
}

// CHECK-LABEL: func @pad_same_permutation_dim_is_sharded_on_both_sides_input_is_larger
func.func @pad_same_permutation_dim_is_sharded_on_both_sides_input_is_larger(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {}, {"y"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"p"}, {}, {"y"}]>}) {
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"x"}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[PAD]] <@mesh_xyzp, [{"p"}, {}, {"y"}]> : tensor<30x26x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<30x26x16xf32>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"p"}, {}, {"y"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_same_permutation_dim_is_sharded_on_both_sides_output_is_larger
func.func @pad_same_permutation_dim_is_sharded_on_both_sides_output_is_larger(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"x"}, {}, {"y"}]>}, %arg1: tensor<f32>) -> (tensor<30x30x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"p"}, {}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"p"}, {}, {"y"}]> : tensor<28x28x16xf32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]], %arg1, low = [1, 1, 0], high = [1, 1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"p"}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x30x16xf32>
  // CHECK-NEXT: return %[[PAD]] : tensor<30x30x16xf32>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, 1, 0], high = [1, 1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"p"}, {}, {"y"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x30x16xf32>
  return %0 : tensor<30x30x16xf32>
}

// CHECK-LABEL: func @pad_input_and_output_permutation_dims_are_sharded_same_way_input_is_larger
func.func @pad_input_and_output_permutation_dims_are_sharded_same_way_input_is_larger(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"z"}, {}, {"y"}]>}, %arg1: tensor<f32>) -> (tensor<30x26x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"p"}, {}, {"y"}]>}) {
  // CHECK: %[[PAD:.*]] = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"z"}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[PAD]] <@mesh_xyzp, [{"p"}, {}, {"y"}]> : tensor<30x26x16xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<30x26x16xf32>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, -1, 0], high = [1, -1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"p"}, {}, {"y"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x26x16xf32>
  return %0 : tensor<30x26x16xf32>
}

// CHECK-LABEL: func @pad_input_and_output_permutation_dims_are_sharded_same_way_output_is_larger
func.func @pad_input_and_output_permutation_dims_are_sharded_same_way_output_is_larger(%arg0: tensor<28x28x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"z"}, {}, {"x"}]>}, %arg1: tensor<f32>) -> (tensor<30x30x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyzp, [{"p"}, {}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyzp, [{"p"}, {}, {"y"}]> : tensor<28x28x16xf32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.pad %[[RESHARD]], %arg1, low = [1, 1, 0], high = [1, 1, 0], interior = [0, 0, 0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"p"}, {}, {"y"}]>]>} : (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x30x16xf32>
  // CHECK-NEXT: return %[[PAD]] : tensor<30x30x16xf32>
  %0 = stablehlo.pad %arg0, %arg1, low = [1, 1, 0], high = [1, 1, 0], interior = [0, 0, 0]  {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyzp, [{"p"}, {}, {"y"}]>]>}: (tensor<28x28x16xf32>, tensor<f32>) -> tensor<30x30x16xf32>
  return %0 : tensor<30x30x16xf32>
}

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<32x4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<32x1x2xf32> {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:32, 1:2, 4:8:2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh, [{}, {}, {}]> : tensor<32x1x2xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<32x1x2xf32>
  %0 = stablehlo.slice %arg0 [0:32, 1:2, 4:8:2] : (tensor<32x4x8xf32>) -> tensor<32x1x2xf32>
  return %0 : tensor<32x1x2xf32>
}

// CHECK-LABEL: func @slice_no_permutation_dim_is_sharded
func.func @slice_no_permutation_dim_is_sharded(%arg0: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"x"}]>}) -> (tensor<4x8x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"y"}]>}) {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_xyz, [{}, {}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.slice %arg0 [8:12, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {"y"}]>]>} : (tensor<32x64x128xf32>) -> tensor<4x8x128xf32>
  return %0 : tensor<4x8x128xf32>
}

// CHECK-LABEL: func @slice_only_input_permutation_dim_is_sharded
func.func @slice_only_input_permutation_dim_is_sharded(%arg0: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"x"}]>}) -> (tensor<4x8x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"y"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{}, {}, {"y"}]>
  // CHECK-NEXT: %[[SLICE:.*]] = stablehlo.slice %[[RESHARD]]
  // CHECK-NEXT: return %[[SLICE]]
  %0 = stablehlo.slice %arg0 [8:12, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {}, {"y"}]>]>} : (tensor<32x64x128xf32>) -> tensor<4x8x128xf32>
  return %0 : tensor<4x8x128xf32>
}

// CHECK-LABEL: func @slice_only_output_permutation_dim_is_sharded
func.func @slice_only_output_permutation_dim_is_sharded(%arg0: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {}, {"x"}]>}) -> (tensor<4x8x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"y"}]>}) {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_xyz, [{"z"}, {}, {"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.slice %arg0 [8:12, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {"y"}]>]>} : (tensor<32x64x128xf32>) -> tensor<4x8x128xf32>
  return %0 : tensor<4x8x128xf32>
}

// CHECK-LABEL: func @slice_both_input_and_output_permutation_dims_are_sharded
func.func @slice_both_input_and_output_permutation_dims_are_sharded(%arg0: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"x"}]>}) -> (tensor<4x8x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{}, {"z"}, {"y"}]>}) {
  // TODO(enver): Consider preferring larger sharding axes along every batch and slice dimension.
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0 [8:12, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {"x"}]>]>} : (tensor<32x64x128xf32>) -> tensor<4x8x128xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_xyz, [{}, {"z"}, {"y"}]> : tensor<4x8x128xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x8x128xf32>
  %0 = stablehlo.slice %arg0 [8:12, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"z"}, {"y"}]>]>}: (tensor<32x64x128xf32>) -> tensor<4x8x128xf32>
  return %0 : tensor<4x8x128xf32>
}

// CHECK-LABEL: func @slice_both_operand_and_result_have_sharded_permutation_factors_result_has_larger_sharding_on_permutation_factor
func.func @slice_both_operand_and_result_have_sharded_permutation_factors_result_has_larger_sharding_on_permutation_factor(%arg0: tensor<2048x1152x192xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x", "y"}, {}, {"z"}]>}) -> (tensor<2048x1024x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x", "z"}, {"y"}, {}]>}) {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:2048, 0:1024, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y"}, {}, {"z"}]>]>} : (tensor<2048x1152x192xf32>) -> tensor<2048x1024x128xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_xyz, [{"x", "z"}, {"y"}, {}]> : tensor<2048x1024x128xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<2048x1024x128xf32>
  %0 = stablehlo.slice %arg0 [0:2048, 0:1024, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "z"}, {"y"}, {}]>]>} : (tensor<2048x1152x192xf32>) -> tensor<2048x1024x128xf32>
  return %0 : tensor<2048x1024x128xf32>
}

// CHECK-LABEL: func @slice_both_operand_and_result_have_sharded_permutation_factors_operand_has_larger_sharding_on_permutation_factor
func.func @slice_both_operand_and_result_have_sharded_permutation_factors_operand_has_larger_sharding_on_permutation_factor(%arg0: tensor<2048x1152x192xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x", "z"}, {}, {"y"}]>}) -> (tensor<2048x1024x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x", "y"}, {"z"}, {}]>}) {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0 [0:2048, 0:1024, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "z"}, {}, {"y"}]>]>} : (tensor<2048x1152x192xf32>) -> tensor<2048x1024x128xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_xyz, [{"x", "y"}, {"z"}, {}]> : tensor<2048x1024x128xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<2048x1024x128xf32>
  %0 = stablehlo.slice %arg0 [0:2048, 0:1024, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y"}, {"z"}, {}]>]>} : (tensor<2048x1152x192xf32>) -> tensor<2048x1024x128xf32>
  return %0 : tensor<2048x1024x128xf32>
}

// CHECK-LABEL: func @slice_input_and_output_permutation_dims_are_sharded_the_same_way
func.func @slice_input_and_output_permutation_dims_are_sharded_the_same_way(%arg0: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"y"}]>}) -> (tensor<16x8x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {}, {"y"}]>}) {
  // CHECK-NOT: sdy.reshard
  %0 = stablehlo.slice %arg0 [8:24, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}, {}, {"y"}]>]>}: (tensor<32x64x128xf32>) -> tensor<16x8x128xf32>
  return %0 : tensor<16x8x128xf32>
}

// CHECK-LABEL: func @slice_input_and_output_permutation_dims_are_sharded_differently_but_nonempty_prefix
func.func @slice_input_and_output_permutation_dims_are_sharded_differently_but_nonempty_prefix(%arg0: tensor<32x64x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z", "x"}, {}, {"y"}]>}) -> (tensor<16x8x128xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z", "y"}, {}, {}]>}) {
  // CHECK: %[[SLICE:.*]] = stablehlo.slice %arg0 [8:24, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z", "x"}, {}, {"y"}]>]>} : (tensor<32x64x128xf32>) -> tensor<16x8x128xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[SLICE]] <@mesh_xyz, [{"z", "y"}, {}, {}]> : tensor<16x8x128xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<16x8x128xf32>
  %0 = stablehlo.slice %arg0 [8:24, 0:64:8, 0:128] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z", "y"}, {}, {}]>]>}: (tensor<32x64x128xf32>) -> tensor<16x8x128xf32>
  return %0 : tensor<16x8x128xf32>
}
