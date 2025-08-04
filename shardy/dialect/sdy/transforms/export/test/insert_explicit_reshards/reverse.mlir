// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>

// CHECK-LABEL: func @reverse_no_permutation_dim_is_sharded_output_sharding_is_larger
func.func @reverse_no_permutation_dim_is_sharded_output_sharding_is_larger(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {}, {}]>}) -> (tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}, {}]>
  // CHECK-NEXT: %[[REVERSE:.*]] = stablehlo.reverse %[[RESHARD]]
  // CHECK-NEXT: return %[[REVERSE]]
  %0 = stablehlo.reverse %arg0, dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}

// CHECK-LABEL: func @reverse_no_permutation_dim_is_sharded_input_sharding_is_larger
func.func @reverse_no_permutation_dim_is_sharded_input_sharding_is_larger(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) -> (tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {}, {}]>}){
  // CHECK: %[[REVERSE:.*]] = stablehlo.reverse %arg0
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[REVERSE]] <@mesh, [{"y"}, {}, {}, {}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.reverse %arg0, dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}, {}, {}]>]>} : tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}

// CHECK-LABEL: func @reverse_only_input_permutation_dim_is_sharded
func.func @reverse_only_input_permutation_dim_is_sharded(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"z"}, {}, {}]>}) -> (tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {}]>}){
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {}]>
  // CHECK-NEXT: %[[REVERSE:.*]] = stablehlo.reverse %[[RESHARD]]
  // CHECK-NEXT: return %[[REVERSE]]
  %0 = stablehlo.reverse %arg0, dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {}]>]>} : tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}

// CHECK-LABEL: func @reverse_multiple_input_permutation_dims_are_sharded
func.func @reverse_multiple_input_permutation_dims_are_sharded(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(1)2}, {"z":(1)2}, {}, {"z":(2)2}]>}) -> (tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(2)2}, {}, {}, {}]>}){
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x":(2)2}, {}, {}, {}]>
  // CHECK-NEXT: %[[REVERSE:.*]] = stablehlo.reverse %[[RESHARD]]
  // CHECK-NEXT: return %[[REVERSE]]
  %0 = stablehlo.reverse %arg0, dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x":(2)2}, {}, {}, {}]>]>} : tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}

// CHECK-LABEL: func @reverse_only_output_permutation_dim_is_sharded
func.func @reverse_only_output_permutation_dim_is_sharded(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {"z":(1)2}, {}]>}) -> (tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {"z"}, {}, {}]>}) {
  // CHECK: %[[REVERSE:.*]] = stablehlo.reverse %arg0
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[REVERSE]] <@mesh_xyz, [{"x"}, {"z"}, {}, {}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.reverse %arg0, dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {"z"}, {}, {}]>]>} : tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}

// CHECK-LABEL: func @reverse_both_input_and_output_permutation_dims_are_sharded
func.func @reverse_both_input_and_output_permutation_dims_are_sharded(%arg0: tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"z":(1)2}, {}, {}]>}) -> (tensor<4x32x8x2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {}, {"z":(2)2}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x"}, {}, {}, {"z":(2)2}]> : tensor<4x32x8x2xf32>
  // CHECK-NEXT: %[[REVERSE:.*]] = stablehlo.reverse %[[RESHARD]], dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {}, {"z":(2)2}]>]>} : tensor<4x32x8x2xf32>
  // CHECK-NEXT: return %[[REVERSE]] : tensor<4x32x8x2xf32>
  %0 = stablehlo.reverse %arg0, dims = [1, 3] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {}, {"z":(2)2}]>]>}: tensor<4x32x8x2xf32>
  return %0 : tensor<4x32x8x2xf32>
}
