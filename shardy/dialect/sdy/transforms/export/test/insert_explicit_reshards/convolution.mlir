// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>

// CHECK-LABEL: func @convolution
func.func @convolution(%arg0 : tensor<2x224x224x192xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}, %arg1 : tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32> {
  // CHECK: %[[CONVOLUTION:.*]] = stablehlo.convolution(%arg0, %arg1)
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CONVOLUTION]] <@mesh, [{}, {}, {}, {}]> : tensor<2x112x112x64xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<2x112x112x64xf32>
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<2x224x224x192xf32>, tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32>
  return %0 : tensor<2x112x112x64xf32>
}

// CHECK-LABEL: func @convolution_batch_group_count
func.func @convolution_batch_group_count(%arg0: tensor<8x224x224x192xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {"z"}, {}, {}]>}, %arg1: tensor<3x3x192x256xf32>) -> (tensor<2x112x112x256xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"z"}, {}, {}]>}) {
  // NOTE: sdy.sharding_rule = ([ij, kl, mn, o], [l, n, o, ip])->([j, k, m, ip]) {i=4, j=2, k=112, l=2, m=112, n=2, o=192, p=64} reduction={l, n, o} permutation={k, m}
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "y"}, {"z"}, {}, {}]> : tensor<8x224x224x192xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{}, {}, {}, {"x"}]> : tensor<3x3x192x256xf32>
  // CHECK-NEXT: %[[CONVOLUTION:.*]] = stablehlo.convolution(%[[RESHARD1]], %[[RESHARD2]])
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {"z"}, {}, {"x"}]>]>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CONVOLUTION]] <@mesh_xyz, [{"y"}, {"z"}, {}, {}]> : tensor<2x112x112x256xf32>
  // CHECK-NEXT: return %[[RESHARD3]]
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 4 : i64,
      feature_group_count = 1 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {"z"}, {}, {}]>]>} : (tensor<8x224x224x192xf32>, tensor<3x3x192x256xf32>) -> tensor<2x112x112x256xf32>
  return %0 : tensor<2x112x112x256xf32>
}

// CHECK-LABEL: func @convolution_batch_group_count_factor_is_not_fully_sharded
func.func @convolution_batch_group_count_factor_is_not_fully_sharded(%arg0: tensor<8x224x224x192xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(1)2}, {"z"}, {}, {}]>}, %arg1: tensor<3x3x192x256xf32>) -> (tensor<2x112x112x256xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"z"}, {}, {}]>}) {
  // NOTE: sdy.sharding_rule = ([ij, kl, mn, o], [l, n, o, ip])->([j, k, m, ip]) {i=4, j=2, k=112, l=2, m=112, n=2, o=192, p=64} reduction={l, n, o} permutation={k, m}
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh_xyz, [{}, {}, {}, {"x":(1)2}]> : tensor<3x3x192x256xf32>
  // CHECK-NEXT: %[[CONVOLUTION:.*]] = stablehlo.convolution(%arg0, %[[RESHARD1]])
  // CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{}, {"z"}, {}, {"x":(1)2}]>]>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CONVOLUTION]] <@mesh_xyz, [{"y"}, {"z"}, {}, {}]>
  // CHECK-NEXT: return %[[RESHARD2]]
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 4 : i64,
      feature_group_count = 1 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>,
      sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {"z"}, {}, {}]>]>} : (tensor<8x224x224x192xf32>, tensor<3x3x192x256xf32>) -> tensor<2x112x112x256xf32>
  return %0 : tensor<2x112x112x256xf32>
}
