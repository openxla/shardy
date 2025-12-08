// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=4]>

// CHECK-LABEL: func @cholesky_sharded_input_batch_dim_only
func.func @cholesky_sharded_input_batch_dim_only(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) -> tensor<8x4x8x8xf32> {
  // CHECK: %[[CHOLESKY:.*]] = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_output_batch_dim_only
func.func @cholesky_sharded_output_batch_dim_only(%arg0: tensor<8x4x8x8xf32>) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}){
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[CHOLESKY]] :  tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_batch_dim_only_different
func.func @cholesky_sharded_batch_dim_only_different(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {}, {}]>}){
  // CHECK: %[[CHOLESKY:.*]] = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{"y"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] :  tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_input_cholesky_dim_only
func.func @cholesky_sharded_input_cholesky_dim_only(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"x"}]>}) -> tensor<8x4x8x8xf32> {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[CHOLESKY]] :  tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_output_cholesky_dim_only
func.func @cholesky_sharded_output_cholesky_dim_only(%arg0: tensor<8x4x8x8xf32>) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"x"}]>}){
  // CHECK: %[[CHOLESKY:.*]] = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{}, {}, {}, {"x"}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] :  tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"x"}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_cholesky_dim_only_different
func.func @cholesky_sharded_cholesky_dim_only_different(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"x"}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"y"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{}, {}, {}, {"y"}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"y"}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_cholesky_dim_only_same
func.func @cholesky_sharded_cholesky_dim_only_same(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"x"}]>}){
  // CHECK: %[[CHOLESKY:.*]] = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{}, {}, {}, {"x"}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] :  tensor<8x4x8x8xf32>
  // TODO(enver): Instead reshard to [{"x"}, {}, {}, {}] and perform the operation on this smaller tensor.
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"x"}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_input_batch_dim_and_output_cholesky_dim_same
func.func @cholesky_sharded_input_batch_dim_and_output_cholesky_dim_same(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"x"}]>}) {
  // CHECK: %[[CHOLESKY:.*]] = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{}, {}, {}, {"x"}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD]] :  tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}, {"x"}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_output_batch_dim_and_input_cholesky_dim_same
func.func @cholesky_sharded_output_batch_dim_and_input_cholesky_dim_same(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {"x"}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {}]>}){
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[CHOLESKY]] : tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_same
func.func @cholesky_sharded_same(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {"y"}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}, {"y"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x", "y"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh, [{"x"}, {}, {}, {"y"}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}, {"y"}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_same_both_cholesky_dimensions
func.func @cholesky_sharded_same_both_cholesky_dimensions(%arg0: tensor<128x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>}) -> (tensor<128x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "y", "z"}, {}, {}, {}]> : tensor<128x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y", "z"}, {}, {}, {}]>]>} : tensor<128x4x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]> : tensor<128x4x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<128x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>]>} : (tensor<128x4x8x8xf32>) -> tensor<128x4x8x8xf32>
  return %0 :  tensor<128x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_same_both_cholesky_dimensions_small_batch_dim
func.func @cholesky_sharded_same_both_cholesky_dimensions_small_batch_dim(%arg0: tensor<16x2x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>}) -> (tensor<16x2x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "y"}, {}, {}, {}]> : tensor<16x2x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y"}, {}, {}, {}]>]>} : tensor<16x2x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]> : tensor<16x2x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x2x8x8xf32>
  // TODO(enver): Instead reshard to [{"x", "y", "z":1(2)}, {}, {}, {}].
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>]>} : (tensor<16x2x8x8xf32>) -> tensor<16x2x8x8xf32>
  return %0 :  tensor<16x2x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_same_both_cholesky_dimensions_small_batch_dim_second_batch_dim_larger
func.func @cholesky_sharded_same_both_cholesky_dimensions_small_batch_dim_second_batch_dim_larger(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>}) -> (tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "y"}, {"z"}, {}, {}]> : tensor<16x8x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y"}, {"z"}, {}, {}]>]>} : tensor<16x8x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]> : tensor<16x8x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x8x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {"y"}, {"z"}]>]>} : (tensor<16x8x8x8xf32>) -> tensor<16x8x8x8xf32>
  return %0 :  tensor<16x8x8x8xf32>
}

// CHECK-LABEL: func @cholesky_batch_and_cholesky_dims_shardings_can_merge
func.func @cholesky_batch_and_cholesky_dims_shardings_can_merge(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(1)2}, {}, {"x":(2)2,"y"}, {"z"}]>}) -> (tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(1)2}, {}, {"x":(2)2, "y"}, {"z"}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "y"}, {"z"}, {}, {}]> : tensor<16x8x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "y"}, {"z"}, {}, {}]>]>} : tensor<16x8x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh_xyz, [{"x":(1)2}, {}, {"x":(2)2, "y"}, {"z"}]> : tensor<16x8x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x8x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x":(1)2}, {}, {"x":(2)2, "y"}, {"z"}]>]>} : (tensor<16x8x8x8xf32>) -> tensor<16x8x8x8xf32>
  return %0 :  tensor<16x8x8x8xf32>
}

// CHECK-LABEL: func @cholesky_cholesky_dims_shardings_can_merge
func.func @cholesky_cholesky_dims_shardings_can_merge(%arg0: tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {"z":(2)2}]>}) -> (tensor<16x8x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {"z":(2)2}]>}){
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x", "z"}, {}, {}, {}]> : tensor<16x8x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD1]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x", "z"}, {}, {}, {}]>]>} : tensor<16x8x8x8xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CHOLESKY]] <@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {"z":(2)2}]> : tensor<16x8x8x8xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<16x8x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {"z":(1)2}, {"z":(2)2}]>]>} : (tensor<16x8x8x8xf32>) -> tensor<16x8x8x8xf32>
  return %0 :  tensor<16x8x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_cholesky_dim_input_only_batch_dim_both_but_input_sharding_larger
func.func @cholesky_sharded_cholesky_dim_input_only_batch_dim_both_but_input_sharding_larger(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {}, {"z"}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {}, {}]>}){
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh_xyz, [{"y"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %1 = stablehlo.cholesky %0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %1 : tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"y"}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @cholesky_sharded_cholesky_dim_input_only_batch_dim_both_but_output_sharding_larger
func.func @cholesky_sharded_cholesky_dim_input_only_batch_dim_both_but_output_sharding_larger(%arg0: tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {}, {}, {"z"}]>}) -> (tensor<8x4x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x"}, {}, {}, {}]>}){
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh_xyz, [{"x"}, {}, {}, {}]> : tensor<8x4x8x8xf32>
  // CHECK-NEXT: %[[CHOLESKY:.*]] = stablehlo.cholesky %[[RESHARD]], lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {}, {}]>]>} : tensor<8x4x8x8xf32>
  // CHECK-NEXT: return %[[CHOLESKY:.*]] : tensor<8x4x8x8xf32>
  %0 = stablehlo.cholesky %arg0, lower = true {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"x"}, {}, {}, {}]>]>} : (tensor<8x4x8x8xf32>) -> tensor<8x4x8x8xf32>
  return %0 :  tensor<8x4x8x8xf32>
}

// CHECK-LABEL: func @triangular_solve
func.func @triangular_solve(%arg0: tensor<8x3x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<8x3x5xf32>) -> tensor<8x3x5xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<8x3x5xf32>
  // CHECK-NEXT: %[[TRIANGULAR_SOLVE:.*]] = "stablehlo.triangular_solve"(%arg0, %[[RESHARD1]])
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[TRIANGULAR_SOLVE]] <@mesh, [{}, {}, {}]> : tensor<8x3x5xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x3x5xf32>
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = false,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>
  } : (tensor<8x3x3xf32>, tensor<8x3x5xf32>) -> tensor<8x3x5xf32>
  return %0 : tensor<8x3x5xf32>
}

// CHECK-LABEL: func @triangular_solve_replicated_dim_is_sharded
func.func @triangular_solve_replicated_dim_is_sharded(%arg0: tensor<8x3x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}, {}]>}, %arg1: tensor<8x3x5xf32>) -> tensor<8x3x5xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}, {}]> : tensor<8x3x3xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<8x3x5xf32>
  // CHECK-NEXT: %[[TRIANGULAR_SOLVE:.*]] = "stablehlo.triangular_solve"(%[[RESHARD1]], %[[RESHARD2]])
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[TRIANGULAR_SOLVE]] <@mesh, [{}, {}, {}]> : tensor<8x3x5xf32>
  // CHECK-NEXT: return %[[RESHARD3]] : tensor<8x3x5xf32>
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) <{
    left_side = true,
    lower = true,
    unit_diagonal = false,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>
  }> {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<8x3x3xf32>, tensor<8x3x5xf32>) -> tensor<8x3x5xf32>
  return %0 : tensor<8x3x5xf32>
}
