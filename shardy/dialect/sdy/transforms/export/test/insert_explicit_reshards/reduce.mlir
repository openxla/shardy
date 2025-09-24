// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @reduce_single_result_reduction_dim_not_sharded
func.func @reduce_single_result_reduction_dim_not_sharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<2x13xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[REDUCE]] <@mesh, [{}, {}]> : tensor<2x13xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<2x13xf32>
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_single_result_reduction_dim_sharded
func.func @reduce_single_result_reduction_dim_sharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> tensor<2x13xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1]
  // CHECK-NOT:  sdy.sharding
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_single_result_multiple_reduction_dims_sharded
func.func @reduce_single_result_multiple_reduction_dims_sharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}, {}]>}) -> tensor<13xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1]
  // CHECK-NOT:  sdy.sharding
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y", "x"} %[[REDUCE]] out_sharding=<@mesh, [{}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<13xf32>
  return %1 : tensor<13xf32>
}

// CHECK-LABEL: func @reduce_single_result_multiple_reduction_dims_sharded_sub_axis
func.func @reduce_single_result_multiple_reduction_dims_sharded_sub_axis(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(2)2}, {"y", "x":(1)2}, {}]>}) -> tensor<13xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1]
  // CHECK-NOT:  sdy.sharding
  // TODO(enver, zixuanjiang): Axes should be canonicalized: merged and sorted.
  // So that it becomes sdy.all_reduce {"x", "y"}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x":(2)2, "y", "x":(1)2} %[[REDUCE]] out_sharding=<@mesh, [{}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<13xf32>
  return %1 : tensor<13xf32>
}

// CHECK-LABEL: func @reduce_multiple_results
func.func @reduce_multiple_results(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<2x64x13xi32>)
    -> (tensor<64xf32>, tensor<64xi32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<2x64x13xi32>
  // CHECK-NEXT: %[[REDUCE:.*]]:2 = stablehlo.reduce(%arg0 init: %cst), (%[[RESHARD]] init: %c) across dimensions = [0, 2]
  // CHECK-NOT:  sdy.sharding
  // CHECK:      %[[ALL_REDUCE1:.*]] = sdy.all_reduce {"x"} %[[REDUCE]]#0 out_sharding=<@mesh, [{}]> : tensor<64xf32>
  // CHECK-NEXT: %[[ALL_REDUCE2:.*]] = sdy.all_reduce {"x"} %[[REDUCE]]#1 out_sharding=<@mesh, [{}]> : tensor<64xi32>
  // CHECK-NEXT: return %[[ALL_REDUCE1]], %[[ALL_REDUCE2]] : tensor<64xf32>, tensor<64xi32>
  %2:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1) across dimensions = [0, 2] :
    (tensor<2x64x13xf32>, tensor<2x64x13xi32>, tensor<f32>, tensor<i32>) -> (tensor<64xf32>, tensor<64xi32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<i32>, %arg5: tensor<i32>)  {
      %3 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %4 = stablehlo.add %arg3, %arg5 : tensor<i32>
      stablehlo.return %3, %4 : tensor<f32>, tensor<i32>
    }
  return %2#0, %2#1 : tensor<64xf32>, tensor<64xi32>
}
