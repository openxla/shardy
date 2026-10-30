// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true mark-partial-result-with-unreduced-axes=true' | FileCheck %s --check-prefixes=CHECK,UNREDUCED
// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true mark-partial-result-with-unreduced-axes=false' | FileCheck %s --check-prefixes=CHECK,NOUNREDUCED

sdy.mesh @mesh = <["x"=4, "y"=2]>
sdy.mesh @mesh_xyz = <["x"=4, "y"=2, "z"=2]>

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
  // UNREDUCED:        %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // NOUNREDUCED:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1]
  // NOUNREDUCED-NOT:  sdy.sharding
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_single_result_multiple_reduction_dims_sharded
func.func @reduce_single_result_multiple_reduction_dims_sharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}, {}]>}) -> tensor<13xf32> {
  // UNREDUCED:        %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}], unreduced={"x", "y"}>]>}
  // NOUNREDUCED:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1]
  // NOUNREDUCED-NOT:  sdy.sharding
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x", "y"} %[[REDUCE]] out_sharding=<@mesh, [{}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<13xf32>
  return %1 : tensor<13xf32>
}

// CHECK-LABEL: func @reduce_single_result_multiple_reduction_dims_sharded_sub_axis
func.func @reduce_single_result_multiple_reduction_dims_sharded_sub_axis(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(2)2}, {"y", "x":(1)2}, {}]>}) -> tensor<13xf32> {
  // UNREDUCED:        %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}], unreduced={"x", "y"}>]>}
  // NOUNREDUCED:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1]
  // NOUNREDUCED-NOT:  sdy.sharding
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x", "y"} %[[REDUCE]] out_sharding=<@mesh, [{}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<13xf32>
  return %1 : tensor<13xf32>
}

// CHECK-LABEL: func @reduce_min_sharded
func.func @reduce_min_sharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> tensor<2x13xf32> {
  // UNREDUCED:        %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.minimum across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced=min{"x"}>]>}
  // NOUNREDUCED:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.minimum across dimensions = [1]
  // NOUNREDUCED-NOT:  sdy.sharding
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce min {"x"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.minimum across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_max_sharded
func.func @reduce_max_sharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> tensor<2x13xf32> {
  // UNREDUCED:        %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced=max{"x"}>]>}
  // NOUNREDUCED:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.maximum across dimensions = [1]
  // NOUNREDUCED-NOT:  sdy.sharding
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce max {"x"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.maximum across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_custom_combiner_without_unreduced
func.func @reduce_custom_combiner_without_unreduced(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> tensor<2x13xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.multiply across dimensions = [1]
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = "stablehlo.reduce"(%arg0, %0) <{dimensions = array<i64: 1>}> ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %2 = stablehlo.multiply %arg1, %arg2 : tensor<f32>
    stablehlo.return %2 : tensor<f32>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_multiple_results
func.func @reduce_multiple_results(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<2x64x13xi32>)
    -> (tensor<64xf32>, tensor<64xi32>) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  // CHECK:            %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}, {}]> : tensor<2x64x13xi32>
  // UNREDUCED-NEXT:   %[[REDUCE:.*]]:2 = stablehlo.reduce(%arg0 init: %{{.*}}), (%[[RESHARD]] init: %{{.*}}) across dimensions = [0, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}], unreduced={"x"}>, <@mesh, [{?}], unreduced={"x"}>]>}
  // NOUNREDUCED-NEXT: %[[REDUCE:.*]]:2 = stablehlo.reduce(%arg0 init: %{{.*}}), (%[[RESHARD]] init: %{{.*}}) across dimensions = [0, 2]
  // NOUNREDUCED-NOT:  sdy.sharding
  // CHECK:            %[[ALL_REDUCE1:.*]] = sdy.all_reduce {"x"} %[[REDUCE]]#0 out_sharding=<@mesh, [{}]> : tensor<64xf32>
  // UNREDUCED-NEXT:   %[[ALL_REDUCE2:.*]] = sdy.all_reduce {"x"} %[[REDUCE]]#1 out_sharding=<@mesh, [{?}]> : tensor<64xi32>
  // NOUNREDUCED-NEXT: %[[ALL_REDUCE2:.*]] = sdy.all_reduce {"x"} %[[REDUCE]]#1 out_sharding=<@mesh, [{}]> : tensor<64xi32>
  // CHECK-NEXT:       return %[[ALL_REDUCE1]], %[[ALL_REDUCE2]] : tensor<64xf32>, tensor<64xi32>
  %2:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1) across dimensions = [0, 2] :
    (tensor<2x64x13xf32>, tensor<2x64x13xi32>, tensor<f32>, tensor<i32>) -> (tensor<64xf32>, tensor<64xi32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<i32>, %arg5: tensor<i32>)  {
      %3 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %4 = stablehlo.add %arg3, %arg5 : tensor<i32>
      stablehlo.return %3, %4 : tensor<f32>, tensor<i32>
    }
  return %2#0, %2#1 : tensor<64xf32>, tensor<64xi32>
}

// CHECK-LABEL: func @reduce_scalar_result_multiple_reduction_dims_sharded
func.func @reduce_scalar_result_multiple_reduction_dims_sharded(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"y"}, {"x"}]>}) -> tensor<f32> {
  // UNREDUCED:        %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [], unreduced={"x", "y"}>]>}
  // UNREDUCED-NEXT:   %[[AR_X:.*]] = sdy.all_reduce {"x"} %[[REDUCE]] out_sharding=<@mesh_xyz, [], unreduced={"y"}> : tensor<f32>
  // UNREDUCED-NEXT:   %[[AR_Y:.*]] = sdy.all_reduce {"y"} %[[AR_X]] out_sharding=<@mesh_xyz, []> : tensor<f32>
  // UNREDUCED-NEXT:   return %[[AR_Y]]
  // NOUNREDUCED:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1]
  // NOUNREDUCED-NOT:  sdy.sharding
  // NOUNREDUCED-NEXT: %[[AR:.*]] = sdy.all_reduce {"x", "y"} %[[REDUCE]] out_sharding=<@mesh_xyz, []> : tensor<f32>
  // NOUNREDUCED-NEXT: return %[[AR]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [0, 1] : (tensor<8x16xf32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: func @reduce_local_scalar_result_multiple_reduction_dims_sharded
func.func @reduce_local_scalar_result_multiple_reduction_dims_sharded(%arg0: tensor<2x8x16xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}, {"y"}, {"x"}]>}) -> (tensor<2xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"z"}]>}) {
  // UNREDUCED:        %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}], unreduced={"x", "y"}>]>}
  // UNREDUCED-NEXT:   %[[AR_X:.*]] = sdy.all_reduce {"x"} %[[REDUCE]] out_sharding=<@mesh_xyz, [{"z"}], unreduced={"y"}> : tensor<2xf32>
  // UNREDUCED-NEXT:   %[[AR_Y:.*]] = sdy.all_reduce {"y"} %[[AR_X]] out_sharding=<@mesh_xyz, [{"z"}]> : tensor<2xf32>
  // UNREDUCED-NEXT:   return %[[AR_Y]]
  // NOUNREDUCED:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}]>]>}
  // NOUNREDUCED-NEXT: %[[AR:.*]] = sdy.all_reduce {"x", "y"} %[[REDUCE]] out_sharding=<@mesh_xyz, [{"z"}]> : tensor<2xf32>
  // NOUNREDUCED-NEXT: return %[[AR]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1, 2] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [{"z"}]>]>} : (tensor<2x8x16xf32>, tensor<f32>) -> tensor<2xf32>
  return %1 : tensor<2xf32>
}

// CHECK-LABEL: func @reduce_scalar_result_sub_axes_same_mesh_axis
func.func @reduce_scalar_result_sub_axes_same_mesh_axis(%arg0: tensor<6x10xf32> {sdy.sharding = #sdy.sharding<@mesh_xyz, [{"x":(1)2}, {"x":(2)2}]>}) -> tensor<f32> {
  // UNREDUCED:        %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xyz, [], unreduced={"x"}>]>}
  // UNREDUCED-NEXT:   %[[AR_MINOR:.*]] = sdy.all_reduce {"x":(2)2} %[[REDUCE]] out_sharding=<@mesh_xyz, [], unreduced={"x":(1)2}> : tensor<f32>
  // UNREDUCED-NEXT:   %[[AR_MAJOR:.*]] = sdy.all_reduce {"x":(1)2} %[[AR_MINOR]] out_sharding=<@mesh_xyz, []> : tensor<f32>
  // UNREDUCED-NEXT:   return %[[AR_MAJOR]]
  // NOUNREDUCED:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1]
  // NOUNREDUCED-NOT:  sdy.sharding
  // NOUNREDUCED-NEXT: %[[AR:.*]] = sdy.all_reduce {"x"} %[[REDUCE]] out_sharding=<@mesh_xyz, []> : tensor<f32>
  // NOUNREDUCED-NEXT: return %[[AR]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [0, 1] : (tensor<6x10xf32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}

// CHECK-LABEL: func @reduce_scalar_result_full_mesh_not_split
func.func @reduce_scalar_result_full_mesh_not_split(%arg0: tensor<6x12xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>}) -> tensor<f32> {
  // UNREDUCED:        %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [], unreduced={"x", "y"}>]>}
  // NOUNREDUCED:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [0, 1]
  // NOUNREDUCED-NOT:  sdy.sharding
  // CHECK-NEXT:       %[[AR:.*]] = sdy.all_reduce {"x", "y"} %[[REDUCE]] out_sharding=<@mesh, []> : tensor<f32>
  // CHECK-NEXT:       return %[[AR]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [0, 1] : (tensor<6x12xf32>, tensor<f32>) -> tensor<f32>
  return %1 : tensor<f32>
}
