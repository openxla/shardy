// RUN: sdy_opt %s -sdy-export-pipeline='enable-insert-explicit-collectives=true remove-all-gather-reduce-scatter-for-cmv1=true' 2>&1 | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2, "z"=2]>

// CHECK-LABEL: func @reduce_scatter_fusion
func.func @reduce_scatter_fusion(%arg0: tensor<16x8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> (tensor<16x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK: %0 = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x8x8xf32>, tensor<f32>) -> tensor<16x8xf32>
  // CHECK-NEXT: %1 = sdy.reduce_scatter [{"x"}, {}] %0 out_sharding=<@mesh, [{"x"}, {}]> : tensor<16x8xf32>
  // CHECK-NEXT: %2 = sdy.all_slice [{}, {"y"}] %1 out_sharding=<@mesh, [{"x"}, {"y"}]> : tensor<16x8xf32>
  // CHECK-NEXT: return %2 : tensor<16x8xf32>
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<16x8x8xf32>, tensor<f32>) -> tensor<16x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{"x"}, {"y"}]> : tensor<16x8xf32>
  return %2 : tensor<16x8xf32>
}

// CHECK-LABEL: func @all_slice_all_gather
func.func @all_slice_all_gather(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %0 = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> : tensor<16x2xf32>
  // CHECK-NEXT: %1 = sdy.all_gather [{"y"}, {}] %0 out_sharding=<@mesh, [{}, {"x"}]> : tensor<16x2xf32>
  // CHECK-NEXT: return %1 : tensor<16x2xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"x"}]> : tensor<16x2xf32>
  return %0 : tensor<16x2xf32>
}

// CHECK-LABEL: func @reshard_of_reshard
func.func @reshard_of_reshard(%arg0 : tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<16x2xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"y"}, {"x"}]>}) {
  // CHECK: %0 = sdy.all_slice [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {"x"}]> : tensor<16x2xf32>
  // CHECK-NEXT: return %0 : tensor<16x2xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"x"}]> : tensor<16x2xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"y"}, {"x"}]> : tensor<16x2xf32>
  return %1 : tensor<16x2xf32>
}

// CHECK-LABEL: func @all_to_all_fusion
func.func @all_to_all_fusion(%arg0 : tensor<64x16x8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"x"}, {"y"}, {}, {}]>}) -> (tensor<64x16x8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{}, {}, {"x"}, {"y"}]>}) {
  // CHECK-NEXT: %0 = sdy.all_to_all [{"x"}: 0->2, {"y"}: 1->3] %arg0 out_sharding=<@mesh, [{}, {}, {"x"}, {"y"}]> : tensor<64x16x8x8xf32>
  // CHECK-NEXT: return %0 : tensor<64x16x8x8xf32>
  %0 = sdy.reshard %arg0 <@mesh, [{}, {"y"}, {"x"}, {}]> : tensor<64x16x8x8xf32>
  %1 = sdy.reshard %0 <@mesh, [{}, {}, {"x"}, {"y"}]> : tensor<64x16x8x8xf32>
  return %1 : tensor<64x16x8x8xf32>
}

// CHECK-LABEL: func @dot_general_with_unreduced_result
func.func @dot_general_with_unreduced_result(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y"}>]>}
  // CHECK-NEXT: %[[REDUCE_SCATTER_0:.*]] = sdy.reduce_scatter [{}, {"y"}] %0 out_sharding=<@mesh, [{"x"}, {"y"}]>
  // CHECK-NEXT: %[[REDUCE_SCATTER_1:.*]] = sdy.reduce_scatter [{}, {"y"}] %0 out_sharding=<@mesh, [{"x"}, {"y"}]>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[REDUCE_SCATTER_0]], %[[REDUCE_SCATTER_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>}
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y"}>]>} :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_general_with_unreduced_result_fully_delayed
func.func @dot_general_with_unreduced_result_fully_delayed(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y", "z"}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y", "z"}]>}) {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y", "z"}>]>}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %[[DOT_GENERAL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y", "z"}>]>}
  // CHECK-NEXT: %[[REDUCE_SCATTER:.*]] = sdy.reduce_scatter [{}, {"y", "z"}] %1 out_sharding=<@mesh, [{"x"}, {"y", "z"}]>
  // CHECK-NEXT: return %[[REDUCE_SCATTER]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y", "z"}>]>} :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y", "z"}>]>} : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @dot_general_with_unreduced_result_partially_delayed
func.func @dot_general_with_unreduced_result_partially_delayed(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y", "z"}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y", "z"}>]>}
  // CHECK-NEXT: %[[REDUCE_SCATTER_Y_0:.*]] = sdy.reduce_scatter [{}, {"y"}] %0 out_sharding=<@mesh, [{"x"}, {"y"}], unreduced={"z"}>
  // CHECK-NEXT: %[[REDUCE_SCATTER_Y_1:.*]] = sdy.reduce_scatter [{}, {"y"}] %0 out_sharding=<@mesh, [{"x"}, {"y"}], unreduced={"z"}>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[REDUCE_SCATTER_Y_0]], %[[REDUCE_SCATTER_Y_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}], unreduced={"z"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE_Z:.*]] = sdy.all_reduce {"z"} %[[ADD]] out_sharding=<@mesh, [{"x"}, {"y"}]>
  // CHECK-NEXT: return %[[ALL_REDUCE_Z]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}], unreduced={"y", "z"}>]>} :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}], unreduced={"z"}>]>} : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// TODO: Add more tests.
