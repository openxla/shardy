// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2, "z"=4]>

// CHECK-LABEL: func @all_reduce_on_func_input
func.func @all_reduce_on_func_input(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %[[ALL_REDUCE]], %arg1
  // CHECK-NEXT: return %[[MUL]]
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @unreduced_func_input_until_return
func.func @unreduced_func_input_until_return(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>})
    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"y"}>]>} : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @lhs_and_result_unreduced_rhs_replicated
func.func @lhs_and_result_unreduced_rhs_replicated(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}, %arg1: tensor<4x8xf32>)
    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"y"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[MUL]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.multiply %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"y"}>]>} : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_fully_delayed_until_return
func.func @all_reduce_fully_delayed_until_return(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> tensor<2x13xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK:      %[[ADD:.*]] = stablehlo.add %[[REDUCE]], %[[REDUCE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[ADD]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : tensor<2x13xf32>
  return %2 : tensor<2x13xf32>
}

// CHECK-LABEL: func @all_reduce_delayed_to_call_site
func.func @all_reduce_delayed_to_call_site(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>}) -> (tensor<2x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK:      %[[ADD:.*]] = stablehlo.add %[[REDUCE]], %[[REDUCE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : tensor<2x13xf32>
  return %2 : tensor<2x13xf32>
}

// CHECK-LABEL: func @all_reduce_partially_delayed_until_return
func.func @all_reduce_partially_delayed_until_return(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>}) -> tensor<2x13xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE_Y:.*]] = sdy.all_reduce {"y"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}], unreduced={"x"}>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ALL_REDUCE_Y]], %[[ALL_REDUCE_Y]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE_X:.*]] = sdy.all_reduce {"x"} %[[ADD]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE_X]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : tensor<2x13xf32>
  return %2 : tensor<2x13xf32>
}

// All-reduces will be deduped during canonicalization.
// CHECK-LABEL: func @all_reduce_delayed_until_op
func.func @all_reduce_delayed_until_op(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>})
    -> (tensor<2x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK:      %[[ADD:.*]] = stablehlo.add %[[REDUCE]], %[[REDUCE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE_1:.*]] = sdy.all_reduce {"x"} %[[ADD]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: %[[ALL_REDUCE_2:.*]] = sdy.all_reduce {"x"} %[[ADD]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: %[[RESHARD_1:.*]] = sdy.reshard %[[ALL_REDUCE_1]] <@mesh, [{}, {"x"}]>
  // CHECK-NEXT: %[[RESHARD_2:.*]] =  sdy.reshard %[[ALL_REDUCE_2]] <@mesh, [{}, {"x"}]>
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %[[RESHARD_1]], %[[RESHARD_2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>}
  // CHECK-NEXT: return %[[MUL]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : tensor<2x13xf32>
  %3 = stablehlo.multiply %2, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : tensor<2x13xf32>
  return %3 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_single_result_fully_unreduced
func.func @reduce_single_result_fully_unreduced(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}, {}]>})
    -> (tensor<2x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x", "y"}>}) {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x", "y"}>]>}
  // CHECK-NEXT: return %[[REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x", "y"}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_single_result_partially_unreduced
func.func @reduce_single_result_partially_unreduced(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y", "x"}, {}]>})
    -> (tensor<2x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}], unreduced={"x"}>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_single_result_unreduced_full_axis
func.func @reduce_single_result_unreduced_full_axis(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x":(1)2, "y"}, {}]>})
    -> (tensor<2x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}], unreduced={"x"}>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_single_result_unreduced_sub_axis
func.func @reduce_single_result_unreduced_sub_axis(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>})
    -> (tensor<2x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x":(1)2}>}) {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x":(1)2}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x":(2)2, "y"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}], unreduced={"x":(1)2}>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x":(1)2}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_single_result_unreduced_resharded
func.func @reduce_single_result_unreduced_resharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>})
    -> (tensor<2x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>}) {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %[[REDUCE]] out_sharding=<@mesh, [{}, {}], unreduced={"x"}>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"y"}, {}], unreduced={"x"}>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}], unreduced={"x"}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK-LABEL: func @reduce_multiple_results_unreduced
func.func @reduce_multiple_results_unreduced(
    %arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}, {}]>},
    %arg1: tensor<2x64x13xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}, {}]>})
    -> (tensor<64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={"x"}>},
        tensor<64xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={"x":(1)2}>}) {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<0> : tensor<i32>
  // CHECK:      %[[REDUCE:.*]]:2 = stablehlo.reduce(%arg0 init: %cst), (%arg1 init: %c) across dimensions = [0, 2]
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}], unreduced={"x"}>, <@mesh, [{}], unreduced={"x"}>]>}
  // CHECK:      %[[ALL_REDUCE1:.*]] = sdy.all_reduce {"y"} %[[REDUCE]]#0 out_sharding=<@mesh, [{}], unreduced={"x"}> : tensor<64xf32>
  // CHECK-NEXT: %[[ALL_REDUCE2:.*]] = sdy.all_reduce {"y"} %[[REDUCE]]#1 out_sharding=<@mesh, [{}], unreduced={"x"}> : tensor<64xi32>
  // CHECK-NEXT: %[[ALL_REDUCE3:.*]] = sdy.all_reduce {"x":(2)2} %[[ALL_REDUCE2]] out_sharding=<@mesh, [{}], unreduced={"x":(1)2}> : tensor<64xi32>
  // CHECK-NEXT: return %[[ALL_REDUCE1]], %[[ALL_REDUCE3]] : tensor<64xf32>, tensor<64xi32>
  %2:2 = stablehlo.reduce(%arg0 init: %0), (%arg1 init: %1) across dimensions = [0, 2]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}], unreduced={"x"}>, <@mesh, [{}], unreduced={"x"}>]>} :
    (tensor<2x64x13xf32>, tensor<2x64x13xi32>, tensor<f32>, tensor<i32>) -> (tensor<64xf32>, tensor<64xi32>)
    reducer(%arg2: tensor<f32>, %arg4: tensor<f32>) (%arg3: tensor<i32>, %arg5: tensor<i32>)  {
      %3 = stablehlo.add %arg2, %arg4 : tensor<f32>
      %4 = stablehlo.add %arg3, %arg5 : tensor<i32>
      stablehlo.return %3, %4 : tensor<f32>, tensor<i32>
    }
  return %2#0, %2#1 : tensor<64xf32>, tensor<64xi32>
}

// CHECK-LABEL: func @manual_computation_all_reduce_free_axis_on_return_value
func.func @manual_computation_all_reduce_free_axis_on_return_value(%arg0: tensor<208xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={"x", "y"}>})
    -> (tensor<208xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK-NEXT: %[[ALL_REDUCE_X:.*]] = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{}], unreduced={"y"}>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE_X]] <@mesh, [{"x"}], unreduced={"y"}>
  // CHECK-NEXT: %[[MANUAL_COMP:.*]] = sdy.manual_computation(%[[RESHARD]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}], unreduced={"y"}>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"}
  // CHECK-SAME:   (%arg1: tensor<52xf32>) {
  // CHECK-NEXT:   %[[ABS:.*]] = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}], unreduced={"y"}>]>}
  // CHECK-NEXT:   %[[ALL_REDUCE_Y:.*]] = sdy.all_reduce {"y"} %[[ABS]] out_sharding=<@mesh, [{}]>
  // CHECK-NEXT:   sdy.return %[[ALL_REDUCE_Y]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[MANUAL_COMP]]
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh, [{"x"}], unreduced={"y"}>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<52xf32>) {
    %1 = stablehlo.abs %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}], unreduced={"y"}>]>} : tensor<52xf32>
    sdy.return %1 : tensor<52xf32>
  } : (tensor<208xf32>) -> tensor<208xf32>
  return %0 : tensor<208xf32>
}

// Note that in sharding is unreduced along manual axis "x", but out sharding
// isn't, this is supported and the user is expected to reduce along the manual
// axis given it becomes replicated (which isn't done in this test case).
// CHECK-LABEL: func @manual_computation_all_reduce_free_axis_on_block_arg
func.func @manual_computation_all_reduce_free_axis_on_block_arg(%arg0: tensor<208xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={"x", "y"}>})
    -> (tensor<208xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}) {
  // CHECK-NEXT: %[[MANUAL_COMP:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME:   in_shardings=[<@mesh, [{}], unreduced={"x", "y"}>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{}]>] manual_axes={"x"}
  // CHECK-SAME:   (%arg1: tensor<208xf32>) {
  // CHECK-NEXT:   %[[ALL_REDUCE_Y:.*]] = sdy.all_reduce {"y"} %arg1 out_sharding=<@mesh, [{}]>
  // CHECK-NEXT:   %[[ABS:.*]] = stablehlo.abs %[[ALL_REDUCE_Y]]
  // CHECK-NEXT:   sdy.return %[[ABS]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[MANUAL_COMP]]
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh, [{}], unreduced={"x", "y"}>] out_shardings=[<@mesh, [{}]>] manual_axes={"x"} (%arg1: tensor<208xf32>) {
    %1 = stablehlo.abs %arg1 : tensor<208xf32>
    sdy.return %1 : tensor<208xf32>
  } : (tensor<208xf32>) -> tensor<208xf32>
  return %0 : tensor<208xf32>
}

// CHECK-LABEL: func @all_reduce_source_has_unreduced_and_target_no_sharding
func.func @all_reduce_source_has_unreduced_and_target_no_sharding(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>})
      -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_source_no_sharding_and_target_has_unreduced
func.func @all_reduce_source_no_sharding_and_target_has_unreduced(
  %arg0: tensor<4x8xf32>)
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}) {
  // CHECK-NEXT: return %arg0
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_both_source_and_target_has_same_unreduced
func.func @all_reduce_both_source_and_target_has_same_unreduced(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}) {
  // CHECK-NEXT: return %arg0
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_target_unreduced_is_strict_subset_of_source_unreduced
func.func @all_reduce_target_unreduced_is_strict_subset_of_source_unreduced(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x", "y"}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{}, {}], unreduced={"x"}>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_source_and_target_has_different_shardings_and_different_unreduced_axes
func.func @all_reduce_source_and_target_has_different_shardings_and_different_unreduced_axes(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x":(1)2}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}], unreduced={"x"}>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}], unreduced={"x"}>
  // CHECK-NEXT: return %[[RESHARD]]
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_source_and_target_has_different_shardings_and_same_unreduced_axes
func.func @all_reduce_source_and_target_has_different_shardings_and_same_unreduced_axes(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x":(1)2}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}], unreduced={"x":(1)2}>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"y"}], unreduced={"x":(1)2}>
  // CHECK-NEXT: return %[[RESHARD]]
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_source_and_target_has_same_shardings_and_different_unreduced_axes
func.func @all_reduce_source_and_target_has_same_shardings_and_different_unreduced_axes(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x":(1)2}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>}) {
  // CHECK-NEXT: return %arg0
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_source_and_target_has_different_shardings_only_source_has_unreduced_axes
func.func @all_reduce_source_and_target_has_different_shardings_only_source_has_unreduced_axes(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}, {}], unreduced={"y"}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(2)2}, {}]>}) {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{"x":(1)2}, {}]>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh, [{"x":(2)2}, {}]>
  // CHECK-NEXT: return %[[RESHARD]]
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_source_and_target_has_same_shardings_only_source_has_unreduced_axes
func.func @all_reduce_source_and_target_has_same_shardings_only_source_has_unreduced_axes(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{"x"}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_source_and_target_has_same_shardings_and_same_unreduced_axes
func.func @all_reduce_source_and_target_has_same_shardings_and_same_unreduced_axes(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x":(1)2}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x":(1)2}>}) {
  // CHECK-NEXT: return %arg0
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_source_and_target_fully_replicated_shardings_and_different_unreduced_axes
func.func @all_reduce_source_and_target_fully_replicated_shardings_and_different_unreduced_axes(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x":(1)2}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) {
  // CHECK-NEXT: return %arg0
  return %arg0 : tensor<4x8xf32>
}

//===----------------------------------------------------------------------===//
// Sharded to unreduced tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @sharded_to_unreduced_1
func.func @sharded_to_unreduced_1(
    %arg0 : tensor<24x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<24x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) {
  // CHECK-NEXT: %0 = sdy.sharded_to_unreduced [{"x"}, {}] %arg0 out_sharding=<@mesh, [{}, {}], unreduced={"x"}>
  // CHECK-NEXT: return %0
  %0 = sdy.reshard %arg0 <@mesh, [{}, {}], unreduced={"x"}> : tensor<24x8xf32>
  return %0 : tensor<24x8xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_single_axis
func.func @sharded_to_unreduced_single_axis(
    %arg0 : tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {"x"}]>})
    -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>}) {
  // CHECK-NEXT: %0 = sdy.sharded_to_unreduced [{}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}], unreduced={"x"}>
  // CHECK-NEXT: return %0
  %0 = sdy.reshard %arg0 <@mesh, [{"y"}, {}], unreduced={"x"}> : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_multiple_axes
func.func @sharded_to_unreduced_multiple_axes(
    %arg0 : tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "z", "y"}, {}]>})
    -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y", "z"}>}) {
  // CHECK-NEXT: %0 = sdy.sharded_to_unreduced [{"z", "y"}, {}] %arg0 out_sharding=<@mesh, [{"x"}, {}], unreduced={"y", "z"}>
  // CHECK-NEXT: return %0
  %0 = sdy.reshard %arg0 <@mesh, [{"x"}, {}], unreduced={"y", "z"}> : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_multiple_dims
func.func @sharded_to_unreduced_multiple_dims(
    %arg0 : tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y", "z"}, {"x"}]>})
    -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x", "z"}>}) {
  // CHECK-NEXT: %0 = sdy.sharded_to_unreduced [{"z"}, {"x"}] %arg0 out_sharding=<@mesh, [{"y"}, {}], unreduced={"x", "z"}>
  // CHECK-NEXT: return %0
  %0 = sdy.reshard %arg0 <@mesh, [{"y"}, {}], unreduced={"x", "z"}> : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_with_subaxis
func.func @sharded_to_unreduced_with_subaxis(
    %arg0 : tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}, {"x"}]>})
    -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z":(1)2}, {"x"}], unreduced={"z":(2)2}>}) {
  // CHECK-NEXT: %0 = sdy.sharded_to_unreduced [{"z":(2)2}, {}] %arg0 out_sharding=<@mesh, [{"z":(1)2}, {"x"}], unreduced={"z":(2)2}>
  // CHECK-NEXT: return %0
 %0 = sdy.reshard %arg0 <@mesh, [{"z":(1)2}, {"x"}], unreduced={"z":(2)2}> :  tensor<32x32xf32>
 return %0 : tensor<32x32xf32>
}

// CHECK-LABEL: func @sharded_to_unreduced_and_replicated_to_unreduced
func.func @sharded_to_unreduced_and_replicated_to_unreduced(
    %arg0 : tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>})
    -> (tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x", "y", "z"}>}) {
  // CHECK-NEXT: %0 = sdy.sharded_to_unreduced [{"x"}, {}] %arg0 out_sharding=<@mesh, [{}, {}], unreduced={"x", "y"}>
  // CHECK-NEXT: %1 = sdy.reshard %0 <@mesh, [{}, {}], unreduced={"x", "y", "z"}>
  // CHECK-NEXT: return %1
 %0 = sdy.reshard %arg0 <@mesh, [{}, {}], unreduced={"x", "y", "z"}> :  tensor<32x32xf32>
 return %0 : tensor<32x32xf32>
}
