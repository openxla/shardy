// RUN: sdy_opt %s -allow-unregistered-dialect  -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2, "z"=2]>
sdy.mesh @mesh_z4 = <["x"=2, "y"=2, "z"=4]>
sdy.mesh @other_mesh = <["x"=2, "y"=2]>
sdy.mesh @mesh_abcd = <["a"=2, "b"=2, "c"=2, "d"=2]>

//===----------------------------------------------------------------------===//
// Unreduced tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @all_reduce_source_has_unreduced_and_target_no_sharding
func.func @all_reduce_source_has_unreduced_and_target_no_sharding(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>})
      -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_source_has_unreduced_and_target_has_sharding_no_unreduced
func.func @all_reduce_source_has_unreduced_and_target_has_sharding_no_unreduced(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{"x"}, {}]>
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

// CHECK-LABEL: func @all_reduce_source_has_sharding_no_unreduced_and_target_has_unreduced
func.func @all_reduce_source_has_sharding_no_unreduced_and_target_has_unreduced(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], unreduced={"y"}>}) {
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

// CHECK-LABEL: func @all_reduce_source_unreduced_is_strict_subset_of_target_unreduced
func.func @all_reduce_source_unreduced_is_strict_subset_of_target_unreduced(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x", "z"}>}) {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {} %arg0 out_sharding=<@mesh, [{}, {}], unreduced={"x", "z"}>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
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

// CHECK-LABEL: func @all_reduce_source_unreduced_and_target_unreduced_have_no_overlap
func.func @all_reduce_source_unreduced_and_target_unreduced_have_no_overlap(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}) {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{}, {}], unreduced={"y"}>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_source_unreduced_and_target_unreduced_has_overlap
func.func @all_reduce_source_unreduced_and_target_unreduced_has_overlap(
  %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x", "y"}>})
     -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x", "z"}>}) {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{}, {}], unreduced={"x", "z"}>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  return %arg0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @all_reduce_on_func_input
func.func @all_reduce_on_func_input(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}, %arg1: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %[[ALL_REDUCE]], %arg1
  // CHECK-NEXT: return %[[MUL]]
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @unreduced_func_input_until_return
func.func @unreduced_func_input_until_return(%arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}, %arg1: tensor<4x8xf32>)
    -> (tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"y"}>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"y"}>]>} : tensor<4x8xf32>
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

// CHECK-LABEL: func @all_reduce_partially_delayed_until_return
func.func @all_reduce_partially_delayed_until_return(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x", "y"}, {}]>}) -> tensor<2x13xf32> {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[REDUCE]], %[[REDUCE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[ADD]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : tensor<2x13xf32>
  return %2 : tensor<2x13xf32>
}

// CHECK-LABEL: func @all_reduce_delayed_until_op
func.func @all_reduce_delayed_until_op(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {}]>})
    -> (tensor<2x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK:      %[[REDUCE:.*]] = stablehlo.reduce(%arg0 init: %cst) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK:      %[[ADD:.*]] = stablehlo.add %[[REDUCE]], %[[REDUCE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE_1:.*]] = sdy.all_reduce {"x"} %[[ADD]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: %[[ALL_REDUCE_2:.*]] = sdy.all_reduce {"x"} %[[ADD]] out_sharding=<@mesh, [{}, {}]>
  // CHECK-NEXT: %[[MUL:.*]] = stablehlo.multiply %[[ALL_REDUCE_1]], %[[ALL_REDUCE_2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>}
  // CHECK-NEXT: return %[[MUL]]
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  %2 = stablehlo.add %1, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : tensor<2x13xf32>
  %3 = stablehlo.multiply %2, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : tensor<2x13xf32>
  return %3 : tensor<2x13xf32>
}

// CHECK-LABEL: func @manual_computation
func.func @manual_computation(%arg0: tensor<208xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], unreduced={"x", "y"}>})
    -> (tensor<208xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  // CHECK-NEXT: %[[ALL_REDUCE_X:.*]] = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{}], unreduced={"y"}>
  // CHECK-NEXT: %[[MANUAL_COMP:.*]] = sdy.manual_computation(%[[ALL_REDUCE_X]])
  // CHECK-SAME:   in_shardings=[<@mesh, [{"x"}], unreduced={"y"}>]
  // CHECK-SAME:   out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"}
  // CHECK-SAME:   (%arg1: tensor<104xf32>) {
  // CHECK-NEXT:   %[[ABS:.*]] = stablehlo.abs %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}], unreduced={"y"}>]>}
  // CHECK-NEXT:   %[[ALL_REDUCE_Y:.*]] = sdy.all_reduce {"y"} %[[ABS]] out_sharding=<@mesh, [{}]>
  // CHECK-NEXT:   sdy.return %[[ALL_REDUCE_Y]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[MANUAL_COMP]]
  %0 = sdy.manual_computation(%arg0)
    in_shardings=[<@mesh, [{"x"}], unreduced={"y"}>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<104xf32>) {
    %1 = stablehlo.abs %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}], unreduced={"y"}>]>} : tensor<104xf32>
    sdy.return %1 : tensor<104xf32>
  } : (tensor<208xf32>) -> tensor<208xf32>
  return %0 : tensor<208xf32>
}

//===----------------------------------------------------------------------===//
// Dot tests
//===----------------------------------------------------------------------===//

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
    %arg0: tensor<4x32xf32> {sdy.sharding=#sdy.sharding<@mesh_z4, [{"z"}, {"x"}]>},
    %arg1: tensor<32x8xf32> {sdy.sharding=#sdy.sharding<@mesh_z4, [{"x"}, {"z":(2)2}]>}) -> tensor<4x8xf32> {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh_z4, [{"z"}, {}]>]>}
  // CHECK-NEXT: return %[[DOT_GENERAL]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh_z4, [{"z"}, {}]>]>} :
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

//===----------------------------------------------------------------------===//
// Concatenate tests
// More tests are in insert_explicit_reshards/concatenate.mlir
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @concatenate_different_shardings
func.func @concatenate_different_shardings(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x48x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {}]>}) -> tensor<4x80x256xf32> {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}, {}, {}]> : tensor<4x32x256xf32>
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %[[RESHARD1]], %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CONCATENATE]] <@mesh, [{}, {}, {}]> : tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<4x80x256xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %0 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concatenate_same_shardings
func.func @concatenate_same_shardings(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x48x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> tensor<4x80x256xf32> {
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[CONCATENATE]] : tensor<4x80x256xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %0 : tensor<4x80x256xf32>
}

// CHECK-LABEL: func @concatenate_same_shardings_func_result_different_sharding
func.func @concatenate_same_shardings_func_result_different_sharding(%arg0: tensor<4x32x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}, %arg1: tensor<4x48x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}, {}]>}) -> (tensor<4x80x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}, {}]>}) {
  // CHECK-NEXT: %[[CONCATENATE:.*]] = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  // CHECK-NEXT: return %[[CONCATENATE]] : tensor<4x80x256xf32>
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %0 : tensor<4x80x256xf32>
}
