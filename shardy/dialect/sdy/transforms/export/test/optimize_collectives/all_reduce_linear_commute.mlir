// RUN: sdy_opt %s -sdy-optimize-collectives | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @add_of_two_all_reduces
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
// CHECK-SAME:    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}
// CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : tensor<8x16xf32>
// CHECK-NEXT:  %[[AR:.*]] = sdy.all_reduce {"x"} %[[ADD]] out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  return %[[AR]] : tensor<8x16xf32>
func.func @add_of_two_all_reduces(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %1 = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @subtract_of_two_all_reduces
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
// CHECK-SAME:    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}
// CHECK-NEXT:  %[[SUB:.*]] = stablehlo.subtract %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : tensor<8x16xf32>
// CHECK-NEXT:  %[[AR:.*]] = sdy.all_reduce {"y"} %[[SUB]] out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
// CHECK-NEXT:  return %[[AR]] : tensor<8x16xf32>
func.func @subtract_of_two_all_reduces(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> tensor<8x16xf32> {
  %0 = sdy.all_reduce {"y"} %arg0 out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  %1 = sdy.all_reduce {"y"} %arg1 out_sharding=<@mesh, [{}, {"x"}]> : tensor<8x16xf32>
  %2 = stablehlo.subtract %0, %1 : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @max_of_two_max_all_reduces
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=max{"x"}>},
// CHECK-SAME:    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=max{"x"}>}
// CHECK-NEXT:  %[[MAX:.*]] = stablehlo.maximum %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}], unreduced=max{"x"}>]>} : tensor<8x16xf32>
// CHECK-NEXT:  %[[AR:.*]] = sdy.all_reduce max {"x"} %[[MAX]] out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  return %[[AR]] : tensor<8x16xf32>
func.func @max_of_two_max_all_reduces(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=max{"x"}>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=max{"x"}>}) -> tensor<8x16xf32> {
  %0 = sdy.all_reduce max {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %1 = sdy.all_reduce max {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %2 = stablehlo.maximum %0, %1 : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @min_of_two_min_all_reduces
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=min{"x"}>},
// CHECK-SAME:    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=min{"x"}>}
// CHECK-NEXT:  %[[MIN:.*]] = stablehlo.minimum %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}], unreduced=min{"x"}>]>} : tensor<8x16xf32>
// CHECK-NEXT:  %[[AR:.*]] = sdy.all_reduce min {"x"} %[[MIN]] out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  return %[[AR]] : tensor<8x16xf32>
func.func @min_of_two_min_all_reduces(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=min{"x"}>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced=min{"x"}>}) -> tensor<8x16xf32> {
  %0 = sdy.all_reduce min {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %1 = sdy.all_reduce min {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %2 = stablehlo.minimum %0, %1 : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @chained_add_three_all_reduces
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
// CHECK-SAME:    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
// CHECK-SAME:    %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}
// CHECK-NEXT:  %[[ADD1:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : tensor<8x16xf32>
// CHECK-NEXT:  %[[ADD2:.*]] = stablehlo.add %[[ADD1]], %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : tensor<8x16xf32>
// CHECK-NEXT:  %[[AR:.*]] = sdy.all_reduce {"x"} %[[ADD2]] out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  return %[[AR]] : tensor<8x16xf32>
func.func @chained_add_three_all_reduces(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
    %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %1 = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %2 = sdy.all_reduce {"x"} %arg2 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %3 = stablehlo.add %0, %1 : tensor<8x16xf32>
  %4 = stablehlo.add %3, %2 : tensor<8x16xf32>
  return %4 : tensor<8x16xf32>
}

// CHECK-LABEL: func @different_reduction_axes_not_folded
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
// CHECK-SAME:    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}
// CHECK-NEXT:  %[[AR0:.*]] = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  %[[AR1:.*]] = sdy.all_reduce {"y"} %arg1 out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[AR0]], %[[AR1]] : tensor<8x16xf32>
// CHECK-NEXT:  return %[[ADD]] : tensor<8x16xf32>
func.func @different_reduction_axes_not_folded(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> tensor<8x16xf32> {
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  %1 = sdy.all_reduce {"y"} %arg1 out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @add_of_two_all_reduces_with_unreduced_axes
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>},
// CHECK-SAME:    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>}
// CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}], unreduced={"x"}>]>} : tensor<8x16xf32>
// CHECK-NEXT:  %[[AR:.*]] = sdy.all_reduce {"x"} %[[ADD]] out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  return %[[AR]] : tensor<8x16xf32>
func.func @add_of_two_all_reduces_with_unreduced_axes(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>}) -> tensor<8x16xf32> {
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %1 = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @different_out_shardings_not_folded
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
// CHECK-SAME:    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}
// CHECK-NEXT:  %[[AR0:.*]] = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  %[[AR1:.*]] = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{}, {"y"}]> : tensor<8x16xf32>
// CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[AR0]], %[[AR1]] : tensor<8x16xf32>
// CHECK-NEXT:  return %[[ADD]] : tensor<8x16xf32>
func.func @different_out_shardings_not_folded(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x16xf32> {
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %1 = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{}, {"y"}]> : tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @different_operand_shardings_not_folded
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>},
// CHECK-SAME:    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}
// CHECK-NEXT:  %[[AR0:.*]] = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  %[[AR1:.*]] = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[AR0]], %[[AR1]] : tensor<8x16xf32>
// CHECK-NEXT:  return %[[ADD]] : tensor<8x16xf32>
func.func @different_operand_shardings_not_folded(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}], unreduced={"x"}>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %1 = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @mismatched_binary_op_sharding_not_folded
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
// CHECK-SAME:    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}
// CHECK-NEXT:  %[[AR0:.*]] = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  %[[AR1:.*]] = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[AR0]], %[[AR1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<8x16xf32>
// CHECK-NEXT:  return %[[ADD]] : tensor<8x16xf32>
func.func @mismatched_binary_op_sharding_not_folded(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %1 = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @non_sum_reduction_op_not_folded
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
// CHECK-SAME:    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}
// CHECK-NEXT:  %[[AR0:.*]] = sdy.all_reduce max {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  %[[AR1:.*]] = sdy.all_reduce max {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[AR0]], %[[AR1]] : tensor<8x16xf32>
// CHECK-NEXT:  return %[[ADD]] : tensor<8x16xf32>
func.func @non_sum_reduction_op_not_folded(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xf32> {
  %0 = sdy.all_reduce max {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %1 = sdy.all_reduce max {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %2 : tensor<8x16xf32>
}

// CHECK-LABEL: func @boolean_add_not_folded
// CHECK-SAME:    %arg0: tensor<8x16xi1> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
// CHECK-SAME:    %arg1: tensor<8x16xi1> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}
// CHECK-NEXT:  %[[AR0:.*]] = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xi1>
// CHECK-NEXT:  %[[AR1:.*]] = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xi1>
// CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[AR0]], %[[AR1]] : tensor<8x16xi1>
// CHECK-NEXT:  return %[[ADD]] : tensor<8x16xi1>
func.func @boolean_add_not_folded(
    %arg0: tensor<8x16xi1> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
    %arg1: tensor<8x16xi1> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> tensor<8x16xi1> {
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xi1>
  %1 = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xi1>
  %2 = stablehlo.add %0, %1 : tensor<8x16xi1>
  return %2 : tensor<8x16xi1>
}

// CHECK-LABEL: func @multiple_uses_not_folded
// CHECK-SAME:    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
// CHECK-SAME:    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}
// CHECK-NEXT:  %[[AR0:.*]] = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  %[[AR1:.*]] = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
// CHECK-NEXT:  %[[ADD:.*]] = stablehlo.add %[[AR0]], %[[AR1]] : tensor<8x16xf32>
// CHECK-NEXT:  return %[[ADD]], %[[AR0]] : tensor<8x16xf32>, tensor<8x16xf32>
func.func @multiple_uses_not_folded(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32>, tensor<8x16xf32>) {
  %0 = sdy.all_reduce {"x"} %arg0 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %1 = sdy.all_reduce {"x"} %arg1 out_sharding=<@mesh, [{"y"}, {}]> : tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x16xf32>
  return %2, %0 : tensor<8x16xf32>, tensor<8x16xf32>
}
