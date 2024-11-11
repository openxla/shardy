// RUN: sdy_opt %s -sdy-close-shardings | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @func_input_sharding_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
func.func @func_input_sharding_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", ?}, {?}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @func_output_sharding_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
func.func @func_output_sharding_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"x",?}]>}) {
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @func_one_dimension_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
func.func @func_one_dimension_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x",?}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @func_empty_dimension_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
func.func @func_empty_dimension_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {?}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @func_multiple_results_one_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}, tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
func.func @func_multiple_results_one_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}, tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x",?}, {?}]>}) {
  return %arg0, %arg0 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @func_multiple_operands_one_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
func.func @func_multiple_operands_one_is_open(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x",?}, {?}]>}, %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @func_with_replicated_axes(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
func.func @func_with_replicated_axes(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}], replicated={"y"}>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @func_already_closed_and_no_replicated(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>})
func.func @func_already_closed_and_no_replicated(%arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  return %arg0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @op_result_is_open_and_with_replicated
func.func @op_result_is_open_and_with_replicated(%arg0: tensor<8x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{},{},{}]>}) -> (tensor<8x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {},{}]>}) {
  // CHECK: stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2}, {"y"}, {}]>]>} : tensor<8x16x2xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x":(1)2,?}, {"y",?},{?}], replicated={"x":(2)2}>]>} : tensor<8x16x2xf32>
  return %0 : tensor<8x16x2xf32>
}

// CHECK-LABEL: func @op_result_is_open_and_no_replicated
func.func @op_result_is_open_and_no_replicated(%arg0: tensor<8x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{},{},{}]>}) -> (tensor<8x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {},{}]>}) {
  // CHECK: stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : tensor<8x16x2xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x",?}, {"y",?},{?}]>]>} : tensor<8x16x2xf32>
  return %0 : tensor<8x16x2xf32>
}

// CHECK-LABEL: func @op_result_is_closed_and_with_replicated
func.func @op_result_is_closed_and_with_replicated(%arg0: tensor<8x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{},{},{}]>}) -> (tensor<8x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {},{}]>}) {
  // CHECK: stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}, {}]>]>} : tensor<8x16x2xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {},{}], replicated={"y"}>]>} : tensor<8x16x2xf32>
  return %0 : tensor<8x16x2xf32>
}

// CHECK-LABEL: func @op_result_is_open_and_fully_replicated
func.func @op_result_is_open_and_fully_replicated(%arg0: tensor<8x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{},{},{}]>}) -> (tensor<8x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {},{}]>}) {
  // CHECK: stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}, {}]>]>} : tensor<8x16x2xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?},{?}]>]>} : tensor<8x16x2xf32>
  return %0 : tensor<8x16x2xf32>
}

// CHECK-LABEL: func @op_result_is_open_and_with_multiple_axes
func.func @op_result_is_open_and_with_multiple_axes(%arg0: tensor<8x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{},{},{}]>}) -> (tensor<8x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {},{}]>}) {
  // CHECK: stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", "y"}, {}, {}]>]>} : tensor<8x16x2xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x","y",?}, {?},{?}]>]>} : tensor<8x16x2xf32>
  return %0 : tensor<8x16x2xf32>
}

// CHECK-LABEL: func @op_result_already_closed_no_replicated
func.func @op_result_already_closed_no_replicated(%arg0: tensor<8x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{},{},{}]>}) -> (tensor<8x16x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {},{}]>}) {
  // CHECK: stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}, {}]>]>} : tensor<8x16x2xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"},{}]>]>} : tensor<8x16x2xf32>
  return %0 : tensor<8x16x2xf32>
}

// CHECK-LABEL: func @all_open_no_replicated_multiple_ops
func.func @all_open_no_replicated_multiple_ops(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"y",?}]>}, %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y",?}, {"x",?}]>}) -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x",?}, {?}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x",?}, {?}]>]>, sdy.sharding_rule = #sdy.op_sharding_rule<([i, k],[k, j])->([i, j]) {i=8, j=16, k=32}>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x",?}, {?}]>]>} : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}
