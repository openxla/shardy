// RUN: sdy_opt %s -sdy-insert-explicit-reshards | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @negate
func.func @negate(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x32xf32>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @negate_input_sharding_is_larger
func.func @negate_input_sharding_is_larger(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh, [{"y"}, {}]> : tensor<4x32xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @negate_output_sharding_is_larger
func.func @negate_output_sharding_is_larger(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {}]> : tensor<4x32xf32>
  // CHECK-NEXT: stablehlo.negate %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @negate_input_and_output_sharded_on_separate_dims
func.func @negate_input_and_output_sharded_on_separate_dims(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  // CHECK: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[NEGATE]] <@mesh, [{}, {"y"}]> : tensor<4x32xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @negate_input_and_output_sharded_on_separate_dims_output_sharding_is_larger
func.func @negate_input_and_output_sharded_on_separate_dims_output_sharding_is_larger(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"x"}]> : tensor<4x32xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : tensor<4x32xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @add
func.func @add(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x32xf32>) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}]> : tensor<4x32xf32>
  // CHECK-NEXT: stablehlo.add %arg0, %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @add_input_sharding_is_larger
func.func @add_input_sharding_is_larger(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x32xf32>) {
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ADD]] <@mesh, [{"y"}, {}]> : tensor<4x32xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @add_output_sharding_is_larger
func.func @add_output_sharding_is_larger(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg1: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ADD]] <@mesh, [{"x"}, {}]> : tensor<4x32xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @add_input_and_output_sharded_on_separate_dims
func.func @add_input_and_output_sharded_on_separate_dims(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}, {"y"}]> : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {"y"}]> : tensor<4x32xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[RESHARD1]], %[[RESHARD2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[ADD]] <@mesh, [{}, {"y"}]> : tensor<4x32xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}

// CHECK-LABEL: func @add_inputs_are_sharded_the_same_way_output_is_unsharded
func.func @add_inputs_are_sharded_the_same_way_output_is_unsharded(%arg0: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> tensor<4x32xf32> {
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : tensor<4x32xf32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ADD]] <@mesh, [{}, {}]> : tensor<4x32xf32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<4x32xf32>
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : tensor<4x32xf32>
  return %0 : tensor<4x32xf32>
}
