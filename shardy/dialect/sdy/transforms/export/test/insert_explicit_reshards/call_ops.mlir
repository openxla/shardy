// RUN: sdy_opt %s -split-input-file -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @call
func.func @call(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x":(1)2}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"x"}]>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[RESHARD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>}
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>}
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ABS]] <@mesh, [{"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// -----
sdy.mesh @mesh = <["x"=4, "y"=2]>

// CHECK-LABEL: func @call_empty_block
func.func @call_empty_block(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>}
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>}
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %1 : tensor<210xf32>
}

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}]>
  // CHECK-NEXT: return %[[RESHARD]]
  return %arg0 : tensor<210xf32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @call_with_shardings
func.func @call_with_shardings(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> tensor<12x2xi32> {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @foo(%0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>, <@mesh, [{}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %1#0 <@mesh, [{}, {"a"}]>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %1#1 <@mesh, [{}, {"a"}]>
  // CHECK-NEXT: %[[CONCAT:.*]] = stablehlo.concatenate %[[RESHARD1]], %[[RESHARD2]], dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>}
  // CHECK-NEXT: %[[RESHARD3:.*]] = sdy.reshard %[[CONCAT]] <@mesh, [{}, {}]>
  %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>, <@mesh, [{}, {}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %1 = stablehlo.concatenate %0#0, %0#1, dim = 0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> tensor<12x2xi32>
  return %1 : tensor<12x2xi32>
}

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, %arg1: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
  -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ABS]] <@mesh, [{}, {"a"}]>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[RESHARD0]] <@mesh, [{"a"}, {}]>
  // CHECK-NEXT: return %[[RESHARD1]], %arg1
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>} : tensor<8x2xi32>
  return %0, %arg1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2, "z"=4]>

// CHECK-LABEL: func @one_argument_to_multiple_calls(
func.func @one_argument_to_multiple_calls(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}]>}) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh, [{"z"}]>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}]>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%[[RESHARD1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>}
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[RESHARD0]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}]>]>}
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[CALL0]] <@mesh, [{"z"}]>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[RESHARD]], %[[CALL1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}]>]>}
  // CHECK-NEXT: return %[[ADD]]
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = call @bar(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}]>]>} : (tensor<210xf32>) -> (tensor<210xf32>)
  %3 = stablehlo.add %0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}]>]>} : tensor<210xf32>
  return %3 : tensor<210xf32>
}

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: return %[[ABS]]
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// CHECK-LABEL: func private @bar
func.func private @bar(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"z"}]>}) {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: return %[[ABS]]
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"z"}]>]>} : tensor<210xf32>
  return %0 : tensor<210xf32>
}

// -----
sdy.mesh @mesh = <["x"=2, "y"=2, "z"=4]>


// CHECK-LABEL: func @different_arguments_to_multiple_calls_with_same_input_output_shardings
func.func @different_arguments_to_multiple_calls_with_same_input_output_shardings(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh, [{"y"}]>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%[[RESHARD0]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>}
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[NEGATE]] <@mesh, [{"y"}]>
  // CHECK-NEXT: %[[CALL1:.*]] = call @foo(%[[RESHARD1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CALL0]], %[[CALL1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>}
  // CHECK-NEXT: return %[[ADD]]
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<210xf32>) -> (tensor<210xf32>)
  %1 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  %2 = call @foo(%1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : (tensor<210xf32>) -> (tensor<210xf32>)
  %4 = stablehlo.add %0, %2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %4 : tensor<210xf32>
}

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> (tensor<210xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0
  // CHECK-NEXT: return %[[ABS]]
  %3 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<210xf32>
  return %3 : tensor<210xf32>
}
