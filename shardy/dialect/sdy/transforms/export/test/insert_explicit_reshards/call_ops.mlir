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

// -----

sdy.mesh @meshA = <["x"=2, "y"=4]>
sdy.mesh @meshB = <["a"=4, "b"=2]>
sdy.mesh @meshC = <["p"=2, "q"=4]>

// CHECK-LABEL: func @call_cross_mesh_replicated_result_and_arg
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{}]>})
func.func @call_cross_mesh_replicated_result_and_arg(
    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{}]>})
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshC, [{}]>}) {
  // CHECK-NEXT: %[[IN_R:.*]] = sdy.reshard %[[ARG0]] <@meshB, [{}]> : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @callee_repl_b(%[[IN_R]]) {sdy.sharding = #sdy.sharding_per_value<[<@meshB, [{}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[OUT_R:.*]] = sdy.reshard %[[CALL]] <@meshC, [{}]> : tensor<8xf32>
  // CHECK-NEXT: return %[[OUT_R]] : tensor<8xf32>
  %0 = call @callee_repl_b(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@meshC, [{}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @callee_repl_b
func.func private @callee_repl_b(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{}]>}) {
  return %arg0 : tensor<8xf32>
}

// CHECK-LABEL: func @call_cross_mesh_tiled_result
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>})
func.func @call_cross_mesh_tiled_result(
    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>})
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>}) {
  // CHECK-NEXT: %[[CALL:.*]] = call @callee_tiled_a(%[[ARG0]]) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[CALL]] <@meshA, [{}]> : tensor<8xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <@meshB, [{}]> : tensor<8xf32>
  // CHECK-NEXT: %[[R2:.*]] = sdy.reshard %[[R1]] <@meshB, [{"a"}]> : tensor<8xf32>
  // CHECK-NEXT: return %[[R2]] : tensor<8xf32>
  %0 = call @callee_tiled_a(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@meshB, [{"a"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @callee_tiled_a
func.func private @callee_tiled_a(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>}) {
  return %arg0 : tensor<8xf32>
}

// CHECK-LABEL: func @call_same_mesh_mismatched_result
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>})
func.func @call_same_mesh_mismatched_result(
    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>})
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"y"}]>}) {
  // CHECK-NEXT: %[[CALL:.*]] = call @callee_tiled_a(%[[ARG0]]) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[R:.*]] = sdy.reshard %[[CALL]] <@meshA, [{"y"}]> : tensor<8xf32>
  // CHECK-NEXT: return %[[R]] : tensor<8xf32>
  %0 = call @callee_tiled_a(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, [{"y"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func @call_unannotated_callee_result_caller_tiled
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xf32>)
func.func @call_unannotated_callee_result_caller_tiled(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>}) {
  // CHECK-NEXT: %[[CALL:.*]] = call @callee_unannotated(%[[ARG0]]) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, [{}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[R:.*]] = sdy.reshard %[[CALL]] <@meshA, [{"x"}]> : tensor<8xf32>
  // CHECK-NEXT: return %[[R]] : tensor<8xf32>
  %0 = call @callee_unannotated(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @callee_unannotated
func.func private @callee_unannotated(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  return %arg0 : tensor<8xf32>
}

// CHECK-LABEL: func @call_multi_result_callee_unannotated_slot_cross_mesh
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8xf32>)
func.func @call_multi_result_callee_unannotated_slot_cross_mesh(
    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>},
    %arg1: tensor<8xf32>)
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>}, tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{}]>}) {
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @callee_multi_res(%[[ARG0]], %[[ARG1]]) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, [{"x"}]>, <@meshA, [{}]>]>} : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[R:.*]] = sdy.reshard %[[CALL]]#1 <@meshB, [{}]> : tensor<8xf32>
  // CHECK-NEXT: return %[[CALL]]#0, %[[R]] : tensor<8xf32>, tensor<8xf32>
  %0:2 = call @callee_multi_res(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, [{"x"}]>, <@meshB, [{}]>]>} : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: func private @callee_multi_res
func.func private @callee_multi_res(
    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>},
    %arg1: tensor<8xf32>)
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>}, tensor<8xf32>) {
  return %arg0, %arg1 : tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: func @call_unannotated_multi_result_mixed_callee_meshes
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
func.func @call_unannotated_multi_result_mixed_callee_meshes(
    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>}) {
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @callee_mixed_results(%[[ARG0]], %[[ARG1]]) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, [{"x"}]>, <@meshB, [{"a"}]>]>} : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: return %[[CALL]]#0, %[[CALL]]#1 : tensor<8xf32>, tensor<8xf32>
  %0:2 = call @callee_mixed_results(%arg0, %arg1) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: func private @callee_mixed_results
func.func private @callee_mixed_results(
    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{"x"}]>},
        tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>}) {
  return %arg0, %arg1 : tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: func @call_unannotated_multi_result_first_callee_unannotated
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xf32>,
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
func.func @call_unannotated_multi_result_first_callee_unannotated(
    %arg0: tensor<8xf32>,
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
    -> (tensor<8xf32>, tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>}) {
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @callee_first_unannotated_second_b(%[[ARG0]], %[[ARG1]]) {sdy.sharding = #sdy.sharding_per_value<[<@meshB, [{}]>, <@meshB, [{"a"}]>]>} : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: return %[[CALL]]#0, %[[CALL]]#1 : tensor<8xf32>, tensor<8xf32>
  %0:2 = call @callee_first_unannotated_second_b(%arg0, %arg1) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: func private @callee_first_unannotated_second_b
func.func private @callee_first_unannotated_second_b(
    %arg0: tensor<8xf32>,
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
    -> (tensor<8xf32>, tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>}) {
  return %arg0, %arg1 : tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: func @call_unreduced_result_cross_mesh
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{}], unreduced={"x"}>})
func.func @call_unreduced_result_cross_mesh(
    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{}], unreduced={"x"}>})
    -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{}], unreduced={"a"}>}) {
  // CHECK-NEXT: %[[CALL:.*]] = call @callee_unreduced_a(%[[ARG0]]) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, [{}], unreduced={"x"}>]>} : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[AR:.*]] = sdy.all_reduce {"x"} %[[CALL]] out_sharding=<@meshA, [{}]> : tensor<8xf32>
  // CHECK-NEXT: %[[R:.*]] = sdy.reshard %[[AR]] <@meshB, [{}]> : tensor<8xf32>
  // CHECK-NEXT: %[[R2U:.*]] = sdy.replicated_to_unreduced {"a"} %[[R]] out_sharding=<@meshB, [{}], unreduced={"a"}> : tensor<8xf32>
  // CHECK-NEXT: return %[[R2U]] : tensor<8xf32>
  %0 = call @callee_unreduced_a(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@meshB, [{}], unreduced={"a"}>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @callee_unreduced_a
func.func private @callee_unreduced_a(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{}], unreduced={"x"}>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{}], unreduced={"x"}>}) {
  return %arg0 : tensor<8xf32>
}

// CHECK-LABEL: func @call_cross_mesh_arg_to_unannotated_callee_slot
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
func.func @call_cross_mesh_arg_to_unannotated_callee_slot(
    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{}]>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
    -> (tensor<8xf32>, tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>}) {
  // CHECK-NEXT: %[[IN_R:.*]] = sdy.reshard %[[ARG0]] <@meshB, [{}]> : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @callee_first_unannotated_second_b(%[[IN_R]], %[[ARG1]]) {sdy.sharding = #sdy.sharding_per_value<[<@meshB, [{}]>, <@meshB, [{"a"}]>]>} : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: return %[[CALL]]#0, %[[CALL]]#1 : tensor<8xf32>, tensor<8xf32>
  %0:2 = call @callee_first_unannotated_second_b(%arg0, %arg1) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %0#0, %0#1 : tensor<8xf32>, tensor<8xf32>
}

// -----

sdy.mesh @meshA = <["x"=2, "y"=4]>
sdy.mesh @meshB = <["a"=4, "b"=2]>

// CHECK-LABEL: func @call_cross_mesh_token_result_no_token_reshard
// CHECK-SAME: (%[[ARG0:.*]]: !stablehlo.token {sdy.sharding = #sdy.sharding<@meshA, []>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
func.func @call_cross_mesh_token_result_no_token_reshard(
    %arg0: !stablehlo.token {sdy.sharding = #sdy.sharding<@meshA, []>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
    -> (!stablehlo.token, tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>}) {
  // CHECK-NOT: sdy.reshard
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @callee_token_and_tensor_b(%[[ARG0]], %[[ARG1]]) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, []>, <@meshB, [{"a"}]>]>} : (!stablehlo.token, tensor<8xf32>) -> (!stablehlo.token, tensor<8xf32>)
  // CHECK-NEXT: return %[[CALL]]#0, %[[CALL]]#1 : !stablehlo.token, tensor<8xf32>
  %0:2 = call @callee_token_and_tensor_b(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, []>, <@meshB, [{"a"}]>]>} : (!stablehlo.token, tensor<8xf32>) -> (!stablehlo.token, tensor<8xf32>)
  return %0#0, %0#1 : !stablehlo.token, tensor<8xf32>
}

// CHECK-LABEL: func @call_cross_mesh_token_and_tensor_result
// CHECK-SAME: (%[[ARG0:.*]]: !stablehlo.token {sdy.sharding = #sdy.sharding<@meshA, []>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
func.func @call_cross_mesh_token_and_tensor_result(
    %arg0: !stablehlo.token {sdy.sharding = #sdy.sharding<@meshA, []>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
    -> (!stablehlo.token, tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshA, [{}]>}) {
  // CHECK-NOT: sdy.reshard
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @callee_token_and_tensor_b(%[[ARG0]], %[[ARG1]]) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, []>, <@meshB, [{"a"}]>]>} : (!stablehlo.token, tensor<8xf32>) -> (!stablehlo.token, tensor<8xf32>)
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[CALL]]#1 <@meshB, [{}]> : tensor<8xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <@meshA, [{}]> : tensor<8xf32>
  // CHECK-NEXT: return %[[CALL]]#0, %[[R1]] : !stablehlo.token, tensor<8xf32>
  %0:2 = call @callee_token_and_tensor_b(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@meshA, []>, <@meshA, [{}]>]>} : (!stablehlo.token, tensor<8xf32>) -> (!stablehlo.token, tensor<8xf32>)
  return %0#0, %0#1 : !stablehlo.token, tensor<8xf32>
}

// CHECK-LABEL: func private @callee_token_and_tensor_b
func.func private @callee_token_and_tensor_b(
    %arg0: !stablehlo.token {sdy.sharding = #sdy.sharding<@meshB, []>},
    %arg1: tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>})
    -> (!stablehlo.token {sdy.sharding = #sdy.sharding<@meshB, []>}, tensor<8xf32> {sdy.sharding = #sdy.sharding<@meshB, [{"a"}]>}) {
  return %arg0, %arg1 : !stablehlo.token, tensor<8xf32>
}
