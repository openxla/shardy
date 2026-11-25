// RUN: sdy_opt %s -sdy-insert-explicit-reshards='enable-full-version=true' | FileCheck %s

sdy.mesh @empty_mesh = <[]>
sdy.mesh @mesh = <["x"=2]>
sdy.mesh @mesh2 = <["y"=2]>
sdy.mesh @single_dev_0 = <[], device_ids=[0]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>

//===----------------------------------------------------------------------===//
// Single-Device Op Results (Single-Device -> Other Mesh)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @single_device_result_to_tiled_consumer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
func.func @single_device_result_to_tiled_consumer(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[IN_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[IN_SINGLE:.*]] = sdy.reshard %[[IN_REPL]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @SomeCustomCall(%[[IN_SINGLE]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>]>}
  // CHECK-NEXT: %[[OUT_REPL:.*]] = sdy.reshard %[[CC]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[OUT_TILED:.*]] = sdy.reshard %[[OUT_REPL]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  %0 = stablehlo.custom_call @SomeCustomCall(%arg0) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>

  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[OUT_TILED]], %[[ARG1]]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>}
  %1 = stablehlo.add %0, %arg1 {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{"x"}, {}]>]>
  } : tensor<8x16xf32>

  // CHECK-NEXT: return %[[ADD]] : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @single_device_result_to_replicated_consumer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
func.func @single_device_result_to_replicated_consumer(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK-NEXT: %[[IN_SINGLE:.*]] = sdy.reshard %[[ARG0]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @SomeCustomCall(%[[IN_SINGLE]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>]>}
  // CHECK-NEXT: %[[OUT_RESHARD:.*]] = sdy.reshard %[[CC]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  %0 = stablehlo.custom_call @SomeCustomCall(%arg0) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>

  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[OUT_RESHARD]]
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  %1 = stablehlo.abs %0 {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh, [{}, {}]>]>
  } : tensor<8x16xf32>

  // CHECK-NEXT: return %[[ABS]] : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @single_device_op0_to_single_device_op1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
func.func @single_device_op0_to_single_device_op1(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK-NEXT: %[[IN_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[IN_SINGLE0:.*]] = sdy.reshard %[[IN_REPL]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[CC0:.*]] = stablehlo.custom_call @CustomOp0(%[[IN_SINGLE0]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>]>}
  %0 = stablehlo.custom_call @CustomOp0(%arg0) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>

  // CHECK-NEXT: %[[MID_REPL:.*]] = sdy.reshard %[[CC0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[IN_SINGLE1:.*]] = sdy.reshard %[[MID_REPL]] <@single_dev_1, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[CC1:.*]] = stablehlo.custom_call @CustomOp1(%[[IN_SINGLE1]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_1, []>]>}
  %1 = stablehlo.custom_call @CustomOp1(%0) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_1, []>]>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>

  // CHECK-NEXT: %[[OUT_REPL:.*]] = sdy.reshard %[[CC1]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[OUT_REPL]] : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @single_device_result_directly_returned
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
func.func @single_device_result_directly_returned(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK-NEXT: %[[IN_SINGLE:.*]] = sdy.reshard %[[ARG0]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @SomeCustomCall(%[[IN_SINGLE]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>]>}
  // CHECK-NEXT: %[[OUT_RESHARD:.*]] = sdy.reshard %[[CC]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  %0 = stablehlo.custom_call @SomeCustomCall(%arg0) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>

  // CHECK-NEXT: return %[[OUT_RESHARD]] : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

//===----------------------------------------------------------------------===//
// Single-Device Op Operands (other Mesh -> Single-Device)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @tiled_operand_to_single_device_consumer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
func.func @tiled_operand_to_single_device_consumer(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD_SINGLE:.*]] = sdy.reshard %[[RESHARD_REPL]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @SomeCustomCall(%[[RESHARD_SINGLE]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>]>}
  // CHECK-NEXT: %[[OUT_REPL:.*]] = sdy.reshard %[[CC]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  %0 = stablehlo.custom_call @SomeCustomCall(%arg0) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>

  // CHECK-NEXT: return %[[OUT_REPL]] : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @replicated_operand_to_single_device_consumer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
func.func @replicated_operand_to_single_device_consumer(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD_SINGLE:.*]] = sdy.reshard %[[ARG0]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @SomeCustomCall(%[[RESHARD_SINGLE]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>]>}
  // CHECK-NEXT: %[[OUT_REPL:.*]] = sdy.reshard %[[CC]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  %0 = stablehlo.custom_call @SomeCustomCall(%arg0) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>

  // CHECK-NEXT: return %[[OUT_REPL]] : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @replicated_operand_on_other_mesh_to_single_device_consumer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
func.func @replicated_operand_on_other_mesh_to_single_device_consumer(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD_GLOBAL_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD_SINGLE:.*]] = sdy.reshard %[[RESHARD_GLOBAL_REPL]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @SomeCustomCall(%[[RESHARD_SINGLE]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>]>}
  // CHECK-NEXT: %[[OUT_REPL:.*]] = sdy.reshard %[[CC]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[OUT_REPL]] : tensor<8x16xf32>
  %0 = stablehlo.custom_call @SomeCustomCall(%arg0) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>

  return %0 : tensor<8x16xf32>
}

//===----------------------------------------------------------------------===//
// Multi-Result Single-Device Ops
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @multi_result_matching_single_device
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
// CHECK-SAME:     tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
func.func @multi_result_matching_single_device(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD_SINGLE:.*]] = sdy.reshard %[[RESHARD_REPL]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[CC:.*]]:2 = stablehlo.custom_call @MultiResultCall(%[[RESHARD_SINGLE]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>, <@single_dev_0, []>]>}
  // CHECK-NEXT: %[[OUT_REPL0:.*]] = sdy.reshard %[[CC]]#0 <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[OUT_REPL1:.*]] = sdy.reshard %[[CC]]#1 <@mesh, [{}, {}]> : tensor<8x16xf32>
  %0:2 = stablehlo.custom_call @MultiResultCall(%arg0) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>, #sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)

  // CHECK-NEXT: return %[[OUT_REPL0]], %[[OUT_REPL1]] : tensor<8x16xf32>, tensor<8x16xf32>
  return %0#0, %0#1 : tensor<8x16xf32>, tensor<8x16xf32>
}

// CHECK-LABEL: func @multi_result_mixed_single_device_and_tiled
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
// CHECK-SAME:     tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
// CHECK-SAME:     !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>}) {
func.func @multi_result_mixed_single_device_and_tiled(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
        !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>}) {
  // CHECK-NEXT: %[[RESHARD_IN_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD_IN_SINGLE:.*]] = sdy.reshard %[[RESHARD_IN_REPL]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[CC:.*]]:3 = stablehlo.custom_call @MixedCall(%[[RESHARD_IN_SINGLE]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>, <@single_dev_0, []>, <@mesh, []>]>}
  // CHECK-NEXT: %[[OUT_REPL0:.*]] = sdy.reshard %[[CC]]#0 <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD_REPL:.*]] = sdy.reshard %[[CC]]#1 <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD_TILED:.*]] = sdy.reshard %[[RESHARD_REPL]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  %0:3 = stablehlo.custom_call @MixedCall(%arg0) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>, #sdy.sharding<@mesh, [{"x"}, {}]>, #sdy.sharding<@mesh, []>]>
  } : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>, !stablehlo.token)

  // CHECK-NEXT: return %[[OUT_REPL0]], %[[RESHARD_TILED]], %[[CC]]#2 : tensor<8x16xf32>, tensor<8x16xf32>, !stablehlo.token
  return %0#0, %0#1, %0#2 : tensor<8x16xf32>, tensor<8x16xf32>, !stablehlo.token
}

// CHECK-LABEL: func @single_device_op_with_token_operand
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
// CHECK-SAME:  %[[TOKEN:.*]]: !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
func.func @single_device_op_with_token_operand(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %token: !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK-NEXT: %[[RESHARD_IN_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD_IN_SINGLE:.*]] = sdy.reshard %[[RESHARD_IN_REPL]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[CC:.*]] = stablehlo.custom_call @CustomCallWithToken(%[[RESHARD_IN_SINGLE]], %[[TOKEN]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>]>}
  // CHECK-NEXT: %[[OUT_REPL:.*]] = sdy.reshard %[[CC]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[OUT_REPL]] : tensor<8x16xf32>
  %0 = stablehlo.custom_call @CustomCallWithToken(%arg0, %token) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>, !stablehlo.token) -> tensor<8x16xf32>

  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @single_device_send_and_recv
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
// CHECK-SAME:  %[[TOKEN:.*]]: !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
// CHECK-SAME:     !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>}) {
func.func @single_device_send_and_recv(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %token: !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>}) {
  // CHECK-NEXT: %[[SEND_IN_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[SEND_IN_SINGLE:.*]] = sdy.reshard %[[SEND_IN_REPL]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[SEND:.*]] = "stablehlo.send"(%[[SEND_IN_SINGLE]], %[[TOKEN]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>]>}
  %0 = "stablehlo.send"(%arg0, %token) <{
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 2>,
    is_host_transfer = true
  }> {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>, !stablehlo.token) -> !stablehlo.token

  // CHECK-NEXT: %[[RECV:.*]]:2 = "stablehlo.recv"(%[[SEND]])
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>, <@single_dev_0, []>]>}
  // CHECK-NEXT: %[[RECV_OUT_REPL:.*]] = sdy.reshard %[[RECV]]#0 <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[RECV_OUT_REPL]], %[[RECV]]#1 : tensor<8x16xf32>, !stablehlo.token
  %1:2 = "stablehlo.recv"(%0) <{
    channel_handle = #stablehlo.channel_handle<handle = 2, type = 3>,
    is_host_transfer = true
  }> {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>, #sdy.sharding<@single_dev_0, []>]>
  } : (!stablehlo.token) -> (tensor<8x16xf32>, !stablehlo.token)

  return %1#0, %1#1 : tensor<8x16xf32>, !stablehlo.token
}

// CHECK-LABEL: func @single_device_non_global_mesh2_operand_and_consumer
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"y"}, {}]>})
func.func @single_device_non_global_mesh2_operand_and_consumer(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"y"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"y"}, {}]>}) {
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[ARG0]] <@mesh2, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R2:.*]] = sdy.reshard %[[R1]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[NEG:.*]] = stablehlo.negate %[[R2]] {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>]>} : tensor<8x16xf32>
  // CHECK-NEXT: %[[R3:.*]] = sdy.reshard %[[NEG]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R4:.*]] = sdy.reshard %[[R3]] <@mesh2, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R5:.*]] = sdy.reshard %[[R4]] <@mesh2, [{"y"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[R5]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"y"}, {}]>]>} : tensor<8x16xf32>
  // CHECK-NEXT: return %[[ABS]] : tensor<8x16xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>} : tensor<8x16xf32>
  %1 = stablehlo.abs %0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh2, [{"y"}, {}]>]>} : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @single_device_cross_mesh_tiled_operand_and_result
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"y"}, {}]>})
func.func @single_device_cross_mesh_tiled_operand_and_result(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"y"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[ARG0]] <@mesh2, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R1:.*]] = sdy.reshard %[[R0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R2:.*]] = sdy.reshard %[[R1]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[NEG:.*]] = stablehlo.negate %[[R2]] {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>]>} : tensor<8x16xf32>
  // CHECK-NEXT: %[[R3:.*]] = sdy.reshard %[[NEG]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R4:.*]] = sdy.reshard %[[R3]] <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[R4]] : tensor<8x16xf32>
  %0 = stablehlo.negate %arg0 {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>} : tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK-LABEL: func @broadcast_single_device_operand_custom_op
func.func @broadcast_single_device_operand_custom_op(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"y"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{"y"}, {}]>}) {
  // CHECK: stablehlo.while
  // CHECK: do {
  // CHECK-NEXT: %[[REPL:.*]] = sdy.reshard %{{.*}} <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[PIVOT:.*]] = sdy.reshard %[[REPL]] <@mesh2, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[CUSTOM:.*]] = stablehlo.custom_call @Custom(%[[PIVOT]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh2, [{"y"}, {}]>]>} : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT: stablehlo.return %{{.*}}, %[[CUSTOM]] : tensor<8x16xf32>, tensor<8x16xf32>
  %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_1 = %arg1) : tensor<8x16xf32>, tensor<8x16xf32>
      attributes {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>, #sdy.sharding<@mesh2, [{"y"}, {}]>]>}
  cond {
    %c = stablehlo.constant dense<true> : tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %1 = stablehlo.custom_call @Custom(%iterArg) {
      sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@mesh2, [{"y"}, {}]>]>
    } : (tensor<8x16xf32>) -> tensor<8x16xf32>
    stablehlo.return %iterArg, %1 : tensor<8x16xf32>, tensor<8x16xf32>
  }
  return %0#1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @broadcast_single_device_operand_no_common_mesh
func.func @broadcast_single_device_operand_no_common_mesh(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK: stablehlo.while
  // CHECK: do {
  // CHECK-NEXT: %[[REPL:.*]] = sdy.reshard %{{.*}} <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[NEG:.*]] = stablehlo.negate %[[REPL]] : tensor<8x16xf32>
  // CHECK-NEXT: stablehlo.return %{{.*}}, %[[NEG]] : tensor<8x16xf32>, tensor<8x16xf32>
  %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_1 = %arg1) : tensor<8x16xf32>, tensor<8x16xf32>
      attributes {sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>, #sdy.sharding<@mesh, [{}, {}]>]>}
  cond {
    %c = stablehlo.constant dense<true> : tensor<i1>
    stablehlo.return %c : tensor<i1>
  } do {
    %1 = stablehlo.negate %iterArg : tensor<8x16xf32>
    stablehlo.return %iterArg, %1 : tensor<8x16xf32>, tensor<8x16xf32>
  }
  return %0#1 : tensor<8x16xf32>
}

// CHECK-LABEL: func @single_device_and_cross_mesh_unreduced_result
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
func.func @single_device_and_cross_mesh_unreduced_result(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{}, {}], unreduced={"y"}>}) {
  // CHECK-NEXT: %[[IN_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[IN_SINGLE:.*]] = sdy.reshard %[[IN_REPL]] <@single_dev_0, []> : tensor<8x16xf32>
  // CHECK-NEXT: %[[MIXED:.*]]:2 = stablehlo.custom_call @MixedResults(%[[IN_SINGLE]]) {sdy.sharding = #sdy.sharding_per_value<[<@single_dev_0, []>, <@single_dev_0, []>]>} : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)
  // CHECK-NEXT: %[[R0:.*]] = sdy.reshard %[[MIXED]]#0 <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R1_GLOBAL:.*]] = sdy.reshard %[[MIXED]]#1 <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R1_MESH2:.*]] = sdy.reshard %[[R1_GLOBAL]] <@mesh2, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[R1_UNRED:.*]] = sdy.replicated_to_unreduced {"y"} %[[R1_MESH2]] out_sharding=<@mesh2, [{}, {}], unreduced={"y"}> : tensor<8x16xf32>
  // CHECK-NEXT: return %[[R0]], %[[R1_UNRED]] : tensor<8x16xf32>, tensor<8x16xf32>
  %0:2 = stablehlo.custom_call @MixedResults(%arg0) {
    sdy.sharding = #sdy.sharding_per_value<[
      #sdy.sharding<@single_dev_0, []>,
      #sdy.sharding<@mesh2, [{}, {}], unreduced={"y"}>
    ]>
  } : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<8x16xf32>)
  return %0#0, %0#1 : tensor<8x16xf32>, tensor<8x16xf32>
}

