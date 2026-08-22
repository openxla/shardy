// RUN: sdy_opt %s -sdy-resolve-single-device-sharding | FileCheck %s

sdy.mesh @mesh = <["x"=2]>
sdy.mesh @mesh2 = <["y"=2]>
sdy.mesh @single_dev_0 = <[], device_ids=[0]>
sdy.mesh @single_dev_1 = <[], device_ids=[1]>

// CHECK-LABEL: func @custom_call_single_device_0
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
// CHECK-SAME:     tensor<4x32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
// CHECK-SAME:     !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>}) {
func.func @custom_call_single_device_0(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        tensor<4x32xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>}) {
  // CHECK-NEXT: %[[IN_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[PART_ID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[PART_ID_I64:.*]] = stablehlo.convert %[[PART_ID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT: %[[IS_DEV0:.*]] = stablehlo.compare EQ, %[[PART_ID_I64]], %[[C0]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-NEXT: %[[IF_RES:.*]]:3 = "stablehlo.if"(%[[IS_DEV0]]) ({
  // CHECK-NEXT:   %[[EXEC:.*]]:3 = stablehlo.custom_call @SomeCustomCall(%[[IN_REPL]]) : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<4x32xi32>, !stablehlo.token)
  // CHECK-NEXT:   stablehlo.return %[[EXEC]]#0, %[[EXEC]]#1, %[[EXEC]]#2 : tensor<8x16xf32>, tensor<4x32xi32>, !stablehlo.token
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   %[[ZEROS0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<8x16xf32>
  // CHECK-NEXT:   %[[ZEROS1:.*]] = stablehlo.constant dense<0> : tensor<4x32xi32>
  // CHECK-NEXT:   %[[TOKEN:.*]] = stablehlo.create_token
  // CHECK-NEXT:   stablehlo.return %[[ZEROS0]], %[[ZEROS1]], %[[TOKEN]] : tensor<8x16xf32>, tensor<4x32xi32>, !stablehlo.token
  // CHECK-NEXT: }) : (tensor<i1>) -> (tensor<8x16xf32>, tensor<4x32xi32>, !stablehlo.token)
  // CHECK-NEXT: %[[ALL_REDUCE0:.*]] = sdy.all_reduce {"x"} %[[IF_RES]]#0 out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE1:.*]] = sdy.all_reduce {"x"} %[[IF_RES]]#1 out_sharding=<@mesh, [{}, {}]> : tensor<4x32xi32>
  %0 = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<8x16xf32>
  %1 = sdy.reshard %0 <@single_dev_0, []> : tensor<8x16xf32>
  %2:3 = stablehlo.custom_call @SomeCustomCall(%1) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>, #sdy.sharding<@single_dev_0, []>, #sdy.sharding<@mesh, []>]>
  } : (tensor<8x16xf32>) -> (tensor<8x16xf32>, tensor<4x32xi32>, !stablehlo.token)
  %3 = sdy.reshard %2#0 <@mesh, [{}, {}]> : tensor<8x16xf32>
  %4 = sdy.reshard %2#1 <@mesh, [{}, {}]> : tensor<4x32xi32>

  // CHECK-NEXT: return %[[ALL_REDUCE0]], %[[ALL_REDUCE1]], %[[IF_RES]]#2 : tensor<8x16xf32>, tensor<4x32xi32>, !stablehlo.token
  return %3, %4, %2#2 : tensor<8x16xf32>, tensor<4x32xi32>, !stablehlo.token
}

// CHECK-LABEL: func @elementwise_single_device_1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
// CHECK-SAME:  %[[ARG1:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
func.func @elementwise_single_device_1(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK-NEXT: %[[PART_ID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[PART_ID_I64:.*]] = stablehlo.convert %[[PART_ID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i64>
  // CHECK-NEXT: %[[IS_DEV1:.*]] = stablehlo.compare EQ, %[[PART_ID_I64]], %[[C1]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-NEXT: %[[IF_RES:.*]] = "stablehlo.if"(%[[IS_DEV1]]) ({
  // CHECK-NEXT:   %[[EXEC:.*]] = stablehlo.add %[[ARG0]], %[[ARG1]] : tensor<8x16xf32>
  // CHECK-NEXT:   stablehlo.return %[[EXEC]] : tensor<8x16xf32>
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   %[[ZEROS:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<8x16xf32>
  // CHECK-NEXT:   stablehlo.return %[[ZEROS]] : tensor<8x16xf32>
  // CHECK-NEXT: }) : (tensor<i1>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[IF_RES]] out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  %0 = sdy.reshard %arg0 <@single_dev_1, []> : tensor<8x16xf32>
  %1 = sdy.reshard %arg1 <@single_dev_1, []> : tensor<8x16xf32>
  %2 = stablehlo.add %0, %1 {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_1, []>]>
  } : tensor<8x16xf32>
  %3 = sdy.reshard %2 <@mesh, [{}, {}]> : tensor<8x16xf32>

  // CHECK-NEXT: return %[[ALL_REDUCE]] : tensor<8x16xf32>
  return %3 : tensor<8x16xf32>
}

// CHECK-LABEL: func @single_device_op0_to_single_device_op1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
func.func @single_device_op0_to_single_device_op1(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  // CHECK-NEXT: %[[IN_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[PART_ID0:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[PART_ID_I64_0:.*]] = stablehlo.convert %[[PART_ID0]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT: %[[IS_DEV0:.*]] = stablehlo.compare EQ, %[[PART_ID_I64_0]], %[[C0]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-NEXT: %[[IF_RES0:.*]] = "stablehlo.if"(%[[IS_DEV0]]) ({
  // CHECK-NEXT:   %[[EXEC0:.*]] = stablehlo.custom_call @CustomOp0(%[[IN_REPL]]) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT:   stablehlo.return %[[EXEC0]] : tensor<8x16xf32>
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   %[[ZEROS0:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<8x16xf32>
  // CHECK-NEXT:   stablehlo.return %[[ZEROS0]] : tensor<8x16xf32>
  // CHECK-NEXT: }) : (tensor<i1>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE0:.*]] = sdy.all_reduce {"x"} %[[IF_RES0]] out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[PART_ID1:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[PART_ID_I64_1:.*]] = stablehlo.convert %[[PART_ID1]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-NEXT: %[[C1:.*]] = stablehlo.constant dense<1> : tensor<i64>
  // CHECK-NEXT: %[[IS_DEV1:.*]] = stablehlo.compare EQ, %[[PART_ID_I64_1]], %[[C1]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-NEXT: %[[IF_RES1:.*]] = "stablehlo.if"(%[[IS_DEV1]]) ({
  // CHECK-NEXT:   %[[EXEC1:.*]] = stablehlo.custom_call @CustomOp1(%[[ALL_REDUCE0]]) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT:   stablehlo.return %[[EXEC1]] : tensor<8x16xf32>
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   %[[ZEROS1:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<8x16xf32>
  // CHECK-NEXT:   stablehlo.return %[[ZEROS1]] : tensor<8x16xf32>
  // CHECK-NEXT: }) : (tensor<i1>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE1:.*]] = sdy.all_reduce {"x"} %[[IF_RES1]] out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  %0 = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<8x16xf32>
  %1 = sdy.reshard %0 <@single_dev_0, []> : tensor<8x16xf32>
  %2 = stablehlo.custom_call @CustomOp0(%1) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %3 = sdy.reshard %2 <@mesh, [{}, {}]> : tensor<8x16xf32>
  %4 = sdy.reshard %3 <@single_dev_1, []> : tensor<8x16xf32>
  %5 = stablehlo.custom_call @CustomOp1(%4) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_1, []>]>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %6 = sdy.reshard %5 <@mesh, [{}, {}]> : tensor<8x16xf32>

  // CHECK-NEXT: return %[[ALL_REDUCE1]] : tensor<8x16xf32>
  return %6 : tensor<8x16xf32>
}

// CHECK-LABEL: func @single_device_op_with_token_result
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
// CHECK-SAME:     !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>}) {
func.func @single_device_op_with_token_result(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>}) {
  // CHECK-NEXT: %[[IN_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[PART_ID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[PART_ID_I64:.*]] = stablehlo.convert %[[PART_ID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT: %[[IS_DEV0:.*]] = stablehlo.compare EQ, %[[PART_ID_I64]], %[[C0]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-NEXT: %[[IF_RES:.*]]:2 = "stablehlo.if"(%[[IS_DEV0]]) ({
  // CHECK-NEXT:   %[[EXEC:.*]]:2 = stablehlo.custom_call @OpWithTokenResult(%[[IN_REPL]]) : (tensor<8x16xf32>) -> (tensor<8x16xf32>, !stablehlo.token)
  // CHECK-NEXT:   stablehlo.return %[[EXEC]]#0, %[[EXEC]]#1 : tensor<8x16xf32>, !stablehlo.token
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   %[[ZEROS:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<8x16xf32>
  // CHECK-NEXT:   %[[TOKEN:.*]] = stablehlo.create_token
  // CHECK-NEXT:   stablehlo.return %[[ZEROS]], %[[TOKEN]] : tensor<8x16xf32>, !stablehlo.token
  // CHECK-NEXT: }) : (tensor<i1>) -> (tensor<8x16xf32>, !stablehlo.token)
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[IF_RES]]#0 out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  %0 = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<8x16xf32>
  %1 = sdy.reshard %0 <@single_dev_0, []> : tensor<8x16xf32>
  %2:2 = stablehlo.custom_call @OpWithTokenResult(%1) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>, #sdy.sharding<@mesh, []>]>
  } : (tensor<8x16xf32>) -> (tensor<8x16xf32>, !stablehlo.token)
  %3 = sdy.reshard %2#0 <@mesh, [{}, {}]> : tensor<8x16xf32>

  // CHECK-NEXT: return %[[ALL_REDUCE]], %[[IF_RES]]#1 : tensor<8x16xf32>, !stablehlo.token
  return %3, %2#1 : tensor<8x16xf32>, !stablehlo.token
}

// CHECK-LABEL: func @single_device_op_reshard_to_other_replicated_mesh
// CHECK-SAME: (%[[ARG0:.*]]: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
// CHECK-SAME: -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{}, {}]>}) {
func.func @single_device_op_reshard_to_other_replicated_mesh(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh2, [{}, {}]>}) {
  // CHECK-NEXT: %[[IN_REPL:.*]] = sdy.reshard %[[ARG0]] <@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[PART_ID:.*]] = stablehlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %[[PART_ID_I64:.*]] = stablehlo.convert %[[PART_ID]] : (tensor<ui32>) -> tensor<i64>
  // CHECK-NEXT: %[[C0:.*]] = stablehlo.constant dense<0> : tensor<i64>
  // CHECK-NEXT: %[[IS_DEV0:.*]] = stablehlo.compare EQ, %[[PART_ID_I64]], %[[C0]] : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-NEXT: %[[IF_RES:.*]] = "stablehlo.if"(%[[IS_DEV0]]) ({
  // CHECK-NEXT:   %[[EXEC:.*]] = stablehlo.custom_call @SomeCustomCall(%[[IN_REPL]]) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  // CHECK-NEXT:   stablehlo.return %[[EXEC]] : tensor<8x16xf32>
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   %[[ZEROS:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<8x16xf32>
  // CHECK-NEXT:   stablehlo.return %[[ZEROS]] : tensor<8x16xf32>
  // CHECK-NEXT: }) : (tensor<i1>) -> tensor<8x16xf32>
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"x"} %[[IF_RES]] out_sharding=<@mesh, [{}, {}]> : tensor<8x16xf32>
  // CHECK-NEXT: %[[RESHARD_MESH2:.*]] = sdy.reshard %[[ALL_REDUCE]] <@mesh2, [{}, {}]> : tensor<8x16xf32>
  %0 = sdy.reshard %arg0 <@mesh, [{}, {}]> : tensor<8x16xf32>
  %1 = sdy.reshard %0 <@single_dev_0, []> : tensor<8x16xf32>
  %2 = stablehlo.custom_call @SomeCustomCall(%1) {
    sdy.sharding = #sdy.sharding_per_value<[#sdy.sharding<@single_dev_0, []>]>
  } : (tensor<8x16xf32>) -> tensor<8x16xf32>
  %3 = sdy.reshard %2 <@mesh2, [{}, {}]> : tensor<8x16xf32>

  // CHECK-NEXT: return %[[RESHARD_MESH2]] : tensor<8x16xf32>
  return %3 : tensor<8x16xf32>
}
