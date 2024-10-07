// RUN: sdy_opt %s -sdy-add-data-flow-edges | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @case
func.func @case(%arg0: tensor<i32>, %arg1: tensor<8xi64>, %arg2: tensor<8xi64>,
                %arg3: tensor<8xi64>, %arg4: tensor<8xi64>)
    -> (tensor<8xi64>, tensor<8xi64>) {
  // CHECK-NEXT: %[[CASE:.*]]:2 = "stablehlo.case"(%arg0)
  // CHECK:      %[[EDGE_1:.*]] = sdy.data_flow_edge %[[CASE]]#0 : tensor<8xi64>
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %[[CASE]]#1 : tensor<8xi64>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[EDGE_2]], %[[EDGE_2]]
  // CHECK-NEXT: return %[[EDGE_1]], %[[ADD]]
  %0:2 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1, %arg2 : tensor<8xi64>, tensor<8xi64>
  }, {
    stablehlo.return %arg3, %arg4 : tensor<8xi64>, tensor<8xi64>
  }) : (tensor<i32>) -> (tensor<8xi64>, tensor<8xi64>)
  %1 = stablehlo.add %0#1, %0#1 : tensor<8xi64>
  return %0#0, %1 : tensor<8xi64>, tensor<8xi64>
}

// CHECK-LABEL: func @case_existing_sharding
func.func @case_existing_sharding(
    %arg0: tensor<i32>, %arg1: tensor<8xi64>, %arg2: tensor<8xi64>,
    %arg3: tensor<8xi64>, %arg4: tensor<8xi64>)
    -> (tensor<8xi64>, tensor<8xi64>) {
  // CHECK-NEXT: %[[CASE:.*]]:2 = "stablehlo.case"(%arg0)
  // CHECK:      }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>, <@mesh, [{?}]>]>}
  // CHECK-NEXT: %[[EDGE_1:.*]] = sdy.data_flow_edge %[[CASE]]#0 sharding=<@mesh, [{"a"}]>
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %[[CASE]]#1 sharding=<@mesh, [{?}]>
  // CHECK-NEXT: return %[[EDGE_1]], %[[EDGE_2]]
  %0:2 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1, %arg2 : tensor<8xi64>, tensor<8xi64>
  }, {
    stablehlo.return %arg3, %arg4 : tensor<8xi64>, tensor<8xi64>
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>, <@mesh, [{?}]>]>}
    : (tensor<i32>) -> (tensor<8xi64>, tensor<8xi64>)
  return %0#0, %0#1 : tensor<8xi64>, tensor<8xi64>
}

// CHECK-LABEL: func @case_token_result_skipped
func.func @case_token_result_skipped(
    %arg0: tensor<i32>, %arg1: !stablehlo.token, %arg2: !stablehlo.token,
    %arg3: tensor<4xi64>, %arg4: tensor<4xi64>)
    -> (!stablehlo.token, tensor<4xi64>) {
  // CHECK-NEXT: %[[CASE:.*]]:2 = "stablehlo.case"(%arg0)
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %[[CASE]]#1
  // CHECK-NEXT: return %[[CASE]]#0, %[[EDGE]]
  %0:2 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1, %arg3 : !stablehlo.token, tensor<4xi64>
  }, {
    stablehlo.return %arg2, %arg4 : !stablehlo.token, tensor<4xi64>
  }) : (tensor<i32>) -> (!stablehlo.token, tensor<4xi64>)
  return %0#0, %0#1 : !stablehlo.token, tensor<4xi64>
}

// CHECK-LABEL: func @optimization_barrier
func.func @optimization_barrier(%arg0: tensor<32x96xf32>, %arg1: tensor<32x96xf32>)
    -> (tensor<32x96xf32>, tensor<32x96xf32>) {
  // CHECK-NEXT: %[[OPT_BARRIER:.*]]:2 = stablehlo.optimization_barrier %arg0, %arg1
  // CHECK:      %[[EDGE_1:.*]] = sdy.data_flow_edge %[[OPT_BARRIER]]#0
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %[[OPT_BARRIER]]#1
  // CHECK-NEXT: return %[[EDGE_1]], %[[EDGE_2]]
  %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<32x96xf32>, tensor<32x96xf32>
  return %0#0, %0#1 : tensor<32x96xf32>, tensor<32x96xf32>
}

// CHECK-LABEL: func @optimization_barrier
func.func @optimization_barrier_dynamic_shaped_tensor_skipped(%arg0: tensor<32x96xf32>, %arg1: tensor<?x?xf32>)
    -> (tensor<32x96xf32>, tensor<?x?xf32>) {
  // CHECK-NEXT: %[[OPT_BARRIER:.*]]:2 = stablehlo.optimization_barrier %arg0, %arg1
  // CHECK:      %[[EDGE_1:.*]] = sdy.data_flow_edge %[[OPT_BARRIER]]#0
  // CHECK-NEXT: return %[[EDGE_1]], %[[OPT_BARRIER]]#1
  %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<32x96xf32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<32x96xf32>, tensor<?x?xf32>
}

// CHECK-LABEL: func @while_unused_result
func.func @while_unused_result(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  // CHECK:      %[[C0:.*]] = stablehlo.constant dense<0>
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = stablehlo.constant dense<1> : tensor<i32>
  %2 = stablehlo.constant dense<32> : tensor<i32>
  // CHECK:      %[[WHILE:.*]]:2 = stablehlo.while
  // CHECK:      %[[EDGE_1:.*]] = sdy.data_flow_edge %[[WHILE]]#0
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %[[WHILE]]#1
  // CHECK-NEXT: return %[[EDGE_1]]
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_2, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_2, %1 : tensor<i32>
    %5 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %5, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// CHECK-LABEL: func @named_computation_multiple_inputs_outputs
func.func @named_computation_multiple_inputs_outputs(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"my_func">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>)  {
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   %[[EDGE_2:.*]] = sdy.data_flow_edge %arg3 : tensor<4x2xi32>
  // CHECK-NEXT:   return %[[EDGE_1]], %[[EDGE_2]] : tensor<8x2xi32>, tensor<4x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[NC]]#0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_4:.*]] = sdy.data_flow_edge %[[NC]]#1 : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_3]], %[[EDGE_4]] : tensor<8x2xi32>, tensor<4x2xi32>
  %0:2 = sdy.named_computation<"my_func">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>)  {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// CHECK-LABEL: func @named_computation_with_shardings
func.func @named_computation_with_shardings(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"my_func">(%arg0, %arg1) in_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{?}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{?}, {}]>] (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>)  {
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg2 sharding=<@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT:   %[[EDGE_2:.*]] = sdy.data_flow_edge %arg3 sharding=<@mesh, [{?}, {}]> : tensor<4x2xi32>
  // CHECK-NEXT:   return %[[EDGE_1]], %[[EDGE_2]] : tensor<8x2xi32>, tensor<4x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[NC]]#0 sharding=<@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_4:.*]] = sdy.data_flow_edge %[[NC]]#1 sharding=<@mesh, [{?}, {}]> : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_3]], %[[EDGE_4]] : tensor<8x2xi32>, tensor<4x2xi32>
  %0:2 = sdy.named_computation<"my_func">(%arg0, %arg1) in_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{?}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>, <@mesh, [{?}, {}]>] (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>)  {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// CHECK-LABEL: func @named_computation_ops_inside_and_outside
func.func @named_computation_ops_inside_and_outside(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"my_func">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>)  {
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   %[[EDGE_2:.*]] = sdy.data_flow_edge %arg3 : tensor<4x2xi32>
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[EDGE_1]], %[[EDGE_1]] : tensor<8x2xi32>
  // CHECK-NEXT:   return %[[ADD]], %[[EDGE_2]] : tensor<8x2xi32>, tensor<4x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[NC]]#0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_4:.*]] = sdy.data_flow_edge %[[NC]]#1 : tensor<4x2xi32>
  // CHECK-NEXT: %[[SUB:.*]] = stablehlo.subtract %[[EDGE_3]], %[[EDGE_3]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[SUB]], %[[EDGE_4]] : tensor<8x2xi32>, tensor<4x2xi32>
  %0:2 = sdy.named_computation<"my_func">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>) {
    %2 = stablehlo.add %arg2, %arg2 : tensor<8x2xi32>
    sdy.return %2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %1 = stablehlo.subtract %0#0, %0#0 : tensor<8x2xi32>
  return %1, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// CHECK-LABEL: func @named_computation_unused_result
func.func @named_computation_unused_result(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"my_func_1">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>)  {
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   %[[EDGE_2:.*]] = sdy.data_flow_edge %arg3 : tensor<4x2xi32>
  // CHECK-NEXT:   return %[[EDGE_1]], %[[EDGE_2]] : tensor<8x2xi32>, tensor<4x2xi32>
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[EDGE_1:.*]] = sdy.data_flow_edge %[[NC]]#0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %[[NC]]#1 : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_1]] : tensor<8x2xi32>
  %0:2 = sdy.named_computation<"my_func_1">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: tensor<4x2xi32>)  {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<4x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0 : tensor<8x2xi32>
}

// CHECK-LABEL: func @named_computation_skip_tokens
func.func @named_computation_skip_tokens(%arg0: tensor<8x2xi32>, %arg1: !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token) {
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"foo">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: !stablehlo.token) {
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[EDGE_1]], %arg3 : tensor<8x2xi32>, !stablehlo.token
  // CHECK-NEXT: } : (tensor<8x2xi32>, !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token)
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %[[NC]]#0 : tensor<8x2xi32>
  // CHECK-NEXT: return %[[EDGE_2]], %[[NC]]#1 : tensor<8x2xi32>, !stablehlo.token
  %0:2 = sdy.named_computation<"foo">(%arg0, %arg1) (%arg2: tensor<8x2xi32>, %arg3: !stablehlo.token) {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, !stablehlo.token
  } : (tensor<8x2xi32>, !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token)
  return %0#0, %0#1 : tensor<8x2xi32>, !stablehlo.token
}
