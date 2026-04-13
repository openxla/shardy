// RUN: sdy_opt %s -sdy-add-data-flow-edges -split-input-file | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

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

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

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

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

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

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

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

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @optimization_barrier
func.func @optimization_barrier_dynamic_shaped_tensor_skipped(%arg0: tensor<32x96xf32>, %arg1: tensor<?x?xf32>)
    -> (tensor<32x96xf32>, tensor<?x?xf32>) {
  // CHECK-NEXT: %[[OPT_BARRIER:.*]]:2 = stablehlo.optimization_barrier %arg0, %arg1
  // CHECK:      %[[EDGE_1:.*]] = sdy.data_flow_edge %[[OPT_BARRIER]]#0
  // CHECK-NEXT: return %[[EDGE_1]], %[[OPT_BARRIER]]#1
  %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<32x96xf32>, tensor<?x?xf32>
  return %0#0, %0#1 : tensor<32x96xf32>, tensor<?x?xf32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

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

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[EDGE_1:.*]] = sdy.data_flow_edge %arg0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %arg1 : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_1]], %[[EDGE_2]] : tensor<8x2xi32>, tensor<4x2xi32>
  return %arg0, %arg1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// CHECK-LABEL: func @call_multiple_inputs_outputs
func.func @call_multiple_inputs_outputs(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @foo(%arg0, %arg1) : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[CALL]]#0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_4:.*]] = sdy.data_flow_edge %[[CALL]]#1 : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_3]], %[[EDGE_4]] : tensor<8x2xi32>, tensor<4x2xi32>
  %0:2 = call @foo(%arg0, %arg1) : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, %arg1: tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {}]>}) {
  // CHECK-NEXT: %[[EDGE_1:.*]] = sdy.data_flow_edge %arg0 sharding=<@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %arg1 sharding=<@mesh, [{?}, {}]> : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_1]], %[[EDGE_2]] : tensor<8x2xi32>, tensor<4x2xi32>
  return %arg0, %arg1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// CHECK-LABEL: func @call_with_func_shardings
func.func @call_with_func_shardings(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @foo(%arg0, %arg1) : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[CALL]]#0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_4:.*]] = sdy.data_flow_edge %[[CALL]]#1 : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_3]], %[[EDGE_4]] : tensor<8x2xi32>, tensor<4x2xi32>
  %0:2 = call @foo(%arg0, %arg1) : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[EDGE_1:.*]] = sdy.data_flow_edge %arg0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %arg1 : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_1]], %[[EDGE_2]] : tensor<8x2xi32>, tensor<4x2xi32>
  return %arg0, %arg1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// CHECK-LABEL: func @call_with_call_result_shardings
func.func @call_with_call_result_shardings(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>, <@mesh, [{?}, {}]>]>}
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[CALL]]#0 sharding=<@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_4:.*]] = sdy.data_flow_edge %[[CALL]]#1 sharding=<@mesh, [{?}, {}]> : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_3]], %[[EDGE_4]] : tensor<8x2xi32>, tensor<4x2xi32>
  %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>, <@mesh, [{?}, {}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[EDGE_1:.*]] = sdy.data_flow_edge %arg0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %arg1 : tensor<4x2xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[EDGE_1]], %[[EDGE_1]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[ADD]], %[[EDGE_2]] : tensor<8x2xi32>, tensor<4x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  return %0, %arg1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// CHECK-LABEL: func @call_ops_inside_and_outside
func.func @call_ops_inside_and_outside(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @foo(%arg0, %arg1) : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[CALL]]#0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_4:.*]] = sdy.data_flow_edge %[[CALL]]#1 : tensor<4x2xi32>
  // CHECK-NEXT: %[[SUB:.*]] = stablehlo.subtract %[[EDGE_3]], %[[EDGE_3]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[SUB]], %[[EDGE_4]] : tensor<8x2xi32>, tensor<4x2xi32>
  %0:2 = call @foo(%arg0, %arg1) : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %1 = stablehlo.subtract %0#0, %0#0 : tensor<8x2xi32>
  return %1, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[EDGE_1:.*]] = sdy.data_flow_edge %arg0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %arg1 : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_1]], %[[EDGE_2]] : tensor<8x2xi32>, tensor<4x2xi32>
  return %arg0, %arg1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// CHECK-LABEL: func @call_unused_result
func.func @call_unused_result(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @foo(%arg0, %arg1) : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[CALL]]#0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_4:.*]] = sdy.data_flow_edge %[[CALL]]#1 : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_3]] : tensor<8x2xi32>
  %0:2 = call @foo(%arg0, %arg1) : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token) {
  // CHECK-NEXT: %[[EDGE_1:.*]] = sdy.data_flow_edge %arg0 : tensor<8x2xi32>
  // CHECK-NEXT: return %[[EDGE_1]], %arg1 : tensor<8x2xi32>, !stablehlo.token
  return %arg0, %arg1 : tensor<8x2xi32>, !stablehlo.token
}

// CHECK-LABEL: func @call_skip_tokens
func.func @call_skip_tokens(%arg0: tensor<8x2xi32>, %arg1: !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token) {
  // CHECK-NEXT: %[[CALL:.*]]:2 = call @foo(%arg0, %arg1) : (tensor<8x2xi32>, !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token)
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %[[CALL]]#0 : tensor<8x2xi32>
  // CHECK-NEXT: return %[[EDGE_2]], %[[CALL]]#1 : tensor<8x2xi32>, !stablehlo.token
  %0:2 = call @foo(%arg0, %arg1) : (tensor<8x2xi32>, !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token)
  return %0#0, %0#1 : tensor<8x2xi32>, !stablehlo.token
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @manual_computation_skip_tokens
func.func @manual_computation_skip_tokens(%arg0: tensor<8x2xi32>, %arg1: !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token) {
  // CHECK-NEXT: %[[MC:.*]]:2 = sdy.manual_computation(%arg0, %arg1)
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg2 sharding=<@mesh, [{"a", ?}, {?}]> : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[EDGE_1]], %arg3 : tensor<8x2xi32>, !stablehlo.token
  // CHECK-NEXT: } : (tensor<8x2xi32>, !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token)
  // CHECK-NEXT: %[[EDGE_2:.*]] = sdy.data_flow_edge %[[MC]]#0 sharding=<@mesh, [{"a", ?}, {?}], replicated={"b"}> : tensor<8x2xi32>
  // CHECK-NEXT: return %[[EDGE_2]], %[[MC]]#1 : tensor<8x2xi32>, !stablehlo.token
  %0:2 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh, [{"a", ?}, {?}], replicated={"b"}>, <@mesh, []>]
      out_shardings=[<@mesh, [{"a", ?}, {?}], replicated={"b"}>, <@mesh, []>]
      manual_axes={"b"}  (%arg2: tensor<8x2xi32>, %arg3: !stablehlo.token) {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, !stablehlo.token
  } : (tensor<8x2xi32>, !stablehlo.token) -> (tensor<8x2xi32>, !stablehlo.token)
  return %0#0, %0#1 : tensor<8x2xi32>, !stablehlo.token
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @manual_computation_multiple_inputs_outputs
func.func @manual_computation_multiple_inputs_outputs(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[MC:.*]]:2 = sdy.manual_computation(%arg0, %arg1)
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg2 sharding=<@mesh, [{"a", ?}, {?}]> : tensor<8x2xi32>
  // CHECK-NEXT:   %[[EDGE_2:.*]] = sdy.data_flow_edge %arg3 sharding=<@mesh, [{?}, {?}]> : tensor<2x2xi32>
  // CHECK-NEXT:   sdy.return %[[EDGE_1]], %[[EDGE_2]] : tensor<8x2xi32>, tensor<2x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[MC]]#0 sharding=<@mesh, [{"a", ?}, {?}], replicated={"b"}> : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE_4:.*]] = sdy.data_flow_edge %[[MC]]#1 sharding=<@mesh, [{"b", ?}, {?}]> : tensor<4x2xi32>
  // CHECK-NEXT: return %[[EDGE_3]], %[[EDGE_4]] : tensor<8x2xi32>, tensor<4x2xi32>
  %0:2 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh, [{"a", ?}, {?}], replicated={"b"}>, <@mesh, [{"b", ?}, {?}]>]
      out_shardings=[<@mesh, [{"a", ?}, {?}], replicated={"b"}>, <@mesh, [{"b", ?}, {?}]>]
      manual_axes={"b"}  (%arg2: tensor<8x2xi32>, %arg3: tensor<2x2xi32>) {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<2x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @manual_computation_user_priority
func.func @manual_computation_user_priority(
    %arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}p2, {?}]>},
    %arg1: tensor<32x32xf32> ) -> tensor<32x32xf32> {
  // CHECK-NEXT: %[[MC:.*]] = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"a", "b"}p1, {?}]>, <@mesh, [{"a", ?}p1, {?}]>] out_shardings=[<@mesh, [{"a", ?}, {"b", ?}p0]>] manual_axes={"a"} (%arg2: tensor<16x32xf32>, %arg3: tensor<16x32xf32>) {
  // CHECK-NEXT:   %[[EDGE_1:.*]] = sdy.data_flow_edge %arg2 sharding=<@mesh, [{"b"}p1, {?}]> : tensor<16x32xf32>
  // CHECK-NEXT:   %[[EDGE_2:.*]] = sdy.data_flow_edge %arg3 sharding=<@mesh, [{?}p1, {?}]> : tensor<16x32xf32>
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[EDGE_1]], %[[EDGE_2]] : tensor<16x32xf32>
  // CHECK-NEXT:   sdy.return %[[ADD]] : tensor<16x32xf32>
  // CHECK-NEXT: } : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  // CHECK-NEXT: %[[EDGE_3:.*]] = sdy.data_flow_edge %[[MC]] sharding=<@mesh, [{"a", ?}, {"b", ?}p0]> : tensor<32x32xf32>
  // CHECK-NEXT: return %[[EDGE_3]] : tensor<32x32xf32>
  %0 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh, [{"a", "b"}p1, {?}]>, <@mesh, [{"a", ?}p1, {?}]>]
      out_shardings=[<@mesh, [{"a", ?}, {"b", ?}p0]>] manual_axes={"a"}
      (%arg2: tensor<16x32xf32>, %arg3: tensor<16x32xf32>) {
    %1 = stablehlo.add %arg2, %arg3 : tensor<16x32xf32>
    sdy.return %1 : tensor<16x32xf32>
  } : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  func.return %0 : tensor<32x32xf32>
}

// -----

// CHECK-LABEL: func @simple_call
func.func @simple_call(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%arg0)
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %0 : tensor<8xf32>
  // CHECK-NEXT: return %[[EDGE]]
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> (tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %arg0
  // CHECK-NEXT: return %[[EDGE]]
  return %arg0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_with_single_argument(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_single_argument(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_multiple_users_on_func_result(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_multiple_users_on_func_result(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  %3 = stablehlo.negate %1 : tensor<8xf32>
  %4 = stablehlo.add %2, %3 : tensor<8xf32>
  return %4 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) ->(tensor<8xf32>, tensor<8xf32>) {
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  %1 = stablehlo.abs %arg0: tensor<8xf32>
  return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_with_multiple_results(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_multiple_results(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]]:2 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %[[CALL]]#0 : tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL]]#1 : tensor<8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[EDGE0]], %[[EDGE1]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1:2 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  %2 = stablehlo.add %1#0, %1#1 : tensor<8xf32>
  %3 = stablehlo.abs %1#1 : tensor<8xf32>
  %4 = stablehlo.multiply %2, %3 : tensor<8xf32>
  return %4 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
func.func private @bar(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xf32> {
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 sharding=<@mesh, [{"a"}]> : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_with_sharded_argument(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_sharded_argument(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.negate %[[EDGE]]
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @multiple_calls_on_same_func(%arg0: tensor<8xf32>)
func.func @multiple_calls_on_same_func(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL0:.*]] = call @bar(%0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %[[CALL0]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[ABS0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL1]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  %3 = call @bar(%2) : (tensor<8xf32>) -> (tensor<8xf32>)
  %4 = stablehlo.abs %3 : tensor<8xf32>
  return %4 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %arg1 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.add %[[EDGE0]], %[[EDGE1]]
  %0 = stablehlo.add %arg0, %arg1: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_with_multiple_argument(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_multiple_argument(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%0, %1) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = stablehlo.abs %arg0 : tensor<8xf32>
  %2 = call @bar(%0, %1) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>)
  %3 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %arg1 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.add %[[EDGE0]], %[[EDGE1]]
  %0 = stablehlo.add %arg0, %arg1: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_on_func_with_multiple_argument_same_operand(%arg0: tensor<8xf32>)
func.func @simple_call_graph_on_func_with_multiple_argument_same_operand(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%0, %0) : (tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0, %0) : (tensor<8xf32>, tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @foo(%arg0: tensor<8xf32>)
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: @simple_chain_call_graph(%arg0: tensor<8xf32>)
func.func @simple_chain_call_graph(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @foo(%arg0: tensor<8xf32>)
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT:                stablehlo.abs %[[EDGE]]
  %0 = stablehlo.abs %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: @simple_non_flat_call_graph(%arg0: tensor<8xf32>)
func.func @simple_non_flat_call_graph(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @bar(%[[NEGATE]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL0]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @foo(%[[NEGATE]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE2:.*]] = sdy.data_flow_edge %[[CALL1]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE2]] : tensor<8xf32>
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  %3 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %4 = stablehlo.abs %3 : tensor<8xf32>
  %5 = stablehlo.add %2, %4 : tensor<8xf32>
  return %5 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @foo(%arg0: tensor<8xf32>)
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: @simple_non_flat_call_graph_one_after_the_other(%arg0: tensor<8xf32>)
func.func @simple_non_flat_call_graph_one_after_the_other(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL0:.*]] = call @bar(%0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %[[CALL0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @foo(%[[EDGE0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL1]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = call @foo(%1) : (tensor<8xf32>) -> (tensor<8xf32>)
  %3 = stablehlo.abs %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @foo(%arg0: tensor<8xf32>)
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[ABS0:.*]] = stablehlo.abs %[[EDGE0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ABS0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS1:.*]] = stablehlo.abs %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @foo(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = stablehlo.abs %1 : tensor<8xf32>
  return %2 : tensor<8xf32>
}

// CHECK-LABEL: @call_on_same_func_twice_input_of_one_is_output_of_the_other(%arg0: tensor<8xf32>)
func.func @call_on_same_func_twice_input_of_one_is_output_of_the_other(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL0:.*]] = call @bar(%0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %[[CALL0]] : tensor<8xf32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[EDGE0]]) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[CALL1]] : tensor<8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[EDGE0]], %[[EDGE1]] : tensor<8xf32>
  %0 = stablehlo.negate %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %2 = call @bar(%1) : (tensor<8xf32>) -> (tensor<8xf32>)
  %3 = stablehlo.add %1, %2 : tensor<8xf32>
  return %3 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_argument_is_input_to_call(%arg0: tensor<8xf32>)
func.func @simple_call_graph_argument_is_input_to_call(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[EDGE]]
  %0 = call @bar(%arg0) : (tensor<8xf32>) -> (tensor<8xf32>)
  %1 = stablehlo.abs %0 : tensor<8xf32>
  return %1 : tensor<8xf32>
}

// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_result_is_the_output_of_call(%arg0: tensor<8xf32>)
func.func @simple_call_graph_result_is_the_output_of_call(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: return %[[EDGE]]
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  %1 = call @bar(%0) : (tensor<8xf32>) -> (tensor<8xf32>)
  return %1 : tensor<8xf32>
}


// -----

// CHECK-LABEL: @bar(%arg0: tensor<8xf32>)
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]] : tensor<8xf32>
  %0 = stablehlo.negate %arg0: tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: @simple_call_graph_entry_contains_call_only(%arg0: tensor<8xf32>)
func.func @simple_call_graph_entry_contains_call_only(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK:      %[[CALL:.*]] = call @bar(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8xf32>
  // CHECK-NEXT: return %[[EDGE]]
  %0 = call @bar(%arg0) : (tensor<8xf32>) -> (tensor<8xf32>)
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @single_call_data_flow_edges_on_manual_computations
func.func @single_call_data_flow_edges_on_manual_computations(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE0:.*]] = sdy.data_flow_edge %[[CALL]] : tensor<8x2xi32>
  // CHECK-NEXT: %[[MC:.*]]:2 = sdy.manual_computation(%1, %arg1)
  // CHECK-SAME: in_shardings=[<@mesh, [{"a", ?}, {?}], replicated={"b"}>, <@mesh, [{"b", ?}, {?}]>]
  // CHECK-SAME: out_shardings=[<@mesh, [{"a", ?}, {?}], replicated={"b"}>, <@mesh, [{"b", ?}, {?}]>] manual_axes={"b"}
  // CHECK-SAME: (%arg2: tensor<8x2xi32>, %arg3: tensor<2x2xi32>) {
  // CHECK-NEXT:   %[[EDGE1:.*]] = sdy.data_flow_edge %arg2 sharding=<@mesh, [{"a", ?}, {?}]> : tensor<8x2xi32>
  // CHECK-NEXT:   %[[EDGE2:.*]] = sdy.data_flow_edge %arg3 sharding=<@mesh, [{?}, {?}]> : tensor<2x2xi32>
  // CHECK-NEXT:   sdy.return %[[EDGE1]], %[[EDGE2]] : tensor<8x2xi32>, tensor<2x2xi32>
  // CHECK-NEXT:  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  // CHECK-NEXT: %[[EDGE3:.*]] = sdy.data_flow_edge %2#0 sharding=<@mesh, [{"a", ?}, {?}], replicated={"b"}> : tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE4:.*]] = sdy.data_flow_edge %[[MC]]#1 sharding=<@mesh, [{"b", ?}, {?}]> : tensor<4x2xi32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[EDGE0]], %[[EDGE3]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[ADD]], %[[EDGE4]] : tensor<8x2xi32>
  %0 = call @foo(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1:2 = sdy.manual_computation(%0, %arg1)
      in_shardings=[<@mesh, [{"a", ?}, {?}], replicated={"b"}>, <@mesh, [{"b", ?}, {?}]>]
      out_shardings=[<@mesh, [{"a", ?}, {?}], replicated={"b"}>, <@mesh, [{"b", ?}, {?}]>]
      manual_axes={"b"}  (%arg2: tensor<8x2xi32>, %arg3: tensor<2x2xi32>) {
    sdy.return %arg2, %arg3 : tensor<8x2xi32>, tensor<2x2xi32>
  } : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %2 = stablehlo.add %0, %1#0 : tensor<8x2xi32>
  return %2, %1#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// CHECK-LABEL: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32>) {
  // CHECK-NEXT: %[[EDGE:.*]] = sdy.data_flow_edge %arg0 : tensor<8x2xi32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[EDGE]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x2xi32>
  %0 = stablehlo.negate %arg0 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

// CHECK-LABEL: func private @top_k_gt_f32_comparator
func.func private @top_k_gt_f32_comparator(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>) -> tensor<i1> {
  // CHECK-NEXT: %0 = stablehlo.compare
  // CHECK-NEXT: return %0
  %0 = stablehlo.compare GT, %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  return %0 : tensor<i1>
}

// CHECK-LABEL: func @custom_call_partial_reduce
func.func @custom_call_partial_reduce(%arg0: tensor<16x4xf32>, %arg1: tensor<16x4xf32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>) {
  %0:2 = stablehlo.custom_call @PartialReduce(%arg0, %arg1, %arg2, %arg3) {
    mhlo.backend_config = {
      aggregate_to_topk = true,
      recall_target = 0.9 : f32,
      reduction_dim = 1 : i64,
      reduction_input_size_override = -1 : i64,
      top_k = 2 : i64},
    called_computations = [@top_k_gt_f32_comparator]} :
    (tensor<16x4xf32>, tensor<16x4xf32>, tensor<f32>, tensor<i32>) -> (tensor<16x2xf32>, tensor<16x2xf32>)
  return %0#0, %0#1 : tensor<16x2xf32>, tensor<16x2xf32>
}
