// RUN: sdy_opt %s -sdy-basic-propagate 2>&1 | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2, "d"=2]>

// A propagation barrier prevents the sharding from arg0 to propagate to %1 but
// %1 still receives the sharding due to the sharding group.
// CHECK-LABEL: func @shard_as_applies_despite_barrier
func.func @shard_as_applies_despite_barrier(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>})
   -> (tensor<8x8xf32>) {
  // CHECK-NEXT: sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} dense<0.000000e+00> : tensor<8x8xf32>
  %0 = sdy.constant dense<0.000000e+00> : tensor<8x8xf32>
  %1 = sdy.constant dense<0.000000e+00> : tensor<8x8xf32>
  // CHECK: stablehlo.tanh %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<8x8xf32>
  %2 = stablehlo.tanh %arg0 : tensor<8x8xf32>
  // CHECK-NEXT: sdy.propagation_barrier %1 allowed_direction=NONE {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<8x8xf32>
  %3 = sdy.propagation_barrier %1 allowed_direction=NONE : tensor<8x8xf32>
  // CHECK-NEXT: stablehlo.add %2, %3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} : tensor<8x8xf32>
  %4 = stablehlo.add %2, %3 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id=0 : tensor<8x8xf32>
  sdy.sharding_group %4 group_id=0 : tensor<8x8xf32>
  return %4 : tensor<8x8xf32>
}

// Validate that sharding groups are propagated across function calls. %0 will
// receive a sharding from %arg0 which should then be propagated across function
// calls to %1 (which otherwise would not receive a sharding).
// CHECK-LABEL: func @shard_as_applies_across_functionA
func.func @shard_as_applies_across_functionA(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>})
   -> (tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.tanh %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", ?}, {?}]>]>} : tensor<8x8xf32>
  %0 = stablehlo.tanh %arg0 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 1 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: func @shard_as_applies_across_functionB
func.func @shard_as_applies_across_functionB(%arg0: tensor<8x8xf32>)
   -> (tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", ?}, {?}]>]>} : tensor<8x8xf32>
  %1 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 1 : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// Verifies sharding group constraint propagation reflects user priority.
// Without the sharding group %1 would have %arg1's sharding, so we validate
// that because of the group, all intermediate values in this function call have
// the same sharding (which had the highest priority).
// CHECK-LABEL: func @shard_as_reflects_user_priority
func.func @shard_as_reflects_user_priority(
  %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}p0, {"c", ?}]>},
  %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c", "a",?}p1, {"b", ?}]>})
   -> (tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: stablehlo.add %0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"c", ?}]>]>} : tensor<8x8xf32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  %1 = stablehlo.add %arg1, %arg1 : tensor<8x8xf32>
  %2 = stablehlo.add %0, %1 : tensor<8x8xf32>
  sdy.sharding_group %0 group_id = 2 : tensor<8x8xf32>
  sdy.sharding_group %1 group_id = 2 : tensor<8x8xf32>
  sdy.sharding_group %2 group_id = 2 : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// Check sharding group constraint is propagated across a DataFlowEdgeOp. %3
// will receive the sharding from %arg0 at which point it will set the sharding
// on %4 (which otherwise would not have a sharding propagated).
// CHECK-LABEL: func @shard_as_across_dataflow_edge
func.func @shard_as_across_dataflow_edge(
  %arg0: tensor<16x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"d"}]>})
  -> tensor<16x16xf32> {
  // CHECK: stablehlo.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"d", ?}]>]>} dense<1.000000e+00> : tensor<16x16xf32>
  %4 = stablehlo.constant dense<1.0> : tensor<16x16xf32>
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %inc = stablehlo.constant dense<1> : tensor<i32>
  %comp = stablehlo.constant dense<32> : tensor<i32>
  // CHECK: %[[RESULT:.*]]:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %c) : tensor<16x16xf32>, tensor<i32>
  %1:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0) : tensor<16x16xf32>, tensor<i32>
    cond {
    %2 = stablehlo.compare  LT, %iterArg_2, %comp : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
  } do {
    %2 = stablehlo.add %iterArg_2, %inc : tensor<i32>
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"d", ?}]>]>} : tensor<16x16xf32>
    %3 = stablehlo.add %iterArg, %iterArg : tensor<16x16xf32>
    sdy.sharding_group %3 group_id = 3 : tensor<16x16xf32>
    stablehlo.return %3, %2 : tensor<16x16xf32>, tensor<i32>
  }
  // CHECK: sdy.data_flow_edge %[[RESULT]]#0 sharding=<@mesh, [{?}, {"d", ?}]> : tensor<16x16xf32>
  %5 = sdy.data_flow_edge %1#0 : tensor<16x16xf32>
  %6 = sdy.data_flow_edge %1#1 : tensor<i32>
  sdy.sharding_group %4 group_id = 3 : tensor<16x16xf32>
  return %5 : tensor<16x16xf32>
}
