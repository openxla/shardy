// RUN: sdy_opt %s -split-input-file -sdy-propagation-pipeline | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @split_constants_different_sharding
func.func @split_constants_different_sharding(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
    -> (tensor<8x16xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %[[CONST_1]], contracting_dims = [1] x [1]
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
  // CHECK-NEXT: return %[[CONST_2]], %[[ADD]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %0, contracting_dims = [1] x [1] : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x8xf32>
  %2 = stablehlo.add %1, %arg0 : tensor<8x8xf32>
  return %0, %2 : tensor<8x16xf32>, tensor<8x8xf32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// This test checks that the propagation pipeline invokes the highest strategy
// in the hierarchy, which is the user-priority propagation.
// CHECK-LABEL: func @user_priorities(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {"b"}]>},
// CHECK-SAME:      %arg2: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {"b"}]>},
// CHECK-SAME:      %arg3: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {"b"}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}, {"b"}]>}) {
func.func @user_priorities(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}p1, {"b"}p1]>},
    %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}, {"b"}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[ADD_0]], %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}, {"b"}]>]>}
  // CHECK-NEXT: stablehlo.divide %[[ADD_1]], %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}, {"b"}]>]>}
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.add %0, %arg2 : tensor<8x8xf32>
  %2 = stablehlo.divide %1, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}p0, {?}]>]>} : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @sharding_constraint_applied
func.func @sharding_constraint_applied(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
    -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// This test verifies that there is no sharding_constraint in the result.
// CHECK-LABEL: func @sharding_constraint_replaced_with_reshard
func.func @sharding_constraint_replaced_with_reshard(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  // CHECK-NEXT: return %arg0, %0
  return %arg0, %0 : tensor<8x8xf32>, tensor<8x8xf32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @size_zero_dim_sharded
// CHECK-SAME: %arg0: tensor<8x0xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>}
// CHECK-SAME: %arg1: tensor<8x0xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"a"}]>})
func.func @size_zero_dim_sharded(%arg0: tensor<8x0xf32>, %arg1: tensor<8x0xf32>) -> tensor<8x0xf32> {
  // CHECK-NEXT: %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>} : tensor<8x0xf32>
  return %0 : tensor<8x0xf32>
}

// -----

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @inlined_mesh(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
// CHECK-SAME:      %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
// CHECK-SAME:  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
func.func @inlined_mesh(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=2]>, [{"a"}, {"b"}]>},
    %arg1: tensor<8x8xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
  // CHECK-NEXT: stablehlo.dot_general %[[ADD]], %arg2
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @add_extra_sharding_constraint_for_incompatible_group_member_shardings(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}
// CHECK-SAME:  ) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
// CHECK-SAME:        tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
func.func @add_extra_sharding_constraint_for_incompatible_group_member_shardings(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[RESHARD_0:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"b"}]>
  // CHECK-NEXT: %[[RESHARD_1:.*]] = sdy.reshard %[[RESHARD_0]] <@mesh, [{"a"}, {"b"}]>
  // CHECK-NEXT: %[[RESHARD_2:.*]] = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]>
  // CHECK-NEXT: %[[RESHARD_3:.*]] = sdy.reshard %[[RESHARD_2]] <@mesh, [{"a"}, {"b"}]>
  // CHECK-NEXT: return %[[RESHARD_1]], %[[RESHARD_3]]
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"b", ?}]> : tensor<8x8xf32>
  sdy.sharding_group %0 group_id=1183 : tensor<8x8xf32>
  %1 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  sdy.sharding_group %1 group_id=1183 : tensor<8x8xf32>
  return %0, %1 : tensor<8x8xf32>, tensor<8x8xf32>
}

// -----

// Verifies the interaction between the `-apply-sharding-constraints` pass and
// sharding group propagation. This is validated by asserting that members of a
// sharding group pick up the sharding of a group member with a sharding
// constraint (the constraint needs to be added to the value in order for it to
// be applied to other group members).
sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: func @sharding_group_on_value_with_sharding_constraint
func.func @sharding_group_on_value_with_sharding_constraint(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  // CHECK: %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>}
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x16xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"a"}]> :  tensor<16x16xf32>
  // CHECK: %2 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>}
  %2 = stablehlo.add %arg1, %arg1 : tensor<16x16xf32>
  sdy.sharding_group %0 group_id = 32 : tensor<16x16xf32>
  sdy.sharding_group %2 group_id = 32 : tensor<16x16xf32>
  return %1, %2 : tensor<16x16xf32>, tensor<16x16xf32>
}

// -----

// Verifies the interaction between the `-sdy-add-data-flow-edges` pass and
// sharding group propagation. This is validated by adding a block argument
// of a while op to a sharding group which has a sharding constraint. This
// should be applied to other members of the group but can only happen if the
// `-sdy-add-data-flow-edges` pass is applied first.

sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: func @sharding_group_on_while_result
func.func @sharding_group_on_while_result(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %inc = stablehlo.constant dense<1> : tensor<i32>
  %comp = stablehlo.constant dense<32> : tensor<i32>
  // CHECK:      stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<16x16xf32>, tensor<i32>
  // CHECK-SAME:   attributes {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>, <@mesh, []>]>}
  %1:2 = stablehlo.while(%iterArg = %arg0, %iterArg_2 = %0) : tensor<16x16xf32>, tensor<i32>
    cond {
    %3 = stablehlo.compare  LT, %iterArg_2, %comp : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %3 : tensor<i1>
  } do {
    %3 = stablehlo.add %iterArg_2, %inc : tensor<i32>
    %4 = stablehlo.add %iterArg, %iterArg : tensor<16x16xf32>
    stablehlo.return %4, %3 : tensor<16x16xf32>, tensor<i32>
  }
  sdy.sharding_group %1#0 group_id = 50 : tensor<16x16xf32>

  // Add a value with an explicit sharding to group_id=50 which will apply an
  // initial sharding to the result of the WhileOp outside of the loop.
  %2 = sdy.sharding_constraint %arg1 <@mesh, [{"a"}, {}]> :  tensor<16x16xf32>
  sdy.sharding_group %2 group_id = 50 : tensor<16x16xf32>
  return %1#0, %2 : tensor<16x16xf32>, tensor<16x16xf32>
}

// -----

sdy.mesh @maximal_mesh = <[], device_ids=[0]>

// Nothing should be propagated, but this verifies the `transformShardings`
// sharding walker is able to handle a maximal sharding with no returned values.
// CHECK-LABEL: func @maximal_sharding_no_results
// CHECK-SAME:      (%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @maximal_sharding_no_results(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.custom_call @xla_python_cpu_callback(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, []>]>} : (tensor<8x8xf32>) -> ()
  // CHECK-NEXT: return %arg0 : tensor<8x8xf32>
  stablehlo.custom_call @xla_python_cpu_callback(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@maximal_mesh, []>]>} : (tensor<8x8xf32>) -> ()
  return %arg0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Nothing should be propagated, but this verifies the `transformShardings`
// sharding walker is able to handle a replicated sharding with no returned
// values.
// CHECK-LABEL: func @replicated_sharding_no_results
// CHECK-SAME:      (%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
func.func @replicated_sharding_no_results(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.custom_call @sdy_testonly(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh, []>]>} : (tensor<8x8xf32>) -> ()
  // CHECK-NEXT: return %arg0 : tensor<8x8xf32>
  stablehlo.custom_call @sdy_testonly(%arg0) {has_side_effect = true, sdy.sharding = #sdy.sharding_per_value<[<@mesh, []>]>} : (tensor<8x8xf32>) -> ()
  return %arg0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @manual_computation_with_tokens
// CHECK-SAME:      %arg0: !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>},
// CHECK-SAME:      %arg1: tensor<4x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
// CHECK-SAME:      -> (!stablehlo.token, tensor<4x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
func.func @manual_computation_with_tokens(
    %arg0: !stablehlo.token {sdy.sharding = #sdy.sharding<@mesh, []>},
    %arg1: tensor<4x4xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {?}]>}
) -> (!stablehlo.token, tensor<4x4xi64>) {
  // CHECK-NEXT: %[[MAN_COMP:.*]]:2 = sdy.manual_computation(%arg0, %arg1)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, []>, <@mesh, [{"a"}, {"b"}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, []>, <@mesh, [{"a"}, {"b"}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg2: !stablehlo.token, %arg3: tensor<4x2xi64>) {
  // CHECK-NEXT:   %[[TOK:.*]] = stablehlo.custom_call @sdy_testonly(%arg2) : (!stablehlo.token) -> !stablehlo.token
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg3, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<4x2xi64>
  // CHECK-NEXT:   sdy.return %[[TOK]], %[[ADD]] : !stablehlo.token, tensor<4x2xi64>
  // CHECK-NEXT: } : (!stablehlo.token, tensor<4x4xi64>) -> (!stablehlo.token, tensor<4x4xi64>)
  // CHECK-NEXT: return %[[MAN_COMP]]#0, %[[MAN_COMP]]#1 : !stablehlo.token, tensor<4x4xi64>
  %0:2 = sdy.manual_computation(%arg0, %arg1)
      in_shardings=[<@mesh, []>, <@mesh, [{?}, {"b"}]>]
      out_shardings=[<@mesh, []>, <@mesh, [{?}, {"b"}]>]
      manual_axes={"b"} (%arg2: !stablehlo.token, %arg3: tensor<4x2xi64>) {
    %1 = stablehlo.custom_call @sdy_testonly(%arg2) : (!stablehlo.token) -> (!stablehlo.token)
    %2 = stablehlo.add %arg3, %arg3 : tensor<4x2xi64>
    sdy.return %1, %2 : !stablehlo.token, tensor<4x2xi64>
  } : (!stablehlo.token, tensor<4x4xi64>) -> (!stablehlo.token, tensor<4x4xi64>)
  return %0#0, %0#1 : !stablehlo.token, tensor<4x4xi64>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @do_not_propagate_manual_axes_to_manual_computation(
// CHECK-SAME:      %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", "b"}]>}) {
func.func @do_not_propagate_manual_axes_to_manual_computation(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[MAN_COMP:.*]] = sdy.manual_computation(%arg0)
  // CHECK-SAME{LITERAL}:   in_shardings=[<@mesh, [{"a"}]>]
  // CHECK-SAME{LITERAL}:   out_shardings=[<@mesh, [{"a"}]>]
  // CHECK-SAME{LITERAL}:   manual_axes={"b"} (%arg1: tensor<8xf32>) {
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}]>]>} : tensor<8xf32>
  // CHECK-NEXT:   sdy.return %[[ADD]] : tensor<8xf32>
  // CHECK-NEXT: } : (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[SINE:.*]] = stablehlo.sine %[[MAN_COMP]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b"}]>]>} : tensor<8xf32>
  // CHECK-NEXT: return %[[SINE]] : tensor<8xf32>
  %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{?}]>] out_shardings=[<@mesh, [{?}]>] manual_axes={"b"} (%arg1: tensor<8xf32>) {
    %1 = stablehlo.add %arg1, %arg1 : tensor<8xf32>
    sdy.return %1 : tensor<8xf32>
  } : (tensor<8xf32>) -> tensor<8xf32>
  %1 = stablehlo.sine %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", "b"}]>]>} : tensor<8xf32>
  return %1 : tensor<8xf32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>
sdy.mesh @mesh_a_2 = <["a"=2]>

// TODO(b/412780544): Add another example this time without broadcast.
// CHECK-LABEL: func @dot_lhs_from_broadcast_and_large_rhs(
// CHECK-SAME:      %arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2, [{"a"}]>}
// CHECK-SAME:      %arg1: tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2, [{"a"}, {}]>}
// CHECK-SAME:  -> tensor<4x1024xf32> {
func.func @dot_lhs_from_broadcast_and_large_rhs(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2, [{"a", ?}]>}, %arg1: tensor<1024x1024xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2, [{"a", ?}, {?}]>}) -> tensor<4x1024xf32> {
  // CHECK-NEXT: %[[BROADCAST_IN_DIM:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2, [{}, {"a"}]>]>}
  // CHECK-NEXT: %[[DOT:.*]] = stablehlo.dot %[[BROADCAST_IN_DIM]], %arg1 :
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[DOT]] :
  // CHECK-NEXT: return %[[NEGATE]] : tensor<4x1024xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<4xf32>) -> tensor<4x1024xf32>
  %1 = stablehlo.dot %0, %arg1 : (tensor<4x1024xf32>, tensor<1024x1024xf32>) -> tensor<4x1024xf32>
  %2 = stablehlo.negate %1 : tensor<4x1024xf32>
  return %2 : tensor<4x1024xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @dot_general_with_unreduced_result(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
// CHECK-SAME:      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
// CHECK-SAME:  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>}) {
func.func @dot_general_with_unreduced_result(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
    %arg1: tensor<8x16xf32>)
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b"}]>}) {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}], unreduced={"b"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE_0:.*]] = sdy.all_reduce {"b"} %[[DOT_GENERAL]] out_sharding=<@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[ALL_REDUCE_1:.*]] = sdy.all_reduce {"b"} %[[DOT_GENERAL]] out_sharding=<@mesh, [{"a"}, {}]>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ALL_REDUCE_0]], %[[ALL_REDUCE_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}]>]>}
  // CHECK-NEXT: return %[[ADD]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?}], unreduced={"b"}>]>} :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @dot_general_with_unreduced_result_fully_delayed(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b", "c"}]>},
// CHECK-SAME:      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "c"}, {}]>})
// CHECK-SAME:  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b", "c"}]>}) {
func.func @dot_general_with_unreduced_result_fully_delayed(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b", "c"}]>},
    %arg1: tensor<8x16xf32>)
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", "c"}]>}) {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}], unreduced={"b", "c"}>]>}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %[[DOT_GENERAL]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}], unreduced={"b", "c"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE:.*]] = sdy.all_reduce {"b", "c"} %[[ADD]] out_sharding=<@mesh, [{"a"}, {}]>
  // CHECK-NEXT: return %[[ALL_REDUCE]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?}], unreduced={"b", "c"}>]>} :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?}], unreduced={"b", "c"}>]>} : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @dot_general_with_unreduced_result_partially_delayed(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b", "c"}]>},
// CHECK-SAME:      %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", "c"}, {}]>})
// CHECK-SAME:  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b", "c"}]>}) {
func.func @dot_general_with_unreduced_result_partially_delayed(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b", "c"}]>},
    %arg1: tensor<8x16xf32>)
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"b", "c"}]>}) {
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}], unreduced={"b", "c"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE_B_0:.*]] = sdy.all_reduce {"b"} %0 out_sharding=<@mesh, [{"a"}, {}], unreduced={"c"}>
  // CHECK-NEXT: %[[ALL_REDUCE_B_1:.*]] = sdy.all_reduce {"b"} %0 out_sharding=<@mesh, [{"a"}, {}], unreduced={"c"}>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[ALL_REDUCE_B_0]], %[[ALL_REDUCE_B_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {"b"}], unreduced={"c"}>]>}
  // CHECK-NEXT: %[[ALL_REDUCE_C:.*]] = sdy.all_reduce {"c"} %[[ADD]] out_sharding=<@mesh, [{"a"}, {"b"}]>
  // CHECK-NEXT: return %[[ALL_REDUCE_C]]
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
    {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?}], unreduced={"b", "c"}>]>} :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  %1 = stablehlo.add %0, %0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?}], unreduced={"c"}>]>} : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}
// -----

// Propagation tests for ops with data-flow edges like CaseOp and WhileOp

sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @case_single_result_func_args_single_sharding(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>}
// CHECK-SAME:      %arg2: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>})
// CHECK-SAME:      -> (tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>})
func.func @case_single_result_func_args_single_sharding(%arg0: tensor<i32>, %arg1: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>}, %arg2: tensor<4xi64>) -> (tensor<4xi64>) {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<4xi64>
  }, {
    stablehlo.return %arg2 : tensor<4xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}]>]>} :
  }) : (tensor<i32>) -> tensor<4xi64>
  return %0 : tensor<4xi64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @case_token_result_skipped(
// CHECK-SAME:      %arg0: tensor<i32>, %arg1: !stablehlo.token, %arg2: !stablehlo.token,
// CHECK-SAME:      %arg3: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>}
// CHECK-SAME:      %arg4: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>})
// CHECK-SAME:      -> (!stablehlo.token, tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>})
func.func @case_token_result_skipped(%arg0: tensor<i32>, %arg1: !stablehlo.token, %arg2: !stablehlo.token,
                                     %arg3: tensor<4xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}]>},
                                     %arg4: tensor<4xi64>) -> (!stablehlo.token, tensor<4xi64>) {
  %0:2 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1, %arg3 : !stablehlo.token, tensor<4xi64>
  }, {
    stablehlo.return %arg2, %arg4 : !stablehlo.token, tensor<4xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, []>, <@mesh_a_2_b_2, [{"a"}]>]>} :
  }) : (tensor<i32>) -> (!stablehlo.token, tensor<4xi64>)
  return %0#0, %0#1 : !stablehlo.token, tensor<4xi64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// This test makes sure we do not propagate any scalar value through the case op
// (even though we try it), since the OpShardingRuleAttr on scalars has no
// factors. Need to stick any sort of sharding on an argument to make sure this
// op has a bound mesh.
// CHECK-LABEL: func @case_scalars(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<i64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, []>},
// CHECK-SAME:      %arg2: tensor<i64>)
// CHECK-SAME:      -> tensor<i64>
func.func @case_scalars(%arg0: tensor<i32>, %arg1: tensor<i64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [], replicated={"a"}>}, %arg2: tensor<i64>) -> tensor<i64> {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<i64>
  }, {
    stablehlo.return %arg2 : tensor<i64>
  }) : (tensor<i32>) -> tensor<i64>
  return %0 : tensor<i64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @case_single_result_func_args_conflict(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "c"}]>})
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}]>})
func.func @case_single_result_func_args_conflict(%arg0: tensor<i32>, %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>}, %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "c"}]>}) -> (tensor<8xi64>) {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a"}]>]>} :
  }) : (tensor<i32>) -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// We find the most compatible major sharding axes, so the fact the first
// sharding is best shouldn't matter.
// CHECK-LABEL: func @case_single_result_func_first_sharding_best_ignored(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", "c"}]>}
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c"}]>})
func.func @case_single_result_func_first_sharding_best_ignored(
  %arg0: tensor<i32>,
  %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", "c"}]>},
  %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c"}]>}) -> (tensor<8xi64>) {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8xi64>
  // Make sure no sharding on case
  // CHECK: }) :
  }) : (tensor<i32>) -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @case_multiple_results_different_sharding(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c"}]>},
// CHECK-SAME:      %arg3: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>},
// CHECK-SAME:      %arg4: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c"}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>},
// CHECK-SAME:          tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c"}]>})
func.func @case_multiple_results_different_sharding(
  %arg0: tensor<i32>,

  %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}]>},
  %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c", ?}]>},

  %arg3: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>},
  %arg4: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"c", ?}]>}
  ) -> (tensor<8xi64>, tensor<8xi64>) {
  %0:2 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1, %arg2 : tensor<8xi64>, tensor<8xi64>
  }, {
    stablehlo.return %arg3, %arg4 : tensor<8xi64>, tensor<8xi64>
  // CHECK:      }) {sdy.sharding = #sdy.sharding_per_value<[
  // CHECK-SAME:     <@mesh_a_2_b_2_c_2, [{"a", "b"}]>,
  // CHECK-SAME      <@mesh_a_2_b_2_c_2, [{"c"}]>]>]>} :
  }) : (tensor<i32>) -> (tensor<8xi64>, tensor<8xi64>)
  return %0#0, %0#1 : tensor<8xi64>, tensor<8xi64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// Make sure we account for when an axis is used in another dimension when
// finding the most compatible major sharding axes.
// In this case we prefer the sharding of %arg1, because both %arg1 and %arg2
// have the same size.
// CHECK-LABEL: func @case_multiple_dim_most_compatible(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8x8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}, {"b"}]>},
// CHECK-SAME:      %arg2: tensor<8x8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}, {}]>}
// CHECK-SAME:      -> (tensor<8x8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}, {"b"}]>})
func.func @case_multiple_dim_most_compatible(
  %arg0: tensor<i32>,
  %arg1: tensor<8x8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{?}, {"b", ?}]>},
  %arg2: tensor<8x8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}, {?}]>}
  ) -> (tensor<8x8xi64>) {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8x8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8x8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a"}, {"b"}]>]>} :
  }) : (tensor<i32>) -> (tensor<8x8xi64>)
  return %0 : tensor<8x8xi64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @case_multiple_results_different_sharding_conflicts(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"b"}]>},
// CHECK-SAME:      %arg3: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>},
// CHECK-SAME:      %arg4: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>},
// CHECK-SAME:          tensor<8xi64>)
func.func @case_multiple_results_different_sharding_conflicts(
  %arg0: tensor<i32>,

  %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}]>},
  %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"b", ?}]>},

  %arg3: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b", ?}]>},
  %arg4: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", ?}]>}
  ) -> (tensor<8xi64>, tensor<8xi64>) {
  %0:2 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1, %arg2 : tensor<8xi64>, tensor<8xi64>
  }, {
    stablehlo.return %arg3, %arg4 : tensor<8xi64>, tensor<8xi64>
  // CHECK:      }) {sdy.sharding = #sdy.sharding_per_value<[
  // CHECK-SAME:     <@mesh_a_2_b_2_c_2, [{"a", "b"}]>,
  // CHECK-SAME:     <@mesh_a_2_b_2_c_2, [{}]>]>} :
  }) : (tensor<i32>) -> (tensor<8xi64>, tensor<8xi64>)
  return %0#0, %0#1 : tensor<8xi64>, tensor<8xi64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @case_closed_sharding(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>}
func.func @case_closed_sharding(
  %arg0: tensor<i32>,
  %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}]>},
  %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>}
  ) -> tensor<8xi64> {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b"}]>]>} :
  }) : (tensor<i32>) -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// Basic case where the shardings are read from an intermediate value and used
// by another op.
// CHECK-LABEL: func @case_not_func_args_or_directly_returned(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>}
func.func @case_not_func_args_or_directly_returned(
  %arg0: tensor<i32>,
  %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a"}]>},
  %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2_c_2, [{"a", "b"}]>}
  ) -> tensor<8xi64> {
  // CHECK: stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b"}]>]>} : tensor<8xi64>
  %0 = stablehlo.add %arg1, %arg1 : tensor<8xi64>
  // CHECK: stablehlo.add %arg2, %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b"}]>]>} : tensor<8xi64>
  %1 = stablehlo.add %arg2, %arg2 : tensor<8xi64>
  // CHECK: %[[CASE:.*]] = "stablehlo.case"
  %2 = "stablehlo.case"(%arg0) ({
    stablehlo.return %0 : tensor<8xi64>
  }, {
    stablehlo.return %1 : tensor<8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b"}]>]>} :
  }) : (tensor<i32>) -> tensor<8xi64>
  // CHECK: stablehlo.add %[[CASE]], %[[CASE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2_c_2, [{"a", "b"}]>]>} : tensor<8xi64>
  %3 = stablehlo.add %2, %2 :  tensor<8xi64>
  return %3 : tensor<8xi64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @case_propagate_from_func_result(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}]>}
func.func @case_propagate_from_func_result(
    %arg0: tensor<i32>, %arg1: tensor<8xi64>,
    %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}]>})
    -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>}) {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b"}]>]>} :
  }) : (tensor<i32>) -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @case_propagate_from_case_op_result(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}]>}
func.func @case_propagate_from_case_op_result(
    %arg0: tensor<i32>, %arg1: tensor<8xi64>,
    %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}]>})
    -> tensor<8xi64> {
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b"}]>]>} :
  }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b", ?}]>]>} : (tensor<i32>) -> tensor<8xi64>
  return %0 : tensor<8xi64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @case_propagate_mulitple_uses(
// CHECK-SAME:      %arg0: tensor<i32>,
// CHECK-SAME:      %arg1: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}]>},
// CHECK-SAME:      %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}]>}
// CHECK-SAME:      -> (tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b"}]>}
func.func @case_propagate_mulitple_uses(
    %arg0: tensor<i32>, %arg1: tensor<8xi64>,
    %arg2: tensor<8xi64> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", "b", ?}]>})
    -> (tensor<8xi64>) {
  // CHECK: %[[CASE:.*]] = "stablehlo.case"(%arg0)
  %0 = "stablehlo.case"(%arg0) ({
    stablehlo.return %arg1 : tensor<8xi64>
  }, {
    stablehlo.return %arg2 : tensor<8xi64>
  // CHECK: }) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b"}]>]>} :
  }) : (tensor<i32>) -> tensor<8xi64>
  // CHECK: %[[ADD:.*]] = stablehlo.add %[[CASE]], %[[CASE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b"}]>]>} :
  %1 = stablehlo.add %0, %0 : tensor<8xi64>
  // CHECK: %[[SUB:.*]] = stablehlo.subtract %[[CASE]], %[[CASE]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b"}]>]>} :
  %2 = stablehlo.subtract %0, %0 : tensor<8xi64>
  // CHECK: stablehlo.divide %[[SUB]], %[[ADD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a", "b"}]>]>} :
  %3 = stablehlo.divide %2, %1 : tensor<8xi64>
  return %3 : tensor<8xi64>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @optimization_barrier(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
// CHECK-SAME:      %arg1: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b"}, {}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
// CHECK-SAME:          tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b"}, {}]>})
func.func @optimization_barrier(%arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {?}]>}, %arg1: tensor<32x96xf32>)
    -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b"}]>},
        tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b"}, {?}]>}) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>}
  %0 = stablehlo.add %arg0, %arg0 : tensor<32x96xf32>
  // CHECK-NEXT: stablehlo.optimization_barrier {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>, <@mesh_a_2_b_2, [{"b"}, {}]>]>}
  %1:2 = stablehlo.optimization_barrier %0, %arg1 : tensor<32x96xf32>, tensor<32x96xf32>
  return %1#0, %1#1 : tensor<32x96xf32>, tensor<32x96xf32>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// Check propagation starting at a return can go through a WhileOp.
// CHECK-LABEL: func @while_func_return(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
func.func @while_func_return(%arg0: tensor<32x96xf32>) -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) {
  // CHECK: %[[C0:.*]] = sdy.constant dense<0>
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  // CHECK: stablehlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]]) : tensor<32x96xf32>, tensor<i32>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>, <@mesh_a_2_b_2, []>]>}
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
    %5 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %5, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// Check propagation starting at a return can go through a WhileOp, but with it
// being based purely on the func block arg.
// CHECK-LABEL: func @while_func_return_directly_on_func_operand(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
func.func @while_func_return_directly_on_func_operand(
    %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a", ?}, {?}]>})
    -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{?}, {"b", ?}]>}) {
  // CHECK: %[[C0:.*]] = sdy.constant dense<0>
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  // CHECK: stablehlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]]) : tensor<32x96xf32>, tensor<i32>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>, <@mesh_a_2_b_2, []>]>}
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
    %5 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %5, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// Check propagation starting at a use of the while result can go through the
// WhileOp.
// CHECK-LABEL: func @while_result_use(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
func.func @while_result_use(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  // CHECK: %[[C0:.*]] = sdy.constant dense<0>
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  // CHECK: stablehlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]]) : tensor<32x96xf32>, tensor<i32>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>, <@mesh_a_2_b_2, []>]>}
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %5 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %5 : tensor<i1>
  } do {
    %5 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
    %6 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %6, %5 : tensor<32x96xf32>, tensor<i32>
  }
  %4 = stablehlo.add %3#0, %3#0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
  return %4 : tensor<32x96xf32>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// Check propagation starting inside the body of the while can go out of the
// body through the block argument of the body. We prevent forwards propagation
// in the body by making the use of the sharded op be not partitionable.
// CHECK-LABEL: func @while_body_propagate_block_arg(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
func.func @while_body_propagate_block_arg(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    %5 = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
    // Not partitionable!
    %6 = stablehlo.custom_call @sdy_testonly(%5) : (tensor<32x96xf32>) -> tensor<32x96xf32>
    stablehlo.return %6, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @while_body_token_block_arg_skipped(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>},
// CHECK-SAME:      %arg1: !stablehlo.token)
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}, !stablehlo.token)
func.func @while_body_token_block_arg_skipped(%arg0: tensor<32x96xf32>, %arg1: !stablehlo.token)
    -> (tensor<32x96xf32>, !stablehlo.token) {
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  %3:3 = stablehlo.while(%iterArg = %arg1, %iterArg_0 = %arg0, %iterArg_1 = %0) : !stablehlo.token, tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_1, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_1, %1 : tensor<i32>
    %5 = stablehlo.add %iterArg_0, %iterArg_0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
    // Not partitionable!
    %6 = stablehlo.custom_call @sdy_testonly(%5) : (tensor<32x96xf32>) -> tensor<32x96xf32>
    stablehlo.return %iterArg, %6, %4 : !stablehlo.token, tensor<32x96xf32>, tensor<i32>
  }
  return %3#1, %3#0 : tensor<32x96xf32>, !stablehlo.token
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// Similar test to while_body_propagate_block_arg, except this makes sure that
// propagation flows to the WhileOp operand through the return op - and not
// through the block argument. We prevent backwards propagation in the body by
// making the defining op of the sharded op's operand be not partitionable.
// CHECK-LABEL: func @while_body_propagate_return_op(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
func.func @while_body_propagate_return_op(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    // Not partitionable!
    %5 = stablehlo.custom_call @sdy_testonly(%iterArg) : (tensor<32x96xf32>) -> tensor<32x96xf32>
    %6 = stablehlo.add %5, %5 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
    stablehlo.return %6, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// Same as above, except the use is not a func operand nor is the result of the
// while directly returned from the func.
// CHECK-LABEL: func @while_body_non_func_operand_result_use(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
func.func @while_body_non_func_operand_result_use(%arg0: tensor<32x96xf32>) -> tensor<32x96xf32> {
  // CHECK: %[[C0:.*]] = sdy.constant dense<0>
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  // CHECK: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
  %3 = stablehlo.add %arg0, %arg0 : tensor<32x96xf32>
  // CHECK: %[[WHILE:.*]]:2 = stablehlo.while(%iterArg = %[[ADD]], %iterArg_0 = %[[C0]]) : tensor<32x96xf32>, tensor<i32>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>, <@mesh_a_2_b_2, []>]>}
  %4:2 = stablehlo.while(%iterArg = %3, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %6 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %6 : tensor<i1>
  } do {
    %6 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    %7 = stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
    stablehlo.return %7, %6 : tensor<32x96xf32>, tensor<i32>
  }
  // CHECK: stablehlo.add %[[WHILE]]#0, %[[WHILE]]#0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
  %5 = stablehlo.add %4#0, %4#0 : tensor<32x96xf32>
  return %5 : tensor<32x96xf32>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>
sdy.mesh @mesh_a_2_b_2_c_2 = <["a"=2, "b"=2, "c"=2]>

// Check propagation starting from a func arg can go through the WhileOp.
// CHECK-LABEL: func @while_func_operand(
// CHECK-SAME:      %arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
// CHECK-SAME:      -> (tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>})
func.func @while_func_operand(%arg0: tensor<32x96xf32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {"b"}]>}) -> tensor<32x96xf32> {
  // CHECK: %[[C0:.*]] = sdy.constant dense<0>
  %0 = sdy.constant dense<0> : tensor<i32>
  %1 = sdy.constant dense<1> : tensor<i32>
  %2 = sdy.constant dense<32> : tensor<i32>
  // CHECK: stablehlo.while(%iterArg = %arg0, %iterArg_0 = %[[C0]]) : tensor<32x96xf32>, tensor<i32>
  // CHECK-SAME: {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>, <@mesh_a_2_b_2, []>]>}
  %3:2 = stablehlo.while(%iterArg = %arg0, %iterArg_0 = %0) : tensor<32x96xf32>, tensor<i32>
    cond {
    %4 = stablehlo.compare  LT, %iterArg_0, %2 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %4 : tensor<i1>
  } do {
    %4 = stablehlo.add %iterArg_0, %1 : tensor<i32>
    // CHECK: stablehlo.add %iterArg, %iterArg {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {"b"}]>]>} : tensor<32x96xf32>
    %5 = stablehlo.add %iterArg, %iterArg : tensor<32x96xf32>
    stablehlo.return %5, %4 : tensor<32x96xf32>, tensor<i32>
  }
  return %3#0 : tensor<32x96xf32>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>

// Check we can propagate forward from outside into the call, then back out.
// test: call_argument_sharding_propagation
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>

// Check we can propagate backwards from outside into the call, then back out.
// test: call_result_sharding_propagation
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>}) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>}) {
// CHECK-NEXT:    %[[SUB:.*]] = stablehlo.subtract %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {}]>]>} : tensor<8x2xi32>
// CHECK-NEXT:    return %[[SUB]] : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = stablehlo.subtract %arg0, %arg0 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>

// Check we can propagate in both directions from inside out.
// test: call_inside_out_propagation
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"b"}, {}]>}) {
// CHECK-NEXT:    %[[RESHARD0:.*]] = sdy.reshard %arg0 <@mesh_a_2_b_2, [{"a"}, {}]> : tensor<8x2xi32>
// CHECK-NEXT:    %[[RESHARD1:.*]] = sdy.reshard %[[RESHARD0]] <@mesh_a_2_b_2, [{"b"}, {}]> : tensor<8x2xi32>
// CHECK-NEXT:    return %[[RESHARD1]] : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = sdy.sharding_constraint %arg0 <@mesh_a_2_b_2, [{"a"}, {}]> : tensor<8x2xi32>
  %1 = sdy.sharding_constraint %0 <@mesh_a_2_b_2, [{"b"}, {}]> : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh_a_2_b_2 = <["a"=2, "b"=2]>

// test: simple non flat. main calls foo twice.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_a_2_b_2, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL1]] : tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_a_2_b_2, [{"a"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2]>

// test: a non flat call graph. main calls foo and bar, foo calls bar. propagates from main.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar_0(%[[CALL0]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL1]] : tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @bar(%1) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @bar_0(%[[ABS]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0: tensor<8x2xi32>
  %1 = call @bar(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }

// CHECK-NOT: func private @bar(%arg0: tensor<8x2xi32>
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}


// -----
sdy.mesh @mesh = <["a"=2, "b"=2]>

// test: a non flat call graph. main calls foo and bar, foo calls bar. propagates from foo.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar_0(%[[CALL0]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL1]] : tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @bar(%1) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @bar_0(%[[ABS]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}: tensor<8x2xi32>
  %1 = call @bar(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }

// CHECK-NOT: func private @bar(%arg0: tensor<8x2xi32>
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2]>

// test: a non flat call graph. main calls foo and bar, foo calls bar. two bar calls have different shardings.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar_0(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL1]] : tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @bar(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %0 <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[RESHARD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0 : tensor<8x2xi32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  %3 = call @bar(%2) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %3 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }
