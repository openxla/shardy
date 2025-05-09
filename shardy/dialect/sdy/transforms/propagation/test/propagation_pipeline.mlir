// RUN: sdy_opt %s -split-input-file -sdy-propagation-pipeline 2>&1 | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @split_constants_different_sharding
func.func @split_constants_different_sharding(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>})
    -> (tensor<8x16xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %[[CONST_0:.*]] = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_1:.*]] = sdy.constant {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", ?}, {?}]>]>} dense<1.000000e+00>
  // CHECK-NEXT: %[[CONST_2:.*]] = sdy.constant dense<1.000000e+00>
  // CHECK-NEXT: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %[[CONST_0]], %[[CONST_1]], contracting_dims = [1] x [1]
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[DOT_GENERAL]], %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: return %[[CONST_2]], %[[ADD]]
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<8x16xf32>
  %1 = stablehlo.dot_general %0, %0, contracting_dims = [1] x [1] : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x8xf32>
  %2 = stablehlo.add %1, %arg0 : tensor<8x8xf32>
  return %0, %2 : tensor<8x16xf32>, tensor<8x8xf32>
}

// This test checks that the propagation pipeline invokes the highest strategy
// in the hierarchy, which is the user-priority propagation.
// CHECK-LABEL: func @user_priorities(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b"}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c", ?}, {"b", ?}]>},
// CHECK-SAME:      %arg2: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c", ?}, {"b", ?}]>},
// CHECK-SAME:      %arg3: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c", ?}, {"b", ?}]>})
// CHECK-SAME:  -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"c", ?}, {"b", ?}]>}) {
func.func @user_priorities(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}p1, {"b"}p1]>},
    %arg1: tensor<8x8xf32>, %arg2: tensor<8x8xf32>, %arg3: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[ADD_0:.*]] = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: %[[ADD_1:.*]] = stablehlo.add %[[ADD_0]], %arg2 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: stablehlo.divide %[[ADD_1]], %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}, {"b", ?}]>]>}
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.add %0, %arg2 : tensor<8x8xf32>
  %2 = stablehlo.divide %1, %arg3 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}p0, {?}]>]>} : tensor<8x8xf32>
  return %2 : tensor<8x8xf32>
}

// CHECK-LABEL: func @sharding_constraint_applied
func.func @sharding_constraint_applied(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
    -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>}
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// This test verifies that there is no sharding_constraint in the result.
// CHECK-LABEL: func @sharding_constraint_replaced_with_reshard
func.func @sharding_constraint_replaced_with_reshard(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: %0 = sdy.reshard %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {"b"}]> : tensor<8x8xf32>
  // CHECK-NEXT: return %arg0, %0
  return %arg0, %0 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @size_zero_dim_sharded
// CHECK-SAME: %arg0: tensor<8x0xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>}
// CHECK-SAME: %arg1: tensor<8x0xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {"a", ?}]>})
func.func @size_zero_dim_sharded(%arg0: tensor<8x0xf32>, %arg1: tensor<8x0xf32>) -> tensor<8x0xf32> {
  // CHECK-NEXT: %0 = stablehlo.add %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>}
  %0 = stablehlo.add %arg0, %arg1 {sdy.sharding=#sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>} : tensor<8x0xf32>
  return %0 : tensor<8x0xf32>
}

// -----

// CHECK: sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @inlined_mesh(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {"b"}]>},
// CHECK-SAME:      %arg1: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>},
// CHECK-SAME:      %arg2: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"b", ?}, {?}]>})
// CHECK-SAME:  -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {?}]>}) {
func.func @inlined_mesh(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<["a"=2, "b"=2]>, [{"a"}, {"b"}]>},
    %arg1: tensor<8x8xf32>, %arg2: tensor<8x16xf32>) -> tensor<8x16xf32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>}
  // CHECK-NEXT: stablehlo.dot_general %[[ADD]], %arg2
  // CHECK-SAME:   {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  %0 = stablehlo.add %arg0, %arg1 : tensor<8x8xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0] :
    (tensor<8x8xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: add_extra_sharding_constraint_for_incompatible_sharding_shardings(
// CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>}
// CHECK-SAME:  ) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a", ?}, {"b", ?}]>}) {
func.func @add_extra_sharding_constraint_for_incompatible_sharding_shardings(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[WSC_0:.*]] = sdy.reshard %arg0 <@mesh, [{}, {"b", ?}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[WSC_1:.*]] = sdy.reshard %[[WSC_0]] <@mesh, [{"a", ?}, {"b", ?}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[WSC_2:.*]] = sdy.reshard %arg0 <@mesh, [{"a"}, {"b", ?}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[WSC_3:.*]] = sdy.reshard %[[WSC_2]] <@mesh, [{"a", ?}, {"b", ?}]> : tensor<8x8xf32>
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[WSC_1]], %[[WSC_3]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>} : tensor<8x8xf32>
  // CHECK-NEXT: return %[[ADD]] : tensor<8x8xf32>
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"b", ?}]> : tensor<8x8xf32>
  sdy.sharding_group %0 group_id=1183 : tensor<8x8xf32>
  %1 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {?}]> : tensor<8x8xf32>
  sdy.sharding_group %1 group_id=1183 : tensor<8x8xf32>
  %2 = stablehlo.add %0, %1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {"b", ?}]>]>} : tensor<8x8xf32>
  func.return %2: tensor<8x8xf32>
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
