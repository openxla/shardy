// RUN: sdy_opt %s -sdy-apply-sharding-constraints | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// CHECK-LABEL: func @input_already_has_sharding
func.func @input_already_has_sharding(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>}
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a", ?}, {?}]>]>} :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @input_has_one_use
func.func @input_has_one_use(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @input_produced_by_data_flow_edge
func.func @input_produced_by_data_flow_edge(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: sdy.data_flow_edge %arg0 : tensor<8x8xf32>
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = sdy.data_flow_edge %arg0 : tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @open_sharding_constraint
func.func @open_sharding_constraint(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b", ?}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @input_is_func_input_with_one_use(
// CHECK-SAMEL    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>})
func.func @input_is_func_input_with_one_use(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %1 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @no_other_sharding_constraint_users
func.func @no_other_sharding_constraint_users(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = stablehlo.add %0, %0 :  tensor<8x8xf32>
  return %0, %1, %2 : tensor<8x8xf32>,  tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @has_other_sharding_constraint_user
func.func @has_other_sharding_constraint_user(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  return %0, %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @has_other_manual_computation_user
func.func @has_other_manual_computation_user(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"a"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>]
      manual_axes={"a"} (%arg2: tensor<4x8xf32>) {
    sdy.return %arg2 : tensor<4x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0, %1, %2 : tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @dangling_and_no_other_sharding_constraint_users
func.func @dangling_and_no_other_sharding_constraint_users(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"b"}]>]>}
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = stablehlo.add %0, %0 :  tensor<8x8xf32>
  return %0, %2 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @dangling_and_has_other_sharding_constraint_user
func.func @dangling_and_has_other_sharding_constraint_user(%arg0: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @dangling_and_has_other_manual_computation_user
func.func @dangling_and_has_other_manual_computation_user(%arg0: tensor<8x8xf32>)
    -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"a"}, {}]>] out_shardings=[<@mesh, [{"a"}, {}]>]
      manual_axes={"a"} (%arg2: tensor<4x8xf32>) {
    sdy.return %arg2 : tensor<4x8xf32>
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0, %2 : tensor<8x8xf32>, tensor<8x8xf32>
}

// CHECK-LABEL: func @manual_computation
func.func @manual_computation(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b", "a"}, {}]>]>}
  // CHECK-NEXT: sdy.manual_computation
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = stablehlo.add %arg1, %arg1 :  tensor<8x8xf32>
  %2 = sdy.manual_computation(%0, %1) in_shardings=[<@mesh, [{"a", "b"}, {?}]>, <@mesh, [{"b", "a"}, {}]>] out_shardings=[<@mesh, [{"a", "b"}, {}]>]
      manual_axes={"a", "b"} (%arg2: tensor<2x8xf32>, %arg3: tensor<2x8xf32>) {
    %3 = stablehlo.add %arg2, %arg3 : tensor<2x8xf32>
    sdy.return %3 : tensor<2x8xf32>
  } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
  func.return %0, %2: tensor<8x8xf32>, tensor<8x8xf32>
}
