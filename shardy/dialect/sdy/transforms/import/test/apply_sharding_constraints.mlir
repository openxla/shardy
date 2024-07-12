// RUN: sdy_opt %s -sdy-apply-sharding-constraints | FileCheck %s

sdy.mesh @mesh = <"a"=2, "b"=2>

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

// CHECK-LABEL: func @input_is_func_input_with_one_use(
// CHECK-SAMEL    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"b"}]>})
func.func @input_is_func_input_with_one_use(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %1 = sdy.sharding_constraint %arg0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @input_has_multiple_uses
func.func @input_has_multiple_uses(%arg0: tensor<8x8xf32>)
  -> (tensor<8x8xf32>, tensor<8x8xf32>, tensor<8x8xf32>) {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = stablehlo.add %0, %0 :  tensor<8x8xf32>
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

// CHECK-LABEL: func @dangling_and_has_other_sharding_constraint_users
func.func @dangling_and_has_other_sharding_constraint_users(%arg0: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  // CHECK-NEXT: stablehlo.add %arg0, %arg0
  // CHECK-NOT: sdy.sharding
  // CHECK-NEXT: sdy.sharding_constraint
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

