// RUN: sdy_opt %s -sdy-propagate-call-sharding-to-func-results -split-input-file | FileCheck %s

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @call_has_sharding_func_does_not
func.func @call_has_sharding_func_does_not(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  return %arg0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @call_multiple_results
func.func @call_multiple_results(%arg0: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  // CHECK-NEXT: call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>, <@mesh, [{}]>]>}
  %0, %1 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>, <@mesh, [{}]>]>} : (tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}, tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>})
func.func private @foo(%arg0: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  return %arg0, %arg0 : tensor<8xf32>, tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @call_overwrites_existing_sharding
func.func @call_overwrites_existing_sharding(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
func.func private @foo(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  return %arg0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @both_call_and_func_have_result_shardings_but_different
func.func @both_call_and_func_have_result_shardings_but_different(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>}
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>})
func.func private @foo(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
  return %arg0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @call_does_not_have_shardings_func_has
func.func @call_does_not_have_shardings_func_has(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> tensor<8xf32> {
func.func private @foo(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  return %arg0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @call_does_have_empty_sharding_func_has_nonempty_sharding
func.func @call_does_have_empty_sharding_func_has_nonempty_sharding(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>}
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}]>]>} : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>})
func.func private @foo(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  return %arg0 : tensor<8xf32>
}
