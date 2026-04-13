// RUN: sdy_opt %s -sdy-equilise-call-and-func-result-shardings -split-input-file | FileCheck %s

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @propagate_func_to_call
func.func @propagate_func_to_call(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %0 = call @foo(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @foo(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  return %arg0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @do_not_overwrite_call_sharding
func.func @do_not_overwrite_call_sharding(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @foo(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  return %arg0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @both_call_and_func_has_empty_result_shardings
func.func @both_call_and_func_has_empty_result_shardings(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @foo(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %0 = call @foo(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_results
func.func @multiple_results(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{"y"}, {}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %0:2 = call @foo(%arg0, %arg1) : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  return %arg0, %arg1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @keep_empty_call_sharding
func.func @keep_empty_call_sharding(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @foo(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  return %arg0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_results_one_same_one_is_empty
func.func @multiple_results_one_same_one_is_empty(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{}, {}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{}, {}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  return %arg0, %arg1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_results_one_different_one_is_empty
func.func @multiple_results_one_different_one_is_empty(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>, <@mesh, [{}, {}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>, <@mesh, [{}, {}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<4x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  return %arg0, %arg1 : tensor<8x2xi32>, tensor<4x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_results_call_no_sharding_func_has_sharding_on_one_no_sharding_on_the_other
func.func @multiple_results_call_no_sharding_func_has_sharding_on_one_no_sharding_on_the_other(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>) {
  // CHECK-NEXT: %0:2 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>, <@mesh, [{?}, {?}]>]>} : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  %0:2 = call @foo(%arg0, %arg1) : (tensor<8x2xi32>, tensor<4x2xi32>) -> (tensor<8x2xi32>, tensor<4x2xi32>)
  return %0#0, %0#1 : tensor<8x2xi32>, tensor<4x2xi32>
}

func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<4x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, tensor<4x2xi32>) {
  return %arg0, %arg1 : tensor<8x2xi32>, tensor<4x2xi32>
}
