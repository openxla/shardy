// RUN: sdy_opt %s -sdy-propagate-unreduced-across-call-boundaries | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

//===----------------------------------------------------------------------===//
// Forward propagation (caller operands -> callee arguments)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @callee_forward(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}], unreduced={"x"}>})
func.func @callee_forward(%arg0: tensor<8x8xf32>) {
  return
}

// CHECK-LABEL: func.func @caller_forward
func.func @caller_forward(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) {
  // CHECK: call @callee_forward(%arg0) : (tensor<8x8xf32>) -> ()
  func.call @callee_forward(%arg0) : (tensor<8x8xf32>) -> ()
  return
}

// CHECK-LABEL: func.func @callee_forward_merge(
// CHECK-SAME:    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>})
func.func @callee_forward_merge(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  return
}

// CHECK-LABEL: func.func @caller_forward_merge
func.func @caller_forward_merge(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) {
  // CHECK: call @callee_forward_merge(%arg0) : (tensor<8x8xf32>) -> ()
  func.call @callee_forward_merge(%arg0) : (tensor<8x8xf32>) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Backward propagation (callee return values -> caller results)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @callee_backward(%arg0: tensor<8x8xf32>)
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>})
func.func @callee_backward(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @caller_backward
func.func @caller_backward(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: %[[RES:.*]] = call @callee_backward(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {?}], unreduced={"x"}>]>}
  // CHECK-NEXT: return %[[RES]]
  %0 = func.call @callee_backward(%arg0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @callee_backward_merge(%arg0: tensor<8x8xf32>)
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>})
func.func @callee_backward_merge(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>}) {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @caller_backward_merge
func.func @caller_backward_merge(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK: %[[RES:.*]] = call @callee_backward_merge(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>}
  // CHECK-NEXT: return %[[RES]]
  %0 = func.call @callee_backward_merge(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @callee_to_func_results(%arg0: tensor<8x8xf32>)
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{?}, {?}], unreduced={"x"}>})
func.func @callee_to_func_results(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @callee_to_func_results_merge(%arg0: tensor<8x8xf32>)
// CHECK-SAME:    -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], unreduced={"x"}>})
func.func @callee_to_func_results_merge(%arg0: tensor<8x8xf32>) -> (tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"x"}>]>} : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
