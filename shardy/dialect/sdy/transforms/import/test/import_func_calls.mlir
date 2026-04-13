// RUN: sdy_opt --split-input-file %s -sdy-import-func-calls | FileCheck %s
// RUN: sdy_opt %s -split-input-file -sdy-import-func-calls -verify-diagnostics

// CHECK-LABEL: func @single_call_data_flow_edges_on_func_and_calls
func.func @single_call_data_flow_edges_on_func_and_calls(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32>) {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[EDGE0:.*]] = sdy.data_flow_edge %arg1 : tensor<8x2xi32>
  // CHECK-NEXT:   %[[NEGATE:.*]] = stablehlo.negate %[[EDGE0]] : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[NEGATE]] : tensor<8x2xi32>
  // CHECK-NEXT: } : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[EDGE1:.*]] = sdy.data_flow_edge %[[NC]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[EDGE1]] : tensor<8x2xi32>
  %0 = call @foo(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = sdy.data_flow_edge %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32>) {
  %0 = sdy.data_flow_edge %arg0 : tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}


