// RUN: sdy_opt %s -split-input-file -sdy-add-func-data-flow-edges -sdy-apply-sharding-constraints -sdy-sink-func-data-flow-edges -sdy-import-func-calls | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @foo(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8x8xf32>) {
  // CHECK-NEXT:   %[[DATA_FLOW_EDGE_0:.*]] = sdy.data_flow_edge %arg1 : tensor<8x8xf32>
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %[[DATA_FLOW_EDGE_0]], %[[DATA_FLOW_EDGE_0]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
  // CHECK-NEXT:   %[[WSC_0:.*]] = sdy.sharding_constraint %[[ADD]] <@mesh, [{"a"}, {}]>
  // CHECK-NEXT:   %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT:   sdy.return %[[WSC_1]] : tensor<8x8xf32>
  // CHECK-NEXT: } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[DATA_FLOW_EDGE_1:.*]] = sdy.data_flow_edge %[[NC]] : tensor<8x8xf32>
  // CHECK-NEXT: return %[[DATA_FLOW_EDGE_1]] : tensor<8x8xf32>
  %0 = call @foo(%arg0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @bar(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg0 :  tensor<8x8xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

func.func @foo(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = call @bar(%arg0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 :  tensor<8x8xf32>
  %2 = sdy.sharding_constraint %1 <@mesh, [{"a"}, {}]> :  tensor<8x8xf32>
  %3 = sdy.sharding_constraint %2 <@mesh, [{}, {"b"}]> :  tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[FOO:.*]] = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8x8xf32>) {
  // CHECK-NEXT:   %[[DATA_FLOW_EDGE_0:.*]] = sdy.data_flow_edge %arg1 : tensor<8x8xf32>
  // CHECK-NEXT:   %[[BAR:.*]] = sdy.named_computation<"bar">(%[[DATA_FLOW_EDGE_0]]) (%arg2: tensor<8x8xf32>) {
  // CHECK-NEXT:     %[[DATA_FLOW_EDGE_1:.*]] = sdy.data_flow_edge %arg2 : tensor<8x8xf32>
  // CHECK-NEXT:     %[[ADD_0:.*]] = stablehlo.add %[[DATA_FLOW_EDGE_1]], %[[DATA_FLOW_EDGE_1]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
  // CHECK-NEXT:     %[[WSC_0:.*]] = sdy.sharding_constraint %[[ADD_0]] <@mesh, [{"a"}, {}]>
  // CHECK-NEXT:     %[[WSC_1:.*]] = sdy.sharding_constraint %[[WSC_0]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT:     sdy.return %[[WSC_1]] : tensor<8x8xf32>
  // CHECK-NEXT:   } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:   %[[DATA_FLOW_EDGE_2:.*]] = sdy.data_flow_edge %[[BAR]] : tensor<8x8xf32>
  // CHECK-NEXT:   %[[ADD_1:.*]] = stablehlo.add %[[DATA_FLOW_EDGE_2]], %[[DATA_FLOW_EDGE_2]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>}
  // CHECK-NEXT:   %[[WSC_2:.*]] = sdy.sharding_constraint %[[ADD_1]] <@mesh, [{"a"}, {}]>
  // CHECK-NEXT:   %[[WSC_3:.*]] = sdy.sharding_constraint %[[WSC_2]] <@mesh, [{}, {"b"}]>
  // CHECK-NEXT:   sdy.return %[[WSC_3]] : tensor<8x8xf32>
  // CHECK-NEXT: } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[DATA_FLOW_EDGE_3:.*]] = sdy.data_flow_edge %[[FOO]] : tensor<8x8xf32>
  // CHECK-NEXT: return %[[DATA_FLOW_EDGE_3]] : tensor<8x8xf32>
  %0 = call @foo(%arg0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

func.func @bar(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x8xf32>
  %1 = sdy.sharding_constraint %arg0 <@mesh, [{"a"}, {}]> : tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

func.func @foo(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  %0 = call @bar(%arg0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  %1 = stablehlo.add %0, %0 : tensor<8x8xf32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  return %1 : tensor<8x8xf32>
}

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT: %[[FOO:.*]] = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8x8xf32>) {
  // CHECK-NEXT:   %[[DATA_FLOW_EDGE_0:.*]] = sdy.data_flow_edge %arg1 : tensor<8x8xf32>
  // CHECK-NEXT:   %[[BAR:.*]] = sdy.named_computation<"bar">(%[[DATA_FLOW_EDGE_0]]) (%arg2: tensor<8x8xf32>) {
  // CHECK-NEXT:     %[[DATA_FLOW_EDGE_1:.*]] = sdy.data_flow_edge %arg2 : tensor<8x8xf32>
  // CHECK-NEXT:     %[[WSC_0:.*]] = sdy.sharding_constraint %[[DATA_FLOW_EDGE_1]] <@mesh, [{"a"}, {}]> : tensor<8x8xf32>
  // CHECK-NEXT:     %[[ADD_0:.*]] = stablehlo.add %[[WSC_0]], %[[WSC_0]] : tensor<8x8xf32>
  // CHECK-NEXT:     sdy.return %[[ADD_0]] : tensor<8x8xf32>
  // CHECK-NEXT:   } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:   %[[DATA_FLOW_EDGE_2:.*]] = sdy.data_flow_edge %[[BAR]] : tensor<8x8xf32>
  // CHECK-NEXT:   %[[WSC_1:.*]] = sdy.sharding_constraint %[[DATA_FLOW_EDGE_2]] <@mesh, [{}, {"b"}]> : tensor<8x8xf32>
  // CHECK-NEXT:   %[[ADD_1:.*]] = stablehlo.add %[[WSC_1]], %[[WSC_1]] : tensor<8x8xf32>
  // CHECK-NEXT:   sdy.return %[[ADD_1]] : tensor<8x8xf32>
  // CHECK-NEXT: } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT: %[[DATA_FLOW_EDGE_3:.*]] = sdy.data_flow_edge %[[FOO]] : tensor<8x8xf32>
  // CHECK-NEXT: return %[[DATA_FLOW_EDGE_3]] : tensor<8x8xf32>
  %0 = call @foo(%arg0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

