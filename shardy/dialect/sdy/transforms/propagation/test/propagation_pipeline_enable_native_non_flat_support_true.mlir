// RUN: sdy_opt %s -split-input-file -sdy-propagation-pipeline='dedup-functions-fully=true' | FileCheck %s

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

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }

// CHECK-NOT: func private @bar(%arg0: tensor<8x2xi32>
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[RESHARD0:.*]] = sdy.reshard %[[ABS]] <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT: %[[RESHARD1:.*]] = sdy.reshard %[[RESHARD0]] <@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @bar_0(%[[RESHARD1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[RESHARD2:.*]] = sdy.reshard %[[CALL]] <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT: return %[[RESHARD2]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0 : tensor<8x2xi32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  %3 = call @bar(%2) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %3 : tensor<8x2xi32>
}
