// RUN: sdy_opt %s -split-input-file -sdy-propagation-pipeline='dedup-functions-fully=true' | FileCheck %s

sdy.mesh @mesh = <["a"=2, "b"=2]>

// test: a non flat call graph. main calls foo and bar, foo calls bar. two bar calls have different shardings.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL1]] : tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @bar(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT:  %[[ABS0:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT:  %[[RESHARD:.*]] = sdy.reshard %[[ABS0]] <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT:  %[[ABS1:.*]] = stablehlo.abs %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT:  %[[CALL:.*]] = call @bar(%[[ABS1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT:  return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0 : tensor<8x2xi32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  %3 = stablehlo.abs %2 : tensor<8x2xi32>
  %4 = call @bar(%3) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %4 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// test: a non flat call graph. main calls foo and bar, foo calls bar. two bar calls have different shardings.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL1]] : tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @bar(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT:  %[[ABS:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT:  %[[CALL:.*]] = call @bar(%[[ABS]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT:  return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0 : tensor<8x2xi32>
  %1 = call @bar(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2]>

// test: a non flat call graph. main calls foo and bar, foo calls bar. two bar calls have different shardings.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT:  %[[ABS0:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT:  %[[RESHARD:.*]] = sdy.reshard %[[ABS0]] <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT:  %[[ABS1:.*]] = stablehlo.abs %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT:  %[[CALL0:.*]] = call @bar(%[[ABS1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT:  %[[CALL1:.*]] = call @bar(%[[ABS1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT:  return %[[CALL0]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0 : tensor<8x2xi32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  %3 = stablehlo.abs %2 : tensor<8x2xi32>
  %4 = call @bar(%3) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %5 = call @bar(%3) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %4 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// test: a non flat call graph. main calls foo and bar, foo calls bar. two bar calls have different shardings.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[CALL0]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL1]] : tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @bar(%1) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT:  %[[ABS:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT:  %[[CALL:.*]] = call @bar(%[[ABS]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT:  return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0 : tensor<8x2xi32>
  %1 = call @bar(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}

// -----
sdy.mesh @mesh = <["a"=2, "b"=2]>

// test: a non flat call graph. main calls foo and bar, foo calls bar. two bar calls have different shardings.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[CALL0]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL1]] : tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @bar(%1) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT:  %[[ABS0:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT:  %[[RESHARD:.*]] = sdy.reshard %[[ABS0]] <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT:  %[[ABS1:.*]] = stablehlo.abs %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT:  %[[CALL:.*]] = call @bar(%[[ABS1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT:  return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0 : tensor<8x2xi32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  %2 = sdy.sharding_constraint %0 <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  %3 = stablehlo.abs %2 : tensor<8x2xi32>
  %4 = call @bar(%3) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %4 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// test: a non flat call graph. main calls foo and bar, foo calls bar. two bar calls have different shardings.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> (tensor<8x2xi32>, tensor<8x2xi32>) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[ADD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[ABS]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL0]], %[[CALL1]] : tensor<8x2xi32>, tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = stablehlo.abs %0 : tensor<8x2xi32>
  %3 = call @bar(%2) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1, %3 : tensor<8x2xi32>, tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT:  %[[ABS0:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT:  %[[RESHARD:.*]] = sdy.reshard %[[ABS0]] <@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT:  %[[ABS1:.*]] = stablehlo.abs %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT:  %[[CALL:.*]] = call @bar(%[[ABS1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT:  return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0 : tensor<8x2xi32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  %2 = stablehlo.abs %1 : tensor<8x2xi32>
  %3 = call @bar(%2) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %3 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8x2xi32>
// CHECK-SAME:      {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// test: a non flat call graph. main calls foo and bar, foo calls bar. two bar calls have different shardings.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}, tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> (tensor<8x2xi32>, tensor<8x2xi32>) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%[[ADD]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %[[ADD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%[[ABS]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL0]], %[[CALL1]] : tensor<8x2xi32>, tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  %1 = call @foo(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = stablehlo.abs %0 : tensor<8x2xi32>
  %3 = call @bar(%2) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1, %3 : tensor<8x2xi32>, tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT:  %[[ABS0:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT:  %[[RESHARD:.*]] = sdy.reshard %[[ABS0]] <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT:  %[[ABS1:.*]] = stablehlo.abs %[[RESHARD]] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT:  %[[CALL:.*]] = call @bar(%[[ABS1]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT:  return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0 : tensor<8x2xi32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"b"}, {}]> : tensor<8x2xi32>
  %2 = stablehlo.abs %1 : tensor<8x2xi32>
  %3 = call @bar(%2) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %3 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}, {}]>}) {
// CHECK-NEXT:    return %arg0 : tensor<8x2xi32>
// CHECK-NEXT:  }
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  return %arg0 : tensor<8x2xi32>
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

// Verifies the interaction between `-sdy-apply-sharding-constraints`,
// `-sdy-sharding-group-import`, and sharding group propagation.
sdy.mesh @mesh = <["a"=2]>

// CHECK-LABEL: func @sharding_group_on_value_with_sharding_constraint
func.func @sharding_group_on_value_with_sharding_constraint(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) -> (tensor<16x16xf32>, tensor<16x16xf32>) {
  // CHECK: %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>}
  %0 = stablehlo.add %arg0, %arg0 : tensor<16x16xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"a"}]> : tensor<16x16xf32>
  // CHECK: %2 = stablehlo.add %arg1, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"a"}]>]>}
  %2 = stablehlo.add %arg1, %arg1 : tensor<16x16xf32>
  sdy.sharding_group %0 group_id = 1234 : tensor<16x16xf32>
  sdy.sharding_group %0 group_id = 2345 : tensor<16x16xf32>
  sdy.sharding_group %2 group_id = 2345 : tensor<16x16xf32>
  sdy.sharding_group %2 group_id = 3456 : tensor<16x16xf32>
  return %1, %2 : tensor<16x16xf32>, tensor<16x16xf32>
}

// -----

// Verifies the interaction between the `-sdy-add-data-flow-edges` pass and
// sharding group propagation.
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
    %3 = stablehlo.compare LT, %iterArg_2, %comp : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %3 : tensor<i1>
  } do {
    %3 = stablehlo.add %iterArg_2, %inc : tensor<i32>
    %4 = stablehlo.add %iterArg, %iterArg : tensor<16x16xf32>
    stablehlo.return %4, %3 : tensor<16x16xf32>, tensor<i32>
  }
  sdy.sharding_group %1#0 group_id = 50 : tensor<16x16xf32>

  // Add a value with an explicit sharding to group_id=50 which will apply an
  // initial sharding to the result of the WhileOp outside of the loop.
  %2 = sdy.sharding_constraint %arg1 <@mesh, [{"a"}, {}]> : tensor<16x16xf32>
  sdy.sharding_group %2 group_id = 50 : tensor<16x16xf32>
  return %1#0, %2 : tensor<16x16xf32>, tensor<16x16xf32>
}

// -----

sdy.mesh @mesh = <["a"=2, "b"=2]>

// Verifies sharding group import and propagation across non-flat function calls.
// CHECK-LABEL: func @main(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
// CHECK-SAME:      %arg1: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>},
// CHECK-SAME:          tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
func.func @main(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>) {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL0:.*]] = call @foo(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL1:.*]] = call @bar(%arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[ADD]], %[[CALL1]] : tensor<8x2xi32>, tensor<8x2xi32>
  %0 = stablehlo.add %arg0, %arg0 : tensor<8x2xi32>
  sdy.sharding_group %0 group_id = 1234 : tensor<8x2xi32>
  sdy.sharding_group %0 group_id = 2345 : tensor<8x2xi32>
  %1 = call @foo(%arg1) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = call @bar(%arg1) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0, %2 : tensor<8x2xi32>, tensor<8x2xi32>
}

// CHECK-LABEL: func private @foo(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[CALL:.*]] = call @bar(%[[ABS]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0 : tensor<8x2xi32>
  %1 = call @bar(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-LABEL: func private @bar(
// CHECK-SAME:      %arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>})
// CHECK-SAME:      -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) {
func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[ABS:.*]] = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : tensor<8x2xi32>
  // CHECK-NEXT: %[[RESHARD:.*]] = sdy.reshard %[[ABS]] <@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  // CHECK-NEXT: return %[[RESHARD]] : tensor<8x2xi32>
  %0 = stablehlo.abs %arg0 : tensor<8x2xi32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"a"}, {}]> : tensor<8x2xi32>
  sdy.sharding_group %0 group_id = 2345 : tensor<8x2xi32>
  sdy.sharding_group %0 group_id = 3456 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// -----

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) {
  // CHECK-NEXT: return
  sdy.sharding_group %arg0 group_id = 1234 : tensor<8x8xf32>
  sdy.sharding_group %arg0 group_id = 2345 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 1234 : tensor<8x8xf32>
  sdy.sharding_group %arg1 group_id = 3456 : tensor<8x8xf32>
  func.return
}
