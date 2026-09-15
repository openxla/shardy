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
