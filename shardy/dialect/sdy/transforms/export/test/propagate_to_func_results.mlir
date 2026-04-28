// RUN: sdy_opt %s -sdy-propagate-to-func-results -split-input-file | FileCheck %s

// test: simple
sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} :  tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// test: terminator value has no sharding, func result does not have either.
sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> tensor<8xf32> {
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// test: terminator value has no sharding, func result has.
sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> tensor<8xf32> {
func.func private @foo(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// test: terminator value and func result have different shardings.
sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
func.func private @foo(%arg0: tensor<8xf32>) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// test: terminator value is func argument.
sdy.mesh @mesh = <["x"=2]>

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = call @foo(%arg0) : (tensor<8xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
func.func private @foo(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<8xf32> {
  return %arg0 : tensor<8xf32>
}

// -----

// test: func has multiple results
sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0:2 = call @foo(%arg0) : (tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) {
func.func private @foo(%arg0: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  %1 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}]>]>} : tensor<8xf32>
  return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// -----

// test: func has multiple results from the same value.
sdy.mesh @mesh = <["x"=2, "y"=2]>

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0:2 = call @foo(%arg0) : (tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>)
  return %0#0 : tensor<8xf32>
}

// CHECK-LABEL: func private @foo
// CHECK-SAME:  -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}
// CHECK-SAME:      tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
func.func private @foo(%arg0: tensor<8xf32>) -> (tensor<8xf32>, tensor<8xf32>) {
  %0 = stablehlo.abs %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<8xf32>
  return %0, %0 : tensor<8xf32>, tensor<8xf32>
}
