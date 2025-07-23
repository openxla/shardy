// RUN: mpmd_opt %s -mpmd-sink-negligible-ops-into-call-op -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: func @main
func.func @main(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i32>) attributes {
  "topology"=#mpmd.topology<<"mesh1": <["z"=2]>>>
} {
  // CHECK-NEXT: mpmd.call @fn(%arg0)
  // CHECK-NEXT: mpmd.call @fn(%arg1)
  // CHECK-NEXT: return
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = mpmd.call @fn(%0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = mpmd.call @fn(%0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1, %2 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func private @fn(%arg0: tensor<i32>)
func.func private @fn(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["z"=2]>>>
} {
  // CHECK-NEXT: %[[C:.*]] = stablehlo.constant
  // CHECK-NEXT: add %[[C]], %arg0
  %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// The op can be sunk, but it can't be DCE's as it's used by another op.

// CHECK-LABEL: func @main
func.func @main(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>) attributes {
  "topology"=#mpmd.topology<<"mesh1": <["z"=2]>>>
} {
  // CHECK-NEXT: %[[C:.*]] = stablehlo.constant
  // CHECK-NEXT: mpmd.call @fn(%arg0)
  // CHECK-NEXT: mpmd.call @fn(%arg1)
  // CHECK-NEXT: return %[[C]],
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = mpmd.call @fn(%0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = mpmd.call @fn(%0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0, %1, %2 : tensor<i32>, tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func private @fn(%arg0: tensor<i32>)
func.func private @fn(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["z"=2]>>>
} {
  // CHECK-NEXT: %[[C:.*]] = stablehlo.constant
  // CHECK-NEXT: add %[[C]], %arg0
  %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// An op with operands cannot be sunk.

// CHECK-LABEL: func @main
func.func @main(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i32>) attributes {
  "topology"=#mpmd.topology<<"mesh1": <["z"=2]>>>
} {
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %arg0, %arg1
  // CHECK-NEXT: mpmd.call @fn(%[[ADD]], %arg0)
  // CHECK-NEXT: mpmd.call @fn(%[[ADD]], %arg1)
  // CHECK-NEXT: return
  %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
  %1 = mpmd.call @fn(%0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = mpmd.call @fn(%0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1, %2 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func private @fn(%arg0: tensor<i32>, %arg1: tensor<i32>)
func.func private @fn(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["z"=2]>>>
} {
  // CHECK-NEXT: add %arg0, %arg1
  %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

// -----

// An op with multiple results cannot be sunk.

// CHECK-LABEL: func @main
func.func @main(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i32>) attributes {
  "topology"=#mpmd.topology<<"mesh1": <["z"=2]>>>
} {
  // CHECK-NEXT: %[[ADD:.*]]:2 = call @my_add
  // CHECK-NEXT: mpmd.call @fn(%[[ADD]]#0, %arg0)
  // CHECK-NEXT: mpmd.call @fn(%[[ADD]]#1, %arg1)
  // CHECK-NEXT: return
  %0:2 = call @my_add(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  %1 = mpmd.call @fn(%0#0, %arg0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = mpmd.call @fn(%0#1, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1, %2 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func private @fn(%arg0: tensor<i32>, %arg1: tensor<i32>)
func.func private @fn(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["z"=2]>>>
} {
  // CHECK-NEXT: add %arg0, %arg1
  %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}

func.func private @my_add(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
  return %0, %0 : tensor<i32>, tensor<i32>
}

// -----

// The op is not consistently used in all calls, so it cannot be sunk.

// CHECK-LABEL: func @main
func.func @main(%arg0 : tensor<i32>, %arg1 : tensor<i32>) -> (tensor<i32>, tensor<i32>) attributes {
  "topology"=#mpmd.topology<<"mesh1": <["z"=2]>>>
} {
  // CHECK-NEXT: %[[C:.*]] = stablehlo.constant
  // CHECK-NEXT: mpmd.call @fn(%arg0, %arg1)
  // CHECK-NEXT: mpmd.call @fn(%[[C]], %arg1)
  %0 = stablehlo.constant dense<0> : tensor<i32>
  %1 = mpmd.call @fn(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = mpmd.call @fn(%0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %1, %2 : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: func private @fn(%arg0: tensor<i32>, %arg1: tensor<i32>)
func.func private @fn(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> attributes {
    "topology"=#mpmd.topology<<"mesh1": <["z"=2]>>>
} {
  // CHECK-NEXT: add %arg0, %arg1
  %0 = stablehlo.add %arg0, %arg1 : tensor<i32>
  return %0 : tensor<i32>
}
