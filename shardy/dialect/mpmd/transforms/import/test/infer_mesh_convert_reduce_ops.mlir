// RUN: mpmd_opt %s -mpmd-infer-mesh-convert-reduce-ops='infer-cross-mesh-reductions=true' 2>&1 | FileCheck --implicit-check-not mpmd.reduce %s

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>>


// CHECK-LABEL: func @simple_reduce_chain
func.func @simple_reduce_chain(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
// CHECK-NEXT: %[[R:.*]] = mpmd.reduce<add> {mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} %arg0, %arg1, %arg2 : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: return %[[R]]
  %4 = stablehlo.add %arg0, %arg1 {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  %5 = stablehlo.add %4, %arg2 {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  func.return %5 : tensor<4x8xf32>
}

// CHECK-LABEL: func @reduce_of_reduces
func.func @reduce_of_reduces(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
// CHECK-NEXT: %[[R:.*]] = mpmd.reduce<max> {mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} %arg0, %arg1, %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: return %[[R]]
  %41 = stablehlo.maximum %arg0, %arg1 {mpmd.reduce = #mpmd.reduction<max>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  %42 = stablehlo.maximum %arg0, %arg1 {mpmd.reduce = #mpmd.reduction<max>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  %5 = stablehlo.maximum %41, %42 {mpmd.reduce = #mpmd.reduction<max>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  func.return %5 : tensor<4x8xf32>
}

// CHECK-LABEL: func @different_reduces_not_flattened
func.func @different_reduces_not_flattened(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
// CHECK-NEXT: %[[ADD:.*]] = mpmd.reduce<add> {mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[MAX:.*]] = mpmd.reduce<max> {mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} %arg0, %arg1, %[[ADD]] : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: return %[[MAX]]
  %41 = stablehlo.maximum %arg0, %arg1 {mpmd.reduce = #mpmd.reduction<max>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  %42 = stablehlo.add %arg0, %arg1 {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  %5 = stablehlo.maximum %41, %42 {mpmd.reduce = #mpmd.reduction<max>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  func.return %5 : tensor<4x8xf32>
}

// CHECK-LABEL: func @intermediate_reduce_multiple_users
// We flatten users, but don't replace them if they still have users.
func.func @intermediate_reduce_multiple_users(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>, tensor<4x8xf32>) attributes {topology=#topology} {
// CHECK-NEXT: %[[ADD0:.*]] = mpmd.reduce<add> {mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[ADD1:.*]] = mpmd.reduce<add> {mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} %arg0, %arg1, %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: return %[[ADD0]], %[[ADD1]]
  %41 = stablehlo.add %arg0, %arg1 {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  %42 = stablehlo.add %arg0, %arg1 {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  %5 = stablehlo.add %41, %42 {mpmd.reduce = #mpmd.reduction<add>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} : tensor<4x8xf32>
  func.return %42, %5 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @concat_reduce
func.func @concat_reduce(%arg0: tensor<4x1x8xf32>, %arg1: tensor<4x1x8xf32>)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
// CHECK-NEXT: stablehlo.constant
// CHECK-NEXT: %[[R0:.*]] = stablehlo.reshape %arg0 : (tensor<4x1x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[R1:.*]] = stablehlo.reshape %arg1 : (tensor<4x1x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[R2:.*]] = stablehlo.reshape %arg0 : (tensor<4x1x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[R3:.*]] = stablehlo.reshape %arg1 : (tensor<4x1x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[MAX:.*]] = mpmd.reduce<max> {mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} %[[R0]], %[[R1]], %[[R2]], %[[R3]] :
// CHECK-SAME:     (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: return %[[MAX]]
  %init = stablehlo.constant dense<1.0> : tensor<f32>
  %concat = "stablehlo.concatenate"(%arg0, %arg1, %arg0, %arg1) <{dimension = 1 : i64}>
    {mpmd.reduce = #mpmd.reduction<max>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} :
    (tensor<4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>) -> tensor<4x4x8xf32>
  %reduce = stablehlo.reduce(%concat init: %init) applies stablehlo.maximum across dimensions = [1]
    {mpmd.reduce = #mpmd.reduction<max>, mpmd.src_set = #mpmd.meshes_with_origins<"m1", "m2">} :
    (tensor<4x4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  func.return %reduce : tensor<4x8xf32>
}

