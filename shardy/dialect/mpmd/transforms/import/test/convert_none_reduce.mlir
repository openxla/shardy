// RUN: mpmd_opt %s -mpmd-infer-mesh-convert-reduce-ops 2>&1 | FileCheck --implicit-check-not mpmd.reduce %s

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=2]>>, <"m3": <["z"=2]>>>

// CHECK-LABEL: func @convert_none_reduce_to_actual_reduce
func.func @convert_none_reduce_to_actual_reduce(%arg0: tensor<4x8xf32> , %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
  // CHECK: %[[REDUCE:.*]] = mpmd.reduce<max>  %arg0, %arg1
  // CHECK-NEXT: %[[REDUCE_NONE:.*]] = mpmd.reduce<none>  %[[REDUCE]]
  // CHECK-NEXT: return %[[REDUCE_NONE]]
  %0 = stablehlo.maximum %arg0, %arg1 {mpmd.reduce = #mpmd.reduction<max>} : tensor<4x8xf32>
  %1 = mpmd.reduce<none> %0 : (tensor<4x8xf32>) -> (tensor<4x8xf32>)
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @cannot_convert_none_reduce_to_actual_reduce_if_reduction_type_not_supported
func.func @cannot_convert_none_reduce_to_actual_reduce_if_reduction_type_not_supported(%arg0: tensor<4x8xf32> , %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
  // CHECK: %[[SUBTRACT:.*]] = stablehlo.subtract %arg0, %arg1
  // CHECK: %[[REDUCE:.*]] = mpmd.reduce<none> %[[SUBTRACT]]
  // CHECK: return %[[REDUCE]]
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<4x8xf32>
  %1 = mpmd.reduce<none> %0 : (tensor<4x8xf32>) -> (tensor<4x8xf32>)
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @do_not_delete_single_none_reduce
func.func @do_not_delete_single_none_reduce(%arg0: tensor<4x8xf32>)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
  // CHECK: %[[REDUCE:.*]] = mpmd.reduce<none> %arg0
  %0 = mpmd.reduce<none> %arg0 : (tensor<4x8xf32>) -> (tensor<4x8xf32>)
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @chain_of_reduces
// CHECK: %[[REDUCE:.*]] = mpmd.reduce<max> %arg0, %arg1, %arg1
// CHECK-NEXT: %[[REDUCE_NONE:.*]] = mpmd.reduce<none>  %[[REDUCE]]
// CHECK-NEXT: return %[[REDUCE_NONE]]
func.func @chain_of_reduces(%arg0: tensor<4x8xf32> , %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
  %0 = stablehlo.maximum %arg0, %arg1 {mpmd.reduce = #mpmd.reduction<max>} : tensor<4x8xf32>
  %1 = stablehlo.maximum %0, %arg1 {mpmd.reduce = #mpmd.reduction<max>} : tensor<4x8xf32>
  %2 = mpmd.reduce<none> %1 : (tensor<4x8xf32>) -> (tensor<4x8xf32>)
  func.return %2 : tensor<4x8xf32>
}

// CHECK-LABEL: func @chain_of_reduces_partially_fold_if_different_type
// CHECK: %[[MIN:.*]] = stablehlo.minimum %arg0, %arg1
// CHECK: %[[REDUCE:.*]] = mpmd.reduce<max> %[[MIN]], %[[MIN]], %[[MIN]], %arg1
// CHECK-NEXT: %[[REDUCE_NONE:.*]] = mpmd.reduce<none>  %[[REDUCE]]
// CHECK-NEXT: return %[[REDUCE_NONE]], %[[MIN]]
func.func @chain_of_reduces_partially_fold_if_different_type(%arg0: tensor<4x8xf32> , %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>, tensor<4x8xf32>) attributes {topology=#topology} {
  %0 = stablehlo.minimum %arg0, %arg1: tensor<4x8xf32>
  %another_max = stablehlo.maximum %0, %arg1 {mpmd.reduce = #mpmd.reduction<max>} : tensor<4x8xf32>
  %max = stablehlo.maximum %0, %another_max {mpmd.reduce = #mpmd.reduction<max>} : tensor<4x8xf32>
  %1 = stablehlo.maximum %0, %max {mpmd.reduce = #mpmd.reduction<max>} : tensor<4x8xf32>
  %2 = mpmd.reduce<none> %1 : (tensor<4x8xf32>) -> (tensor<4x8xf32>)
  func.return %2, %0 : tensor<4x8xf32>, tensor<4x8xf32>
}

// CHECK-LABEL: func @push_through_unary_ops
func.func @push_through_unary_ops(%arg0: tensor<4x8xf32> , %arg1: tensor<4x8xf32>)
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
// CHECK:  %[[REDUCE:.*]] = mpmd.reduce<max>  %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK:  %[[SQRT:.*]] = stablehlo.sqrt %[[REDUCE]] : tensor<4x8xf32>
// CHECK:  %[[ABS:.*]] = stablehlo.abs %[[SQRT]] : tensor<4x8xf32>
// CHECK:  %[[REDUCE_NONE:.*]] = mpmd.reduce<none>  %[[ABS]] : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = stablehlo.maximum %arg0, %arg1 {mpmd.reduce = #mpmd.reduction<max>}: tensor<4x8xf32>
  %sqrt = stablehlo.sqrt %0 : tensor<4x8xf32>
  %abs = stablehlo.abs %sqrt : tensor<4x8xf32>
  %1 = mpmd.reduce<none> %abs : (tensor<4x8xf32>) -> (tensor<4x8xf32>)
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @concat_reduce
func.func @concat_reduce(%arg0: tensor<4x1x8xf32>, %arg1: tensor<4x1x8xf32>)
// CHECK-NEXT: stablehlo.constant
// CHECK-NEXT: %[[R0:.*]] = stablehlo.reshape %arg0 : (tensor<4x1x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[R1:.*]] = stablehlo.reshape %arg1 : (tensor<4x1x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[R2:.*]] = stablehlo.reshape %arg0 : (tensor<4x1x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[R3:.*]] = stablehlo.reshape %arg1 : (tensor<4x1x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[MAX:.*]] = mpmd.reduce<max> %[[R0]], %[[R1]], %[[R2]], %[[R3]] :
// CHECK-SAME:     (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: %[[REDUCE_NONE:.*]] = mpmd.reduce<none> %[[MAX]] : (tensor<4x8xf32>) -> tensor<4x8xf32>
// CHECK-NEXT: return %[[REDUCE_NONE]]
  -> (tensor<4x8xf32>) attributes {topology=#topology} {
  %init = stablehlo.constant dense<1.0> : tensor<f32>
  %concat = "stablehlo.concatenate"(%arg0, %arg1, %arg0, %arg1) <{dimension = 1 : i64}> :
    (tensor<4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>, tensor<4x1x8xf32>) -> tensor<4x4x8xf32>
  %reduce = stablehlo.reduce(%concat init: %init) applies stablehlo.maximum across dimensions = [1]
    {mpmd.reduce = #mpmd.reduction<max>} :
    (tensor<4x4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  %reduce_none = mpmd.reduce<none> %reduce : (tensor<4x8xf32>) -> (tensor<4x8xf32>)
  func.return %reduce_none : tensor<4x8xf32>
}
