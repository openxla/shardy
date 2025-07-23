// RUN: mpmd_opt %s -mpmd-validate-named-ops-in-mpmd-func 2>&1 | FileCheck %s

// CHECK-LABEL: func @named_computation_inside_non_mpmd_function
func.func @named_computation_inside_non_mpmd_function(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
// CHECK-ERROR: {{.*}}{{Named computations can only be nested in mpmd functions or mpmd ops.}}
  %1 = mpmd.named_computation<"f1"> (%arg0) (%arg3: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg3, %arg3 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @named_computation_inside_non_mpmd_ops
func.func @named_computation_inside_non_mpmd_ops(%arg0: tensor<i32>) -> tensor<i32> {
// CHECK-ERROR: {{.*}}{{Named computations can only be nested in mpmd functions or mpmd ops.}}
  %0 = "stablehlo.case"(%arg0) ({
    %named_tensor = mpmd.named_tensor %arg0 name="tensor" : tensor<i32>
    stablehlo.return %named_tensor : tensor<i32>
  }, {
    stablehlo.return %arg0 : tensor<i32>
  }) : (tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func @named_tensor_inside_non_mpmd_function
func.func @named_tensor_inside_non_mpmd_function(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
// CHECK-ERROR: {{.*}}{{Named tensors can only be nested in mpmd functions or mpmd ops.}}
  %0 = mpmd.named_tensor %arg0 name="tensor" : tensor<4x8xf32>
  return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @named_tensor_inside_non_mpmd_ops
func.func @named_tensor_inside_non_mpmd_ops(%arg0: tensor<i32>) -> tensor<i32> attributes {} {
// CHECK-ERROR: {{.*}}{{Named tensors can only be nested in mpmd functions or mpmd ops.}}
  %0 = "stablehlo.case"(%arg0) ({
    %named_tensor = mpmd.named_tensor %arg0 name="tensor" : tensor<i32>
    stablehlo.return %named_tensor : tensor<i32>
  }, {
    stablehlo.return %arg0 : tensor<i32>
  }) : (tensor<i32>) -> tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: func @named_tensor_inside_named_computation_is_ok
func.func @named_tensor_inside_named_computation_is_ok(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %1 = mpmd.named_computation<"f1"> (%arg0) (%arg3: tensor<4x8xf32>) {
    %named_tensor = mpmd.named_tensor %arg3 name="tensor" : tensor<4x8xf32>
    %10 = stablehlo.add %named_tensor, %arg3 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @named_computation_inside_named_computation_is_ok
func.func @named_computation_inside_named_computation_is_ok(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %1 = mpmd.named_computation<"outer"> (%arg0) (%arg3: tensor<4x8xf32>) {
    %inner_named_computation = mpmd.named_computation<"inner"> (%arg3) (%arg4: tensor<4x8xf32>) {
      mpmd.return %arg4 : tensor<4x8xf32>
    } : (tensor<4x8xf32>) -> tensor<4x8xf32>
    %10 = stablehlo.add %inner_named_computation, %arg3 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>
// CHECK-LABEL: func @named_computation_inside_mpmd_function_is_ok
func.func @named_computation_inside_mpmd_function_is_ok(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#topology} {
  %1 = mpmd.named_computation<"f1"> (%arg0, %arg0) (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
    %10 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
    mpmd.return %10 : tensor<4x8xf32>
  } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

