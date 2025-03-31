// RUN: sdy_opt %s --sdy-constant-merger | FileCheck %s

// CHECK-LABEL: func @merge_constants_sdy
func.func @merge_constants_sdy() -> tensor<f32> {
  // CHECK: %[[C0:.*]] = sdy.constant dense<1.0{{.*}}> : tensor<f32>
  // CHECK-NOT: sdy.constant
  %0 = sdy.constant dense<1.000000e+00> : tensor<f32>
  %1 = sdy.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: stablehlo.add %[[C0]], %[[C0]] : tensor<f32>
  %2 = stablehlo.add %0, %1 : tensor<f32>
  return %2 : tensor<f32>
}

// CHECK-LABEL: func @merge_constants_constant_like
func.func @merge_constants_constant_like() -> tensor<f32> {
  // CHECK: %[[C0:.*]] = stablehlo.constant dense<1.0{{.*}}> : tensor<f32>
  // CHECK-NOT: stablehlo.constant
  %0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: stablehlo.add %[[C0]], %[[C0]] : tensor<f32>
  %2 = stablehlo.add %0, %1 : tensor<f32>
  return %2 : tensor<f32>
}

// CHECK-LABEL: func @dont_merge_constants_different_regions
func.func @dont_merge_constants_different_regions(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: %[[C0:.*]] = sdy.constant dense<1> : tensor<i32>
  // CHECK: stablehlo.while{{.*}} %[[C0]]
  %c = sdy.constant dense<1> : tensor<i32>
  %1 = stablehlo.while(%iterArg = %c) : tensor<i32>
   cond {
    // CHECK: %[[C1:.*]] = sdy.constant dense<1> : tensor<i32>
    // CHECK-NEXT: stablehlo.compare{{.*}} %[[C1]]
    %c_0 = sdy.constant dense<1> : tensor<i32>
    %2 = stablehlo.compare  LT, %iterArg, %c_0 : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %2 : tensor<i1>
   } do {
    // CHECK: %[[C2:.*]] = sdy.constant dense<1> : tensor<i32>
    // CHECK-NEXT: return %[[C2]]
    %c_0 = sdy.constant dense<1> : tensor<i32>
    stablehlo.return %c_0 : tensor<i32>
  }
  return %1 : tensor<i32>
}

// CHECK-LABEL: func @merge_constants_same_annotations
func.func @merge_constants_same_annotations() -> tensor<f32> {
  // CHECK: %[[C0:.*]] = sdy.constant {some.annotation = "AA"} dense<1.0{{.*}}> : tensor<f32>
  // CHECK-NOT: sdy.constant
  %0 = sdy.constant {some.annotation = "AA"} dense<1.000000e+00> : tensor<f32>
  %1 = sdy.constant {some.annotation = "AA"} dense<1.000000e+00> : tensor<f32>
  // CHECK: stablehlo.add %[[C0]], %[[C0]] : tensor<f32>
  %2 = stablehlo.add %0, %1 : tensor<f32>
  return %2 : tensor<f32>
}

// CHECK-LABEL: func @dont_merge_constants_different_annotations
func.func @dont_merge_constants_different_annotations() -> tensor<f32> {
  // CHECK: %[[C0:.*]] = sdy.constant {some.annotation = "AA"} dense<1.0{{.*}}> : tensor<f32>
  // CHECK: %[[C1:.*]] = sdy.constant {some.annotation = "BB"} dense<1.0{{.*}}> : tensor<f32>
  %0 = sdy.constant {some.annotation = "AA"} dense<1.000000e+00> : tensor<f32>
  %1 = sdy.constant {some.annotation = "BB"} dense<1.000000e+00> : tensor<f32>
  // CHECK: stablehlo.add %[[C0]], %[[C1]] : tensor<f32>
  %2 = stablehlo.add %0, %1 : tensor<f32>
  return %2 : tensor<f32>
}
