// RUN: sdy_opt %s --sdy-constant-or-scalar-merger | FileCheck %s

sdy.mesh @mesh = <["x"=2, "y"=2]>

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

// CHECK-LABEL: func @dont_merge_non_scalar_constants_different_annotations
func.func @dont_merge_non_scalar_constants_different_annotations() -> tensor<32xf32> {
  // CHECK: %[[C0:.*]] = sdy.constant {some.annotation = "AA"} dense<1.0{{.*}}> : tensor<32xf32>
  // CHECK: %[[C1:.*]] = sdy.constant {some.annotation = "BB"} dense<1.0{{.*}}> : tensor<32xf32>
  %0 = sdy.constant {some.annotation = "AA"} dense<1.000000e+00> : tensor<32xf32>
  %1 = sdy.constant {some.annotation = "BB"} dense<1.000000e+00> : tensor<32xf32>
  // CHECK: stablehlo.add %[[C0]], %[[C1]] : tensor<32xf32>
  %2 = stablehlo.add %0, %1 : tensor<32xf32>
  return %2 : tensor<32xf32>
}

// CHECK-LABEL: func @merge_scalar_constants_same_annotations
func.func @merge_scalar_constants_same_annotations() -> tensor<f32> {
  // CHECK: %[[C0:.*]] = sdy.constant {some.annotation = "AA"} dense<1.0{{.*}}> : tensor<f32>
  // CHECK-NOT: sdy.constant
  %0 = sdy.constant {some.annotation = "AA"} dense<1.000000e+00> : tensor<f32>
  %1 = sdy.constant {some.annotation = "AA"} dense<1.000000e+00> : tensor<f32>
  // CHECK: stablehlo.add %[[C0]], %[[C0]] : tensor<f32>
  %2 = stablehlo.add %0, %1 : tensor<f32>
  return %2 : tensor<f32>
}

// CHECK-LABEL: func @dont_merge_broadcasts_different_regions
func.func @dont_merge_broadcasts_different_regions(%arg0: tensor<f32>, %arg1: tensor<i32>) -> tensor<2x64xf32> {
  // CHECK: %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %1 = "stablehlo.case"(%arg1) ({
  // CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT:   stablehlo.return %3 : tensor<2x64xf32>
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   %3 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT:   %4 = stablehlo.add %0, %3 : tensor<2x64xf32>
  // CHECK-NEXT:   stablehlo.return %4 : tensor<2x64xf32>
  // CHECK-NEXT: }) : (tensor<i32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.add %0, %1 : tensor<2x64xf32>
  // CHECK-NEXT: return %2 : tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %1 = "stablehlo.case"(%arg1) ({
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    stablehlo.return %3 : tensor<2x64xf32>
  }, {
    %3 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
    %4 = stablehlo.add %0, %3 : tensor<2x64xf32>
    stablehlo.return %4 : tensor<2x64xf32>
  }) : (tensor<i32>) -> tensor<2x64xf32>
  %2 = stablehlo.add %0, %1 : tensor<2x64xf32>
  return %2 : tensor<2x64xf32>
}

// CHECK-LABEL: func @merge_broadcasts_same_annotations
func.func @merge_broadcasts_same_annotations(%arg0: tensor<f32>) -> tensor<2x64xf32> {
  // CHECK: %[[C0:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [] {some.annotation = "AA"} : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NOT: stablehlo.broadcast_in_dim
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] {some.annotation = "AA"} : (tensor<f32>) -> tensor<2x64xf32>
  %1 = stablehlo.broadcast_in_dim %arg0, dims = [] {some.annotation = "AA"} : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK: stablehlo.add %[[C0]], %[[C0]] : tensor<2x64xf32>
  %2 = stablehlo.add %0, %1 : tensor<2x64xf32>
  return %2 : tensor<2x64xf32>
}

// CHECK-LABEL: func @dont_merge_broadcasts_different_annotations
func.func @dont_merge_broadcasts_different_annotations(%arg0: tensor<f32>) -> tensor<2x64xf32> {
  // CHECK: %[[C0:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [] {some.annotation = "AA"} : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK: %[[C1:.*]] = stablehlo.broadcast_in_dim %arg0, dims = [] {some.annotation = "BB"} : (tensor<f32>) -> tensor<2x64xf32>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [] {some.annotation = "AA"} : (tensor<f32>) -> tensor<2x64xf32>
  %1 = stablehlo.broadcast_in_dim %arg0, dims = [] {some.annotation = "BB"} : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK: stablehlo.add %[[C0]], %[[C1]] : tensor<2x64xf32>
  %2 = stablehlo.add %0, %1 : tensor<2x64xf32>
  return %2 : tensor<2x64xf32>
}

// CHECK-LABEL: func @merge_broadcasts
func.func @merge_broadcasts() -> tensor<2x64xf32> {
  // CHECK: %0 = sdy.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.add %1, %1 : tensor<2x64xf32>
  // CHECK-NEXT: return %2 : tensor<2x64xf32>
  %0 = sdy.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %2 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %3 = stablehlo.add %1, %2 : tensor<2x64xf32>
  return %3 : tensor<2x64xf32>
}

// CHECK-LABEL: func @does_not_merge_broadcasts_different_sharding
func.func @does_not_merge_broadcasts_different_sharding() -> tensor<2x64xf32> {
  // CHECK: %0 = sdy.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %0, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.add %1, %2 : tensor<2x64xf32>
  // CHECK-NEXT: return %3 : tensor<2x64xf32>
  %0 = sdy.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %2 = stablehlo.broadcast_in_dim %0, dims = [] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>} : (tensor<f32>) -> tensor<2x64xf32>
  %3 = stablehlo.add %1, %2 : tensor<2x64xf32>
  return %3 : tensor<2x64xf32>
}

// CHECK-LABEL: func @does_not_merge_broadcasts_with_different_operands
func.func @does_not_merge_broadcasts_with_different_operands(%arg0: tensor<f32>) -> tensor<2x64xf32> {
  // CHECK: %0 = sdy.constant dense<1.000000e+00> : tensor<f32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.add %1, %2 : tensor<2x64xf32>
  // CHECK-NEXT: return %3 : tensor<2x64xf32>
  %0 = sdy.constant dense<1.000000e+00> : tensor<f32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %2 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<2x64xf32>
  %3 = stablehlo.add %1, %2 : tensor<2x64xf32>
  return %3 : tensor<2x64xf32>
}

// CHECK-LABEL: func @does_not_merge_broadcasts_on_non_scalars
func.func @does_not_merge_broadcasts_on_non_scalars() -> tensor<2x64xf32> {
  // CHECK: %0 = sdy.constant dense<1.000000e+00> : tensor<2xf32>
  // CHECK-NEXT: %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<2xf32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %2 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<2xf32>) -> tensor<2x64xf32>
  // CHECK-NEXT: %3 = stablehlo.add %1, %2 : tensor<2x64xf32>
  // CHECK-NEXT: return %3 : tensor<2x64xf32>
  %0 = sdy.constant dense<1.000000e+00> : tensor<2xf32>
  %1 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<2xf32>) -> tensor<2x64xf32>
  %2 = stablehlo.broadcast_in_dim %0, dims = [0] : (tensor<2xf32>) -> tensor<2x64xf32>
  %3 = stablehlo.add %1, %2 : tensor<2x64xf32>
  return %3 : tensor<2x64xf32>
}
