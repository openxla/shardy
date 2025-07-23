// RUN: mpmd_opt %s -mpmd-inline-nested-user-exposed-ops='assignment=f1@m1,f2@m1' 2>&1 | FileCheck %s


// CHECK-LABEL: func @nested_named_comp_assignment
func.func @nested_named_comp_assignment(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>} {
// CHECK-NEXT: mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>)
// CHECK-NEXT:   %[[CONST:.*]] = stablehlo.constant dense<1.000000e+00>
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %[[CONST]]
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
  %1 = mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>) {
    %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
    %11 = mpmd.named_computation<"f2"> (%arg1, %0) (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
      %10 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
      mpmd.return %10 : tensor<4x8xf32>
    } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    mpmd.return %11 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @nested_named_comp_may_be_unassigned
func.func @nested_named_comp_may_be_unassigned(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>} {
// CHECK-NEXT: mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>)
// CHECK-NEXT:   %[[CONST:.*]] = stablehlo.constant dense<1.000000e+00>
// CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %[[CONST]]
// CHECK-NEXT:   mpmd.return %[[ADD]]
// CHECK-NEXT: }
  %1 = mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>) {
    %0 = stablehlo.constant dense<1.0> : tensor<4x8xf32>
    %11 = mpmd.named_computation<"a_name_not_in_the_assignment"> (%arg1, %0) (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
      %10 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
      mpmd.return %10 : tensor<4x8xf32>
    } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    mpmd.return %11 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// CHECK-LABEL: func @nested_unary_mpmd_ops
func.func @nested_unary_mpmd_ops(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>} {
// CHECK-NEXT: mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>)
// CHECK-NEXT:   mpmd.return %arg1
// CHECK-NEXT: }
  %0 = mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>) {
    %1 = mpmd.named_tensor %arg1 name="f2" : tensor<4x8xf32> // Same assignment as the parent.
    %2 = mpmd.named_tensor %1 name="f3" : tensor<4x8xf32>  // Assignment not in the user assignment map.
    %3 = mpmd.broadcast %2 : tensor<4x8xf32>
    %4 = mpmd.reduce<none> %3 : (tensor<4x8xf32>) -> tensor<4x8xf32>
    mpmd.return %4 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// CHECK-LABEL: func @doubly_nested_named_computation
func.func @doubly_nested_named_computation(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>} {
// CHECK-NEXT: mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>)
// CHECK-NEXT:   mpmd.return %arg1
// CHECK-NEXT: }
  %0 = mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>) {
    %1 = mpmd.named_computation<"f1"> (%arg1) (%arg2: tensor<4x8xf32>) {
      %2 = mpmd.named_computation<"f1"> (%arg2) (%arg3: tensor<4x8xf32>) {
        mpmd.return %arg3 : tensor<4x8xf32>
      }: (tensor<4x8xf32>) -> tensor<4x8xf32>
      mpmd.return %2 : tensor<4x8xf32>
    } : (tensor<4x8xf32>) -> tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
