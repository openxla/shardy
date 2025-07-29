// RUN: mpmd_opt %s -split-input-file -mpmd-inline-nested-user-exposed-ops='assignment=f1@m1/1,f2@m2,f3@m1/2' -verify-diagnostics

func.func @nested_named_comp_assignment_different_mesh(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>} {
  %1 = mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<4x8xf32>
    // expected-error@+1 {{NamedComputation 'f2' is nested in a NamedComputation 'f1' which has a different mesh or stage assignment.}}
    %11 = mpmd.named_computation<"f2"> (%arg1, %0) (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
      %10 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
      mpmd.return %10 : tensor<4x8xf32>
    } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    mpmd.return %11 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// -----

func.func @nested_different_named_comp_assignment_different_stage(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>} {
  %1 = mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<4x8xf32>
    // expected-error@+1 {{NamedComputation 'f3' is nested in a NamedComputation 'f1' which has a different mesh or stage assignment.}}
    %11 = mpmd.named_computation<"f3"> (%arg1, %0) (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
      %10 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
      mpmd.return %10 : tensor<4x8xf32>
    } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    mpmd.return %11 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// -----

func.func @nested_named_computation_assignment_different_name(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>} {
  %1 = mpmd.named_computation<"f2"> (%arg0) (%arg1: tensor<4x8xf32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<4x8xf32>
    // expected-error@+1 {{NamedComputation 'f1' is nested in a NamedComputation 'f2' which has a different mesh or stage assignment.}}
    %11 = mpmd.named_computation<"f1"> (%arg1, %0) (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
      %10 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
      mpmd.return %10 : tensor<4x8xf32>
    } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    mpmd.return %11 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// -----

func.func @nested_named_computation_assignment_different_stage(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>} {
  %1 = mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>) {
    %0 = stablehlo.constant dense<0.0> : tensor<4x8xf32>
    // expected-error@+1 {{NamedComputation 'f3' is nested in a NamedComputation 'f1' which has a different mesh or stage assignment.}}
    %11 = mpmd.named_computation<"f3"> (%arg1, %0) (%arg3: tensor<4x8xf32>, %arg4: tensor<4x8xf32>) {
      %10 = stablehlo.add %arg3, %arg4 : tensor<4x8xf32>
      mpmd.return %10 : tensor<4x8xf32>
    } : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    mpmd.return %11 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %1 : tensor<4x8xf32>
}

// -----

func.func @nested_unary_mpmd_ops(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> attributes {
    "topology"=#mpmd.topology<<"m1": <["x"=2]>>,<"m2": <["x"=2]>>>} {
  %0 = mpmd.named_computation<"f1"> (%arg0) (%arg1: tensor<4x8xf32>) {
    // expected-error@+1 {{NamedTensor 'f2' is nested in a NamedComputation 'f1' which has a different mesh assignment.}}
    %1 = mpmd.named_tensor %arg1 name="f2" : tensor<4x8xf32>
    mpmd.return %1 : tensor<4x8xf32>
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
