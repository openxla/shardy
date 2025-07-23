// RUN: mpmd_opt %s -mpmd-import-pipeline -verify-diagnostics

#topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["x"=2]>>>
func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>) attributes {topology=#topology} {
  %c = stablehlo.constant dense<1> : tensor<i32>
  %c_0 = stablehlo.constant dense<10> : tensor<i32>
  %0:2 = stablehlo.while(%iterArg = %arg0, %iterArg_1 = %arg1) : tensor<i32>, tensor<i32>
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
       %new_sum = func.call @None(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      stablehlo.return %1 : tensor<i1>
    } do {
      %1 = stablehlo.add %iterArg, %c : tensor<i32>
      %2 = stablehlo.add %iterArg_1, %1 : tensor<i32>
      stablehlo.return %1, %2 : tensor<i32>, tensor<i32>
  }
  return %0#1 : tensor<i32>
}

func.func public @None(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // expected-error @below {{Named computations can only be nested in mpmd functions or mpmd ops.}}
  %1 = mpmd.named_computation<"f1"> (%arg0) (%arg3: tensor<i32>) {
    %10 = stablehlo.add %arg3, %arg3 : tensor<i32>
    mpmd.return %10 : tensor<i32>
  } : (tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}
