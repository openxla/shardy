// RUN: mpmd_opt %s -split-input-file -verify-diagnostics


#topology = #mpmd.topology<<"m1" : <["x"=1]>>>
module {

  func.func public @call_not_in_function_not_ok(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
    attributes {topology = #topology}
  {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %0:2 = stablehlo.while(%iterArg = %c, %iterArg_0 = %arg0) : tensor<i32>, tensor<3x5xf32>
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %iterArg,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
    // expected-error @+1 {{Mpmd CallOp on "f" can only be used in a function or for_op block but was called from inside op stablehlo.while}}
      %1 = mpmd.call @f(%iterArg_0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
      stablehlo.return %iterArg, %1 : tensor<i32>, tensor<3x5xf32>
    }
    return %0#1 : tensor<3x5xf32>
  }

  func.func private @f(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
    attributes {topology = #topology} {
    return %arg0 : tensor<3x5xf32>
  }
}

// -----

#topology = #mpmd.topology<<"m1" : <["x"=1]>>>
module {

  func.func public @nested_mpmd_call_not_ok(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
    attributes {topology = #topology}
  {
    %1 = mpmd.call @f(%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
    return %1 : tensor<3x5xf32>
  }

  func.func private @f(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
    attributes {topology = #topology} {
    // expected-error @+1 {{Mpmd CallOp on "g" in an Mpmd function can only be used directly by the entrypoint function, i.e. the main function, but was called from "f"}}
    %1 = mpmd.call @g(%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
    return %1 : tensor<3x5xf32>
  }

  func.func private @g(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
    attributes {topology = #topology} {
    return %arg0 : tensor<3x5xf32>
  }
}

// -----

#topology = #mpmd.topology<<"m1" : <["x"=1]>>>
module {

  func.func public @mpmd_call_in_nested_in_func_call_is_ok(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
    attributes {topology = #topology}
  {
    %1 = func.call @f(%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
    return %1 : tensor<3x5xf32>
  }

  func.func private @f(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32> {
    %1 = mpmd.call @g(%arg0) : (tensor<3x5xf32>) -> tensor<3x5xf32>
    return %1 : tensor<3x5xf32>
  }

  func.func private @g(%arg0: tensor<3x5xf32>) -> tensor<3x5xf32>
    attributes {topology = #topology} {
    return %arg0 : tensor<3x5xf32>
  }
}
