// RUN: sdy_opt %s -sdy-manual-axes-cleanup -verify-diagnostics

func.func @manual_computation_no_inputs_or_outputs_with_manual_axes() {
  // expected-error @+1 {{op has manual_axes when there are no in/out shardings and the body is not empty}}
  sdy.manual_computation() in_shardings=[] out_shardings=[] manual_axes={"a"} () {
    %0 = sdy.constant dense<1.000000e+00> : tensor<8xf32>
    sdy.return
  } : () -> ()
  func.return
}
