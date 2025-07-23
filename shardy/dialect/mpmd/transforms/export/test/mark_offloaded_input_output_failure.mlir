// RUN: mpmd_opt %s -mpmd-mark-offloaded-input-output -split-input-file -verify-diagnostics 2>&1

!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>


// expected-error@+1 {{Memory kind mismatch between users of arg 0: <<NULL ATTRIBUTE>> vs "pinned_host"}}
func.func @func_arg_multiple_different_fragment_users(%func_arg: !m1_16) -> (!m1_16, !m1_16)
  attributes {topology=#topology} {

  %f1 = mpmd.fragment<mesh="m1", origin=["f"]> (%func_arg)
    {arg_attrs = [{mhlo.memory_kind = "pinned_host"}], res_attrs = [{}]}
    (%arg0: tensor<16xf32>) {
    %7 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    mpmd.return %7 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  %f2 = mpmd.fragment<mesh="m1", origin=["g"]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    mpmd.return %arg0 : tensor<16xf32>
  } : (!m1_16) -> !m1_16
  func.return %f1, %f2 : !m1_16, !m1_16
}

// -----

!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>

// expected-error@+1 {{Memory kind mismatch between users of arg 0: <<NULL ATTRIBUTE>> vs "pinned_host"}}
func.func @func_arg_multiple_different_users(%func_arg: !m1_16) -> (!m1_16, !m1_16)
  attributes {topology=#topology} {

  %f1 = mpmd.fragment<mesh="m1", origin=["f"]> (%func_arg)
    {arg_attrs = [{mhlo.memory_kind = "pinned_host"}], res_attrs = [{}]}
    (%arg0: tensor<16xf32>) {
    %7 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    mpmd.return %7 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  // Note that we don't support the transferring of offloaded tensors right now.
  %f2 = mpmd.transfer %func_arg : (!m1_16) -> !m1_16
  func.return %f1, %f2 : !m1_16, !m1_16
}

// -----

!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>

// expected-error@+1 {{Memory kind mismatch between arg 0 and users: "device" vs "pinned_host"}}
func.func @func_arg_mismatch(%func_arg: !m1_16 {mhlo.memory_kind = "device"}) -> !m1_16
  attributes {topology=#topology} {

  %f1 = mpmd.fragment<mesh="m1", origin=["f"]> (%func_arg)
    {arg_attrs = [{mhlo.memory_kind = "pinned_host"}], res_attrs = [{}]}
    (%arg0: tensor<16xf32>) {
    %7 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    mpmd.return %7 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  func.return %f1 : !m1_16
}

// -----

!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>

func.func @func_result_mismatch(%func_arg: !m1_16) -> (!m1_16 {mhlo.memory_kind = "device"})
  attributes {topology=#topology} {

  %f1 = mpmd.fragment<mesh="m1", origin=["f"]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    %7 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    mpmd.return %7 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

// expected-error@+1 {{Memory kind mismatch between result 0 and defining op: "device" vs "pinned_host"}}
  func.return %f1 : !m1_16
}

// -----

!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>

// expected-error@+1 {{Memory kind mismatch between arg 0 and users: "pinned_host" vs "device"}}
func.func @func_arg_returned_mismatch(%func_arg: !m1_16 {mhlo.memory_kind = "pinned_host"})
    -> (!m1_16 {mhlo.memory_kind = "device"})
  attributes {topology=#topology} {
// expected-error@+1 {{Memory kind mismatch between result 0 and defining op: "device" vs "pinned_host"}}
  func.return %func_arg : !m1_16
}

