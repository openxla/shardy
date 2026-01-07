// RUN: mpmd_opt %s -mpmd-mark-offloaded-input-output 2>&1 | FileCheck -implicit-check-not=mhlo.memory_kind %s

!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
!m1_4x4 = !mpmd.mesh_tensor<"m1", tensor<4x4xf32>>
!m1_16x16x16 = !mpmd.mesh_tensor<"m1", tensor<16x16x16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8]>>>

// CHECK-LABEL: func @simple(%arg0: {{.*}} {mhlo.memory_kind = "pinned_host"}) ->
// CHECK-SAME:    {mhlo.memory_kind = "pinned_host"})
func.func @simple(%func_arg: !m1_16 {mhlo.memory_kind = "pinned_host"}) ->
  (!m1_16 {mhlo.memory_kind = "pinned_host"})
  attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{mhlo.memory_kind = "pinned_host"}], res_attrs = [{mhlo.memory_kind = "pinned_host"}]}
  %f = mpmd.fragment<mesh="m1", origin=[]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    %7 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    %8 = stablehlo.add %7, %7 : tensor<16xf32>

    %9 = stablehlo.custom_call @annotate_device_placement(%8) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}
      } : (tensor<16xf32>) -> tensor<16xf32>
    mpmd.return %9 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  func.return %f : !m1_16
}

// CHECK-LABEL: func @simple_unpinned_host(%arg0: {{.*}} {mhlo.memory_kind = "unpinned_host"}) ->
// CHECK-SAME:    {mhlo.memory_kind = "unpinned_host"})
func.func @simple_unpinned_host(%func_arg: !m1_16 {mhlo.memory_kind = "unpinned_host"}) ->
  (!m1_16 {mhlo.memory_kind = "unpinned_host"})
  attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{mhlo.memory_kind = "unpinned_host"}], res_attrs = [{mhlo.memory_kind = "unpinned_host"}]}
  %f = mpmd.fragment<mesh="m1", origin=[]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    %7 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    %8 = stablehlo.add %7, %7 : tensor<16xf32>

    %9 = stablehlo.custom_call @annotate_device_placement(%8) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "unpinned_host"}
      } : (tensor<16xf32>) -> tensor<16xf32>
    mpmd.return %9 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  func.return %f : !m1_16
}

// CHECK-LABEL: func @with_optimization_barrier(%arg0
// CHECK-SAME:    %arg1: {{.*}} {mhlo.memory_kind = "pinned_host"}) ->
func.func @with_optimization_barrier(%func_arg0: !m1_16,
  %func_arg1: !m1_16 {mhlo.memory_kind = "pinned_host"}) -> !m1_16
  attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{}, {mhlo.memory_kind = "pinned_host"}], res_attrs = [{}]}
  %f = mpmd.fragment<mesh="m1", origin=[]> (%func_arg0, %func_arg1)
    (%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) {
    %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<16xf32>, tensor<16xf32>
    %1:2 = stablehlo.optimization_barrier %0#0, %0#1 : tensor<16xf32>, tensor<16xf32>

    %7 = stablehlo.custom_call @annotate_device_placement(%1#1) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    %8 = stablehlo.add %7, %7 : tensor<16xf32>
    mpmd.return %8 : tensor<16xf32>
  } : (!m1_16, !m1_16) -> !m1_16

  func.return %f : !m1_16
}

// CHECK-LABEL: func @in_while_loop_with_update
// CHECK-SAME:  -> ({{.*}}mesh_tensor{{.*}}mesh_tensor{{.*}} {mhlo.memory_kind = "pinned_host"})
func.func @in_while_loop_with_update(%func_arg: !m1_16x16x16) -> (!m1_16x16x16, !m1_16x16x16 {mhlo.memory_kind = "pinned_host"})
  attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{}], res_attrs = [{}, {mhlo.memory_kind = "pinned_host"}]}
  %f:2 = mpmd.fragment<mesh="m1", origin=[]> (%func_arg) (%arg0: tensor<16x16x16xf32>) {
    %0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = stablehlo.constant dense<1> : tensor<i32>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<16x16x16xf32>
    %6:4 = stablehlo.while(%iterArg = %1, %iterArg_0 = %0, %iterArg_1 = %5, %iterArg_2 = %5)
      : tensor<i32>, tensor<f32>, tensor<16x16x16xf32>, tensor<16x16x16xf32>
      cond {
      %7 = stablehlo.compare  LT, %iterArg, %iterArg,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %7 : tensor<i1>
    } do {
      %10 = stablehlo.dynamic_slice %iterArg_2, %iterArg, %iterArg, %iterArg, sizes = [1, 16, 16] : (tensor<16x16x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x16x16xf32>
      %11 = stablehlo.reshape %10 : (tensor<1x16x16xf32>) -> tensor<16x16xf32>
      %13 = stablehlo.broadcast_in_dim %iterArg_0, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
      %14 = stablehlo.add %11, %13 : tensor<16x16xf32>
      %15 = stablehlo.sine %14 : tensor<16x16xf32>
      %16 = stablehlo.cosine %14 : tensor<16x16xf32>
      %17 = stablehlo.custom_call @annotate_device_placement(%16) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}} : (tensor<16x16xf32>) -> tensor<16x16xf32>
      %18 = stablehlo.reshape %15 : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %19 = stablehlo.dynamic_update_slice %iterArg_1, %18, %iterArg, %iterArg, %iterArg : (tensor<16x16x16xf32>, tensor<1x16x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x16x16xf32>
      %20 = stablehlo.reshape %17 : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %21 = stablehlo.dynamic_update_slice %iterArg_2, %20, %iterArg, %iterArg, %iterArg : (tensor<16x16x16xf32>, tensor<1x16x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x16x16xf32>
      stablehlo.return %iterArg, %iterArg_0, %19, %21 : tensor<i32>, tensor<f32>, tensor<16x16x16xf32>, tensor<16x16x16xf32>
    }
    %cc = stablehlo.custom_call @Sharding(%6#3) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
    %cc1 = stablehlo.custom_call @SPMDFullToShardShape(%cc) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
    %cc2 = stablehlo.custom_call @SPMDShardToFullShape(%cc1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
    mpmd.return %6#2, %cc2 : tensor<16x16x16xf32>, tensor<16x16x16xf32>
  } : (!m1_16x16x16) -> (!m1_16x16x16, !m1_16x16x16)
  func.return %f#0, %f#1: !m1_16x16x16, !m1_16x16x16
}

// CHECK-LABEL: func @in_while_loop_with_slice(%arg0: {{.*}} {mhlo.memory_kind = "pinned_host"})
func.func @in_while_loop_with_slice(
    %func_arg: !m1_16x16x16 {mhlo.memory_kind = "pinned_host"}
  ) -> !m1_16x16x16 attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{mhlo.memory_kind = "pinned_host"}], res_attrs = [{}]}
  %f = mpmd.fragment<mesh="m1", origin=[]> (%func_arg)
    (%arg0: tensor<16x16x16xf32>) {
    %0 = stablehlo.constant dense<1> : tensor<i32>
    %3 = stablehlo.constant dense<0.000000e+00> : tensor<16x16x16xf32>
    %cc = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
    %cc1 = stablehlo.custom_call @SPMDFullToShardShape(%cc) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
    %cc2 = stablehlo.custom_call @SPMDShardToFullShape(%cc1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
    %4:3 = stablehlo.while(%iterArg = %0, %iterArg_0 = %3, %iterArg_5 = %cc2) : tensor<i32>, tensor<16x16x16xf32>, tensor<16x16x16xf32>
      cond {
      %5 = stablehlo.compare  LT, %iterArg, %iterArg,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %5 : tensor<i1>
    } do {
      %10 = stablehlo.dynamic_slice %iterArg_0, %iterArg, %iterArg, %iterArg, sizes = [1, 16, 16] : (tensor<16x16x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x16x16xf32>
      %11 = stablehlo.reshape %10 : (tensor<1x16x16xf32>) -> tensor<16x16xf32>
      %12 = stablehlo.dynamic_slice %iterArg_5, %iterArg, %iterArg, %iterArg, sizes = [1, 16, 16] : (tensor<16x16x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x16x16xf32>
      %13 = stablehlo.reshape %12 : (tensor<1x16x16xf32>) -> tensor<16x16xf32>
      %14:2 = stablehlo.optimization_barrier %13, %11 : tensor<16x16xf32>, tensor<16x16xf32>
      %15 = stablehlo.custom_call @annotate_device_placement(%14#0) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "device"}} : (tensor<16x16xf32>) -> tensor<16x16xf32>
      %17 = stablehlo.reshape %15 : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %18 = stablehlo.dynamic_update_slice %iterArg_0, %17, %iterArg, %iterArg, %iterArg : (tensor<16x16x16xf32>, tensor<1x16x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x16x16xf32>
      stablehlo.return %iterArg, %18, %iterArg_0 : tensor<i32>, tensor<16x16x16xf32>, tensor<16x16x16xf32>
    }
    mpmd.return %4#1 : tensor<16x16x16xf32>
  } : (!m1_16x16x16) -> !m1_16x16x16
  return %f : !m1_16x16x16
}

// CHECK-LABEL: func @place_device_with_incompatible_reshape_and_custom_call(
// CHECK-NOT:     mhlo.memory_kind
func.func @place_device_with_incompatible_reshape_and_custom_call(%func_arg0: !m1_16, %func_arg1: !m1_16) -> (!m1_4x4, !m1_16)
  attributes {topology=#topology} {
  %f:2 = mpmd.fragment<mesh="m1", origin=[]> (%func_arg0, %func_arg1)
    (%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) {
    %0 = stablehlo.reshape %arg0 : (tensor<16xf32>) -> tensor<4x4xf32>
    %6 = stablehlo.custom_call @annotate_device_placement(%0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<4x4xf32>) -> tensor<4x4xf32>

    %cc = stablehlo.custom_call @Something(%arg1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<16xf32>) -> tensor<16xf32>
    %7 = stablehlo.custom_call @annotate_device_placement(%cc) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<16xf32>) -> tensor<16xf32>
    mpmd.return %6, %7: tensor<4x4xf32>, tensor<16xf32>
  }  : (!m1_16, !m1_16) -> (!m1_4x4, !m1_16)
  func.return %f#0, %f#1 : !m1_4x4, !m1_16
}

// CHECK-LABEL: func @place_host_with_incompatible_reshape_and_custom_call(
// CHECK-NOT:     mhlo.memory_kind
func.func @place_host_with_incompatible_reshape_and_custom_call(%func_arg0: !m1_16, %func_arg1: !m1_16) -> (!m1_4x4, !m1_16)
  attributes {topology=#topology} {
  %f:2 = mpmd.fragment<mesh="m1", origin=[]> (%func_arg0, %func_arg1)
    (%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) {
    %6 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}
      } : (tensor<16xf32>) -> tensor<16xf32>
    %0 = stablehlo.reshape %6 : (tensor<16xf32>) -> tensor<4x4xf32>

    %7 = stablehlo.custom_call @annotate_device_placement(%arg1) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}
      } : (tensor<16xf32>) -> tensor<16xf32>
    %cc = stablehlo.custom_call @Something(%7) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<16xf32>) -> tensor<16xf32>
    mpmd.return %0, %cc: tensor<4x4xf32>, tensor<16xf32>
  }  : (!m1_16, !m1_16) -> (!m1_4x4, !m1_16)
  func.return %f#0, %f#1 : !m1_4x4, !m1_16
}

// CHECK-LABEL: func @func_arg_multiple_matching_users_pinned_host
// CHECK-SAME:    (%arg0: {{.*}} {mhlo.memory_kind = "pinned_host"})
func.func @func_arg_multiple_matching_users_pinned_host(
  %func_arg: !m1_16 {mhlo.memory_kind = "pinned_host"}) -> (!m1_16, !m1_16)
  attributes {topology=#topology} {

  // CHECK: fragment{{.*}} origin=["f"]
  // CHECK-SAME: {arg_attrs = [{mhlo.memory_kind = "pinned_host"}]
  %f1 = mpmd.fragment<mesh="m1", origin=["f"]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    %7 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    mpmd.return %7 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  // CHECK: fragment{{.*}} origin=["g"]
  // CHECK-SAME: {arg_attrs = [{mhlo.memory_kind = "pinned_host"}]
  %f2 = mpmd.fragment<mesh="m1", origin=["g"]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    %7 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    mpmd.return %7 : tensor<16xf32>
  } : (!m1_16) -> !m1_16
  func.return %f1, %f2 : !m1_16, !m1_16
}

// CHECK-LABEL: func @func_arg_multiple_matching_users_unpinned_host
// CHECK-SAME:    (%arg0: {{.*}} {mhlo.memory_kind = "unpinned_host"})
func.func @func_arg_multiple_matching_users_unpinned_host(
  %func_arg: !m1_16 {mhlo.memory_kind = "unpinned_host"}) -> (!m1_16, !m1_16)
  attributes {topology=#topology} {

  // CHECK: fragment{{.*}} origin=["f"]
  // CHECK-SAME: {arg_attrs = [{mhlo.memory_kind = "unpinned_host"}]
  %f1 = mpmd.fragment<mesh="m1", origin=["f"]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    %7 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    mpmd.return %7 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  // CHECK: fragment{{.*}} origin=["g"]
  // CHECK-SAME: {arg_attrs = [{mhlo.memory_kind = "unpinned_host"}]
  %f2 = mpmd.fragment<mesh="m1", origin=["g"]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    %7 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    mpmd.return %7 : tensor<16xf32>
  } : (!m1_16) -> !m1_16
  func.return %f1, %f2 : !m1_16, !m1_16
}

// CHECK-LABEL: func @func_arg_match
// CHECK-SAME:    (%arg0: {{.*}} {mhlo.memory_kind = "pinned_host"})
func.func @func_arg_match(%func_arg: !m1_16 {mhlo.memory_kind = "pinned_host"}) -> !m1_16
  attributes {topology=#topology} {

  // CHECK: fragment{{.*}} origin=["f"]
  // CHECK-SAME: {arg_attrs = [{mhlo.memory_kind = "pinned_host"}]
  %f1 = mpmd.fragment<mesh="m1", origin=["f"]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    %7 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "device"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    mpmd.return %7 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  func.return %f1 : !m1_16
}


// CHECK-LABEL: func @func_result_match
// CHECK-SAME:  -> ({{.*}} {mhlo.memory_kind = "pinned_host"})
func.func @func_result_match(%func_arg: !m1_16) -> (!m1_16 {mhlo.memory_kind = "pinned_host"})
  attributes {topology=#topology} {

  // CHECK: fragment{{.*}} origin=["f"]
  // CHECK-SAME: {arg_attrs = [{}], res_attrs = [{mhlo.memory_kind = "pinned_host"}]}
  %f1 = mpmd.fragment<mesh="m1", origin=["f"]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    %7 = stablehlo.custom_call @annotate_device_placement(%arg0) {
        backend_config = "", has_side_effect = true,
        mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}
      } : (tensor<16xf32>) -> tensor<16xf32>

    mpmd.return %7 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  func.return %f1 : !m1_16
}

// CHECK-LABEL: func @func_arg_returned_match(%arg0: {{.*}} {mhlo.memory_kind = "pinned_host"}) ->
// CHECK-SAME:    {mhlo.memory_kind = "pinned_host"})
func.func @func_arg_returned_match(%func_arg: !m1_16 {mhlo.memory_kind = "pinned_host"})
    -> (!m1_16 {mhlo.memory_kind = "pinned_host"})
  attributes {topology=#topology} {

  func.return %func_arg : !m1_16
}

// CHECK-LABEL: func @noop_if_no_non_return_user(
// CHECK-NOT:     mhlo.memory_kind
func.func @noop_if_no_non_return_user(%func_arg0: !m1_16, %func_arg1: !m1_16) -> (!m1_4x4, !m1_16)
  attributes {topology=#topology} {

  %f:2 = mpmd.fragment<mesh="m1", origin=[]> (%func_arg0, %func_arg1)
    (%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) {
    %0 = stablehlo.reshape %arg0 : (tensor<16xf32>) -> tensor<4x4xf32>
    mpmd.return %0, %arg1: tensor<4x4xf32>, tensor<16xf32>
  }  : (!m1_16, !m1_16) -> (!m1_4x4, !m1_16)

  func.return %f#0, %f#1 : !m1_4x4, !m1_16
}

// CHECK-LABEL: func @no_mismatch_between_func_device_and_frag_missing(%arg0: {{.*}} {mhlo.memory_kind = "device"}) ->
// CHECK-SAME:    {mhlo.memory_kind = "device"})
func.func @no_mismatch_between_func_device_and_frag_missing(%func_arg: !m1_16 {mhlo.memory_kind = "device"}) ->
  (!m1_16 {mhlo.memory_kind = "device"})
  attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{}], res_attrs = [{}]}
  %f = mpmd.fragment<mesh="m1", origin=[]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    mpmd.return %arg0 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  func.return %f : !m1_16
}

// CHECK-LABEL: func @no_mismatch_between_func_missing_and_frag_device(
// CHECK-NOT:    {mhlo.memory_kind = "device"}
func.func @no_mismatch_between_func_missing_and_frag_device(%func_arg: !m1_16 ) -> !m1_16
  attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{mhlo.memory_kind = "device"}], res_attrs = [{mhlo.memory_kind = "device"}]}
  %f = mpmd.fragment<mesh="m1", origin=[]> (%func_arg)
    {arg_attrs = [{mhlo.memory_kind = "device"}], res_attrs = [{mhlo.memory_kind = "device"}]}
    (%arg0: tensor<16xf32>) {
    mpmd.return %arg0 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  func.return %f : !m1_16
}

// CHECK-LABEL: func @simple_compute_on_host(%arg0: {{.*}} {mhlo.memory_kind = "pinned_host"}) ->
// CHECK-SAME:    {mhlo.memory_kind = "pinned_host"})
func.func @simple_compute_on_host(%func_arg: !m1_16 {mhlo.memory_kind = "pinned_host"}) ->
  (!m1_16 {mhlo.memory_kind = "pinned_host"})
  attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{mhlo.memory_kind = "pinned_host"}], res_attrs = [{mhlo.memory_kind = "pinned_host"}]}
  %f = mpmd.fragment<mesh="m1", origin=[]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    %8 = stablehlo.add %arg0, %arg0 {
      mhlo.frontend_attributes = {_xla_compute_type = "host"}
    } : tensor<16xf32>
    mpmd.return %8 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  func.return %f : !m1_16
}

// CHECK-LABEL: func @propagate_memory_kind_with_broadcast
// CHECK-SAME:  -> ({{.*}}mesh_tensor{{.*}}mesh_tensor{{.*}} {mhlo.memory_kind = "pinned_host"})
func.func @propagate_memory_kind_with_broadcast(%func_arg: !m1_16x16x16) -> (!m1_16x16x16, !m1_16x16x16 {mhlo.memory_kind = "pinned_host"})
  attributes {topology=#topology} {
  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{}], res_attrs = [{}, {mhlo.memory_kind = "pinned_host"}]}
  %f:2 = mpmd.fragment<mesh="m1", origin=[]> (%func_arg) (%arg0: tensor<16x16x16xf32>) {
    %0 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %1 = stablehlo.constant dense<1> : tensor<i32>
    %5 = stablehlo.constant dense<0.000000e+00> : tensor<16x16x16xf32>
    %6:4 = stablehlo.while(%iterArg = %1, %iterArg_0 = %0, %iterArg_1 = %5, %iterArg_2 = %5)
      : tensor<i32>, tensor<f32>, tensor<16x16x16xf32>, tensor<16x16x16xf32>
      cond {
      %7 = stablehlo.compare  LT, %iterArg, %iterArg,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %7 : tensor<i1>
    } do {
      %10 = stablehlo.dynamic_slice %iterArg_2, %iterArg, %iterArg, %iterArg, sizes = [1, 16, 16] : (tensor<16x16x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x16x16xf32>
      %11 = stablehlo.reshape %10 : (tensor<1x16x16xf32>) -> tensor<16x16xf32>
      %13 = stablehlo.broadcast_in_dim %iterArg_0, dims = [] : (tensor<f32>) -> tensor<16x16xf32>
      %14 = stablehlo.add %11, %13 : tensor<16x16xf32>
      %15 = stablehlo.sine %14 : tensor<16x16xf32>
      %16 = stablehlo.cosine %14 : tensor<16x16xf32>
      %17 = stablehlo.custom_call @annotate_device_placement(%16) {backend_config = "", has_side_effect = true, mhlo.frontend_attributes = {_xla_buffer_placement = "pinned_host"}} : (tensor<16x16xf32>) -> tensor<16x16xf32>
      %18 = stablehlo.broadcast_in_dim %15, dims = [1, 2] : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %19 = stablehlo.dynamic_update_slice %iterArg_1, %18, %iterArg, %iterArg, %iterArg : (tensor<16x16x16xf32>, tensor<1x16x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x16x16xf32>
      %20 = stablehlo.reshape %17 : (tensor<16x16xf32>) -> tensor<1x16x16xf32>
      %21 = stablehlo.dynamic_update_slice %iterArg_2, %20, %iterArg, %iterArg, %iterArg : (tensor<16x16x16xf32>, tensor<1x16x16xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<16x16x16xf32>
      stablehlo.return %iterArg, %iterArg_0, %19, %21 : tensor<i32>, tensor<f32>, tensor<16x16x16xf32>, tensor<16x16x16xf32>
    }
    %cc = stablehlo.custom_call @Sharding(%6#3) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
    %cc1 = stablehlo.custom_call @SPMDFullToShardShape(%cc) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
    %cc2 = stablehlo.custom_call @SPMDShardToFullShape(%cc1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<16x16x16xf32>) -> tensor<16x16x16xf32>
    mpmd.return %6#2, %cc2 : tensor<16x16x16xf32>, tensor<16x16x16xf32>
  } : (!m1_16x16x16) -> (!m1_16x16x16, !m1_16x16x16)
  func.return %f#0, %f#1: !m1_16x16x16, !m1_16x16x16
}

