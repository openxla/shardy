// RUN: mpmd_opt %s -mpmd-mark-input-output-with-layouts -split-input-file -verify-diagnostics 2>&1 | FileCheck %s

!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
!m2_16x16 = !mpmd.mesh_tensor<"m2", tensor<16x16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>, <"m2": <["x"=8,"y"=8]>>>

// CHECK-LABEL: func @func_default_layout_to_frag_layout(%arg0: {{.*}} {mhlo.layout_mode = "auto"}) ->
// CHECK-SAME:    {mhlo.layout_mode = "auto"})
func.func @func_default_layout_to_frag_layout(
  %func_arg: !m1_16 {mhlo.layout_mode = "auto"}) ->
  (!m1_16 {mhlo.layout_mode = "auto"}) attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{mhlo.layout_mode = "auto"}], res_attrs = [{mhlo.layout_mode = "auto"}]}
  %f = mpmd.fragment<mesh="m1", origin=[]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    mpmd.return %arg0 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  func.return %f : !m1_16
}

// CHECK-LABEL: func @func_mix_of_no_layout_and_auto_layout_to_frag(%arg0: {{.*}}, %arg1: {{.*}} {mhlo.layout_mode = "auto"}) ->
// CHECK-SAME:    (!mpmd.mesh_tensor<"m1", tensor<16xf32>>, !mpmd.mesh_tensor<"m1", tensor<16xf32>> {mhlo.layout_mode = "auto"})
func.func @func_mix_of_no_layout_and_auto_layout_to_frag(
  %func_arg0: !m1_16, %func_arg1: !m1_16 {mhlo.layout_mode = "auto"}) ->
  (!m1_16, !m1_16 {mhlo.layout_mode = "auto"}) attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{}, {mhlo.layout_mode = "auto"}], res_attrs = [{}, {}, {mhlo.layout_mode = "auto"}]}
  %f:3 = mpmd.fragment<mesh="m1", origin=[]> (%func_arg0, %func_arg1)
    (%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) {
    mpmd.return %arg0, %arg0, %arg1 : tensor<16xf32>, tensor<16xf32>, tensor<16xf32>
  } : (!m1_16, !m1_16) -> (!m1_16, !m1_16, !m1_16)

  func.return %f#0, %f#2 : !m1_16, !m1_16
}

// Verifies that the func arg and res layouts are refined before they are
// propagated to the fragment.

// CHECK-LABEL: func @layouts_propagated_between_func_args_and_func_res_before_frag(%arg0: {{.*}} {mhlo.layout_mode = "{0}"}, %arg1: {{.*}}) ->
// CHECK-SAME:    (!mpmd.mesh_tensor<"m1", tensor<16xf32>> {mhlo.layout_mode = "auto"}, !mpmd.mesh_tensor<"m1", tensor<16xf32>>, !mpmd.mesh_tensor<"m1", tensor<16xf32>> {mhlo.layout_mode = "{0}"})
func.func @layouts_propagated_between_func_args_and_func_res_before_frag(
  %func_arg0: !m1_16 {mhlo.layout_mode = "auto"},
  %func_arg1: !m1_16 {mhlo.layout_mode = "default"})
    -> (!m1_16 {mhlo.layout_mode = "auto"},
        !m1_16 {mhlo.layout_mode = "default"},
        !m1_16 {mhlo.layout_mode = "{0}"})
       attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{mhlo.layout_mode = "{0}"}, {}], res_attrs = [{mhlo.layout_mode = "auto"}, {}]}
  %f:2 = mpmd.fragment<mesh="m1", origin=[]> (%func_arg0, %func_arg1)
    (%arg0: tensor<16xf32>, %arg1: tensor<16xf32>) {
    mpmd.return %arg0, %arg1 : tensor<16xf32>, tensor<16xf32>
  } : (!m1_16, !m1_16) -> (!m1_16, !m1_16)

  func.return %f#0, %f#1, %func_arg0: !m1_16, !m1_16, !m1_16
}

// CHECK-LABEL: func @func_arg_multiple_users_and_func_res_multiple_producers
// CHECK-SAME:    (%arg0: {{.*}} {mhlo.layout_mode = "auto"})
func.func @func_arg_multiple_users_and_func_res_multiple_producers(
  %func_arg: !m1_16 {mhlo.layout_mode = "auto"}) ->
  (!m1_16 {mhlo.layout_mode = "auto"}, !m1_16 {mhlo.layout_mode = "auto"})
  attributes {topology=#topology} {

  // CHECK: fragment{{.*}} origin=["f"]
  // CHECK-SAME: {arg_attrs = [{mhlo.layout_mode = "auto"}], res_attrs = [{mhlo.layout_mode = "auto"}]}
  %f1 = mpmd.fragment<mesh="m1", origin=["f"]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    mpmd.return %arg0 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  // CHECK: fragment{{.*}} origin=["g"]
  // CHECK-SAME: {arg_attrs = [{mhlo.layout_mode = "auto"}], res_attrs = [{mhlo.layout_mode = "auto"}]}
  %f2 = mpmd.fragment<mesh="m1", origin=["g"]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    mpmd.return %arg0 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  func.return %f1, %f2 : !m1_16, !m1_16
}

// CHECK-LABEL: func @two_args_w_custom_layouts_to_two_frags(%arg0: {{.*}} {mhlo.layout_mode = "{0, 1}"}, %arg1: {{.*}} {mhlo.layout_mode = "{1, 0}"}) ->
// CHECK-SAME:    {mhlo.layout_mode = "{0, 1}"}
// CHECK-SAME:    {mhlo.layout_mode = "{1, 0}"})
func.func @two_args_w_custom_layouts_to_two_frags(
    %func_arg_0: !m2_16x16 {mhlo.layout_mode = "{0, 1}"},
    %func_arg_1: !m2_16x16 {mhlo.layout_mode = "{1, 0}"})
    -> (!m2_16x16 {mhlo.layout_mode = "{0, 1}"},
        !m2_16x16 {mhlo.layout_mode = "{1, 0}"})
        attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{mhlo.layout_mode = "{0, 1}"}], res_attrs = [{mhlo.layout_mode = "{0, 1}"}]}
  %f1 = mpmd.fragment<mesh="m2", origin=[]> (%func_arg_0)
    (%arg0: tensor<16x16xf32>) {
    %8 = stablehlo.add %arg0, %arg0 : tensor<16x16xf32>
    mpmd.return %8 : tensor<16x16xf32>
  } : (!m2_16x16) -> !m2_16x16

  // CHECK: fragment
  // CHECK-SAME: {arg_attrs = [{mhlo.layout_mode = "{0, 1}"}, {mhlo.layout_mode = "{1, 0}"}], res_attrs = [{mhlo.layout_mode = "{1, 0}"}]}
  %f2 = mpmd.fragment<mesh="m2", origin=[]> (%func_arg_0, %func_arg_1)
    (%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) {
    %8 = stablehlo.add %arg0, %arg1 : tensor<16x16xf32>
    mpmd.return %8 : tensor<16x16xf32>
  } : (!m2_16x16, !m2_16x16) -> !m2_16x16

  func.return %f1, %f2 : !m2_16x16, !m2_16x16
}

// CHECK-LABEL: func @no_op_func_arg_returned(%arg0: {{.*}} {mhlo.layout_mode = "auto"}) ->
// CHECK-SAME:    {mhlo.layout_mode = "auto"})
func.func @no_op_func_arg_returned(%func_arg: !m1_16 {mhlo.layout_mode = "auto"})
    -> (!m1_16 {mhlo.layout_mode = "auto"})
  attributes {topology=#topology} {

  func.return %func_arg : !m1_16
}

// -----
!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>>

// CHECK-LABEL: func @args_w_layouts_propagated_to_return(%arg0: {{.*}}, %arg1: {{.*}} {mhlo.layout_mode = "{0}"}, %arg2: {{.*}}, %arg3: {{.*}}) ->
// CHECK-SAME:    {mhlo.layout_mode = "auto"}
// CHECK-SAME:    {mhlo.layout_mode = "{0}"}
// CHECK-NOT:     {mhlo.layout_mode = "default"}
func.func @args_w_layouts_propagated_to_return(
  %arg0: !m1_16 {mhlo.layout_mode = "auto"},
  %arg1: !m1_16 {mhlo.layout_mode = "{0}"},
  %arg2: !m1_16,
  %arg3: !m1_16 {mhlo.layout_mode = "default"})
    -> (!m1_16 {mhlo.layout_mode = "auto"}, !m1_16 {mhlo.layout_mode = "auto"},
        !m1_16, !m1_16 {mhlo.layout_mode = "auto"})
        attributes {topology=#topology} {
  func.return %arg0, %arg1, %arg2, %arg3 : !m1_16, !m1_16, !m1_16, !m1_16
}

// -----
!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>>

// CHECK-LABEL: func @args_w_layouts_propagated_from_return(%arg0: {{.*}} {mhlo.layout_mode = "{0}"}, %arg1: {{.*}}, %arg2: {{.*}}) ->
// CHECK-SAME:    {mhlo.layout_mode = "{0}"}
// CHECK-NOT: {mhlo.layout_mode = "default"}
func.func @args_w_layouts_propagated_from_return(
  %arg0: !m1_16 {mhlo.layout_mode = "auto"},
  %arg1: !m1_16,
  %arg2: !m1_16 {mhlo.layout_mode = "auto"})
    -> (!m1_16 {mhlo.layout_mode = "{0}"},
        !m1_16, !m1_16 {mhlo.layout_mode = "default"})
    attributes {topology=#topology} {
  func.return %arg0, %arg1, %arg2 : !m1_16, !m1_16, !m1_16
}

// -----
!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>>

// CHECK-LABEL: func @convert_returned_auto_layouts(%arg0: {{.*}}, %arg1: {{.*}} {mhlo.layout_mode = "{0}"}) ->
// CHECK-SAME:    {mhlo.layout_mode = "{0}"})
// CHECK-NOT: {mhlo.layout_mode = "default"}
func.func @convert_returned_auto_layouts(
  %arg0: !m1_16 {mhlo.layout_mode = "auto"},
  %arg1: !m1_16 {mhlo.layout_mode = "{0}"})
    -> (!m1_16 {mhlo.layout_mode = "default"},
        !m1_16 {mhlo.layout_mode = "auto"})
    attributes {topology=#topology} {
  func.return %arg0, %arg1 : !m1_16, !m1_16
}

// -----
!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>>

// CHECK-LABEL: func @refine_both_arg_and_res_auto_layouts_from_default_result_layout(%arg0: {{.*}}) ->
// CHECK-NOT: {mhlo.layout_mode = "default"}
func.func @refine_both_arg_and_res_auto_layouts_from_default_result_layout(
  %arg0: !m1_16 {mhlo.layout_mode = "auto"})
    -> (!m1_16 {mhlo.layout_mode = "auto"},
        !m1_16 {mhlo.layout_mode = "default"})
    attributes {topology=#topology} {
  func.return %arg0, %arg0 : !m1_16, !m1_16
}

// -----
!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>>

// CHECK-LABEL: func @refine_func_return_layout_from_fragment_result(%arg0: {{.*}}) ->
// CHECK-NOT: {mhlo.layout_mode = "default"}
func.func @refine_func_return_layout_from_fragment_result(%func_arg: !m1_16) ->
  (!m1_16 {mhlo.layout_mode = "auto"}, !m1_16 {mhlo.layout_mode = "default"})
  attributes {topology=#topology} {

  // CHECK-NEXT: fragment
  // CHECK-SAME: {arg_attrs = [{}], res_attrs = [{}]}
  %f = mpmd.fragment<mesh="m1", origin=[]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    mpmd.return %arg0 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  func.return %f, %f : !m1_16, !m1_16
}

// -----

!m1_16x16 = !mpmd.mesh_tensor<"m1", tensor<16x16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8,"y"=8]>>>

// expected-error@+1 {{Arg #0 is returned as result #0, but with incompatible layouts: "{0, 1}" vs. "{1, 0}"}}
func.func @error_return_func_return_with_custom_layout(
    %arg0: !m1_16x16 {mhlo.layout_mode = "{0, 1}"})
      -> (!m1_16x16 {mhlo.layout_mode = "{1, 0}"})
      attributes {topology=#topology} {
  func.return %arg0 : !m1_16x16
}

// -----

!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>>

// expected-error@+1 {{Arg #0 is returned as result #1 and result #0, but with incompatible layouts: "{0}" vs. "default"}}
func.func @error_incompatible_func_return_layouts(
  %arg0: !m1_16 {mhlo.layout_mode = "auto"})
    -> (!m1_16 {mhlo.layout_mode = "default"},
        !m1_16 {mhlo.layout_mode = "{0}"})
    attributes {topology=#topology} {
  func.return %arg0, %arg0 : !m1_16, !m1_16
}

// -----

!m1_16 = !mpmd.mesh_tensor<"m1", tensor<16xf32>>
#topology = #mpmd.topology<<"m1": <["x"=8]>>>

func.func @error_return_twice_with_incompatible_layouts(%func_arg: !m1_16) ->
  (!m1_16 {mhlo.layout_mode = "{0}"}, !m1_16 {mhlo.layout_mode = "default"})
  attributes {topology=#topology} {

  %f = mpmd.fragment<mesh="m1", origin=[]> (%func_arg)
    (%arg0: tensor<16xf32>) {
    mpmd.return %arg0 : tensor<16xf32>
  } : (!m1_16) -> !m1_16

  // expected-error@+1 {{Result #0 is also returned as result #1, but with incompatible layouts: "{0}" vs. "default"}}
  func.return %f, %f : !m1_16, !m1_16
}
