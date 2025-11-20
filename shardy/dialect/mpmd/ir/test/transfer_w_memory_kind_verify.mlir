// RUN: mpmd_opt %s -verify-diagnostics -split-input-file

!m_undefined = !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>>
!m_device = !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>, memory_kind = "device">
!m_host = !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>, memory_kind = "device">
!m_invalid = !mpmd.mesh_tensor<"mesh", tensor<12x16xf32>, sharding=<@mesh, [{?}, {"y"}]>, memory_kind = "qwerty">

func.func @f(%arg0 : !m_device)
    attributes {"topology"=#mpmd.topology<<"mesh": <["x"=2, "y"=4]>>>}
{
  %t1 = mpmd.transfer %arg0 : (!m_device) -> !m_undefined  // No error.
  %t2 = mpmd.transfer %t1 : (!m_undefined) -> !m_host      // No error.
  // expected-error@+1 {{memory kind must be either 'pinned_host' or 'unpinned_host' or 'device'. Found 'qwerty'.}}
  %t3 = mpmd.transfer %t2 : (!m_host) -> !m_invalid
  func.return
}

// -----

!m1_type = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!m2_type = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func private @transfer_on_device_is_allowed(%arg0 : !m1_type {mhlo.memory_kind = "device"}) -> !m2_type
  attributes {topology=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>, <"m2": <["x"=2, "y"=4]>>>}
{
  %t = mpmd.transfer %arg0 : (!m1_type) -> !m2_type
  func.return %t : !m2_type
}

// -----

!m1_type = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!m2_type = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func private @arg_cannot_be_pinned_to_host_if_transferred(%arg0 : !m1_type {mhlo.memory_kind = "pinned_host"}) -> !m2_type
  attributes {topology=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>, <"m2": <["x"=2, "y"=4]>>>}
{
// expected-error@+1 {{Transfers from host with attributes are not supported. Memory kinds must be expressed in the type.}}
  %t = mpmd.transfer %arg0 : (!m1_type) -> !m2_type
  func.return %t : !m2_type
}

// -----

!m1_type = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!m2_type = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>


func.func private @fragment_result_cannot_be_pinned_to_host_if_transferred(%arg0 : !m1_type {mhlo.memory_kind = "pinned_host"}) -> !m2_type
  attributes {topology=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>, <"m2": <["x"=2, "y"=4]>>>}
{
  %f:2 = mpmd.fragment<mesh="m1", origin=[]> (%arg0)
    {arg_attrs = [{mhlo.memory_kind = "pinned_host"}],
     res_attrs = [{mhlo.memory_kind = "pinned_host"}, {mhlo.memory_kind = "device"}]}
  (%arg1: tensor<4x8xf32>) {
    mpmd.return %arg1, %arg1 : tensor<4x8xf32>, tensor<4x8xf32>
  } : (!m1_type) -> (!m1_type, !m1_type)

// expected-error@+1 {{Transfers from host with attributes are not supported. Memory kinds must be expressed in the type.}}
  %t = mpmd.transfer %f#0 : (!m1_type) -> !m2_type
  func.return %t : !m2_type
}

// -----

!m1_type = !mpmd.mesh_tensor<"m1", tensor<4x8xf32>>
!m2_type = !mpmd.mesh_tensor<"m2", tensor<4x8xf32>>

func.func private @transfer_to_host_with_attributes_is_not_allowed(%arg0 : !m1_type) -> !m2_type
  attributes {topology=#mpmd.topology<<"m1": <["x"=2, "y"=4]>>, <"m2": <["x"=2, "y"=4]>>>}
{
// expected-error@+1 {{Transfers to host with attributes are not supported. Memory kinds must expressed be in the type.}}
  %t = mpmd.transfer {mhlo.memory_kind = "pinned_host"} %arg0 : (!m1_type) -> !m2_type
  func.return %t : !m2_type
}
