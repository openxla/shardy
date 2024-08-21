// RUN: sdy_opt %s 2>&1 | FileCheck %s

// CHECK: sdy.mesh @empty_mesh = <>
sdy.mesh @empty_mesh = <>

// CHECK: sdy.mesh @maximal_mesh_1 = <device_ids=[0]>
sdy.mesh @maximal_mesh_1 = <device_ids=[0]>

// CHECK: sdy.mesh @maximal_mesh_2 = <device_ids=[3]>
sdy.mesh @maximal_mesh_2 = <device_ids=[3]>

// CHECK: sdy.mesh @single_axis = <["a"=2]>
sdy.mesh @single_axis = <["a"=2]>

// CHECK: sdy.mesh @single_axis_explicit_device_ids = <["a"=2]device_ids=[1, 0]>
sdy.mesh @single_axis_explicit_device_ids = <["a"=2] device_ids=[1, 0]>

// CHECK: sdy.mesh @two_axes = <["a"=2, "b"=1]>
sdy.mesh @two_axes = <["a"=2, "b"=1]>

// CHECK: sdy.mesh @two_axes_explicit_device_ids = <["a"=2, "b"=1]device_ids=[1, 0]>
sdy.mesh @two_axes_explicit_device_ids = <["a"=2, "b"=1] device_ids=[1, 0]>
