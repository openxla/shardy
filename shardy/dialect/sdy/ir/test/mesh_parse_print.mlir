// RUN: sdy_opt %s 2>&1 | FileCheck %s

// CHECK: sdy.mesh @mesh = <device_ids=[0]>
sdy.mesh @mesh = <device_ids=[0]>

// CHECK: sdy.mesh @mesh2 = <["a"=2]>
sdy.mesh @mesh2 = <["a"=2]>

// CHECK: sdy.mesh @mesh3 = <["a"=2, "b"=1]>
sdy.mesh @mesh3 = <["a"=2, "b"=1]>

// CHECK: sdy.mesh @mesh4 = <>
sdy.mesh @mesh4 = <>
