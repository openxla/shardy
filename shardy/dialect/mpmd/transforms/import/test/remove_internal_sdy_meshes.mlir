// RUN: mpmd_opt %s -mpmd-remove-internal-sdy-meshes | FileCheck %s

// CHECK-NOT: sdy.mesh @__mesh1
// CHECK: sdy.mesh @mesh2
module {
  sdy.mesh @__mesh1 = <[]>
  sdy.mesh @mesh2 = <[]>
}
