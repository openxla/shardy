// RUN: mpmd_opt %s -mpmd-import-pipeline 2>&1 | FileCheck %s

// CHECK-LABEL: module @integration_test
module @integration_test {
  // CHECK-DAG: sdy.mesh @m1 = <["x"=2]>
  // CHECK-DAG: sdy.mesh @m2 = <["y"=4]>
  // CHECK-NOT: sdy.mesh @mesh
  // CHECK-NOT: sdy.mesh @__mesh
  sdy.mesh @mesh = <["x"=2, "y"=4]>

  // CHECK: func.func @main
  func.func @main() -> () attributes {
    topology = #mpmd.topology<<"m1": <["x"=2]>>, <"m2": <["y"=4]>>>
  } {
    return
  }
}
