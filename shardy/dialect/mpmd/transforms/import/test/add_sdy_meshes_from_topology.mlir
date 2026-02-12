// RUN: mpmd_opt %s -mpmd-add-sdy-meshes-from-topology 2>&1 | FileCheck %s

// CHECK-LABEL: module @add_meshes
module @add_meshes {
  // CHECK-DAG: sdy.mesh @mesh = <["x"=2, "y"=4, "z"=8]>
  // CHECK-DAG: sdy.mesh @m1 = <["x"=2, "y"=4]>
  // CHECK-DAG: sdy.mesh @m2 = <["z"=8]>
  sdy.mesh @mesh = <["x"=2, "y"=4, "z"=8]>

  // CHECK: func.func @main
  func.func @main() -> () attributes {
    topology = #mpmd.topology<<"m1": <["x"=2, "y"=4]>>, <"m2": <["z"=8]>>>
  } {
    return
  }
}
