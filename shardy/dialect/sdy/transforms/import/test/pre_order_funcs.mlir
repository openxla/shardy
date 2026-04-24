// RUN: sdy_opt %s -sdy-pre-order-funcs | FileCheck %s

// CHECK: func.func @main
// CHECK: func.func @func1
// CHECK: func.func @func2

func.func @func2() {
  return
}

func.func @func1() {
  func.call @func2() : () -> ()
  return
}

func.func @main() {
  func.call @func1() : () -> ()
  return
}
