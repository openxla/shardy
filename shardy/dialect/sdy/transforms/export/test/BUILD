# Lit tests for the SDY export passes.

load("//shardy:lit.bzl", "glob_lit_tests")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "test_data",
    testonly = True,
    data = [
        "//shardy/tools:sdy_opt",
        "@llvm-project//llvm:FileCheck",
    ],
)

glob_lit_tests(
    name = "all_tests",
    data = [":test_data"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)
