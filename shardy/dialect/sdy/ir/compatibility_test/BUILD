# Lit tests for the SDY dialect.

load("//shardy:lit.bzl", "glob_lit_tests")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "test_data",
    testonly = True,
    data = [
        ":compatibility_test.0_0_0.mlir.bc",
        ":compatibility_test.0_0_1.mlir.bc",
        ":compatibility_test.0_0_2.mlir.bc",
        "//shardy/tools:sdy_opt",
        "//shardy/tools:sdy_translate",
        "@llvm-project//llvm:FileCheck",
        # copybara:uncomment "@llvm-project//mlir:run_lit",
    ],
)

glob_lit_tests(
    name = "compatibility_tests",
    data = [":test_data"],
    driver = "@llvm-project//mlir:run_lit.sh",
    test_file_exts = ["mlir"],
)
