# The SDY transformations common library.

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "op_properties",
    srcs = ["op_properties.cc"],
    hdrs = ["op_properties.h"],
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
        "@stablehlo//:base",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
)

cc_test(
    name = "op_properties_test",
    srcs = ["op_properties_test.cc"],
    deps = [
        ":op_properties",
        "//shardy/dialect/sdy/ir:register",
        "@com_google_googletest//:gtest_main",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Parser",
        "@llvm-project//mlir:Support",
        "@stablehlo//:stablehlo_ops",
    ],
)

cc_library(
    name = "sharding_walker",
    srcs = ["sharding_walker.cc"],
    hdrs = ["sharding_walker.h"],
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Support",
    ],
)
