package(default_visibility = ["//visibility:public"])

cc_library(
    name = "file_utils",
    srcs = [
        "file_utils.cc",
        "save_module_op.cc",
    ],
    hdrs = [
        "file_utils.h",
        "save_module_op.h",
    ],
    deps = [
        "//shardy/dialect/sdy/ir:dialect",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)
