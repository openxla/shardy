load("//third_party/py:py_extension.bzl", "py_extension")
load("//third_party/py:symlink_files.bzl", "symlink_files")

package(
    default_visibility = ["//visibility:public"],
)

# These flags are needed for parse_headers feature.
COPTS = [
    "-fexceptions",
    "-frtti",
]

##---------------------------------------------------------------------------##
# Python extensions.
##---------------------------------------------------------------------------##

py_extension(
    name = "_mlir",
    srcs = [
        "@llvm-project//mlir:lib/Bindings/Python/MainModule.cpp",
    ],
    copts = COPTS,
    # pytype_srcs = [
    #     ":_mlirPyi",
    # ],
    deps = [
        "@llvm-project//mlir:MLIRBindingsPythonCore",
        "@llvm-project//mlir:MLIRBindingsPythonHeadersAndDeps",
    ],
)
