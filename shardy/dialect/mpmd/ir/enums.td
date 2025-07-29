/* Copyright 2025 The MPMD Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef MPMD_ENUMS
#define MPMD_ENUMS

include "mlir/IR/AttrTypeBase.td"
include "shardy/dialect/mpmd/ir/dialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"


def Mpmd_ReductionType : I32EnumAttr<"ReductionType", "reduction type attribute", [
                           // The `none` reduction type is a special case where
                           // all tensors to reduce are identical and the
                           // reduction acts as an identity function.
                           I32EnumAttrCase<"kNone", 0, "none">,
                           I32EnumAttrCase<"kAdd", 1, "add">,
                           I32EnumAttrCase<"kMax", 2, "max">,
                           I32EnumAttrCase<"kMin", 3, "min">,
                           I32EnumAttrCase<"kMul", 4, "mul">,
                           I32EnumAttrCase<"kOr",  5, "or">,
                           I32EnumAttrCase<"kAnd", 6, "and">]> {
  let genSpecializedAttr = 0;
  let cppNamespace = Mpmd_Dialect.cppNamespace;
}

#endif  // MPMD_ENUMS
