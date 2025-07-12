/* Copyright 2024 The Shardy Authors.

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

#ifndef SDY_ENUMS
#define SDY_ENUMS

include "shardy/dialect/sdy/ir/dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

// Direction of propagation.
//
// The value of the enum is important in what direction is the most
// expressive/important. Specifically that `BOTH` is the largest value,
// and that `NONE` is the smallest value. `BACKWARD` being a higher value than
// `FORWARD` is arbitrary, and they should be considered equivalent.
def Sdy_PropagationDirection :
    I32EnumAttr<"PropagationDirection",
        "propagation direction enum", [
        I32EnumAttrCase<"NONE", 0>,
        I32EnumAttrCase<"FORWARD", 1>,
        I32EnumAttrCase<"BACKWARD", 2>,
        I32EnumAttrCase<"BOTH", 3>]> {
  let genSpecializedAttr = 1;
  let cppNamespace = Sdy_Dialect.cppNamespace;
}

// Type of value referenced by a propagation edge.
//
// Edge shardings correspond to either the operand or result of an operation.
// This enum distinguishes between the two.
def Sdy_EdgeNodeType :
    I32EnumAttr<"EdgeNodeType",
        "edge node type enum", [
        I32EnumAttrCase<"OPERAND", 0, "operand">,
        I32EnumAttrCase<"RESULT", 1, "result">]> {
  let genSpecializedAttr = 1;
  let cppNamespace = Sdy_Dialect.cppNamespace;
}

#endif  // SDY_ENUMS
