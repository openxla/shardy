/* Copyright 2025 The Shardy Authors.

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

#include "shardy/round_trip_import/utils.h"

#include <cstdint>
#include <functional>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/round_trip_import/constants.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace sdy {

using ::mlir::func::FuncOp;
using ::mlir::stablehlo::CustomCallOp;

DictionaryAttr getFrontendAttrs(Operation* op) {
  return op->getAttrOfType<DictionaryAttr>(kFrontendAttributesAttr);
}

DictionaryAttr getFuncArgFrontendAttrs(FuncOp funcOp, unsigned int index) {
  return funcOp.getArgAttrOfType<DictionaryAttr>(index,
                                                 kFrontendAttributesAttr);
}

namespace {

SmallVector<NamedAttribute> getExistingFrontendAttributes(
    DictionaryAttr frontendAttributes, StringRef excludedAttribute) {
  SmallVector<NamedAttribute> dictEntries;
  if (!frontendAttributes) {
    return dictEntries;
  }
  for (NamedAttribute entry : frontendAttributes) {
    if (entry.getName() != excludedAttribute) {
      dictEntries.push_back(entry);
    }
  }
  return dictEntries;
}

void removeFrontendAttribute(
    DictionaryAttr frontendAttributes, StringRef attributeName,
    std::function<void(ArrayRef<NamedAttribute>)> setAttr,
    std::function<void()> removeAttr) {
  SmallVector<NamedAttribute> existingAttributes =
      getExistingFrontendAttributes(frontendAttributes, attributeName);
  if (!existingAttributes.empty()) {
    setAttr(existingAttributes);
  } else {
    removeAttr();
  }
}

void setFrontendAttrs(Operation* op, ArrayRef<NamedAttribute> frontendAttrs) {
  return op->setAttr(kFrontendAttributesAttr,
                     DictionaryAttr::get(op->getContext(), frontendAttrs));
}

void setFuncArgFrontendAttrs(FuncOp funcOp, unsigned int index,
                             ArrayRef<NamedAttribute> frontendAttrs) {
  funcOp.setArgAttr(index, kFrontendAttributesAttr,
                    DictionaryAttr::get(funcOp.getContext(), frontendAttrs));
}

}  // namespace

void removeFrontendAttribute(Operation* op, StringRef attributeName) {
  removeFrontendAttribute(
      getFrontendAttrs(op), attributeName,
      [&](ArrayRef<NamedAttribute> newDict) { setFrontendAttrs(op, newDict); },
      [&]() { op->removeAttr(kFrontendAttributesAttr); });
}

void removeFrontendAttribute(FuncOp funcOp, StringRef attributeName,
                             int64_t argNum) {
  removeFrontendAttribute(
      getFuncArgFrontendAttrs(funcOp, argNum), attributeName,
      [&](ArrayRef<NamedAttribute> newDict) {
        setFuncArgFrontendAttrs(funcOp, argNum, newDict);
      },
      [&]() { funcOp.removeArgAttr(argNum, kFrontendAttributesAttr); });
}

bool hasFrontendAttr(mlir::Operation* op, mlir::StringRef key) {
  return hasKey(getFrontendAttrs(op), key);
}

bool hasKey(mlir::DictionaryAttr dictAttr, mlir::StringRef key) {
  return dictAttr && dictAttr.contains(key);
}

CustomCallOp cloneCustomCallWithNewResultTypes(CustomCallOp op,
                                               mlir::TypeRange resultTypes,
                                               mlir::IRRewriter& rewriter) {
  auto customCallOp = rewriter.create<CustomCallOp>(
      op.getLoc(), resultTypes, op.getOperands(), op.getCallTargetNameAttr(),
      op.getHasSideEffectAttr(), op.getBackendConfigAttr(),
      op.getApiVersionAttr(), op.getCalledComputations(),
      op.getOperandLayoutsAttr(), op.getResultLayoutsAttr(),
      op.getOutputOperandAliases());
  customCallOp->setDiscardableAttrs(mlir::DictionaryAttr::get(
      op->getContext(), llvm::to_vector(op->getDiscardableAttrs())));
  return customCallOp;
};

bool isPythonCallbackCustomCall(mlir::stablehlo::CustomCallOp op) {
  mlir::StringRef targetName = op.getCallTargetName();
  return targetName == kPythonCpuCallbackCustomCallTargetName ||
         targetName == kPythonGpuCallbackCustomCallTargetName ||
         targetName == kFFIPythonCpuCallbackCustomCallTargetName ||
         targetName == kFFIPythonGpuCallbackCustomCallTargetName;
}

std::string cunescape(llvm::StringRef escapedValue) {
  std::string unescapedValue;
  unescapedValue.reserve(escapedValue.size());

  for (int i = 0; i < escapedValue.size(); i++) {
    if (escapedValue[i] == '\\' && i + 1 < escapedValue.size()) {
      switch (escapedValue[i + 1]) {
        case 'n':
          unescapedValue += '\n';
          break;

        case 't':
          unescapedValue += '\t';
          break;

        case '\\':
          unescapedValue += '\\';
          break;
        case '"':
          unescapedValue += '"';
          break;

        default:
          unescapedValue += escapedValue[i];
          i--;  // To accommodate i++ after this.
          break;
      }
      i++;
    } else {
      unescapedValue += escapedValue[i];
    }
  }

  return unescapedValue;
}

}  // namespace sdy
}  // namespace mlir
