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

#include "shardy/integrations/c/mpmd/attributes.h"

#include <cstdint>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/mpmd/ir/dialect.h"

namespace {

template <typename AttrTy>
AttrTy unwrapAttr(MlirAttribute attr) {
  return mlir::cast<AttrTy>(unwrap(attr));
}

}  // namespace

extern "C" {

//===----------------------------------------------------------------------===//
// UserOriginAttr
//===----------------------------------------------------------------------===//

bool mpmdAttributeIsAUserOriginAttr(MlirAttribute attr) {
  return mlir::isa<mlir::mpmd::UserOriginAttr>(unwrap(attr));
}

MlirAttribute mpmdUserOriginAttrGet(MlirContext ctx, MlirStringRef userName,
                                    int64_t transposeCount) {
  return wrap(mlir::mpmd::UserOriginAttr::get(
      unwrap(ctx), mlir::StringAttr::get(unwrap(ctx), unwrap(userName)),
      transposeCount));
}

MlirStringRef mpmdUserOriginAttrGetUserName(MlirAttribute attr) {
  return wrap(
      unwrapAttr<mlir::mpmd::UserOriginAttr>(attr).getUserName().getValue());
}

int64_t mpmdUserOriginAttrGetTransposeCount(MlirAttribute attr) {
  return unwrapAttr<mlir::mpmd::UserOriginAttr>(attr).getTransposeCount();
}

}  // extern "C"
