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

#ifndef SDY_CANONICALIZATION
#define SDY_CANONICALIZATION

include "shardy/dialect/sdy/ir/ops.td"
include "mlir/IR/PatternBase.td"

def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

def ReshardOfReshardPattern :
    Pat<(Sdy_ReshardOp (Sdy_ReshardOp:$inner_reshard $tensor, $inner_sharding), $outer_sharding),
        (Sdy_ReshardOp $tensor, $outer_sharding), [(HasOneUse:$inner_reshard)]>;


def AllGatherNoopPattern : Pat<(Sdy_AllGatherOp $tensor, Sdy_EmptyAxesPerDim:$gatheringAxes, $resultSharding),
                               (replaceWithValue $tensor)>;

#endif  // SDY_CANONICALIZATION
