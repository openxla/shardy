# Copyright 2025 The MPMD Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines options for MPMD partitioning."""

MPMD_BOOLEAN_OPTIONS = frozenset({
    'mpmd_infer_transfers',
    'mpmd_infer_cross_mesh_reductions',
    'mpmd_merge_inferred_with_cloning_during_import',
    'mpmd_fragment_remat',
    'mpmd_merge_remat_fragments',
    'mpmd_split_bwd_fragments',
    'mpmd_assume_homogeneous_devices',
    'mpmd_absorb_inferred_fragments_on_entry_point_function',
    'mpmd_copy_constant_creation_from_producer_to_consumer',
    'mpmd_apply_merge_transfers_pass',
    'mpmd_merge_inferred_after_scheduling',
})

MPMD_PIPELINE_SCHEDULE_OPTION = 'mpmd_pipeline_schedule'

MPMD_PIPELINE_SCHEDULE_VALUES = frozenset({
    'None',
    '1F1B',
    'GPipe',
    'Circular',
    'CircularWithReversedBackward',
    'GPipeBut1F1BForLastMesh',
    'ZeroBubbleH1',
    'ZeroBubbleH2ZeroTxLatency',
    'ZeroBubbleH2HalfTxLatency',
    'ZeroBubbleH2FullTxLatency',
    'ParallelPipelinesWithWrapAround',
})

MPMD_OPTIONS = MPMD_BOOLEAN_OPTIONS | frozenset({MPMD_PIPELINE_SCHEDULE_OPTION})
