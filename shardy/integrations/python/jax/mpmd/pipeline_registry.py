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

"""Pipeline schedule registry.

Central registry mapping schedule names to PipelineSchedule objects. Each
schedule defines fragment merging and ordering using binary predicate functions.

Usage:
    schedule = get_pipeline_schedule('ONE_FWD_ONE_BWD')
    config = make_config(
        topology=topology,
        name_to_mesh_assignment=mesh_assignment,
        pipeline_schedule=schedule,
    )
"""

import functools
from typing import Mapping

from shardy.integrations.python.jax.mpmd import pipeline
from shardy.integrations.python.jax.mpmd import types


PIPELINE_SCHEDULES: Mapping[str, types.PipelineSchedule] = {
    'ONE_FWD_ONE_BWD': types.PipelineSchedule(
        merge_predicates=None,
        schedule_predicates=[pipeline.one_fwd_one_bwd_schedule_predicate],
        required_mpmd_options={
            'partir_mpmd_pipeline_schedule': 'ONE_FWD_ONE_BWD'
        },
    ),
    'GPIPE': types.PipelineSchedule(
        merge_predicates=None,
        schedule_predicates=[pipeline.gpipe_schedule_predicate],
        required_mpmd_options={'partir_mpmd_pipeline_schedule': 'GPIPE'},
    ),
    'GPIPE_BUT_1F1B_FOR_LAST_MESH': types.PipelineSchedule(
        merge_predicates=None,
        schedule_predicates=[
            pipeline.gpipe_with_1f1b_on_last_mesh_schedule_predicate
        ],
        required_mpmd_options={
            'partir_mpmd_pipeline_schedule': 'GPIPE_BUT_1F1B_FOR_LAST_MESH'
        },
    ),
    'ZERO_BUBBLE_H1': types.PipelineSchedule(
        merge_predicates=None,
        schedule_predicates=[pipeline.zero_bubble_h1_schedule_predicate],
        required_mpmd_options={
            'partir_mpmd_pipeline_schedule': 'ZERO_BUBBLE_H1'
        },
    ),
    'ZERO_BUBBLE_H2_ZERO_TX_LATENCY': types.PipelineSchedule(
        merge_predicates=None,
        schedule_predicates=[
            functools.partial(
                pipeline.latency_hiding_zero_bubble_h2_schedule_predicate,
                latency_stage_fraction=0.0,
            )
        ],
        required_mpmd_options={
            'partir_mpmd_split_bwd_fragments': True,
            'partir_mpmd_pipeline_schedule': 'ZERO_BUBBLE_H2_ZERO_TX_LATENCY',
        },
    ),
    'ZERO_BUBBLE_H2_HALF_TX_LATENCY': types.PipelineSchedule(
        merge_predicates=None,
        schedule_predicates=[
            functools.partial(
                pipeline.latency_hiding_zero_bubble_h2_schedule_predicate,
                latency_stage_fraction=0.5,
            )
        ],
        required_mpmd_options={
            'partir_mpmd_split_bwd_fragments': True,
            'partir_mpmd_pipeline_schedule': 'ZERO_BUBBLE_H2_HALF_TX_LATENCY',
        },
    ),
    'ZERO_BUBBLE_H2_FULL_TX_LATENCY': types.PipelineSchedule(
        merge_predicates=None,
        schedule_predicates=[
            functools.partial(
                pipeline.latency_hiding_zero_bubble_h2_schedule_predicate,
                latency_stage_fraction=1.0,
            )
        ],
        required_mpmd_options={
            'partir_mpmd_split_bwd_fragments': True,
            'partir_mpmd_pipeline_schedule': 'ZERO_BUBBLE_H2_FULL_TX_LATENCY',
        },
    ),
    'PARALLEL_PIPELINES_WITH_WRAP_AROUND': types.PipelineSchedule(
        merge_predicates=None,
        schedule_predicates=[
            pipeline.parallel_pipelines_with_wraparound_schedule_predicate
        ],
        required_mpmd_options={
            'partir_mpmd_pipeline_schedule': (
                'PARALLEL_PIPELINES_WITH_WRAP_AROUND'
            ),
        },
    ),
    'CIRCULAR': types.PipelineSchedule(
        merge_predicates=None,
        schedule_predicates=[
            functools.partial(
                pipeline.circular_schedule_predicate_base,
                reverse_backward=False,
            )
        ],
        required_mpmd_options={
            'partir_mpmd_pipeline_schedule': 'CIRCULAR',
        },
    ),
    'CIRCULAR_WITH_REVERSED_BACKWARD': types.PipelineSchedule(
        merge_predicates=None,
        schedule_predicates=[
            functools.partial(
                pipeline.circular_schedule_predicate_base,
                reverse_backward=True,
            )
        ],
        required_mpmd_options={
            'partir_mpmd_pipeline_schedule': 'CIRCULAR_WITH_REVERSED_BACKWARD',
        },
    ),
}


def get_pipeline_schedule(schedule_name: str) -> types.PipelineSchedule:
  """Get a PipelineSchedule object for the given schedule name."""
  if schedule_name not in PIPELINE_SCHEDULES:
    valid_schedules = sorted(PIPELINE_SCHEDULES.keys())
    raise KeyError(
        f"Unknown pipeline schedule '{schedule_name}'. "
        f'Valid schedules are: {valid_schedules}'
    )
  return PIPELINE_SCHEDULES[schedule_name]
