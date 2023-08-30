import dataclasses
from typing import Callable, Sequence, Tuple, Optional

import numpy as np
import torch

from constant import BLANK_ID
from data_process import note_event_codec
from data_process.data_classes import NoteEventItem, PreprocessDataItem

from data_process.note_event_codec import Event


def encode_and_index_events(
        event_times: Sequence[float],
        event_values: Sequence[NoteEventItem],
        encode_event_fn: Callable[[NoteEventItem], Sequence[note_event_codec.Event]],
        codec: note_event_codec.Codec,
        frame_times: Sequence[float]
) -> Tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
    indices = np.argsort(event_times, kind='stable')
    event_steps = [round(event_times[i] * codec.steps_per_second)
                   for i in indices]
    event_values = [event_values[i] for i in indices]
    events = []
    state_events = []
    event_start_indices = []
    state_event_indices = []

    cur_step = 0
    cur_event_idx = 0
    cur_state_event_idx = 0

    def fill_event_start_indices_to_cur_step():
        while (len(event_start_indices) < len(frame_times) and
               frame_times[len(event_start_indices)] <
               cur_step / codec.steps_per_second):
            event_start_indices.append(cur_event_idx)
            state_event_indices.append(cur_state_event_idx)

    for event_step, event_value in zip(event_steps, event_values):
        while event_step > cur_step:
            events.append(codec.encode_event(Event(type='shift', value=1)))
            cur_step += 1
            fill_event_start_indices_to_cur_step()
            cur_event_idx = len(events)
            cur_state_event_idx = len(state_events)
        for e in encode_event_fn(event_value):
            events.append(codec.encode_event(e))

    while cur_step / codec.steps_per_second <= frame_times[-1]:
        events.append(codec.encode_event(Event(type='shift', value=1)))
        cur_step += 1
        fill_event_start_indices_to_cur_step()
        cur_event_idx = len(events)

    event_end_indices = event_start_indices[1:] + [len(events)]
    events = np.array(events)
    state_events = np.array(state_events)
    event_start_indices = np.array(event_start_indices)
    event_end_indices = np.array(event_end_indices)
    state_event_indices = np.array(state_event_indices)

    return (events, event_start_indices, event_end_indices,
            state_events, state_event_indices)


def encoding_target_sequence_with_indices(dataset: PreprocessDataItem) -> PreprocessDataItem:
    target_start_idx = dataset.input_event_start_indic[0]
    target_end_idx = dataset.input_event_end_indic[-1]
    dataset.targets = dataset.targets[target_start_idx:target_end_idx]
    return dataset


def encoding_shifts(dataset: PreprocessDataItem, codec: note_event_codec.Codec) -> PreprocessDataItem:
    shift_steps = 0
    total_shift_steps = 0
    events = torch.tensor(dataset.targets)
    # print(events)
    output = torch.tensor([])

    for event in events:
        if codec.is_shift_event_index(event):
            shift_steps += 1
            total_shift_steps += 1

        else:
            if shift_steps > 0:
                shift_steps = total_shift_steps
                while shift_steps > 0:
                    output_steps = min(codec.max_shift_steps, shift_steps)
                    output = torch.concat([output, torch.tensor([output_steps])], axis=0)
                    shift_steps -= output_steps
            output = torch.concat([output, torch.tensor([event])], axis=0)
    # print(shift_steps,total_shift_steps)
    dataset.targets = output.long()
    return dataset
