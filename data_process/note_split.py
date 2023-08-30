import random
from math import ceil

import note_seq
import numpy as np
import torch

from data_process import midi_note
from data_process.data_classes import SplitToken
from data_process.midi_note import trans_note_sequence_to_onsets_and_offsets_event
from data_process.spectrograms import SpectrogramConfig
from data_process.vocabulary import build_codec, STEPS_PER_SECOND


def split_note_seq(pred: note_seq.NoteSequence, target: note_seq.NoteSequence, config: SpectrogramConfig) -> SplitToken:
    codec = build_codec()
    # 处理prediction的tokens
    pred_times, pred_events = trans_note_sequence_to_onsets_and_offsets_event(pred)
    pred_indices = np.argsort(pred_times)
    pred_event_steps = [round(pred_times[i] * codec.steps_per_second)
                        for i in pred_indices]
    pred_event_values = [pred_events[i] for i in pred_indices]
    # 处理targets的tokens
    target_times, target_events = trans_note_sequence_to_onsets_and_offsets_event(target)
    target_indices = np.argsort(target_times)
    target_event_steps = [round(target_times[i] * codec.steps_per_second)
                          for i in target_indices]
    target_event_values = [target_events[i] for i in target_indices]
    total_time = max(pred.total_time, target.total_time)
    seconds_per_frame = 1.0 / config.frames_per_second * config.num_mel_bins
    shift = random.uniform(0, seconds_per_frame * codec.steps_per_second / STEPS_PER_SECOND)
    frame_num = ceil((total_time + shift) / (seconds_per_frame * codec.steps_per_second / STEPS_PER_SECOND))
    results = []
    for i in range(frame_num):
        results.append(SplitToken(start_step=round(i * seconds_per_frame * codec.steps_per_second - shift),
                                  end_step=round((i + 1) * seconds_per_frame * codec.steps_per_second - shift), pred=[],
                                  target=[]))
        for time, value in zip(pred_event_steps, pred_event_values):
            if time > results[-1].start_step and time <= results[-1].end_step:
                events = midi_note.note_event_data_to_events(value)
                results[-1].pred.append(time - results[-1].start_step)
                for e in events:
                    results[-1].pred.append(codec.encode_event(e))
        results[-1].pred = torch.LongTensor(results[-1].pred)
        for time, value in zip(target_event_steps, target_event_values):
            if time > results[-1].start_step and time <= results[-1].end_step:
                events = midi_note.note_event_data_to_events(value)
                results[-1].target.append(time - results[-1].start_step)
                for e in events:
                    results[-1].target.append(codec.encode_event(e))
        results[-1].target = torch.LongTensor(results[-1].target)
    return results
