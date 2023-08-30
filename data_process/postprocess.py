from typing import Sequence, Mapping, Any

import note_seq
import numpy
import numpy as np

from constant import TokenConfig
from data_process import note_event_codec
from data_process.midi_note import decode_note_event, flush_decoding_result_from_state
from data_process.preprocess import TestItem
from data_process.tokens_decoding import decode_events, NoteDecodingState
from data_process.vocabulary import TokensVocabulary, build_codec


def postprocess(tokens: numpy.ndarray, input_times: numpy.ndarray, codec: note_event_codec.Codec,
                vocabulary: TokensVocabulary):
    start, end = np.argmin(tokens == vocabulary.start_of_tokens), np.argmax(tokens == vocabulary.end_of_tokens)
    tokens = tokens[start:end]
    tokens = tokens - vocabulary.num_special_tokens
    start_time = input_times[0]
    start_time -= start_time % (1 / codec.steps_per_second)
    return {
        'est_tokens': tokens,
        'start_time': start_time,
        'raw_inputs': []
    }


def decode_and_combine_predictions(state: NoteDecodingState, predictions: Sequence[Mapping[str, Any]],
                                   codec: note_event_codec.Codec):
    sorted_predictions = sorted(predictions, key=lambda pred: pred['start_time'])
    total_invalid_events = 0
    total_dropped_events = 0
    for pred_idx, pred in enumerate(sorted_predictions):
        max_decode_time = None
        if pred_idx < len(sorted_predictions) - 1:
            max_decode_time = sorted_predictions[pred_idx + 1]['start_time']

        invalid_events, dropped_events = decode_events(
            state, pred['est_tokens'], pred['start_time'], max_decode_time, codec, decode_note_event)
        total_invalid_events += invalid_events
        total_dropped_events += dropped_events
    return flush_decoding_result_from_state(state), total_invalid_events, total_dropped_events


def decode_predictions_to_note_sequence(state: NoteDecodingState, predictions: Sequence[Mapping[str, Any]],
                                        codec: note_event_codec.Codec):
    ns, total_invalid_events, total_dropped_events = decode_and_combine_predictions(
        state, predictions, codec)
    sorted_predictions = sorted(predictions, key=lambda pred: pred['start_time'])
    raw_inputs = np.concatenate(
        [pred['raw_inputs'] for pred in sorted_predictions], axis=0)
    start_times = [pred['start_time'] for pred in sorted_predictions]

    return {
        'raw_inputs': raw_inputs,
        'start_times': start_times,
        'est_ns': ns,
        'est_invalid_events': total_invalid_events,
        'est_dropped_events': total_dropped_events,
    }


def trans_tokens_to_midi(items: Sequence[TestItem], midi_path: str):
    tokens_config = TokenConfig()
    codec = build_codec()
    state = NoteDecodingState()
    vocabularies = TokensVocabulary(codec.num_classes)
    predictions = []
    for item in items:
        item.pred = item.pred.cpu().numpy()
        res = postprocess(item.pred, item.input_times.numpy(), codec, vocabularies)
        predictions.append(res)
    pred = decode_predictions_to_note_sequence(state, predictions, codec)["est_ns"]
    note_seq.sequence_proto_to_midi_file(pred, midi_path)
    return pred
