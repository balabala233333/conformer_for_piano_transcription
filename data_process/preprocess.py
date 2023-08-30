import dataclasses
import logging
import os.path
import random
from typing import Sequence, Tuple

import numpy
import numpy as np
import torch

from data_process import spectrograms, datasets, note_event_codec, midi_note, tokens_encoding
from constant import TokenConfig
from data_process.data_classes import PreprocessDataset, TrainItem, TestItem
from data_process.note_event_codec import Codec
from data_process.spectrograms import SpectrogramConfig
from data_process.tokens_encoding import PreprocessDataItem
from data_process.vocabulary import TokensVocabulary, build_codec


def audio_to_frames(
        samples: Sequence[float],
        spectrogram_config: spectrograms.SpectrogramConfig,
) -> Tuple[Sequence[Sequence[int]], np.ndarray]:
    """

    Args:
        samples: 采样结果
        spectrogram_config: 频谱图的配置

    Returns: 按帧分类的结果

    """
    frame_size = spectrogram_config.hop_width
    logging.info('Padding %d samples to multiple of %d', len(samples), frame_size)
    samples = np.pad(samples,
                     [0, frame_size - len(samples) % frame_size],
                     mode='constant')
    frames = spectrograms.split_audio(samples, spectrogram_config)

    num_frames = len(samples) // frame_size
    logging.info('Encoded %d samples to %d frames (%d samples each)',
                 len(samples), num_frames, frame_size)

    times = np.arange(num_frames) / spectrogram_config.frames_per_second
    times = torch.Tensor(times)
    return frames, times


def split_data(dataset: PreprocessDataset, config: TokenConfig) -> Sequence[PreprocessDataItem]:
    inputs_tmp_tensor = torch.tensor(dataset.inputs)
    input_times_tmp_tensor = torch.tensor(dataset.input_times)
    input_event_start_indices_tmp_tensor = torch.tensor(dataset.input_event_start_indices)
    input_event_end_indices_tmp_tensor = torch.tensor(dataset.input_event_end_indices)

    max_length = config.max_num_cached_frames
    inputs_slices = torch.split(inputs_tmp_tensor, max_length)
    input_times_slices = torch.split(input_times_tmp_tensor, max_length)
    input_event_start_indices_slices = torch.split(input_event_start_indices_tmp_tensor, max_length)
    input_event_end_indices_slices = torch.split(input_event_end_indices_tmp_tensor, max_length)

    data = (inputs_slices, input_times_slices, input_event_start_indices_slices, input_event_end_indices_slices)


    split_cuts = []
    for inputs, input_times, input_event_start_indic, input_event_end_indic in zip(inputs_slices, input_times_slices,
                                                                                   input_event_start_indices_slices,
                                                                                   input_event_end_indices_slices):
        split_cuts.append(
            PreprocessDataItem(inputs=inputs, input_times=input_times, input_event_start_indic=input_event_start_indic,
                               input_event_end_indic=input_event_end_indic, targets=dataset.targets))

    return split_cuts


def compute_spectrogram(dataset: PreprocessDataItem, spectrogram_config) -> PreprocessDataItem:
    samples = np.reshape(dataset.inputs.numpy(), [-1])
    dataset.inputs = spectrograms.compute_spectrogram(samples, spectrogram_config)
    dataset.inputs = torch.tensor(dataset.inputs)
    return dataset


def select_random_chunk(item: PreprocessDataItem, config: TokenConfig) -> PreprocessDataItem:
    length = config.inputs_length
    n_tokens = item.inputs.shape[0]
    # print(length)
    start = random.randint(-length + 1, n_tokens - 1)
    end = min(start + length, n_tokens)
    start = max(start, 0)
    item.inputs = item.inputs[start:end]
    item.input_times = item.input_times[start:end]
    item.input_event_end_indic = item.input_event_end_indic[start:end]
    item.input_event_start_indic = item.input_event_start_indic[start:end]
    return item


def tokenize_example(pair: datasets.MidiAudioPair,
                     spectrogram_config: spectrograms.SpectrogramConfig,
                     codec: note_event_codec.Codec) -> PreprocessDataset:
    """

    Args:
        pair: midi和audio对
        spectrogram_config: 频谱的配置
        codec: 字母表

    Returns:

    """
    audio = pair.audio
    ns = pair.midi
    print(f"process wav_file {pair.audio_file_name} midi_file {pair.midi_file_name} ")
    frames, frame_times = audio_to_frames(audio, spectrogram_config)
    times, values = midi_note.trans_note_sequence_to_onsets_and_offsets_event(ns)
    (events, event_start_indices, event_end_indices,
     state_events, state_event_indices) = tokens_encoding.encode_and_index_events(event_times=times,
                                                                                  event_values=values,
                                                                                  encode_event_fn=midi_note.note_event_data_to_events,
                                                                                  codec=codec,
                                                                                  frame_times=frame_times)
    return PreprocessDataset(id=pair.id, inputs=frames, input_times=frame_times, targets=events,
                             input_event_start_indices=event_start_indices, input_event_end_indices=event_end_indices)


def tokenize_pedal(pair: datasets.MidiAudioPair,
                   spectrogram_config: spectrograms.SpectrogramConfig,
                   codec: note_event_codec.Codec) -> PreprocessDataset:
    """

    Args:
        pair: midi和audio对
        spectrogram_config: 频谱的配置
        codec: 字母表

    Returns:

    """
    audio = pair.audio
    ns = pair.midi
    print(f"process wav_file {pair.audio_file_name} midi_file {pair.midi_file_name} ")
    frames, frame_times = audio_to_frames(audio, spectrogram_config)
    times, values = midi_note.trans_note_sequence_to_padel_event(ns)
    (events, event_start_indices, event_end_indices,
     state_events, state_event_indices) = tokens_encoding.encode_and_index_events(event_times=times,
                                                                                  event_values=values,
                                                                                  encode_event_fn=midi_note.note_event_data_to_events,
                                                                                  codec=codec,
                                                                                  frame_times=frame_times)
    return PreprocessDataset(id=pair.id, inputs=frames, input_times=frame_times, targets=events,
                             input_event_start_indices=event_start_indices, input_event_end_indices=event_end_indices)


def tokenize_note_pedal(pair: datasets.MidiAudioPair,
                   spectrogram_config: spectrograms.SpectrogramConfig,
                   codec: note_event_codec.Codec) -> PreprocessDataset:
    """

    Args:
        pair: midi和audio对
        spectrogram_config: 频谱的配置
        codec: 字母表

    Returns:

    """
    audio = pair.audio
    ns = pair.midi
    print(f"process wav_file {pair.audio_file_name} midi_file {pair.midi_file_name} ")
    frames, frame_times = audio_to_frames(audio, spectrogram_config)
    times, values = midi_note.trans_note_sequence_to_padel_note_event(ns)
    (events, event_start_indices, event_end_indices,
     state_events, state_event_indices) = tokens_encoding.encode_and_index_events(event_times=times,
                                                                                  event_values=values,
                                                                                  encode_event_fn=midi_note.note_event_data_to_events,
                                                                                  codec=codec,
                                                                                  frame_times=frame_times)
    return PreprocessDataset(id=pair.id, inputs=frames, input_times=frame_times, targets=events,
                             input_event_start_indices=event_start_indices, input_event_end_indices=event_end_indices)


def trans_preprocess_data_item_to_train_data(data_item: PreprocessDataItem, vocabulary: TokensVocabulary,
                                             token_config: TokenConfig) -> TrainItem:
    targets = vocabulary.decode_tokens(data_item.targets)
    inputs = data_item.inputs
    inputs = torch.nn.functional.pad(inputs, (0, 0, 0, 2*token_config.inputs_length - inputs.shape[0]), mode='constant',
                                     value=0)

    target_lengths = torch.tensor([targets.shape[0]])
    input_lengths = torch.tensor([inputs.shape[0]])
    train_item = TrainItem(inputs=inputs, input_lengths=input_lengths, targets=targets, target_lengths=target_lengths)
    return train_item


def preprocess_midi_and_audio_to_tokens(pair: datasets.FileNamePair,
                                        spectrogram_config: spectrograms.SpectrogramConfig,
                                        codec: note_event_codec.Codec,
                                        token_config: TokenConfig, vocabulary: TokensVocabulary) -> Sequence[TrainItem]:
    cache_path = os.path.join(pair.cache_data_path, pair.id + ".pt")

    if os.path.exists(cache_path):
        print(f"load cache {cache_path}")
        split_data_items = torch.load(cache_path)
    else:
        pair = datasets.trans_path_to_raw_data(pair)
        preprocessed_data = tokenize_example(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
        split_data_items = split_data(dataset=preprocessed_data, config=token_config, cache_path=pair.cache_path)
    result = []

    for split_data_item in split_data_items:
        random_data_item = select_random_chunk(split_data_item, token_config)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, codec)
        target_sequences = compute_spectrogram(target_sequences, spectrogram_config)
        train_item = trans_preprocess_data_item_to_train_data(target_sequences, vocabulary, token_config)
        result.append(train_item)

    return result


def test_midi_and_audio_to_tokens(pair: datasets.FileNamePair, spectrogram_config: SpectrogramConfig, codec: Codec,
                                  tokens_config: TokenConfig,
                                  vocabularies: TokensVocabulary):
    """
    直接将一个wav和audio对处理好，然后切片
    Args:
        pair: wav和audio对
        spectrogram_config: 频谱图配置
        codec: 单词表
        tokens_config:  模型配置
        vocabularies: 单词表

    Returns: 切好的音频和midi段

    """
    tokens_config.max_num_cached_frames = tokens_config.inputs_length
    pair = datasets.trans_path_to_raw_data(pair)
    preprocessed_data = tokenize_example(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
    split_data_items = split_data(dataset=preprocessed_data, config=tokens_config)

    result = []
    for split_data_item in split_data_items:
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(split_data_item)
        input_times = target_sequences.input_times
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, codec)
        target_sequences = compute_spectrogram(target_sequences, spectrogram_config)
        test_item = trans_preprocess_data_item_to_train_data(target_sequences, vocabularies, TokenConfig)
        result.append(TestItem(input_times=input_times, inputs=test_item.inputs, input_lengths=test_item.input_lengths,
                               target_lengths=test_item.target_lengths, targets=test_item.targets, pred=None,
                               midi_name=pair.midi_file_name))

    return result


def test_midi_and_audio_to_pedal_tokens(pair: datasets.FileNamePair, spectrogram_config: SpectrogramConfig, codec: Codec,
                                  tokens_config: TokenConfig,
                                  vocabularies: TokensVocabulary):
    """
    直接将一个wav和audio对处理好，然后切片
    Args:
        pair: wav和audio对
        spectrogram_config: 频谱图配置
        codec: 单词表
        tokens_config:  模型配置
        vocabularies: 单词表

    Returns: 切好的音频和踏板段

    """
    tokens_config.max_num_cached_frames = tokens_config.inputs_length
    pair = datasets.trans_path_to_raw_data(pair)
    preprocessed_data = tokenize_pedal(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
    split_data_items = split_data(dataset=preprocessed_data, config=tokens_config)

    result = []
    for split_data_item in split_data_items:
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(split_data_item)
        input_times = target_sequences.input_times
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, codec)
        target_sequences = compute_spectrogram(target_sequences, spectrogram_config)
        test_item = trans_preprocess_data_item_to_train_data(target_sequences, vocabularies, TokenConfig)
        result.append(TestItem(input_times=input_times, inputs=test_item.inputs, input_lengths=test_item.input_lengths,
                               target_lengths=test_item.target_lengths, targets=test_item.targets, pred=None,
                               midi_name=pair.midi_file_name))

    return result

def test_midi_and_audio_to_note_pedal_tokens(pair: datasets.FileNamePair, spectrogram_config: SpectrogramConfig, codec: Codec,
                                  tokens_config: TokenConfig,
                                  vocabularies: TokensVocabulary):
    """
    直接将一个wav和audio对处理好，然后切片
    Args:
        pair: wav和audio对
        spectrogram_config: 频谱图配置
        codec: 单词表
        tokens_config:  模型配置
        vocabularies: 单词表

    Returns: 切好的音频和踏板段

    """
    tokens_config.max_num_cached_frames = tokens_config.inputs_length
    pair = datasets.trans_path_to_raw_data(pair)
    preprocessed_data = tokenize_note_pedal(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
    split_data_items = split_data(dataset=preprocessed_data, config=tokens_config)

    result = []
    for split_data_item in split_data_items:
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(split_data_item)
        input_times = target_sequences.input_times
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, codec)
        target_sequences = compute_spectrogram(target_sequences, spectrogram_config)
        test_item = trans_preprocess_data_item_to_train_data(target_sequences, vocabularies, TokenConfig)
        result.append(TestItem(input_times=input_times, inputs=test_item.inputs, input_lengths=test_item.input_lengths,
                               target_lengths=test_item.target_lengths, targets=test_item.targets, pred=None,
                               midi_name=pair.midi_file_name))

    return result


def trans_conformer_test_item_to_ismir2021_test_item(test_items: Sequence[TestItem], vocabulary: TokensVocabulary,
                                                     token_config: TokenConfig):
    encoder_input_tokens = []
    decoder_target_tokens = []
    decoder_input_tokens = []
    decoder_loss_weights = []

    for test_item in test_items:
        start, end = np.argmin(test_item.targets == vocabulary.start_of_tokens), np.argmax(
            test_item.targets == vocabulary.end_of_tokens)
        encoder_input_tokens.append(test_item.inputs)
        decoder_input_tokens.append(test_item.targets[start:])
        decoder_target_tokens.append(test_item.targets[start:end])

    encoder_input_tokens = torch.stack(encoder_input_tokens)
    decoder_target_tokens = torch.nn.utils.rnn.pad_sequence(decoder_target_tokens, batch_first=True)
    decoder_input_tokens = torch.nn.utils.rnn.pad_sequence(decoder_input_tokens, batch_first=True)
    decoder_input_tokens = torch.cat((decoder_input_tokens, torch.zeros(decoder_input_tokens.shape[0],
                                                                        token_config.targets_length -
                                                                        decoder_input_tokens.shape[1])), dim=1)
    decoder_target_tokens = torch.cat((decoder_target_tokens, torch.zeros(decoder_target_tokens.shape[0],
                                                                          token_config.targets_length -
                                                                          decoder_target_tokens.shape[1])), dim=1)
    print(encoder_input_tokens.shape, decoder_target_tokens.shape, decoder_input_tokens.shape)


