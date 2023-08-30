import os
import random

import note_seq
import numpy as np
import torch
from torch.utils.data import Dataset

from constant import TokenConfig, BLANK_ID
from data_process import datasets, spectrograms, tokens_encoding
from data_process.data_classes import PreprocessDataItem
from data_process.preprocess import tokenize_example, split_data, select_random_chunk, compute_spectrogram, \
    trans_preprocess_data_item_to_train_data, test_midi_and_audio_to_tokens, tokenize_pedal, tokenize_note_pedal
from data_process.tokens_decoding import NoteDecodingState
from data_process.vocabulary import TokensVocabulary, build_codec


class RNNTDataset(Dataset):
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.train_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_data_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.path = []
        self.chuck = []
        self.use_cache = use_cache
        for pair in pairs:
            cache_path = os.path.join(config.cache_data_path, pair.id + ".pt")
            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_example(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items, cache_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        random_data_item = select_random_chunk(split_data_item, self.token_config)
        targets = random_data_item.targets
        input_times = random_data_item.input_times
        len = random_data_item.inputs.size(0)
        output = torch.tensor([])
        total_shift_steps = 0
        shift_steps = 0
        for i in range(len):
            events = random_data_item.targets[
                     random_data_item.input_event_start_indic[i]:random_data_item.input_event_end_indic[i]]
            # print(events.shape)
            if (events.shape[0] == 0 or events.shape[0] == 1):
                if events.shape[0]==0:
                    output = torch.cat([output, torch.tensor([-3])], axis=0)
                else:
                    output = torch.cat([output, torch.tensor([-3])], axis=0)
                    shift_steps += 1
                    total_shift_steps += 1
                continue
            for event in events:
                if self.codec.is_shift_event_index(event):
                    shift_steps += 1
                    total_shift_steps += 1
                else:
                    if shift_steps > 0:
                        shift_steps = total_shift_steps
                        while shift_steps > 0:
                            output_steps = min(self.codec.max_shift_steps, shift_steps)
                            output = torch.concat([output, torch.tensor([output_steps])], axis=0)
                            shift_steps -= output_steps
                    output = torch.concat([output, torch.tensor([event])], axis=0)
            output = torch.cat((output, torch.tensor([0])))
        for i in range(self.token_config.inputs_length - random_data_item.inputs.shape[0]):
            output = torch.cat((output, torch.tensor([-3])))
        # print(output)
        targets = output.long()
        targets = self.vocabulary.decode_tokens(targets)
        # print(torch.sum(targets==0))
        inputs = torch.nn.functional.pad(random_data_item.inputs,
                                         (0, 0, 0, self.token_config.inputs_length - random_data_item.inputs.shape[0]),
                                         mode='constant',
                                         value=0)

        target_lengths = torch.tensor([targets.shape[0]])
        input_lengths = torch.tensor([inputs.shape[0]])
        samples = np.reshape(inputs.numpy(), [-1])
        inputs = spectrograms.compute_spectrogram(samples, self.spectrogram_config)
        inputs = torch.tensor(inputs)
        return inputs, input_lengths, targets, target_lengths


class TrainDataset(Dataset):
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.train_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_data_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.path = []
        self.chuck = []
        self.use_cache = use_cache
        for pair in pairs:
            cache_path = os.path.join(config.cache_data_path, pair.id + ".pt")
            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_example(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items, cache_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        random_data_item = select_random_chunk(split_data_item, self.token_config)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        train_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)

        return train_item.inputs, train_item.input_lengths, train_item.targets, train_item.target_lengths


class TestDataset(Dataset):
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.test_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_data_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.path = []
        self.chuck = []
        self.use_cache = use_cache
        for pair in pairs:
            cache_path = os.path.join(config.cache_data_path, pair.id + ".pt")
            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_example(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items, cache_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        random_data_item = select_random_chunk(split_data_item, self.token_config)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        train_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)

        return train_item.inputs, train_item.input_lengths, train_item.targets, train_item.target_lengths


class PedalNoteDataset:
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.train_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_pedal_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.cache_path = []
        self.note_path = []
        self.chuck = []
        self.pedal = []
        self.use_cache = use_cache
        for pair in pairs:
            cache_path = os.path.join(config.cache_data_path, pair.id + ".pt")
            pedal_path = os.path.join(config.cache_pedal_path, pair.id + ".pt")

            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                note_pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_example(pair=note_pair, spectrogram_config=spectrogram_config, codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items, cache_path)
            if os.path.exists(pedal_path) and use_cache:
                print(f"load cache {pedal_path}")
                split_pedal_items = torch.load(pedal_path)
            else:
                pedal_pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_pedal(pair=pedal_pair, spectrogram_config=spectrogram_config, codec=codec)
                split_pedal_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_pedal_items, pedal_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1
            for split_pedal_item in split_pedal_items:
                self.pedal.append(split_pedal_item)

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        split_pedal_item = self.pedal[n]
        length = self.token_config.inputs_length
        n_tokens = split_data_item.inputs.shape[0]
        start = random.randint(-length + 1, n_tokens - 1)
        end = min(start + length, n_tokens)
        start = max(start, 0)
        note_target = split_data_item.targets
        pedal_target = split_pedal_item.targets
        inputs = split_data_item.inputs[start:end]
        input_times = split_data_item.input_times[start:end]
        input_event_end_indic = split_data_item.input_event_end_indic[start:end]
        input_event_start_indic = split_data_item.input_event_start_indic[start:end]
        input_pedal_end_indic = split_pedal_item.input_event_end_indic[start:end]
        input_pedal_start_indic = split_pedal_item.input_event_start_indic[start:end]
        random_data_item = PreprocessDataItem(targets=note_target, inputs=inputs, input_times=input_times,
                                              input_event_start_indic=input_event_start_indic,
                                              input_event_end_indic=input_event_end_indic)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        note_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)
        random_pedal_item = PreprocessDataItem(targets=pedal_target, inputs=inputs, input_times=input_times,
                                               input_event_start_indic=input_pedal_start_indic,
                                               input_event_end_indic=input_pedal_end_indic)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_pedal_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        pedal_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)

        return note_item.inputs, note_item.input_lengths, pedal_item.targets, pedal_item.target_lengths, \
               note_item.targets, note_item.target_lengths


class PedalNoteTestset:
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.test_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_pedal_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.cache_path = []
        self.note_path = []
        self.chuck = []
        self.pedal = []
        self.use_cache = use_cache
        for pair in pairs:
            cache_path = os.path.join(config.cache_data_path, pair.id + ".pt")
            pedal_path = os.path.join(config.cache_pedal_path, pair.id + ".pt")
            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_example(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items, cache_path)
            if os.path.exists(pedal_path) and use_cache:
                print(f"load cache {pedal_path}")
                split_pedal_items = torch.load(pedal_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_pedal(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
                split_pedal_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_pedal_items, pedal_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1
            for split_pedal_item in split_pedal_items:
                self.pedal.append(split_pedal_item)

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        split_pedal_item = self.pedal[n]
        length = self.token_config.inputs_length
        n_tokens = split_data_item.inputs.shape[0]
        start = random.randint(-length + 1, n_tokens - 1)
        end = min(start + length, n_tokens)
        start = max(start, 0)
        note_target = split_data_item.targets
        pedal_target = split_pedal_item.targets
        inputs = split_data_item.inputs[start:end]
        input_times = split_data_item.input_times[start:end]
        input_event_end_indic = split_data_item.input_event_end_indic[start:end]
        input_event_start_indic = split_data_item.input_event_start_indic[start:end]
        input_pedal_end_indic = split_pedal_item.input_event_end_indic[start:end]
        input_pedal_start_indic = split_pedal_item.input_event_start_indic[start:end]
        random_data_item = PreprocessDataItem(targets=note_target, inputs=inputs, input_times=input_times,
                                              input_event_start_indic=input_event_start_indic,
                                              input_event_end_indic=input_event_end_indic)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        note_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)
        random_pedal_item = PreprocessDataItem(targets=pedal_target, inputs=inputs, input_times=input_times,
                                               input_event_start_indic=input_pedal_start_indic,
                                               input_event_end_indic=input_pedal_end_indic)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_pedal_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        pedal_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)

        return note_item.inputs, note_item.input_lengths, pedal_item.targets, pedal_item.target_lengths, \
               note_item.targets, note_item.target_lengths


class PedalDataset:
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.train_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_pedal_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.path = []
        self.chuck = []
        self.use_cache = use_cache

        for pair in pairs:
            cache_path = os.path.join(config.cache_pedal_path, pair.id + ".pt")
            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_pedal(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items, cache_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        random_data_item = select_random_chunk(split_data_item, self.token_config)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        train_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)

        return train_item.inputs, train_item.input_lengths, train_item.targets, train_item.target_lengths


class PedalTestDataset:
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.test_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_pedal_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.path = []
        self.chuck = []
        self.use_cache = use_cache

        for pair in pairs:
            cache_path = os.path.join(config.cache_pedal_path, pair.id + ".pt")
            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_pedal(pair=pair, spectrogram_config=spectrogram_config, codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items, cache_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        random_data_item = select_random_chunk(split_data_item, self.token_config)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        train_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)

        return train_item.inputs, train_item.input_lengths, train_item.targets, train_item.target_lengths


class NotePedalDataset:
    def __init__(self, config: datasets.DatasetConfig, use_cache=True):
        pairs = config.train_pairs
        token_config = TokenConfig()
        spectrogram_config = spectrograms.SpectrogramConfig()
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        self.split_data_path = config.cache_split_pedal_path
        self.token_config = token_config
        self.codec = codec
        self.spectrogram_config = spectrogram_config
        self.vocabulary = vocabulary
        self.cnt = 0
        self.path = []
        self.chuck = []
        self.use_cache = use_cache

        for pair in pairs:
            cache_path = os.path.join(config.cache_note_pedal_path, pair.id + ".pt")
            if os.path.exists(cache_path) and use_cache:
                print(f"load cache {cache_path}")
                split_data_items = torch.load(cache_path)
            else:
                pair = datasets.trans_path_to_raw_data(pair)
                preprocessed_data = tokenize_note_pedal(pair=pair, spectrogram_config=spectrogram_config,
                                                        codec=codec)
                split_data_items = split_data(dataset=preprocessed_data, config=token_config)
                torch.save(split_data_items, cache_path)
            for split_data_item in split_data_items:
                self.chuck.append(split_data_item)
                self.cnt = self.cnt + 1

    def __len__(self):
        return self.cnt

    def __getitem__(self, n):
        split_data_item = self.chuck[n]
        random_data_item = select_random_chunk(split_data_item, self.token_config)
        target_sequences = tokens_encoding.encoding_target_sequence_with_indices(random_data_item)
        target_sequences = tokens_encoding.encoding_shifts(target_sequences, self.codec)
        target_sequences = compute_spectrogram(target_sequences, self.spectrogram_config)
        train_item = trans_preprocess_data_item_to_train_data(target_sequences, self.vocabulary, self.token_config)

        return train_item.inputs, train_item.input_lengths, train_item.targets, train_item.target_lengths


def collate_fn(batch):
    inputs = []
    targets = []
    input_lengths = []
    for input, input_length, target, target_length in batch:
        inputs.append(input)
        targets.append(target)
        input_lengths.append(input_length)
    target_lengths = torch.IntTensor([s.size(0) for s in targets])
    inputs = torch.stack(inputs, dim=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(dtype=torch.int)
    input_lengths = torch.stack(input_lengths, dim=0)
    return inputs, input_lengths, targets, target_lengths
