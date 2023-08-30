import json
import os

import note_seq

from constant import BASE_MAESTROV3_PATH, BASE_MAESTROV3_CACHE_PATH, BASE_MAESTROV3_SPLIT_PATH, BASE_MAESTROV1_PATH, \
    BASE_MAESTROV1_CACHE_PATH, BASE_MAESTROV1_SPLIT_PATH, BASE_MAESTROV2_PATH, BASE_MAESTROV2_CACHE_PATH, \
    BASE_MAESTROV2_SPLIT_PATH, SAMPLE_RATE, BASE_MAESTROV1_PEDAL_CACHE_PATH, BASE_MAESTROV1_PEDAL_SPLIT_PATH, \
    BASE_MAESTROV2_PEDAL_CACHE_PATH, BASE_MAESTROV2_PEDAL_SPLIT_PATH, BASE_MAESTROV3_PEDAL_SPLIT_PATH, \
    BASE_MAESTROV3_PEDAL_CACHE_PATH, PEDAL_EXTEND, BASE_MAESTROV1_PEDAL_NOTE_CACHE_PATH, \
    BASE_MAESTROV1_PEDAL_NOTE_SPLIT_PATH, BASE_MAESTROV3_PEDAL_NOTE_CACHE_PATH, BASE_MAESTROV2_PEDAL_NOTE_CACHE_PATH, \
    BASE_MAESTROV2_PEDAL_NOTE_SPLIT_PATH, BASE_MAESTROV3_PEDAL_NOTE_SPLIT_PATH
from data_process import spectrograms, midi_note
from data_process.data_classes import DatasetConfig, FileNamePair, MidiAudioPair


def build_maestrov3_dataset() -> DatasetConfig:
    config = DatasetConfig('maestrov3', BASE_MAESTROV3_PATH, BASE_MAESTROV3_CACHE_PATH, BASE_MAESTROV3_SPLIT_PATH,
                           BASE_MAESTROV3_PEDAL_CACHE_PATH, BASE_MAESTROV3_PEDAL_SPLIT_PATH,
                           BASE_MAESTROV3_PEDAL_NOTE_CACHE_PATH, BASE_MAESTROV3_PEDAL_NOTE_SPLIT_PATH, [],
                           [], [])
    json_path = os.path.join(config.base_file_path, "maestro-v3.0.0.json")
    with open(json_path, 'r') as file:
        data_dir = json.load(file)
    split = data_dir['split']
    audio_filename = data_dir['audio_filename']
    midi_filename = data_dir['midi_filename']
    cache_path = config.cache_data_path
    for k, v in split.items():
        id = audio_filename[k][5:-4]
        midi_path = os.path.join(config.base_file_path, midi_filename[k])
        audio_path = os.path.join(config.base_file_path, audio_filename[k])
        if v == 'train':
            config.train_pairs.append(
                FileNamePair(id=id, midi_file_name=midi_path, audio_file_name=audio_path, cache_data_path=cache_path))
        elif v == 'test':
            config.test_pairs.append(
                FileNamePair(id=id, midi_file_name=midi_path, audio_file_name=audio_path, cache_data_path=cache_path))
        elif v == 'validation':
            config.validation_pairs.append(
                FileNamePair(id=id, midi_file_name=midi_path, audio_file_name=audio_path, cache_data_path=cache_path))
        else:
            raise ValueError(f"Unknown split {v}")
    return config


def build_maestrov1_dataset() -> DatasetConfig:
    config = DatasetConfig('maestrov1', BASE_MAESTROV1_PATH, BASE_MAESTROV1_CACHE_PATH, BASE_MAESTROV1_SPLIT_PATH,
                           BASE_MAESTROV1_PEDAL_CACHE_PATH, BASE_MAESTROV1_PEDAL_SPLIT_PATH,
                           BASE_MAESTROV1_PEDAL_NOTE_CACHE_PATH, BASE_MAESTROV1_PEDAL_NOTE_SPLIT_PATH, [],
                           [], [])
    json_path = os.path.join(config.base_file_path, "maestro-v1.0.0.json")
    with open(json_path, 'r') as file:
        data_dir = json.load(file)
    cache_path = config.cache_data_path
    for item in data_dir:
        id = item['audio_filename'][5:-4]
        audio_path = os.path.join(config.base_file_path, item['audio_filename'])
        midi_path = os.path.join(config.base_file_path, item['midi_filename'])
        if item['split'] == 'train':
            config.train_pairs.append(
                FileNamePair(id=id, midi_file_name=midi_path, audio_file_name=audio_path, cache_data_path=cache_path))
        elif item['split'] == 'test':
            config.test_pairs.append(
                FileNamePair(id=id, midi_file_name=midi_path, audio_file_name=audio_path, cache_data_path=cache_path))
        elif item['split'] == 'validation':
            config.validation_pairs.append(
                FileNamePair(id=id, midi_file_name=midi_path, audio_file_name=audio_path, cache_data_path=cache_path))
        else:
            raise ValueError(f"Unknown split {item['split']}")
    return config


def build_maestrov2_dataset() -> DatasetConfig:
    config = DatasetConfig('maestrov2', BASE_MAESTROV2_PATH, BASE_MAESTROV2_CACHE_PATH, BASE_MAESTROV2_SPLIT_PATH,
                           BASE_MAESTROV2_PEDAL_CACHE_PATH, BASE_MAESTROV2_PEDAL_SPLIT_PATH,
                           BASE_MAESTROV2_PEDAL_NOTE_CACHE_PATH, BASE_MAESTROV2_PEDAL_NOTE_SPLIT_PATH, [],
                           [], [])
    json_path = os.path.join(config.base_file_path, "maestro-v2.0.0.json")
    with open(json_path, 'r') as file:
        data_dir = json.load(file)
    cache_path = config.cache_data_path
    for item in data_dir:
        id = item['audio_filename'][5:-4]
        audio_path = os.path.join(config.base_file_path, item['audio_filename'])
        midi_path = os.path.join(config.base_file_path, item['midi_filename'])
        if item['split'] == 'train':
            config.train_pairs.append(
                FileNamePair(id=id, midi_file_name=midi_path, audio_file_name=audio_path, cache_data_path=cache_path))
        elif item['split'] == 'test':
            config.test_pairs.append(
                FileNamePair(id=id, midi_file_name=midi_path, audio_file_name=audio_path, cache_data_path=cache_path))
        elif item['split'] == 'validation':
            config.validation_pairs.append(
                FileNamePair(id=id, midi_file_name=midi_path, audio_file_name=audio_path, cache_data_path=cache_path))
        else:
            raise ValueError(f"Unknown split {item['split']}")
    return config


def trans_path_to_raw_data(pair: FileNamePair) -> MidiAudioPair:
    cache_path = pair.cache_data_path
    audio_file_name = pair.audio_file_name
    midi_file_name = pair.midi_file_name
    id = pair.id
    audio = spectrograms.read_wave(audio_file_name, SAMPLE_RATE)
    midi = note_seq.midi_file_to_note_sequence(midi_file_name)
    for cc in midi.control_changes:
        if cc.control_number!=64:
            midi.control_changes.remove(cc)
        if cc.control_value >= 64:
            cc.control_value = 127
        else:
            cc.control_value = 0
    if PEDAL_EXTEND:
        midi = note_seq.apply_sustain_control_changes(midi)
    midi_audio_pair = MidiAudioPair(id=id, midi_file_name=midi_file_name, audio_file_name=audio_file_name, audio=audio,
                                    midi=midi, cache_path=cache_path)
    return midi_audio_pair
