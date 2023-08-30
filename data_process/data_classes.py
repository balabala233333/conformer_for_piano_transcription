import dataclasses
from typing import Sequence, Optional, MutableMapping, Tuple

import note_seq
import numpy
import torch

from constant import DEFAULT_VELOCITY


@dataclasses.dataclass
class FileNamePair:
    id: str
    # midi文件名
    midi_file_name: str
    # wav文件名
    audio_file_name: str
    # 缓存路径
    cache_data_path: str


@dataclasses.dataclass
class DatasetConfig:
    name: str
    base_file_path: str
    cache_data_path: str
    cache_split_data_path: str
    cache_pedal_path: str
    cache_split_pedal_path: str
    cache_note_pedal_path: str
    cache_split_note_pedal_path: str
    train_pairs: Sequence[FileNamePair]
    test_pairs: Sequence[FileNamePair]
    validation_pairs: Sequence[FileNamePair]


@dataclasses.dataclass
class MidiAudioPair:
    # 这是一个midi和wav文件的原始数据的pair
    id: str
    midi: note_seq
    audio: numpy.ndarray
    midi_file_name: str
    audio_file_name: str
    cache_path: str


@dataclasses.dataclass
class SplitToken:
    start_step: int
    end_step: int
    pred: Sequence[int]
    target: Sequence[int]


@dataclasses.dataclass
class PreprocessDataItem:
    inputs: torch.Tensor
    input_times: torch.Tensor
    targets: torch.Tensor
    input_event_start_indic: torch.Tensor
    input_event_end_indic: torch.Tensor


@dataclasses.dataclass
class NoteEventItem:
    pitch: int
    velocity: Optional[int] = None
    program: Optional[int] = None
    pedal: Optional[int] = None
    pedal_velocity: Optional = None


@dataclasses.dataclass
class PreprocessDataset:
    # 这是描述预处理中间过程的类
    id: str

    inputs: numpy.ndarray
    input_times: numpy.ndarray
    targets: Sequence[int]
    input_event_start_indices: Sequence[int]
    input_event_end_indices: Sequence[int]


@dataclasses.dataclass
class TestItem:
    inputs: torch.tensor
    input_lengths: torch.tensor
    targets: torch.tensor
    target_lengths: torch.tensor
    input_times: torch.tensor
    pred: torch.tensor
    midi_name: str


@dataclasses.dataclass
class TrainItem:
    inputs: torch.tensor
    input_lengths: torch.tensor
    targets: torch.tensor
    target_lengths: torch.tensor


@dataclasses.dataclass
class NoteDecodingState:
    """表示解码的当前状态"""

    # 表示当前时间
    current_time: float = 0.0
    # 当前音量
    current_velocity: int = DEFAULT_VELOCITY
    # 当前乐器
    current_program: int = 0
    # 当前激活的音高(音高乐器的激活时间)
    current_pedal: int = 0

    current_pedal_velocity: int = 0
    activate_pitches: MutableMapping[Tuple[int, int],
                                     Tuple[float, int]] = dataclasses.field(
        default_factory=dict)
    activate_pedals: MutableMapping[Tuple[int, int],
                                     Tuple[float, int]] = dataclasses.field(
        default_factory=dict)
    # 已经解码的音符序列
    note_sequence: note_seq.NoteSequence = dataclasses.field(
        default_factory=lambda: note_seq.NoteSequence(ticks_per_quarter=220))
