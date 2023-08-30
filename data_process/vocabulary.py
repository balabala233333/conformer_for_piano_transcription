import dataclasses

from typing import Sequence
import note_seq
import torch

from constant import STEPS_PER_SECOND, MAX_SHIFT_SECONDS, NUM_VELOCITY_BINS, DECODED_EOS_ID, DECODED_INVALID_ID
from data_process import note_event_codec




@dataclasses.dataclass
class VocabularyConfig:
    # 此类定义了词汇表的配置
    step_per_second: int = STEPS_PER_SECOND
    max_shift_seconds: int = MAX_SHIFT_SECONDS
    num_velocity_bins: int = NUM_VELOCITY_BINS


class TokensVocabulary:
    # 此类表示的是词汇表
    def __init__(self, num_words: int):
        # 0表示空格,1表示end of tokens,2表示start of tokens
        self.num_special_tokens = 3
        self.num_words = num_words

    @property
    def end_of_tokens(self) -> int:
        return 1

    @property
    def start_of_tokens(self) -> int:
        return 2

    @property
    def vocabulary_size(self) -> int:
        return self.num_special_tokens + self.num_words

    # def encode_tokens(self, tokens: torch.Tensor) -> Sequence[int]:

    #     """
    #     讲token变成词汇表token,就是加sos或者eos这些的
    #     Args:
    #         tokens: 单词tokens
    #
    #     Returns: 词汇表tokens
    #
    #     """
    #     result_tokens = tokens+3
    #     return result_tokens

    def decode_one_token(self, token_id) -> int:
        """
        解码单个token
        Args:
            token_id: 单个vocabulary_token的id

        Returns: 单词token的id

        """
        if token_id == self.end_of_tokens:
            return DECODED_EOS_ID
        elif token_id < self.num_special_tokens:
            return DECODED_INVALID_ID
        elif token_id > self.vocabulary_size:
            return DECODED_INVALID_ID
        else:
            return token_id - self.num_special_tokens

    def decode_tokens(self, ids: Sequence[int]) -> Sequence[int]:
        """
        解码一串tokens
        Args:
            ids:  一串vocabulary_token

        Returns: 一串单词_token

        """
        ids = ids + self.num_special_tokens
        ids = torch.cat((torch.tensor([self.start_of_tokens]), ids, torch.tensor([self.end_of_tokens])),dim=0)
        return ids


event_ranges = [
    note_event_codec.EventRange('pitch', note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_VELOCITY),
    note_event_codec.EventRange('velocity', 0, VocabularyConfig.num_velocity_bins),
    note_event_codec.EventRange('tie', 0, 0),
    note_event_codec.EventRange('pedal_velocity', note_seq.MIN_MIDI_PROGRAM, note_seq.MAX_MIDI_PROGRAM),
    note_event_codec.EventRange('pedal', note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH)
]


# 构建单词(token)映射
def build_codec() -> note_event_codec.Codec:
    max_shift_steps = VocabularyConfig.step_per_second * VocabularyConfig.max_shift_seconds
    steps_per_second = VocabularyConfig.step_per_second
    codec = note_event_codec.Codec(max_shift_steps, steps_per_second, event_ranges)
    return codec


# TODO 这个也许没用未来要把他删了
def drop_programs(tokens, codec: note_event_codec.Codec):
    min_value, max_value = codec.event_type_range('program')
    return tokens[(tokens < min_value) | (tokens > max_value)]
