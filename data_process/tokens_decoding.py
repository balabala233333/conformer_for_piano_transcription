from typing import Optional, Callable, Tuple

import numpy as np

from data_process import note_event_codec
from data_process.data_classes import NoteDecodingState


def decode_events(state: NoteDecodingState,
                  tokens: np.ndarray,
                  start_time: int,
                  max_time: Optional[int],
                  codec: note_event_codec.Codec,
                  decode_event_fn: Callable[
                      [NoteDecodingState, float, note_event_codec.Event, note_event_codec.Codec]
                      , None]
                  ) -> Tuple[int, int]:
    """
    解码一段tokens
    Args:
        state: 当前状态
        tokens: 要解码的tokens
        start_time: 开始时间
        max_time: 最长时间
        codec: 单词字典
        decode_event_fn: 解码单个token的方法

    Returns: 无效的token个数和删除的token个数

    """
    invalid_events = 0
    dropped_events = 0
    cur_time = start_time

    for token_idx, token in enumerate(tokens):
        try:
            event = codec.decode_event_value(token)
        except ValueError:
            invalid_events += 1
            continue
        if event.type == 'shift':
            cur_steps = event.value
            cur_time = start_time + cur_steps / codec.steps_per_second

            # 如果出现shift比最长时间长的情况，那么当前token之后的所有token都是无效的
            if max_time and cur_time > max_time:
                dropped_events = len(tokens) - token_idx
                break
        else:
            try:
                decode_event_fn(state, cur_time, event, codec)
            except ValueError:
                invalid_events += 1
                print(f'解码得到一个无效的token{event}在时间点:{cur_time},这是第{invalid_events}个无效的token')
                continue

    return invalid_events, dropped_events
