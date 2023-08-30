import dataclasses
from typing import List, Tuple


@dataclasses.dataclass
class EventRange:
    """这个类表示token的范围表示什么东西"""

    type: str
    min_value: int
    max_value: int


@dataclasses.dataclass
class Event:
    """表示一个音符事件"""

    type: str
    value: int


class Codec:

    # 这是一个对每一个token进行编码的config 类

    def __init__(self, max_shift_steps: int, steps_per_second: float, event_ranges: List[EventRange]):
        """

        Args:
            max_shift_steps: 表示偏移，通常表示每个frame里面的时间
            steps_per_second: 表示每秒有多少个时间戳
            event_ranges: 表示所有音符事件的范围
        """

        self.steps_per_second = steps_per_second
        self.shift_range = EventRange(type='shift',
                                      min_value=0,
                                      max_value=max_shift_steps)
        self.event_ranges = [self.shift_range] + event_ranges
        assert len(self.event_ranges) == len(set([er.type for er in self.event_ranges]))

    @property
    def num_classes(self) -> int:
        """

        Returns: token的总范围,用于decoder的词嵌入

        """
        return sum(er.max_value - er.min_value + 1 for er in self.event_ranges)

    @property
    def max_shift_steps(self) -> int:
        return self.shift_range.max_value

    def is_shift_event_index(self, index: int) -> bool:
        return (self.shift_range.min_value <= index) and (
                index <= self.shift_range.max_value)

    def event_type_range(self, event_type: str) -> Tuple[int, int]:
        """

        Args:
            event_type: 根据不同的事件类型

        Returns: 不同的token范围

        """
        token_num = 0
        for event_range in self.event_ranges:
            if event_type == event_range.type:
                return token_num, token_num + (event_range.max_value - event_range.min_value)
            token_num += event_range.max_value - event_range.min_value + 1

        raise ValueError(f'unknown type :{event_type}')

    def decode_event_value(self, index: int) -> Event:
        """
        将索引映射到对应的事件
        Args:
            index: tokens的索引

        Returns: 对应的事件

        """
        time_step = 0
        for event_range in self.event_ranges:
            if time_step <= index <= time_step + event_range.max_value - event_range.min_value:
                return Event(type=event_range.type, value=event_range.min_value + index - time_step)
            time_step += event_range.max_value - event_range.min_value + 1
        raise ValueError(f'未知index:{index}')

    def encode_event(self, event: Event) -> int:
        """Encode an event to an index."""
        offset = 0
        for er in self.event_ranges:
            if event.type == er.type:
                if not er.min_value <= event.value <= er.max_value:
                    raise ValueError(
                        f'Event value {event.value} is not within valid range '
                        f'[{er.min_value}, {er.max_value}] for type {event.type}')
                return offset + event.value - er.min_value
            offset += er.max_value - er.min_value + 1

        raise ValueError(f'Unknown event type: {event.type}')
