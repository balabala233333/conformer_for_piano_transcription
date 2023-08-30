from typing import Sequence, Tuple

import note_seq

from constant import MIN_NOTE_DURATION
from data_process import note_event_codec
from data_process.tokens_decoding import NoteDecodingState
from data_process.tokens_encoding import NoteEventItem


def load_midi_file(file_path: str) -> note_seq:
    """
    加载一个midi文件
    Args:
        file_path: 文件路径

    Returns: midi对应的音符序列

    """
    ns = note_seq.midi_file_to_note_sequence(file_path)
    return ns


def add_control_change_to_sequence(ns: note_seq.NoteSequence, time: float, control_number: int,
                                   control_value: int) -> None:
    """

    :param ns: 音符序列
    :param time: 控制器时间
    :param control_number: 踏板编号
    :param control_value: 踏板力度
    :return: None
    """
    ns.control_changes.add(control_value=control_value, control_number=control_number, time=time)
    ns.total_time = max(time, ns.total_time)


def add_note_to_sequence(ns: note_seq.NoteSequence, start_time: float, end_time: float, pitch: int, velocity: int,
                         program: int = 0, is_drum: bool = False) -> None:
    """
    往音符序列当中加一个音符
    Args:
        ns: 音符序列
        start_time: 开始时间
        end_time: 结束时间
        pitch: 音高
        velocity: 音量
        program: 乐器
        is_drum: 是否是鼓点

    """
    end_time = max(end_time, start_time + MIN_NOTE_DURATION)
    ns.notes.add(start_time=start_time,
                 end_time=end_time,
                 pitch=pitch,
                 velocity=velocity,
                 program=program,
                 is_drum=is_drum)
    ns.total_time = max(ns.total_time, end_time)


def validate_note_sequence(ns: note_seq.NoteSequence) -> None:
    """
    验证音符序列是否有效
    Args:
        ns: 一个音符序列

    Returns: 是否有效

    """
    for note in ns.notes:
        if note.start_time >= note.end_time:
            raise ValueError(f'note has start time >= end time: {note.start_time} >= {note.end_time}')
        if note.velocity == 0:
            raise ValueError('note has zero velocity')
    return True


def trans_note_sequence_to_padel_note_event(ns: note_seq.NoteSequence) -> Tuple[Sequence[float], Sequence[NoteEventItem]]:
    notes = sorted(ns.notes, key=lambda note: note.pitch)
    pedals = sorted(ns.control_changes, key=lambda note: note.control_number)
    times = ([note.end_time for note in notes] + [note.start_time for note in notes] + [pedal.time for pedal in pedals])
    event_values = ([NoteEventItem(pitch=note.pitch, velocity=0, pedal=None, pedal_velocity=None) for note in notes] + [
        NoteEventItem(pitch=note.pitch, velocity=note.velocity, pedal=None, pedal_velocity=None) for note in notes] + [
        NoteEventItem(pedal=pedal.control_number, velocity=None, pitch=None, pedal_velocity=pedal.control_value) for
        pedal in
        pedals])
    return times, event_values

def trans_note_sequence_to_padel_event(ns: note_seq.NoteSequence) -> Tuple[Sequence[float], Sequence[NoteEventItem]]:
    pedals = sorted(ns.control_changes, key=lambda note: note.control_number)
    times = ([pedal.time for pedal in pedals])
    event_values = [
        NoteEventItem(pedal=pedal.control_number, velocity=None, pitch=None, pedal_velocity=pedal.control_value) for
        pedal in
        pedals]
    return times, event_values


def trans_note_sequence_to_onsets_and_offsets_event(
        ns: note_seq.NoteSequence
) -> Tuple[Sequence[float], Sequence[NoteEventItem]]:
    """
    将音符序列变成onset和offset事件
    Args:
        ns: 音符序列

    Returns: 根据onset和offset拆分之后的音符事件

    """
    notes = sorted(ns.notes, key=lambda note: note.pitch)
    times = ([note.end_time for note in notes] + [note.start_time for note in notes])
    event_values = ([NoteEventItem(pitch=note.pitch, velocity=0, pedal=None, pedal_velocity=None) for note in notes] + [
        NoteEventItem(pitch=note.pitch, velocity=note.velocity, pedal=None, pedal_velocity=None) for note in notes])
    return times, event_values


def decode_note_event(
        state: NoteDecodingState,
        time: float,
        event: note_event_codec.Event,
        codec: note_event_codec.Codec
) -> None:
    """
    处理当前事件，并更新解码状态
    Args:
        state: 当前解码状态
        time: 当前时间
        event: 当前event
        codec:  单词表

    """
    if time < state.current_time:
        return ValueError(f'event time 小于当前时间{time} < {state.current_time}')
    state.current_time = time
    pitch = event.value
    if event.type == 'pitch':
        if state.current_velocity == 0:
            # 这个事件是offset事件
            if (pitch, state.current_program) not in state.activate_pitches:
                raise ValueError(f'乐器为:{state.current_program}音高为:{pitch}的乐器没有被激活')
            onset_time, onset_velocity = state.activate_pitches.pop((pitch, state.current_program))
            add_note_to_sequence(state.note_sequence, start_time=onset_time, end_time=time, pitch=pitch,
                                 velocity=onset_velocity, program=state.current_program)
        else:
            # onset事件
            if (pitch, state.current_program) in state.activate_pitches:
                onset_time, onset_velocity = state.activate_pitches.pop((pitch, state.current_program))
                add_note_to_sequence(state.note_sequence, start_time=onset_time, end_time=time, pitch=pitch,
                                     velocity=onset_velocity, program=state.current_program)
            state.activate_pitches[(pitch, state.current_program)] = (time, state.current_velocity)
    elif event.type == 'velocity':
        velocity = event.value
        state.current_velocity = velocity
    elif event.type == 'program':
        program = event.value
        state.current_program = program
    elif event.type == 'pedal':
        pedal = event.value
        state.current_pedal = pedal
        add_control_change_to_sequence(state.note_sequence, time=time, control_number=pedal,
                                       control_value=state.current_pedal_velocity)
    elif event.type == 'pedal_velocity':
        pedal_velocity = event.value
        state.current_pedal_velocity = pedal_velocity
    else:
        raise ValueError(f'unknown event type:{event.type}')


def flush_decoding_result_from_state(state: NoteDecodingState) -> note_seq.NoteSequence:
    for onset_time, _ in state.activate_pitches.values():
        state.current_time = max(state.current_time, onset_time + MIN_NOTE_DURATION)
    for pitch, program in list(state.activate_pitches.keys()):
        onset_time, onset_velocity = state.activate_pitches.pop((pitch, program))
        add_note_to_sequence(state.note_sequence, start_time=onset_time, end_time=state.current_time, pitch=pitch,
                             velocity=onset_velocity, program=program)
    return state.note_sequence


def note_event_data_to_events(
        value: NoteEventItem,
) -> Sequence[note_event_codec.Event]:
    """

    Args:
        value:音符事件

    Returns: 每个token所表达的事件

    """
    velocity_bin = value.velocity
    if value.program is None:
        if value.pedal:
            return [note_event_codec.Event('pedal_velocity', value.pedal_velocity),
                    note_event_codec.Event('pedal', value.pedal)]
        else:
            # onsets + offsets + velocities only, no programs
            return [note_event_codec.Event('velocity', velocity_bin),
                    note_event_codec.Event('pitch', value.pitch)]
    else:
        if value.pedal:
            # drum events use a separate vocabulary
            return [note_event_codec.Event('pedal_velocity', value.pedal_velocity),
                    note_event_codec.Event('pedal', value.pedal)]
        else:
            # program + velocity + pitch
            return [note_event_codec.Event('program', value.program),
                    note_event_codec.Event('velocity', velocity_bin),
                    note_event_codec.Event('pitch', value.pitch)]
