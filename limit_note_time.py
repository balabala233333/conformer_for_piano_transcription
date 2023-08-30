import os.path

import note_seq
import torch

from constant import TEST_NOTE_CHECKPOINT_PATH, TokenConfig
from data_process import spectrograms
from data_process.datasets import build_maestrov1_dataset
from data_process.postprocess import trans_tokens_to_midi
from data_process.preprocess import test_midi_and_audio_to_tokens
from data_process.spectrograms import SpectrogramConfig
from data_process.vocabulary import TokensVocabulary, build_codec
from evaluation.metrics import get_scores
from model.listen_attend_and_spell import load_conformer_listen_attend_and_spell_from_checkpoint
import tensorflow as tf
import pandas as pd

tf.config.set_visible_devices([], 'GPU')
config = build_maestrov1_dataset()



if __name__ == '__main__':
    for i in range(4,11):
        df = pd.DataFrame()
        name = []
        mir_eval_onset_presion = []
        mir_eval_onset_recall = []
        mir_eval_onset_f1 = []
        mir_eval_onset_offset_presion = []
        mir_eval_onset_offset_recall = []
        mir_eval_onset_offset_f1 = []
        frame_precision = []
        frame_recall = []
        frame_f1 = []
        mir_eval_vel_f1 = []
        mir_eval_vel_recall = []
        mir_eval_vel_presion = []
        for pair in config.test_pairs:
            target_ns = note_seq.midi_file_to_note_sequence(
                pair.midi_file_name)
            pred_ns = note_seq.midi_file_to_note_sequence(
                os.path.join("/data/lobby/mt3/pred", pair.midi_file_name.split("/")[-1]))
            res = get_scores(target_ns, pred_ns, 5)
            mir_eval_onset_presion.append(res["onset_score"].precision_score)
            mir_eval_onset_recall.append(res["onset_score"].recall_score)
            mir_eval_onset_f1.append(res["onset_score"].f1_score)
            mir_eval_onset_offset_presion.append(res["onset_offset_score"].precision_score)
            mir_eval_onset_offset_recall.append(res["onset_offset_score"].recall_score)
            mir_eval_onset_offset_f1.append(res["onset_offset_score"].f1_score)
            mir_eval_vel_presion.append(res["onset_offset_velocity_score"].precision_score)
            mir_eval_vel_recall.append(res["onset_offset_velocity_score"].recall_score)
            mir_eval_vel_f1.append(res["onset_offset_velocity_score"].f1_score)
            frame_precision.append(res["frame_score"].precision_score)
            frame_recall.append(res["frame_score"].recall_score)
            frame_f1.append(res["frame_score"].f1_score)
            print(res)
        df["name"] = name
        df["mir_eval_onset_presion"] = mir_eval_onset_presion
        df["mir_eval_onset_recall"] = mir_eval_onset_recall
        df["mir_eval_onset_f1"] = mir_eval_onset_f1
        df["mir_eval_onset_offset_presion"] = mir_eval_onset_offset_presion
        df["mir_eval_onset_offset_recall"] = mir_eval_onset_offset_recall
        df["mir_eval_onset_offset_f1"] = mir_eval_onset_offset_f1
        df["mir_eval_vel_f1"] = mir_eval_vel_f1
        df["mir_eval_vel_recall"] = mir_eval_vel_recall
        df["mir_eval_vel_presion"] = mir_eval_vel_presion
        df["frame_f1"] = frame_f1
        df["frame_recall"] = frame_recall
        df["frame_precision"] = frame_precision
        df.to_excel(f"res-{i}.xlsx")
        print("onset_precision:", sum(mir_eval_onset_presion) / len(mir_eval_onset_presion))
        print("onset_recall:", sum(mir_eval_onset_recall) / len(mir_eval_onset_recall))
        print("onset_f1:", sum(mir_eval_onset_f1) / len(mir_eval_onset_f1))
        print("onset_offset_precision:", sum(mir_eval_onset_offset_presion) / len(mir_eval_onset_offset_presion))
        print("onset_offset_recall:", sum(mir_eval_onset_offset_recall) / len(mir_eval_onset_offset_recall))
        print("onset_offset_f1:", sum(mir_eval_onset_offset_f1) / len(mir_eval_onset_offset_f1))
        print("onset_offset_velocity_precision:", sum(mir_eval_vel_presion) / len(mir_eval_vel_presion))
        print("onset_offset_velocity_recall:", sum(mir_eval_vel_recall) / len(mir_eval_vel_recall))
        print("onset_offset_velocity_f1:", sum(mir_eval_vel_f1) / len(mir_eval_vel_f1))
        print("frame_precision:", sum(frame_precision) / len(frame_precision))
        print("frame_recall:", sum(frame_recall) / len(frame_recall))
        print("frame_f1:", sum(frame_f1) / len(frame_f1))