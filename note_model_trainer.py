import os
import gc

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from constant import NOTE_MODEL_EPOCHS, NOTE_MODEL_SCHEDULER_WARMUP_RATIO, TokenConfig, CHECKPOINT_PATH, LEARNING_RATE, \
    BETAS, EPS, WEIGHT_DECAY, BATCH_SIZE
from data_process.datasets import build_maestrov3_dataset
from data_set.data_set import TrainDataset, collate_fn, TestDataset
from model.listen_attend_and_spell import build_conformer_listen_attend_and_spell_from_config, \
    load_conformer_listen_attend_and_spell_from_checkpoint
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

file = open("loss.txt", 'w')


def note_model_train_step():
    token_config = TokenConfig()
    config = build_maestrov3_dataset()
    dataset = TrainDataset(config, use_cache=True)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, prefetch_factor=2,shuffle=True
                             )
    testset = TestDataset(config, use_cache=True)
    test_loader = DataLoader(testset, batch_size=10, collate_fn=collate_fn, num_workers=4, prefetch_factor=2,shuffle=True
                             )
    epochs = NOTE_MODEL_EPOCHS
    num_training_steps = epochs * len(data_loader)
    scheduler_warmup_ratio = NOTE_MODEL_SCHEDULER_WARMUP_RATIO
    num_warmup_steps = int(num_training_steps * scheduler_warmup_ratio)

    # model = build_conformer_lstm()
    model = build_conformer_listen_attend_and_spell_from_config()
    # model = load_conformer_listen_attend_and_spell_from_checkpoint("/home/ylwang/share/mt3/conformer/conformer_las_145")
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=BETAS, eps=EPS,
                      weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler('polynomial',
                              optimizer,
                              num_warmup_steps=num_warmup_steps,
                              num_training_steps=num_training_steps,
                              )
    result = []
    # cnt = 0
    for i in range(epochs):
        model.train()
        torch.cuda.empty_cache()
        cnt = 0
        for inputs, input_lengths, targets, target_lengths in data_loader:
            # gc.collect()
            # cnt = cnt + 1
            # print("train",cnt)
            # continue
            optimizer.zero_grad()
            targets_in = targets[:, :-1].to(token_config.device)
            targets_out = targets[:, 1:].to(token_config.device)
            inputs = inputs.to(token_config.device)
            input_lengths = input_lengths.to(token_config.device)
            target_lengths = target_lengths.to(token_config.device)
            res = model(inputs, input_lengths, targets_in, target_lengths)

            bz, t, _ = res.size()
            loss = criterion(torch.squeeze(res, 0).view(bz * t, -1), torch.squeeze(targets_out.long(), 0).view(-1))
            # scheduler.get_lr()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # tokens = res.argmax(-1)[-1]
            cnt = cnt + 1
            if cnt % 100 == 0:
                print("========================train{}:loss:{},lr:{}==========================".format(i, loss,
                                                                                                       scheduler.get_last_lr()))
            # print(tokens)

        torch.cuda.empty_cache()
        tot = 0
        cnt = 0
        model.eval()
        for inputs, input_lengths, targets, target_lengths in test_loader:
            # print("test",i)
            # continue

            targets_in = targets[:, :-1].to(token_config.device)
            targets_out = targets[:, 1:].to(token_config.device)
            inputs = inputs.to(token_config.device)
            input_lengths = input_lengths.to(token_config.device)
            target_lengths = target_lengths.to(token_config.device)
            res = model(inputs, input_lengths, targets_in, target_lengths)
            bz, t, _ = res.size()
            loss = criterion(torch.squeeze(res, 0).view(bz * t, -1), torch.squeeze(targets_out.long(), 0).view(-1))
            # loss.backward()
            # tokens = res.argmax(-1)[-1]
            tot = tot + loss.item()
            cnt = cnt + 1
            # print(tot,cnt,tot/cnt)
            if cnt % 100 == 0:
                print("========================test{}:loss:{}==========================".format(i, loss))
            # print(tokens)
        result.append(tot / cnt)
        print(result)
        torch.save(model.state_dict(),
                   os.path.join(CHECKPOINT_PATH, "conformer_las_{}".format(i)))


if __name__ == '__main__':
    note_model_train_step()
