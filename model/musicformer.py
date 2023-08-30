from typing import List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from constant import TokenConfig, MUSICFORMER_TARGETS_LENGTH
from data_process.vocabulary import build_codec, TokensVocabulary
from model.decoder import TransformerDecoder
from model.encoder import Conformer, TransformerEncoder


class MusicFormer(nn.Module):
    def __init__(self, pedal_encoder, wav_encoder, decoder):
        super(MusicFormer, self).__init__()
        self.token_config = TokenConfig()
        codec = build_codec()
        self.pedal_embedding = nn.Embedding(codec.num_classes, self.token_config.encoder_input_dim)
        self.pedal_encoder = pedal_encoder
        self.wav_encoder = wav_encoder
        self.decoder = decoder
        self.out_layer = nn.Linear(MUSICFORMER_TARGETS_LENGTH + self.token_config.encoder_input_dim,
                                   512)
        self.out_cat = nn.Linear(decoder.output_dim, codec.num_classes)

    def forward(self, pedal_inputs, pedal_inputs_length, wav_inputs,
                wav_inputs_length, targets):
        pedal_inputs = self.pedal_embedding(pedal_inputs)
        pedal_out = self.pedal_encoder(pedal_inputs, pedal_inputs_length)[0]
        wav_out = self.wav_encoder(wav_inputs, wav_inputs_length)[0]
        out = torch.cat([pedal_out, wav_out], dim=-2)

        out = torch.transpose(out, -2, -1)
        out = self.out_layer(out)
        out = torch.transpose(out, -2, -1)
        out = self.decoder(targets, out)[0]
        out = self.out_cat(out)
        out = F.log_softmax(out, -1)
        return out

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) -> str:

        encoder_output = encoder_output.unsqueeze(0)
        codec = build_codec()
        vocabulary = TokensVocabulary(codec.num_classes)
        targets = torch.IntTensor([vocabulary.start_of_tokens]).to(encoder_output.device)

        for i in range(max_length):

            targets = targets.unsqueeze(0)

            decoder_outputs, hidden_state = self.decoder(
                targets, encoder_output
            )
            outputs = F.log_softmax(self.out(decoder_outputs), -1)

            last_token = outputs.squeeze(0).argmax(-1)[-1]

            targets = torch.concat((targets.squeeze(0), last_token.unsqueeze(0)), -1)

            if last_token == vocabulary.end_of_tokens:
                break

        return targets

    @torch.no_grad()
    def recognize(self, wav_inputs: Tensor, wav_inputs_length: Tensor, pedal_inputs: Tensor,
                  pedal_inputs_length: Tensor) -> List[str]:
        pedal_inputs = self.pedal_embedding(pedal_inputs)
        pedal_out = self.pedal_encoder(pedal_inputs, pedal_inputs_length)[0]
        wav_out = self.wav_encoder(wav_inputs, wav_inputs_length)[0]
        out = torch.cat([pedal_out, wav_out], dim=-2)

        out = torch.transpose(out, -2, -1)
        out = self.out_layer(out)
        out = torch.transpose(out, -2, -1)

        outputs = list()
        max_length = 1024

        for encoder_output in out:
            predict = self.decode(encoder_output, max_length)
            outputs.append(predict)

        return outputs


def load_musicformer_from_config(path: str):
    token_config = TokenConfig()
    pedal_encoder = TransformerEncoder(input_dim=token_config.encoder_input_dim, d_model=token_config.encoder_input_dim,
                                       nhead=token_config.num_encoder_attention_heads,
                                       num_layers=token_config.num_encoder_layers)
    wav_encoder = Conformer(input_dim=token_config.encoder_input_dim, encoder_dim=token_config.encoder_input_dim,
                            num_encoder_layers=token_config.num_encoder_layers,
                            num_attention_heads=token_config.num_encoder_attention_heads)
    codec = build_codec()
    transformer_decoder = TransformerDecoder(n_class=codec.num_classes, d_model=token_config.decoder_input_dim,
                                             nhead=token_config.num_decoder_attention_heads,
                                             num_layers=token_config.num_decoder_layers,device=token_config.device)
    musicformer = MusicFormer(pedal_encoder, wav_encoder, transformer_decoder)

    state_dict = torch.load(path, map_location=token_config.device)
    musicformer.load_state_dict(state_dict)

    return musicformer
