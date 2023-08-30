from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from constant import TokenConfig, LSTM_LAYERS
from data_process.vocabulary import build_codec, TokensVocabulary
from model.decoder import LSTMDecoder, TransformerDecoder
from model.encoder import Conformer


class RNNTModel(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            n_class: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out = nn.Sequential(
            nn.Linear(encoder.output_dim + decoder.output_dim, encoder.output_dim),
            nn.Tanh(),
            nn.Linear(encoder.output_dim, n_class, bias=False),
        )

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
            targets: Tensor,
            target_lengths: Tensor,
    ) -> Tensor:
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs, hidden_state = self.decoder(targets,encoder_outputs)

        outputs = self.joint(encoder_outputs, decoder_outputs)
        return outputs, encoder_output_lengths

    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        """
        Joint `encoder_outputs` and `decoder_outputs`.
        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            decoder_outputs (torch.FloatTensor): A output sequence of decoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        Returns:
            * outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outputs`..
        """
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            input_length = encoder_outputs.size(1)
            target_length = decoder_outputs.size(1)

            encoder_outputs = encoder_outputs.unsqueeze(2)
            decoder_outputs = decoder_outputs.unsqueeze(1)

            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.out(outputs)
        outputs = F.log_softmax(outputs, -1)

        return outputs

    @torch.no_grad()
    def decode(self, encoder_output: Tensor, max_length: int) -> str:
        """
        Decode `encoder_outputs`.
        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step
        Returns:
            * predicted_log_probs (torch.FloatTensor): Log probability of model predictions.
        """
        pred_tokens, hidden_state = list(), None
        decoder_input = torch.IntTensor([[2]]).to(self.device)

        for t in range(max_length):
            decoder_output, hidden_state = self.decoder(
                decoder_input, hidden_state=hidden_state
            )
            step_output = self.joint(
                encoder_output[t].view(-1), decoder_output.view(-1)
            )
            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)
            decoder_input = torch.LongTensor([[pred_token]]).to(self.device)

        pred_tokens = torch.LongTensor(pred_tokens)

        return pred_tokens

    @torch.no_grad()
    def recognize(self, inputs: Tensor, input_lengths: Tensor) -> List[str]:
        """
        Recognize input speech. This method consists of the forward of the encoder and the decode() of the decoder.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        outputs = list()

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        max_length = 1024

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        return outputs


def build_conformer_transducer():
    codec = build_codec()
    token_config = TokenConfig()

    conformer_encoder = Conformer(input_dim=token_config.encoder_input_dim, encoder_dim=token_config.encoder_input_dim,
                                  num_encoder_layers=token_config.num_encoder_layers,
                                  num_attention_heads=token_config.num_encoder_attention_heads)
    transformer_decoder = TransformerDecoder(n_class=codec.num_classes, d_model=token_config.decoder_input_dim,
                                             nhead=token_config.num_decoder_attention_heads,
                                             num_layers=token_config.num_decoder_layers, device=token_config.device)

    model = RNNTModel(encoder=conformer_encoder, decoder=transformer_decoder, n_class=TokensVocabulary(codec.num_classes).vocabulary_size).to(
        token_config.device)
    return model


def load_conformer_transducer_from_checkpoint(path: str):
    codec = build_codec()
    token_config = TokenConfig()

    conformer_encoder = Conformer(input_dim=token_config.encoder_input_dim, encoder_dim=token_config.encoder_input_dim,
                                  num_encoder_layers=token_config.num_encoder_layers,
                                  num_attention_heads=token_config.num_encoder_attention_heads)
    transformer_decoder = TransformerDecoder(n_class=codec.num_classes, d_model=token_config.decoder_input_dim,
                                             nhead=token_config.num_decoder_attention_heads,
                                             num_layers=token_config.num_decoder_layers, device=token_config.device)

    model = RNNTModel(encoder=conformer_encoder, decoder=transformer_decoder,
                      n_class=TokensVocabulary(codec.num_classes).vocabulary_size).to(
        token_config.device)

    state_dict = torch.load(path, map_location=token_config.device)
    model.load_state_dict(state_dict)

    return model
