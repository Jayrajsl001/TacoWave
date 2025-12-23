# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from math import sqrt
# from torch.autograd import Variable
# from layers import ConvNorm, LinearNorm
# from utils import to_gpu, get_mask_from_lengths

# class GeluGatedLayer(nn.Module):
#     def __init__(self, input_dim, output_dim, bias=True):
#         super(GeluGatedLayer, self).__init__()
#         self.input_linear = nn.Linear(input_dim, output_dim, bias=bias)
#         self.activation = nn.GELU()

#     def forward(self, src):
#         output = self.activation(self.input_linear(src))
#         return output

# class MultiHeadSelfAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(MultiHeadSelfAttention, self).__init__()
#         assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.head_dim = embed_dim // num_heads

#         self.query = LinearNorm(embed_dim, embed_dim, bias=False)
#         self.key = LinearNorm(embed_dim, embed_dim, bias=False)
#         self.value = LinearNorm(embed_dim, embed_dim, bias=False)
#         self.out = LinearNorm(embed_dim, embed_dim, bias=False)
#         self.layer_norm = nn.LayerNorm(embed_dim)
#         self.gelu = nn.GELU()

#     def forward(self, x, mask=None):
#         batch_size, seq_len, embed_dim = x.size()
        
#         Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
#         V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

#         scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.head_dim)
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
#         attn_weights = F.softmax(scores, dim=-1)
#         attn_output = torch.matmul(attn_weights, V)

#         attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
#         output = self.gelu(self.out(attn_output))
#         output = self.layer_norm(output + x)
#         return output, attn_weights

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]
#         return x

# class LocationLayer(nn.Module):
#     def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
#         super(LocationLayer, self).__init__()
#         padding = int((attention_kernel_size - 1) / 2)
#         self.location_conv = ConvNorm(2, attention_n_filters,
#                                       kernel_size=attention_kernel_size,
#                                       padding=padding, bias=False, stride=1,
#                                       dilation=1)
#         self.location_dense = LinearNorm(attention_n_filters, attention_dim,
#                                          bias=False, w_init_gain='tanh')

#     def forward(self, attention_weights_cat):
#         processed_attention = self.location_conv(attention_weights_cat)
#         processed_attention = processed_attention.transpose(1, 2)
#         processed_attention = self.location_dense(processed_attention)
#         return processed_attention

# class Attention(nn.Module):
#     def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
#                  attention_location_n_filters, attention_location_kernel_size):
#         super(Attention, self).__init__()
#         self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
#                                       bias=False, w_init_gain='tanh')
#         self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
#                                        w_init_gain='tanh')
#         self.v = LinearNorm(attention_dim, 1, bias=False)
#         self.location_layer = LocationLayer(attention_location_n_filters,
#                                             attention_location_kernel_size,
#                                             attention_dim)
#         self.score_mask_value = -float("inf")

#     def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
#         processed_query = self.query_layer(query.unsqueeze(1))
#         processed_attention_weights = self.location_layer(attention_weights_cat)
#         energies = self.v(torch.tanh(
#             processed_query + processed_attention_weights + processed_memory))
#         energies = energies.squeeze(-1)
#         return energies

#     def forward(self, attention_hidden_state, memory, processed_memory,
#                 attention_weights_cat, mask):
#         alignment = self.get_alignment_energies(
#             attention_hidden_state, processed_memory, attention_weights_cat)

#         if mask is not None:
#             alignment.data.masked_fill_(mask, self.score_mask_value)

#         attention_weights = F.softmax(alignment, dim=1)
#         attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
#         attention_context = attention_context.squeeze(1)
#         return attention_context, attention_weights

# class Prenet(nn.Module):
#     def __init__(self, in_dim, sizes):
#         super(Prenet, self).__init__()
#         in_sizes = [in_dim] + sizes[:-1]
#         self.layers = nn.ModuleList(
#             [LinearNorm(in_size, out_size, bias=False)
#              for (in_size, out_size) in zip(in_sizes, sizes)])

#     def forward(self, x):
#         for linear in self.layers:
#             x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
#         return x

# class Postnet(nn.Module):
#     def __init__(self, hparams):
#         super(Postnet, self).__init__()
#         self.convolutions = nn.ModuleList()

#         self.convolutions.append(
#             nn.Sequential(
#                 ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
#                          kernel_size=hparams.postnet_kernel_size, stride=1,
#                          padding=int((hparams.postnet_kernel_size - 1) / 2),
#                          dilation=1, w_init_gain='tanh'),
#                 nn.LayerNorm(hparams.postnet_embedding_dim))
#         )

#         for i in range(1, hparams.postnet_n_convolutions - 1):
#             self.convolutions.append(
#                 nn.Sequential(
#                     ConvNorm(hparams.postnet_embedding_dim,
#                              hparams.postnet_embedding_dim,
#                              kernel_size=hparams.postnet_kernel_size, stride=1,
#                              padding=int((hparams.postnet_kernel_size - 1) / 2),
#                              dilation=1, w_init_gain='tanh'),
#                     nn.LayerNorm(hparams.postnet_embedding_dim))
#             )

#         self.convolutions.append(
#             nn.Sequential(
#                 ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
#                          kernel_size=hparams.postnet_kernel_size, stride=1,
#                          padding=int((hparams.postnet_kernel_size - 1) / 2),
#                          dilation=1, w_init_gain='linear'),
#                 nn.LayerNorm(hparams.n_mel_channels))
#             )

#     def forward(self, x):
#         for i in range(len(self.convolutions) - 1):
#             x = self.convolutions[i][0](x)
#             x = x.transpose(1, 2)
#             x = F.dropout(torch.tanh(self.convolutions[i][1](x)), 0.5, self.training)
#             x = x.transpose(1, 2)
#         x = self.convolutions[-1][0](x)
#         x = x.transpose(1, 2)
#         x = F.dropout(self.convolutions[-1][1](x), 0.5, self.training)
#         x = x.transpose(1, 2)
#         return x

# class Encoder(nn.Module):
#     def __init__(self, hparams):
#         super(Encoder, self).__init__()
#         self.encoder_embedding_dim = hparams.encoder_embedding_dim
#         self.encoder_n_convolutions = hparams.encoder_n_convolutions - 1
#         self.encoder_kernel_size = hparams.encoder_kernel_size
#         self.num_heads = getattr(hparams, 'num_attention_heads', 8)

#         convolutions = []
#         self.gelu_layers = nn.ModuleList()
#         for _ in range(self.encoder_n_convolutions):
#             conv_layer = nn.Sequential(
#                 ConvNorm(hparams.encoder_embedding_dim,
#                          hparams.encoder_embedding_dim,
#                          kernel_size=hparams.encoder_kernel_size, stride=1,
#                          padding=int((hparams.encoder_kernel_size - 1) / 2),
#                          dilation=1, w_init_gain='relu'),
#                 nn.LayerNorm(hparams.encoder_embedding_dim),
#                 nn.Dropout(0.5)
#             )
#             convolutions.append(conv_layer)
#             self.gelu_layers.append(GeluGatedLayer(hparams.encoder_embedding_dim, hparams.encoder_embedding_dim))
#         self.convolutions = nn.ModuleList(convolutions)

#         self.self_attention = MultiHeadSelfAttention(hparams.encoder_embedding_dim, self.num_heads)
#         self.positional_encoding = PositionalEncoding(hparams.encoder_embedding_dim)

#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Conv1d, nn.Linear, GeluGatedLayer)):
#             if hasattr(module, 'weight') and module.weight is not None:
#                 torch.nn.init.xavier_uniform_(module.weight)
#             if hasattr(module, 'bias') and module.bias is not None:
#                 torch.nn.init.constant_(module.bias, 0)

#     def forward(self, x, input_lengths):
#         x = self.positional_encoding(x.transpose(1, 2)).transpose(1, 2)
#         for i, conv in enumerate(self.convolutions):
#             x = conv[0](x)
#             x = x.transpose(1, 2)
#             x = F.relu(conv[1](x))
#             x = x.transpose(1, 2)
#             x = conv[2](x)
#             gelu_layer = self.gelu_layers[i]
#             batch, channels, time = x.size()
#             x = x.permute(0, 2, 1).reshape(-1, channels)
#             x = gelu_layer(x)
#             x = x.reshape(batch, time, channels).permute(0, 2, 1)

#         x = x.transpose(1, 2)
#         mask = get_mask_from_lengths(input_lengths).unsqueeze(1).unsqueeze(1)
#         x, attn_weights = self.self_attention(x, mask)
#         return x

#     def inference(self, x):
#         x = self.positional_encoding(x.transpose(1, 2)).transpose(1, 2)
#         for i, conv in enumerate(self.convolutions):
#             x = conv[0](x)
#             x = x.transpose(1, 2)
#             x = F.relu(conv[1](x))
#             x = x.transpose(1, 2)
#             x = conv[2](x)
#             gelu_layer = self.gelu_layers[i]
#             batch, channels, time = x.size()
#             x = x.permute(0, 2, 1).reshape(-1, channels)
#             x = gelu_layer(x)
#             x = x.reshape(batch, time, channels).permute(0, 2, 1)

#         x = x.transpose(1, 2)
#         x, _ = self.self_attention(x)
#         return x

# class Decoder(nn.Module):
#     def __init__(self, hparams):
#         super(Decoder, self).__init__()
#         self.n_mel_channels = hparams.n_mel_channels
#         self.n_frames_per_step = hparams.n_frames_per_step
#         self.encoder_embedding_dim = hparams.encoder_embedding_dim
#         self.attention_rnn_dim = hparams.attention_rnn_dim
#         self.decoder_rnn_dim = hparams.decoder_rnn_dim
#         self.prenet_dim = hparams.prenet_dim
#         self.max_decoder_steps = hparams.max_decoder_steps
#         self.gate_threshold = hparams.gate_threshold
#         self.p_attention_dropout = hparams.p_attention_dropout
#         self.p_decoder_dropout = hparams.p_decoder_dropout

#         self.prenet = Prenet(
#             hparams.n_mel_channels * hparams.n_frames_per_step,
#             [hparams.prenet_dim, hparams.prenet_dim])

#         self.attention_rnn = nn.LSTMCell(
#             hparams.prenet_dim + hparams.encoder_embedding_dim,
#             hparams.attention_rnn_dim)

#         self.attention_layer = Attention(
#             hparams.attention_rnn_dim, hparams.encoder_embedding_dim,
#             hparams.attention_dim, hparams.attention_location_n_filters,
#             hparams.attention_location_kernel_size)

#         self.decoder_rnn = nn.LSTMCell(
#             hparams.attention_rnn_dim + hparams.encoder_embedding_dim,
#             hparams.decoder_rnn_dim, 1)

#         self.linear_projection = LinearNorm(
#             hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,
#             hparams.n_mel_channels * hparams.n_frames_per_step)

#         self.gate_layer = LinearNorm(
#             hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
#             bias=True, w_init_gain='sigmoid')

#     def get_go_frame(self, memory):
#         B = memory.size(0)
#         decoder_input = Variable(memory.data.new(
#             B, self.n_mel_channels * self.n_frames_per_step).zero_())
#         return decoder_input

#     def initialize_decoder_states(self, memory, mask):
#         B = memory.size(0)
#         MAX_TIME = memory.size(1)

#         self.attention_hidden = Variable(memory.data.new(
#             B, self.attention_rnn_dim).zero_())
#         self.attention_cell = Variable(memory.data.new(
#             B, self.attention_rnn_dim).zero_())

#         self.decoder_hidden = Variable(memory.data.new(
#             B, self.decoder_rnn_dim).zero_())
#         self.decoder_cell = Variable(memory.data.new(
#             B, self.decoder_rnn_dim).zero_())

#         self.attention_weights = Variable(memory.data.new(
#             B, MAX_TIME).zero_())
#         self.attention_weights_cum = Variable(memory.data.new(
#             B, MAX_TIME).zero_())
#         self.attention_context = Variable(memory.data.new(
#             B, self.encoder_embedding_dim).zero_())

#         self.memory = memory
#         self.processed_memory = self.attention_layer.memory_layer(memory)
#         self.mask = mask

#     def parse_decoder_inputs(self, decoder_inputs):
#         decoder_inputs = decoder_inputs.transpose(1, 2)
#         decoder_inputs = decoder_inputs.view(
#             decoder_inputs.size(0),
#             int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
#         decoder_inputs = decoder_inputs.transpose(0, 1)
#         return decoder_inputs

#     def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
#         alignments = torch.stack(alignments).transpose(0, 1)
#         gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()
#         mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
#         mel_outputs = mel_outputs.view(
#             mel_outputs.size(0), -1, self.n_mel_channels)
#         mel_outputs = mel_outputs.transpose(1, 2)
#         return mel_outputs, gate_outputs, alignments

#     def decode(self, decoder_input):
#         cell_input = torch.cat((decoder_input, self.attention_context), -1)
#         self.attention_hidden, self.attention_cell = self.attention_rnn(
#             cell_input, (self.attention_hidden, self.attention_cell))
#         self.attention_hidden = F.dropout(
#             self.attention_hidden, self.p_attention_dropout, self.training)

#         attention_weights_cat = torch.cat(
#             (self.attention_weights.unsqueeze(1),
#              self.attention_weights_cum.unsqueeze(1)), dim=1)
#         self.attention_context, self.attention_weights = self.attention_layer(
#             self.attention_hidden, self.memory, self.processed_memory,
#             attention_weights_cat, self.mask)

#         self.attention_weights_cum += self.attention_weights
#         decoder_input = torch.cat(
#             (self.attention_hidden, self.attention_context), -1)
#         self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
#             decoder_input, (self.decoder_hidden, self.decoder_cell))
#         self.decoder_hidden = F.dropout(
#             self.decoder_hidden, self.p_decoder_dropout, self.training)

#         # Use decoder_input directly (already contains attention_hidden and attention_context)
#         decoder_output = self.linear_projection(decoder_input)
#         gate_prediction = self.gate_layer(decoder_input)
#         return decoder_output, gate_prediction, self.attention_weights

#     def forward(self, memory, decoder_inputs, memory_lengths):
#         decoder_input = self.get_go_frame(memory).unsqueeze(0)
#         decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
#         decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
#         decoder_inputs = self.prenet(decoder_inputs)

#         self.initialize_decoder_states(
#             memory, mask=~get_mask_from_lengths(memory_lengths))

#         mel_outputs, gate_outputs, alignments = [], [], []
#         while len(mel_outputs) < decoder_inputs.size(0) - 1:
#             decoder_input = decoder_inputs[len(mel_outputs)]
#             mel_output, gate_output, attention_weights = self.decode(
#                 decoder_input)
#             mel_outputs += [mel_output.squeeze(1)]
#             gate_outputs += [gate_output.squeeze(1)]
#             alignments += [attention_weights]

#         mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
#             mel_outputs, gate_outputs, alignments)
#         return mel_outputs, gate_outputs, alignments

#     def inference(self, memory):
#         decoder_input = self.get_go_frame(memory)
#         self.initialize_decoder_states(memory, mask=None)

#         mel_outputs, gate_outputs, alignments = [], [], []
#         while True:
#             decoder_input = self.prenet(decoder_input)
#             mel_output, gate_output, alignment = self.decode(decoder_input)

#             mel_outputs += [mel_output.squeeze(1)]
#             gate_outputs += [gate_output]
#             alignments += [alignment]

#             if torch.sigmoid(gate_output.data) > self.gate_threshold:
#                 break
#             elif len(mel_outputs) == self.max_decoder_steps:
#                 print("Warning! Reached max decoder steps")
#                 break

#             decoder_input = mel_output

#         mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
#             mel_outputs, gate_outputs, alignments)
#         return mel_outputs, gate_outputs, alignments

# class Tacotron2(nn.Module):
#     def __init__(self, hparams):
#         super(Tacotron2, self).__init__()
#         self.mask_padding = hparams.mask_padding
#         self.fp16_run = hparams.fp16_run
#         self.n_mel_channels = hparams.n_mel_channels
#         self.n_frames_per_step = hparams.n_frames_per_step
#         self.embedding = nn.Embedding(
#             hparams.n_symbols, hparams.symbols_embedding_dim)
#         std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
#         val = sqrt(3.0) * std
#         self.embedding.weight.data.uniform_(-val, val)
#         self.encoder = Encoder(hparams)
#         self.decoder = Decoder(hparams)
#         self.postnet = Postnet(hparams)

#     def parse_batch(self, batch):
#         text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
#         text_padded = to_gpu(text_padded).long()
#         input_lengths = to_gpu(input_lengths).long()
#         max_len = torch.max(input_lengths.data).item()
#         mel_padded = to_gpu(mel_padded).float()
#         gate_padded = to_gpu(gate_padded).float()
#         output_lengths = to_gpu(output_lengths).long()
#         return (
#             (text_padded, input_lengths, mel_padded, max_len, output_lengths),
#             (mel_padded, gate_padded))

#     def parse_output(self, outputs, output_lengths=None):
#         if self.mask_padding and output_lengths is not None:
#             mask = ~get_mask_from_lengths(output_lengths)
#             mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
#             mask = mask.permute(1, 0, 2)
#             outputs[0].data.masked_fill_(mask, 0.0)
#             outputs[1].data.masked_fill_(mask, 0.0)
#             outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)
#         return outputs

#     def forward(self, inputs):
#         text_inputs, text_lengths, mels, max_len, output_lengths = inputs
#         text_lengths, output_lengths = text_lengths.data, output_lengths.data

#         embedded_inputs = self.embedding(text_inputs).transpose(1, 2)
#         encoder_outputs = self.encoder(embedded_inputs, text_lengths)
#         mel_outputs, gate_outputs, alignments = self.decoder(
#             encoder_outputs, mels, memory_lengths=text_lengths)

#         mel_outputs_postnet = self.postnet(mel_outputs)
#         mel_outputs_postnet = mel_outputs + mel_outputs_postnet

#         return self.parse_output(
#             [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
#             output_lengths)

#     def inference(self, inputs):
#         embedded_inputs = self.embedding(inputs).transpose(1, 2)
#         encoder_outputs = self.encoder.inference(embedded_inputs)
#         mel_outputs, gate_outputs, alignments = self.decoder.inference(
#             encoder_outputs)

#         mel_outputs_postnet = self.postnet(mel_outputs)
#         mel_outputs_postnet = mel_outputs + mel_outputs_postnet

#         outputs = self.parse_output(
#             [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
#         return outputs
# model.py: Final TacoWave Model Implementation
# End-to-End TTS with Forward Attention and BiLSTM Encoder
# Compatible with Python 3.7 (uses Variable, older torch compat)
# Integrates with BigVGAN for waveform;
# Usage: model = TacoWave(hparams); outputs = model(batch) or model.inference(text_ids)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import LinearNorm, ConvNorm, to_gpu, get_mask_from_lengths
from hparams import create_tacowave_v2_hparams  # V2 hparams

class GeluActivation(nn.Module):
    def __init__(self):
        super(GeluActivation, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x)

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.hidden_dim = hidden_dim * 2  # Bidirectional output

    def forward(self, x):
        output, _ = self.lstm(x)
        return output

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

class ForwardAttention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(ForwardAttention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")
        self.eps = 1e-8  # For normalization stability

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        processed_query = self.query_layer(query.unsqueeze(1)).squeeze(1)
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query.unsqueeze(1) + processed_attention_weights + processed_memory))
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask, prev_attention_weights=None):
        # Compute raw energies
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        # Softmax for raw alpha_t
        alpha_t = F.softmax(alignment, dim=1)

        if prev_attention_weights is not None:
            # Forward shift: Pad left by 1 and slice [:-1]
            B, T = prev_attention_weights.shape
            tilde_alpha_prev = F.pad(prev_attention_weights.unsqueeze(1), (1, 0), mode='constant', value=0).squeeze(1)[:, :-1]
            # Reweight: alpha' = alpha_t * tilde_alpha_prev
            alpha_prime = alpha_t * tilde_alpha_prev
            # Normalize: alpha_t = alpha_prime / (sum(alpha_prime) + eps)
            alpha_t = alpha_prime / (torch.sum(alpha_prime, dim=1, keepdim=True) + self.eps)
            # Renormalize to ensure softmax-like
            alpha_t = F.softmax(alpha_t + 1e-8, dim=1)  # Small epsilon for stability

        attention_weights = alpha_t
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return attention_context, attention_weights

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes, dropout=0.05):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])
        self.dropout = dropout

    def forward(self, x):
        for linear in self.layers[:-1]:
            x = F.relu(linear(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)  # No activation on last
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class Postnet(nn.Module):
    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.gelu = GeluActivation()
        self.tanh = nn.Tanh()

        # 7 conv layers as per diagram (5 core + 2 refinement)
        for i in range(hparams.postnet_n_convolutions + 2):  # Extend to 7
            if i == 0:
                in_channels = hparams.n_mel_channels
                out_channels = hparams.postnet_embedding_dim
                gain = 'tanh'
            elif i == hparams.postnet_n_convolutions + 1:
                in_channels = hparams.postnet_embedding_dim
                out_channels = hparams.n_mel_channels
                gain = 'linear'
            else:
                in_channels = out_channels = hparams.postnet_embedding_dim
                gain = 'tanh'

            conv = ConvNorm(in_channels, out_channels,
                            kernel_size=hparams.postnet_kernel_size, stride=1,
                            padding=int((hparams.postnet_kernel_size - 1) / 2),
                            dilation=1, w_init_gain=gain)
            self.convolutions.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(out_channels))

    def forward(self, x):
        # No initial residual to avoid dim mismatch (80 vs 768); process sequentially
        for i in range(len(self.convolutions)):
            x = self.convolutions[i](x)
            x = self.batch_norms[i](x)
            x = x.transpose(1, 2)  # BN on time dim
            x = self.gelu(x)
            if i < len(self.convolutions) - 1:
                x = self.tanh(x)  # Tanh except last
            else:
                x = x  # No final tanh
            x = x.transpose(1, 2)
            x = F.dropout(x, p=0.5, training=self.training)

        return x  # Add to input Mel outside in TacoWave.forward

class Encoder(nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.embedding_dim = hparams.symbols_embedding_dim
        self.n_convolutions = hparams.encoder_n_convolutions
        self.kernel_size = hparams.encoder_kernel_size

        self.embedding = nn.Embedding(hparams.n_symbols, self.embedding_dim)
        self.convolutions = nn.ModuleList()
        self.gelu = GeluActivation()

        for _ in range(self.n_convolutions):
            conv = ConvNorm(self.embedding_dim, self.embedding_dim,
                            kernel_size=self.kernel_size, stride=1,
                            padding=int((self.kernel_size - 1) / 2),
                            dilation=1, w_init_gain='relu')
            self.convolutions.append(conv)

        # BiLSTM: 256 hidden per dir -> 512 out
        self.bilstm = BiLSTM(self.embedding_dim, 256)
        self.output_dim = 512  # BiLSTM out

    def forward(self, x, input_lengths):
        x = self.embedding(x.long()).transpose(1, 2)  # (B, dim, T)

        for conv in self.convolutions:
            x = conv(x)
            x = x.transpose(1, 2)
            x = self.gelu(x)
            x = x.transpose(1, 2)

        x = x.transpose(1, 2)  # To (B, T, dim) for LSTM
        x = self.bilstm(x)  # (B, T, 512)

        return x

    def inference(self, x):
        # Similar to forward, no lengths
        x = self.embedding(x.long()).transpose(1, 2)
        for conv in self.convolutions:
            x = conv(x)
            x = x.transpose(1, 2)
            x = self.gelu(x)
            x = x.transpose(1, 2)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = 512  # BiLSTM output dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,
            [self.prenet_dim, self.prenet_dim])  # 384 x2

        self.attention_rnn = nn.LSTMCell(
            self.prenet_dim + self.encoder_embedding_dim,
            self.attention_rnn_dim)

        self.attention_layer = ForwardAttention(
            self.attention_rnn_dim, self.encoder_embedding_dim,
            hparams.attention_dim, hparams.attention_location_n_filters,
            hparams.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            self.attention_rnn_dim + self.encoder_embedding_dim,
            self.decoder_rnn_dim, 1)  # Single layer; stack if multi

        self.linear_projection = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            hparams.n_mel_channels * hparams.n_frames_per_step)

        self.gate_layer = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        alignments = torch.stack(alignments).transpose(0, 1)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1).contiguous()
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input, prev_attention_weights=None):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask, prev_attention_weights)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_output = self.linear_projection(decoder_input)
        gate_prediction = self.gate_layer(decoder_input)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        prev_attn = None
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input, prev_attn)
            prev_attn = attention_weights
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        prev_attn = None
        for step in range(self.max_decoder_steps):
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input, prev_attn)
            prev_attn = alignment

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

class TacoWave(nn.Module):
    def __init__(self, hparams):
        super(TacoWave, self).__init__()
        self.mask_padding = hparams.mask_padding
        self.fp16_run = hparams.fp16_run
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)
        self.encoder_embedding_dim = self.encoder.output_dim  # 512 from BiLSTM

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()
        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)
        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        encoder_outputs = self.encoder(text_inputs, text_lengths)
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths)

    def inference(self, inputs):
        encoder_outputs = self.encoder.inference(inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
        return outputs

# Usage Example:
# hparams = create_tacowave_v2_hparams()
# model = TacoWave(hparams)
# For BigVGAN: Integrate separately for waveform from mel_outputs_postnet[1]