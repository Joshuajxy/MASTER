import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math

from base_model import SequenceModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model/2]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [N, T, d_model]
        return x + self.pe[:x.shape[1], :]  # [N, T, d_model]


class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]
        self.ktrans = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]
        self.vtrans = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)

        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),  # [d_model, d_model]
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),  # [d_model, d_model]
            Dropout(p=dropout)
        )

    def forward(self, x):
        # x: [N, T, d_model]
        x = self.norm1(x)  # [N, T, d_model]
        q = self.qtrans(x).transpose(0, 1)  # [T, N, d_model]
        k = self.ktrans(x).transpose(0, 1)  # [T, N, d_model]
        v = self.vtrans(x).transpose(0, 1)  # [T, N, d_model]

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]  # [T, N, d_model/nhead]
                kh = k[:, :, i * dim:]  # [T, N, d_model/nhead]
                vh = v[:, :, i * dim:]  # [T, N, d_model/nhead]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]  # [T, N, d_model/nhead]
                kh = k[:, :, i * dim:(i + 1) * dim]  # [T, N, d_model/nhead]
                vh = v[:, :, i * dim:(i + 1) * dim]  # [T, N, d_model/nhead]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)  # [T, N, N]
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))  # [N, T, d_model/nhead]
        att_output = torch.concat(att_output, dim=-1)  # [N, T, d_model]

        # FFN
        xt = x + att_output  # [N, T, d_model]
        xt = self.norm2(xt)  # [N, T, d_model]
        att_output = xt + self.ffn(xt)  # [N, T, d_model]

        return att_output


class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]
        self.ktrans = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]
        self.vtrans = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        # input LayerNorm
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        # FFN layerNorm
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        # FFN
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),  # [d_model, d_model]
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),  # [d_model, d_model]
            Dropout(p=dropout)
        )

    def forward(self, x):
        # x: [N, T, d_model]
        x = self.norm1(x)  # [N, T, d_model]
        q = self.qtrans(x)  # [N, T, d_model]
        k = self.ktrans(x)  # [N, T, d_model]
        v = self.vtrans(x)  # [N, T, d_model]

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]  # [N, T, d_model/nhead]
                kh = k[:, :, i * dim:]  # [N, T, d_model/nhead]
                vh = v[:, :, i * dim:]  # [N, T, d_model/nhead]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]  # [N, T, d_model/nhead]
                kh = k[:, :, i * dim:(i + 1) * dim]  # [N, T, d_model/nhead]
                vh = v[:, :, i * dim:(i + 1) * dim]  # [N, T, d_model/nhead]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)  # [N, T, T]
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))  # [N, T, d_model/nhead]
        att_output = torch.concat(att_output, dim=-1)  # [N, T, d_model]

        # FFN
        xt = x + att_output  # [N, T, d_model]
        xt = self.norm2(xt)  # [N, T, d_model]
        att_output = xt + self.ffn(xt)  # [N, T, d_model]

        return att_output


class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)  # [d_input, d_output]
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        # gate_input: [N, d_input]
        output = self.trans(gate_input)  # [N, d_output]
        output = torch.softmax(output / self.t, dim=-1)  # [N, d_output]
        return self.d_output * output  # [N, d_output]


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)  # [d_model, d_model]

    def forward(self, z):
        # z: [N, T, d_model]
        h = self.trans(z)  # [N, T, d_model]
        query = h[:, -1, :].unsqueeze(-1)  # [N, d_model, 1]
        lam = torch.matmul(h, query).squeeze(-1)  # [N, T]
        lam = torch.softmax(lam, dim=1).unsqueeze(1)  # [N, 1, T]
        output = torch.matmul(lam, z).squeeze(1)  # [N, d_model]
        return output


class MASTER(nn.Module):
    def __init__(self, d_feat, d_model, t_nhead, s_nhead, T_dropout_rate, S_dropout_rate, gate_input_start_index, gate_input_end_index, beta):
        super(MASTER, self).__init__()
        # market
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)  # F'
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.layers = nn.Sequential(
            # feature layer
            nn.Linear(d_feat, d_model),  # [d_feat, d_model]
            PositionalEncoding(d_model),  # [N, T, d_model]
            # intra-stock aggregation
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),  # [N, T, d_model]
            # inter-stock aggregation
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),  # [N, T, d_model]
            TemporalAttention(d_model=d_model),  # [N, d_model]
            # decoder
            nn.Linear(d_model, 1)  # [d_model, 1]
        )

    def forward(self, x):
        # x: [N, T, D]
        src = x[:, :, :self.gate_input_start_index]  # [N, T, D']
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]  # [N, F']
        # Apply the feature gate to the input features
        # src: [N, T, D'] (batch size, time steps, feature dimension)
        # gate_input: [N, F'] (batch size, selected feature dimension)
        # feature_gate(gate_input): [N, D'] (batch size, transformed feature dimension)
        # torch.unsqueeze(self.feature_gate(gate_input), dim=1): [N, 1, D']
        # Resulting src: [N, T, D'] (scaled by the feature gate)
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        output = self.layers(src).squeeze(-1)  # [N]

        return output


class MASTERModel(SequenceModel):
    def __init__(
            self, d_feat, d_model, t_nhead, s_nhead, gate_input_start_index, gate_input_end_index,
            T_dropout_rate, S_dropout_rate, beta, **kwargs,
    ):
        super(MASTERModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_feat = d_feat

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta

        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                                   T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                                   gate_input_start_index=self.gate_input_start_index,
                                   gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        super(MASTERModel, self).init_model()
