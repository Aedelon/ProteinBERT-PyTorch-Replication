#!/usr/bin/env python
# *** coding: utf-8 ***

"""modules.py: Contains the different modules which will be used to make a ProteinBERT model.

   * Author: Delanoe PIRARD
   * Email: delanoe.pirard.pro@gmail.com
   * Licence: MIT

   * Paper: "ProteinBERT: A universal deep-learning model of protein sequence and function. "
   * Paper's authors: Brandes, N., Ofer, D., Peleg, Y., Rappoport, N. and Linial, M. .
   * Paper DOI: https://doi.org/10.1093/bioinformatics/btac020
"""

# IMPORTS -------------------------------------------------
import torch
import torch.nn as nn


# CLASSES -------------------------------------------------
class GlobalAttentionHead(nn.Module):
    def __init__(self,
                 local_dim: int,
                 global_dim: int,
                 value_dim: int,
                 key_dim: int,
                 device: torch.device = ('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.key_dim = key_dim
        self.global_dim = global_dim

        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.Wv_parameter = nn.Parameter(
            data=torch.randn(local_dim, value_dim, device=device),
            requires_grad=True
        )
        self.Wk_parameter = nn.Parameter(
            data=torch.randn(local_dim, key_dim, device=device),
            requires_grad=True
        )
        self.Wq_parameter = nn.Parameter(
            data=torch.randn(global_dim, key_dim, device=device),
            requires_grad=True
        )

    def forward(self, x):
        x_local = x["local"]
        x_global = x["global"].repeat(self.key_dim, 1, 1).permute(1, 0, 2)

        Q = self.tanh(torch.matmul(x_global, self.Wq_parameter))
        K = self.tanh(torch.matmul(x_local, self.Wk_parameter))
        V = self.gelu(torch.matmul(x_local, self.Wv_parameter))

        return torch.matmul(
            self.softmax(torch.matmul(Q, K.permute(0, 2, 1)) / torch.sqrt(torch.tensor(self.key_dim))),
            V
        )


class GlobalAttention(nn.Module):
    def __init__(self,
                 num_heads: int,
                 local_dim: int,
                 global_dim: int,
                 value_dim: int,
                 key_dim: int,
                 device: torch.device = ('cuda' if torch.cuda.is_available() else 'cpu')
                 ):
        super().__init__()
        self.global_attention_heads = [
            GlobalAttentionHead(
                local_dim=local_dim,
                global_dim=global_dim,
                value_dim=value_dim,
                key_dim=key_dim,
                device=device
            ) for _ in range(num_heads)
        ]
        self.W_parameter = nn.Parameter(
            data=torch.randn(key_dim, device=device),
            requires_grad=True
        )

    def forward(self, x):
        attention_tensors = []
        for global_attention_head in self.global_attention_heads:
            attention_tensors.append(global_attention_head(x))

        return torch.matmul(self.W_parameter, torch.cat(attention_tensors, dim=2))


class ProteinBERTBlock(nn.Module):
    def __init__(self,
                 sequences_length,
                 local_dim: int,
                 global_dim: int,
                 num_heads: int,
                 key_dim: int,
                 conv_kernel_size: int = 9,
                 wide_conv_dilation: int = 5,
                 device: torch.device = ('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()

        # Assert the global_dim must be divisible by num_heads
        assert global_dim % num_heads == 0, \
            f"Global_dim must be divisible by num_heads. \
            Global_dim: {global_dim}. Num_heads: {num_heads}"

        self.sequences_length = sequences_length
        self.local_dim = local_dim

        self.global_attention_layer = GlobalAttention(
            num_heads=num_heads,
            local_dim=local_dim,
            global_dim=global_dim,
            value_dim=int(global_dim / num_heads),
            key_dim=key_dim,
            device=device
        )

        self.local_narrow_conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=local_dim,
                out_channels=local_dim,
                kernel_size=conv_kernel_size,
                stride=1,
                dilation=1,
                padding="same",
                device=device
            ),
            nn.GELU()
        )
        self.local_wide_conv_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=local_dim,
                out_channels=local_dim,
                kernel_size=conv_kernel_size,
                stride=1,
                dilation=wide_conv_dilation,
                padding="same",
                device=device
            ),
            nn.GELU()
        )
        self.local_norm_1 = nn.LayerNorm(
            normalized_shape=(sequences_length, local_dim),
            device=device
        )

        self.local_linear_layer = nn.Sequential(
            nn.Linear(
                in_features=local_dim,
                out_features=local_dim,
                device=device
            ),
            nn.GELU()
        )
        self.local_norm_2 = nn.LayerNorm(
            normalized_shape=(sequences_length, local_dim),
            device=device
        )

        self.global_to_local_linear_layer = nn.Sequential(
            nn.Linear(
                in_features=global_dim,
                out_features=local_dim,
                device=device
            ),
            nn.GELU()
        )

        self.global_linear_layer_1 = nn.Sequential(
            nn.Linear(
                in_features=global_dim,
                out_features=global_dim,
                device=device
            ),
            nn.GELU()
        )
        self.global_norm_1 = nn.LayerNorm(
            normalized_shape=global_dim,
            device=device
        )

        self.global_linear_layer_2 = nn.Sequential(
            nn.Linear(
                in_features=global_dim,
                out_features=global_dim,
                device=device
            ),
            nn.GELU()
        )
        self.global_norm_2 = nn.LayerNorm(
            normalized_shape=global_dim,
            device=device
        )

    def forward(self, x):
        x_local = x["local"]
        x_global = x["global"]

        x_hidden_local_narrow = self.local_narrow_conv_layer(x_local)
        x_hidden_local_wide = self.local_wide_conv_layer(x_local)

        x_hidden_global_to_local = self.global_to_local_linear_layer(x_global)
        x_hidden_global_to_local = x_hidden_global_to_local.repeat(self.sequences_length, 1, 1).permute(1, 2, 0)

        x_hidden_local = x_local + x_hidden_local_narrow + x_hidden_local_wide + x_hidden_global_to_local
        x_hidden_local = self.local_norm_1(x_hidden_local.permute(0, 2, 1))

        x_hidden_local_linear = self.local_linear_layer(x_hidden_local)

        x_hidden_local = x_hidden_local + x_hidden_local_linear
        x_hidden_local = self.local_norm_2(x_hidden_local)

        x_hidden_local_to_global = self.global_attention_layer({"local": x_hidden_local, "global": x_global})

        x_hidden_global = self.global_linear_layer_1(x_global)

        x_hidden_global = x_global + x_hidden_global + x_hidden_local_to_global
        x_hidden_global = self.global_norm_1(x_hidden_global)

        x_hidden_global_linear = self.global_linear_layer_2(x_hidden_global)

        x_hidden_global = x_hidden_global + x_hidden_global_linear
        x_hidden_global = self.global_norm_2(x_hidden_global)

        return {"local": x_hidden_local.permute(0, 2, 1), "global": x_hidden_global}


class ProteinBERT(nn.Module):
    def __init__(self,
                 sequences_length: int,
                 num_annotations: int,
                 local_dim: int,
                 global_dim: int,
                 key_dim: int,
                 num_heads: int,
                 num_blocks: int,
                 conv_kernel_size: int = 9,
                 wide_conv_dilation: int = 5,
                 vocab_size: int = 26,
                 device: torch.device = ('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()

        self.local_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=local_dim,
            device=device
        )

        self.global_linear_layer = nn.Sequential(
            nn.Linear(
                in_features=num_annotations,
                out_features=global_dim,
                device=device
            ),
            nn.GELU()
        )

        self.proteinBERT_blocks = nn.Sequential(*[
            ProteinBERTBlock(
                sequences_length=sequences_length,
                global_dim=global_dim,
                local_dim=local_dim,
                key_dim=key_dim,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                wide_conv_dilation=wide_conv_dilation,
                device=device
            ) for _ in range(num_blocks)
        ])

        self.pretraining_local_output = nn.Sequential(
            nn.Linear(
                in_features=local_dim,
                out_features=vocab_size,
                device=device
            ),
            nn.Softmax()
        )

        self.pretraining_global_output = nn.Sequential(
            nn.Linear(
                in_features=global_dim,
                out_features=num_annotations,
                device=device
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_local = x["local"]
        x_global = x["global"]

        x = self.proteinBERT_blocks({
            "local": self.local_embedding(x_local).permute(0, 2, 1),
            "global": self.global_linear_layer(x_global)
        })

        return self.pretraining_local_output(x["local"].permute(0, 2, 1)), self.pretraining_global_output(x["global"])

