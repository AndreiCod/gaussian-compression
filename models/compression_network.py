import torch
from torch import nn
import numpy as np


class CompressionNet(nn.Module):
    def __init__(
        self,
        input_size,
        hash_encoding_size,
        output_size,
        hidden_size,
        num_hidden_layers,
        num_freq,
        output_activation,
        output_split_size,
        dtype=torch.float32,
    ):
        super(CompressionNet, self).__init__()

        self.dtype = dtype
        self.input_size = input_size * num_freq * 2 + hash_encoding_size
        self.hash_encoding_size = hash_encoding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.output_activation = output_activation
        self.num_freq = num_freq
        self.freq_bands = nn.Parameter(
            2.0
            ** (torch.linspace(0.0, self.num_freq - 1, self.num_freq, dtype=self.dtype))
            * torch.pi,
            requires_grad=False,
        )

        self.block1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, dtype=self.dtype),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size, dtype=self.dtype),
                    nn.ReLU(),
                )
                for _ in range(self.num_hidden_layers // 2)
            ],
        )
        self.block2 = nn.Sequential(
            nn.Linear(
                self.hidden_size + self.input_size, self.hidden_size, dtype=self.dtype
            ),
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size, dtype=self.dtype),
                    nn.ReLU(),
                )
                for _ in range(self.num_hidden_layers // 2, self.num_hidden_layers)
            ],
        )

        assert np.sum(output_split_size) == output_size

        self.output_blocks = nn.ModuleList()
        for split_size in output_split_size:
            self.output_blocks.append(
                nn.Sequential(
                    nn.Linear(
                        self.hidden_size + self.input_size,
                        self.hidden_size // 2,
                        dtype=self.dtype,
                    ),
                    nn.ReLU(),
                    nn.Linear(self.hidden_size // 2, split_size, dtype=self.dtype),
                )
            )

    def embed(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(-1) * self.freq_bands
        x = x.view(-1, self.num_freq).unsqueeze(-1)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=1).view(batch_size, -1)
        return x

    def forward(self, x, hash_encoding):
        if self.num_freq > 0:
            embedded_x = self.embed(x)
            embedded_x = torch.cat([embedded_x, hash_encoding], dim=1)
        else:
            embedded_x = hash_encoding.to(self.dtype)

        x = self.block1(embedded_x)
        x = torch.cat([x, embedded_x], dim=1)
        x = self.block2(x)
        x = torch.cat([x, embedded_x], dim=1)

        outputs = []
        for output_block in self.output_blocks:
            outputs.append(output_block(x))
        x = torch.cat(outputs, dim=1)

        for i, activation in enumerate(self.output_activation):
            if activation == "relu":
                x[:, i] = torch.relu(x[:, i])
            elif activation == "sigmoid":
                x[:, i] = torch.sigmoid(x[:, i])
            elif activation == "tanh":
                x[:, i] = torch.tanh(x[:, i])
            elif activation == "softmax":
                x[:, i] = torch.softmax(x[:, i])
            elif activation == "none":
                pass
            else:
                raise ValueError("Invalid activation function")

        return x
