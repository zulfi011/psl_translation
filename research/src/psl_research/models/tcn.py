import torch.nn as nn

from ..registry import MODELS


class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size, padding=pad, dilation=dilation), nn.BatchNorm1d(c_out), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel_size, padding=pad, dilation=dilation), nn.BatchNorm1d(c_out),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))


@MODELS.register("tcn")
class TCN(nn.Module):
    """
    Configurable TCN. Differences from FYP v1 (`tcn_v1` below): dilated convolutions
    for a wider receptive field, dropout inside blocks, an input projection, and
    channel widths set from config.
    """

    def __init__(self, input_size, num_classes, channels=(64, 128, 128), kernel_size=3,
                 dilations=None, dropout=0.3, head_dropout=0.4, input_proj=None):
        super().__init__()
        dilations = dilations or [2 ** i for i in range(len(channels))]
        self.proj = nn.Conv1d(input_size, input_proj, 1) if input_proj else nn.Identity()
        c = input_proj or input_size
        blocks = []
        for c_out, d in zip(channels, dilations):
            blocks.append(TemporalBlock(c, c_out, kernel_size, d, dropout))
            c = c_out
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(head_dropout), nn.Linear(c, num_classes))

    def forward(self, x):                        # x: (B, T, F)
        return self.head(self.blocks(self.proj(x.transpose(1, 2))))


@MODELS.register("tcn_v1")
class TCNv1(nn.Module):
    """Exact FYP v1 architecture (497,240 params at 258→24). Use as the baseline."""

    def __init__(self, input_size, num_classes, dropout=0.4):
        super().__init__()

        def block(i, o):
            return _V1Block(i, o)
        self.block1, self.block2, self.block3 = block(input_size, 64), block(64, 128), block(128, 256)
        self.pool, self.dropout, self.fc = nn.AdaptiveAvgPool1d(1), nn.Dropout(dropout), nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block3(self.block2(self.block1(x.permute(0, 2, 1))))
        return self.fc(self.dropout(self.pool(x).squeeze(-1)))


class _V1Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv1d(i, o, 3, padding=1), nn.BatchNorm1d(o)
        self.conv2, self.bn2 = nn.Conv1d(o, o, 3, padding=1), nn.BatchNorm1d(o)
        self.relu = nn.ReLU()
        self.skip = nn.Conv1d(i, o, 1) if i != o else nn.Identity()

    def forward(self, x):
        r = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x)) + r)
