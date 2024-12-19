import torchinfo
import torch
import torch.nn as nn
from mamba_ssm.models.mixer_seq_simple import (
    create_block,
    _init_weights,
)
from rich import print
from torch.nn import Conv1d, ConvTranspose1d, ModuleList
from functools import partial

activations = {"relu": nn.ReLU, "silu": nn.SiLU, "mish": nn.Mish, "prelu": nn.PReLU}


class DownSample(nn.Module):
    def __init__(
        self, features_in, features_out, kernel_size, stride, num_blocks, act_fn
    ):
        super(DownSample, self).__init__()
        self.conv1d = Conv1d(
            in_channels=features_in,
            out_channels=features_out,
            kernel_size=kernel_size,
            stride=stride,
        )

        self.mamba_layer = MambaLayer(dim=features_out, num_blocks=num_blocks)
        self.act = activations[act_fn]()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.act(x)

        x, res_forward = self.mamba_layer(x)

        return x, res_forward


class UpSample(nn.Module):
    def __init__(
        self, features_in, features_out, kernel_size, stride, num_blocks, act_fn
    ):
        super(UpSample, self).__init__()

        self.convT1d = ConvTranspose1d(
            in_channels=features_in,
            out_channels=features_out,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv1d_fusion = Conv1d(
            in_channels=features_in, out_channels=features_out, kernel_size=1
        )

        self.mamba_layer = MambaLayer(
            dim=features_out, num_blocks=num_blocks, start_idx=0
        )
        
        self.act = activations[act_fn]()

    def pad(self, x, match):
        if match.shape[2] > x.shape[2]:
            zero_pad = torch.full(
                [x.shape[0], match.shape[1], match.shape[2] - x.shape[2]], 0
            ).cuda()
            return torch.cat((x, zero_pad), dim=2)
        else:
            return x

    def forward(self, x, skip):
        x = self.convT1d(x)
        x = self.act(x)

        x = self.pad(x, match=skip)
        x = torch.concat([skip, x], dim=1)
        x = self.conv1d_fusion(x)
        x = self.act(x)

        x, _ = self.mamba_layer(x)

        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, num_blocks, start_idx=0):
        super(MambaLayer, self).__init__()
        self.forward_mamba_block = MambaBlock(
            dim=dim, num_blocks=num_blocks, start_idx=start_idx
        )

    def forward(self, x, res_for=None, res_back=None):
        x = x.permute(0, 2, 1)
        x_forward = x.clone()

        x_forward, res_forward = self.forward_mamba_block(x_forward, res_for)

        x_forward = x_forward + res_forward

        x = x_forward.clone()
        x = x.permute(0, 2, 1)

        return x, res_forward


class MambaBlock(nn.Module):
    def __init__(self, dim, start_idx=0, num_blocks=3):
        super(MambaBlock, self).__init__()

        self.mamba_blocks = ModuleList(
            [
                create_block(d_model=dim, layer_idx=i)
                for i in range(start_idx, start_idx + num_blocks)
            ]
        )

        self.apply(partial(_init_weights, n_layer=num_blocks))

    def forward(self, x, residual=None):
        for block in self.mamba_blocks:
            x, residual = block(x, residual)

        return x, residual


class SepMambaCausal(nn.Module):
    def __init__(
        self,
        dim,
        kernel_sizes=[12, 12, 12],
        strides=[2, 2, 2],
        num_blocks=6,
        act_fn="relu",
        
    ):
        super(SepMambaCausal, self).__init__()
        assert len(kernel_sizes) == len(strides)

        self.down = ModuleList(
            [
                DownSample(
                    features_in=1,
                    features_out=dim,
                    kernel_size=kernel_sizes[0],
                    stride=strides[0],
                    num_blocks=num_blocks,
                    act_fn=act_fn
                ),
                DownSample(
                    features_in=dim,
                    features_out=dim * 2,
                    kernel_size=kernel_sizes[1],
                    stride=strides[1],
                    num_blocks=num_blocks,
                    act_fn=act_fn
                ),
                DownSample(
                    features_in=dim * 2,
                    features_out=dim * 4,
                    kernel_size=kernel_sizes[2],
                    stride=strides[2],
                    num_blocks=num_blocks,
                    act_fn=act_fn
                ),
            ]
        )

        self.up = ModuleList(
            [
                UpSample(
                    features_in=dim * 4,
                    features_out=dim * 2,
                    kernel_size=kernel_sizes[2],
                    stride=strides[2],
                    num_blocks=num_blocks,
                    act_fn=act_fn
                ),
                UpSample(
                    features_in=dim * 2,
                    features_out=dim,
                    kernel_size=kernel_sizes[1],
                    stride=strides[1],
                    num_blocks=num_blocks,
                    act_fn=act_fn
                ),
            ]
        )

        self.decoder = ConvTranspose1d(
            in_channels=dim,
            out_channels=2,
            kernel_size=kernel_sizes[0],
            stride=strides[0],
        )

    def forward(self, x):
        skips = []

        for down in self.down:
            x, _ = down(x)
            skips.append(x)

        skips.pop()

        for up in self.up:
            x = up(
                x,
                skip=skips.pop()
            )

        x = self.decoder(x)

        return x


if __name__ == "__main__":
    dim = 64
    num_blocks = 8
    kernel_sizes = [16, 16, 16]
    strides = [2, 2, 2]
    act_fn = "relu"

    model = SepMambaCausal(
        dim=dim,
        num_blocks=num_blocks,
        kernel_sizes=kernel_sizes,
        strides=strides,
        act_fn=act_fn,
    )
    model = model.cuda()

    torchinfo.summary(model)

    x = torch.randn(1, 1, 32000, requires_grad=True).cuda()
    y = model(x)

    output = y[:, :, 0]

    # Compute gradients
    grad = torch.autograd.grad(
        outputs=output,
        inputs=x,
        grad_outputs=torch.ones_like(output),
        create_graph=True
    )[0]

    print((grad != 0).sum())
    # print(grad[0, 0, :110])

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")