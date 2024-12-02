from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

from continuity_contrastive_ecg.utils.module import Module


class IngestECG(nn.Module):
    def __init__(
        self,
        ch_in: int = 1,
        ch_out: int = 64,
        group_mode: str = "mix",
        filt_len: int = 16,
        filt_step: int = 1,
        layer_padding: str = "same",
        filt_order: int = 1,
        pool_step: int = 1,
    ) -> None:
        super().__init__()

        if group_mode == "parallel":
            groups = ch_in
        else:
            groups = 1

        self.conv_prl_ord = nn.ModuleList()
        layer_0 = nn.Sequential(
            nn.Conv1d(
                ch_in,
                ch_out,
                groups=groups,
                kernel_size=filt_len,
                stride=filt_step,
                padding=layer_padding,
                bias=False,
            ),
            nn.BatchNorm1d(ch_out),
        )
        self.conv_prl_ord.append(layer_0)
        if filt_order > 1:
            ord_filt_len = filt_len
            if group_mode == "parallel":
                groups = ch_out
            else:
                groups = 1
            for _ in range(1, filt_order):
                layer_l = nn.Sequential(
                    nn.Conv1d(
                        ch_out,
                        ch_out,
                        groups=groups,
                        kernel_size=ord_filt_len,
                        stride=1,
                        padding="same",
                        bias=False,
                    ),
                    nn.BatchNorm1d(ch_out),
                )
                self.conv_prl_ord.append(layer_l)
        self.pool = nn.Sequential(
            nn.ReLU(), nn.MaxPool1d(kernel_size=pool_step, stride=pool_step, padding=0, ceil_mode=True)
        )

    def forward(self, x):
        layer_0 = self.conv_prl_ord[0]
        out = layer_0(x)

        if len(self.conv_prl_ord) > 1:
            out_curr = out
            out_list = [out_curr]
            for layer_ in range(1, len(self.conv_prl_ord)):
                layer_l = self.conv_prl_ord[layer_]
                out_curr = layer_l(out_curr)
                out_list.append(out_curr)
            out = torch.cat(out_list, dim=1)
        out = self.pool(out)

        return out


def hidden_layer(feat_in: int, feat_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(feat_in, feat_out, bias=False), nn.BatchNorm1d(feat_out), nn.ReLU(inplace=True)
    )


class DigestToRegress(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        logger.info(f"Init Digest with num outcomes {opt.num_outcomes}")
        try:
            hidden_layer_list = [int(tempvar) for tempvar in opt.digest_hidden_layers.split(",")]
            opt.digest_num_hidden_layers = len(hidden_layer_list)
        except Exception:
            logger.warning("Digest Hidden Layers not specified")
            opt.digest_num_hidden_layers = 0

        if opt.digest_num_hidden_layers < 1:
            logger.info("Digest: Only Regression with num outcomes", opt.num_outcomes)
            self.digest_layer = nn.Linear(opt.digest_ch_in, opt.num_outcomes)
        else:
            hidden_layer_list = [int(tempvar) for tempvar in opt.digest_hidden_layers.split(",")]
            if len(hidden_layer_list) < 1:  # .split(',')
                raise NotImplementedError("Spec of Hidden Layers unknown")
            self.digest_layer = nn.Sequential()
            for layer_ in range(len(hidden_layer_list)):
                if len == 0:
                    feat_in = opt.digest_ch_in
                else:
                    feat_in = hidden_layer_list[len - 1]
                feat_out = hidden_layer_list[layer_]
                self.digest_layer.add_module(
                    "digest_layer" + str(layer_), hidden_layer(feat_in=feat_in, feat_out=feat_out)
                )

            self.digest_layer.add_module(
                "digest_layer" + str(len(hidden_layer_list)),
                nn.Linear(hidden_layer_list[-1], opt.num_outcomes),
            )
            logger.info(f"Digest: Hidden Layers with num outcomes {opt.num_outcomes}")

    def forward(self, x):
        return self.digest_layer(x)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        filt_len: int = 16,
        group_mode: str = "mix",
        layer_padding: str = "valid",
        downsample: nn.Module | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        groups = 1
        groups2 = 1
        if (group_mode == "parallel") and (out_ch % in_ch == 0):
            groups = in_ch
            groups2 = out_ch
        self.conv_blk = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=filt_len,
                stride=stride,
                padding=layer_padding,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_ch, out_ch, kernel_size=filt_len, stride=1, padding="same", groups=groups2, bias=False
            ),
        )
        self.downsample = downsample
        self.stride = stride
        self.filt_len = filt_len
        self.act = nn.Sequential(norm_layer(out_ch), nn.ReLU(inplace=True))

    def forward(self, inputs):
        x, y = inputs
        out = self.conv_blk(x)
        y_new = y[..., self.filt_len // 2 : (y.shape[-1] - ((self.filt_len - 1) // 2))]
        if self.downsample is not None:
            identity = self.downsample(y_new)
        else:
            identity = y_new
        out += identity
        y = out
        x = self.act(out)
        return (x, y)


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv1d:
    """1x1 convolution."""
    return nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding="same", bias=False)


class ResNet(nn.Module):
    def __init__(
        self,
        opt,
        block: type[BasicBlock],
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        resnet_layer_list = [int(tempvar) for tempvar in opt.resnet_layers.split(",")]
        resnet_ch_out_list = [int(tempvar) for tempvar in opt.resnet_ch_list.split(",")]
        opt.resnet_ch_out = resnet_ch_out_list[-1]
        resnet_stride_list = [int(tempvar) for tempvar in opt.resnet_skip_stride.split(",")]
        num_layers = len(resnet_layer_list)
        print("resnetlayers:" + str(resnet_layer_list))
        if num_layers <= 1:
            raise NotImplementedError("Number of Layers <= 1 in ResNet")

        opt.resnet_ch_out = resnet_ch_out_list[-1]
        resnet_ch_in = opt.resnet_ch_in  # opt.ingest_ch_out * opt.ingest_filt_ord
        resnet_sig_out = 1

        self.resnet_layer = nn.Sequential()
        self.resnet_layer_ch_in = resnet_ch_in
        for layer_ in range(num_layers):
            layer_name = "resnet_layer_" + str(layer_)
            block_cnt = resnet_layer_list[layer_]
            if len == 0:
                layer_ch_in = resnet_ch_in
            elif layer_ch_in is not None:
                layer_ch_in = layer_ch_in
            else:
                raise ValueError("layer_ch_in not defined")
            layer_ch_out = resnet_ch_out_list[layer_]
            block_stride = resnet_stride_list[layer_]
            curr_layer = self._make_layer(
                block=block,
                layer_ch_in=layer_ch_in,
                layer_ch_out=layer_ch_out,
                block_cnt=block_cnt,
                stride=block_stride,
            )
            self.resnet_layer.add_module(layer_name, curr_layer)

        self.resnet_pool = nn.AdaptiveAvgPool1d(resnet_sig_out)

    def _make_layer(
        self,
        block: type[BasicBlock],
        layer_ch_in: int,
        layer_ch_out: int,
        block_cnt: int,
        filt_len: int = 16,
        group_mode: str = "mix",
        layer_padding: str = "valid",
        stride: int = 1,
    ) -> nn.Sequential:
        print(layer_ch_in, layer_ch_out, block_cnt, filt_len, group_mode, layer_padding, stride)
        norm_layer = self._norm_layer
        downsample = None
        if (stride > 1) or (layer_ch_in != layer_ch_out):
            downsample = nn.Sequential(
                nn.MaxPool1d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True),
                conv1x1(layer_ch_in, layer_ch_out),
            )
        locallayers = []
        locallayers.append(
            block(
                in_ch=layer_ch_in,
                out_ch=layer_ch_out,
                stride=stride,
                filt_len=filt_len,
                group_mode=group_mode,
                layer_padding=layer_padding,
                downsample=downsample,
                norm_layer=norm_layer,
            )
        )
        for _ in range(1, block_cnt):
            locallayers.append(
                block(
                    in_ch=layer_ch_out,
                    out_ch=layer_ch_out,
                    stride=stride,
                    filt_len=filt_len,
                    group_mode=group_mode,
                    layer_padding=layer_padding,
                    downsample=nn.Sequential(
                        nn.MaxPool1d(kernel_size=stride, stride=stride, padding=0, ceil_mode=True),
                        conv1x1(layer_ch_out, layer_ch_out),
                    ),
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*locallayers)

    def forward(self, x: Tensor) -> Tensor:
        x, y = self.resnet_layer((x, x))
        out = self.resnet_pool(x)
        out = torch.flatten(out, 1)
        return out


class ResNet4ECG(nn.Module, Module):
    def __init__(
        self,
        opt,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        self.opt = opt

        self._norm_layer = nn.BatchNorm1d

        if opt.ingest_conv_stride > 1:
            opt.ingest_layer_padding = "valid"

        opt.ingest_sig_in = opt.ecg_len_sec * opt.sampling_rate
        self.ingest_blk = IngestECG(
            ch_in=opt.ingest_ch_in,  # 1
            ch_out=opt.ingest_ch_out,  # 64
            group_mode=opt.ingest_mode,  # "mix", "parallel"
            filt_len=opt.ingest_kernel_size,  # 16
            filt_step=opt.ingest_conv_stride,  # 1
            layer_padding=opt.ingest_layer_padding,  # "same", "valid"
            filt_order=opt.ingest_filt_ord,  # 1
            pool_step=opt.ingest_pool_stride,  # 1
        )
        if (opt.ingest_layer_padding == "same") and (opt.ingest_conv_stride == 1):
            opt.ingest_sig_out = float(np.ceil(opt.ingest_sig_in / opt.ingest_pool_stride))
        elif (opt.ingest_layer_padding == "valid") and (opt.ingest_conv_stride == 1):
            opt.ingest_sig_out = float(
                np.ceil((opt.ingest_sig_in - opt.ingest_kernel_size + 1) / opt.ingest_pool_stride)
            )
        else:
            opt.ingest_sig_out = float(
                np.ceil(
                    np.floor(1 + (opt.ingest_sig_in - opt.ingest_kernel_size) / opt.ingest_conv_stride)
                    / opt.ingest_pool_stride
                )
            )

        opt.resnet_sig_in = opt.ingest_sig_out  # ??
        opt.resnet_ch_in = opt.ingest_ch_out * opt.ingest_filt_ord
        self.resnet_blk = ResNet(opt=opt, block=BasicBlock)

        resnet_ch_out_list = [int(tempvar) for tempvar in opt.resnet_ch_list.split(",")]
        opt.resnet_ch_out = resnet_ch_out_list[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                print("Initializing Conv1d")
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                print("Initializing Linear")
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                print("Initializing BatchNorm1d")
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    print("initializing BasicBlock BN2")
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _forward_impl(self, x: Tensor) -> Tensor:
        x_in = self.ingest_blk(x)
        h = self.resnet_blk(x_in)
        return h

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
