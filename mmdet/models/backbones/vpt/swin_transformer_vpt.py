import math
import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
from functools import reduce
from operator import mul
from torch.nn import Conv2d, Dropout

from timm.models.layers import to_2tuple, trunc_normal_

from .swin_vpt_utils import (
    BasicLayer, PatchMerging, SwinTransformer, SwinTransformerBlock,
    window_partition, window_reverse, WindowAttention
    )
from mmdet.utils import get_root_logger
from mmcv_custom import load_checkpoint
from ...builder import BACKBONES


@BACKBONES.register_module()
class SwinTransformer_vpt(SwinTransformer):
    def __init__(
        self, prompt_config,
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            **kwargs
    ):
        if prompt_config.LOCATION == "pad":
            img_size += 2 * prompt_config.NUM_TOKENS
        super(SwinTransformer_vpt, self).__init__(
            img_size, patch_size, in_chans, embed_dim, depths,
            num_heads, window_size, mlp_ratio, qkv_bias, qk_scale, drop_rate,
            attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm,
            use_checkpoint, out_indices=(0, 1, 2, 3), frozen_stages=-1, **kwargs
        )
        self.prompt_config = prompt_config
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        if self.prompt_config.LOCATION == "add":
            num_tokens = self.embeddings.position_embeddings.shape[1]
        elif self.prompt_config.LOCATION == "add-1":
            num_tokens = 1
        else:
            num_tokens = self.prompt_config.NUM_TOKENS

        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)
        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT
            self.prompt_proj = nn.Linear(
                prompt_dim, embed_dim)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            self.prompt_proj = nn.Identity()

        # build layers
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2 ** i_layer),
                    self.patches_resolution[1] // (2 ** i_layer)
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                block_module=PromptedSwinTransformerBlock,
                downsample=PromptedPatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                num_prompts=num_tokens,
                prompt_location=self.prompt_config.LOCATION,
                deep_prompt=self.prompt_config.DEEP
            )
            self.layers.append(layer)

        if self.prompt_config.INITIATION == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + embed_dim))  # noqa

            if self.prompt_config.LOCATION == "below":
                self.patch_embed.proj = Conv2d(
                    in_channels=num_tokens+3,
                    out_channels=embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size
                )
                # add xavier_uniform initialization
                nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
                nn.init.zeros_(self.patch_embed.proj.bias)

                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, img_size[0], img_size[1]))
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)


            elif self.prompt_config.LOCATION == "pad":
                self.prompt_embeddings_tb = nn.Parameter(torch.zeros(
                    1, 3, 2 * num_tokens, img_size[0]
                ))
                self.prompt_embeddings_lr = nn.Parameter(torch.zeros(
                    1, 3, img_size[0] - 2 * num_tokens, 2 * num_tokens
                ))

                nn.init.uniform_(self.prompt_embeddings_tb.data, 0.0, 1.0)
                nn.init.uniform_(self.prompt_embeddings_lr.data, 0.0, 1.0)

                self.prompt_norm = tv.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )

            else:
                # for "prepend"
                self.prompt_embeddings = nn.Parameter(torch.zeros(
                    1, num_tokens, embed_dim))
                nn.init.uniform_(self.prompt_embeddings.data, -val, val)

                if self.prompt_config.DEEP:
                    # NOTE: only for 4 layers, need to be more flexible
                    self.deep_prompt_embeddings_0 = nn.Parameter(
                        torch.zeros(
                            depths[0] - 1, num_tokens, embed_dim
                    ))
                    nn.init.uniform_(
                        self.deep_prompt_embeddings_0.data, -val, val)
                    self.deep_prompt_embeddings_1 = nn.Parameter(
                        torch.zeros(
                            depths[1], num_tokens, embed_dim * 2
                    ))
                    nn.init.uniform_(
                        self.deep_prompt_embeddings_1.data, -val, val)
                    self.deep_prompt_embeddings_2 = nn.Parameter(
                        torch.zeros(
                            depths[2], num_tokens, embed_dim * 4
                    ))
                    nn.init.uniform_(
                        self.deep_prompt_embeddings_2.data, -val, val)
                    self.deep_prompt_embeddings_3 = nn.Parameter(
                        torch.zeros(
                            depths[3], num_tokens, embed_dim * 8
                    ))
                    nn.init.uniform_(
                        self.deep_prompt_embeddings_3.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

        '''wxk新增'''
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        self.out_indices = out_indices
        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        self._freeze_stages()

        # 2. 执行 VPT 特有的冻结逻辑：强制冻结所有 Backbone，只留 Prompt 可训练
        # 这一步会覆盖上面的逻辑，确保即使 frozen_stages=-1，Backbone 也是冻结的
        self._freeze_prompt_backbone()

    def _freeze_prompt_backbone(self):
        """
        Freeze all parameters in the backbone except for prompt-related parameters.
        This is the core logic for Visual Prompt Tuning.
        """
        # 需要解冻（可训练）的关键字列表
        # 包括：prompt_embeddings, deep_prompt_embeddings, prompt_proj, prompt_dropout
        # 注意：prompt_dropout 没有 parameter，但如果包含 BatchNorm 类似的层也需要处理

        for name, param in self.named_parameters():
            # 判断是否包含 prompt 相关的关键字
            # 注意：要避免误判，例如 'prompt' 这个词可能出现在其他地方，这里用具体的前缀比较稳妥
            if 'prompt_embeddings' in name or \
                    'prompt_proj' in name or \
                    'prompt_dropout' in name or \
                    'deep_prompt' in name:  # 兼容 deep prompt

                param.requires_grad = True
                # print(f"Unfreezing: {name}") # 调试用
            else:
                # 冻结所有非 Prompt 参数
                param.requires_grad = False
                # print(f"Freezing: {name}") # 调试用

        # 同时，为了保证 BN 层统计量不更新，需要强制将非 Prompt 模块设为 eval 模式
        # 虽然在 train() 方法里会做这个，但在初始化时做一次更保险
        self.eval()
        # 如果 prompt_dropout 或 prompt_proj 包含需要处于 train 模式的层（如 Dropout），
        # 在 train() 方法中会重新开启，这里全部 eval 是安全的初始状态

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]

        if self.prompt_config.LOCATION == "prepend":
            # after CLS token, all before image patches
            x, Wh, Ww = self.get_patch_embeddings(x)  # x维度(b, n, c)  # '''wxk修改'''
            prompt_embd = self.prompt_dropout(
                self.prompt_embeddings.expand(B, -1, -1))
            x = torch.cat((
                    prompt_embd, x
                ), dim=1)
            # (batch_size, n_prompt + n_patches, hidden_dim)

        elif self.prompt_config.LOCATION == "add":
            # add to the input patches + CLS
            # assert self.prompt_config.NUM_TOKENS == x.shape[1]
            x, Wh, Ww = self.get_patch_embeddings(x)  # x维度(b, n+1, c)  # '''wxk修改'''
            x = x + self.prompt_dropout(
                self.prompt_embeddings.expand(B, -1, -1))
            # (batch_size, n_patches, hidden_dim)

        elif self.prompt_config.LOCATION == "add-1":
            x, Wh, Ww = self.get_patch_embeddings(x)   # x维度(b, n+1, c)  # '''wxk修改'''
            L = x.shape[1]
            prompt_emb = self.prompt_dropout(
                self.prompt_embeddings.expand(B, -1, -1))
            x = x + prompt_emb.expand(-1, L, -1)
            # (batch_size, cls_token + n_patches, hidden_dim)

        elif self.prompt_config.LOCATION == "pad":
            prompt_emb_lr = self.prompt_norm(
                self.prompt_embeddings_lr).expand(B, -1, -1, -1)
            prompt_emb_tb = self.prompt_norm(
                self.prompt_embeddings_tb).expand(B, -1, -1, -1)

            x = torch.cat((
                prompt_emb_lr[:, :, :, :self.num_tokens],
                x, prompt_emb_lr[:, :, :, self.num_tokens:]
                ), dim=-1)
            x = torch.cat((
                prompt_emb_tb[:, :, :self.num_tokens, :],
                x, prompt_emb_tb[:, :, self.num_tokens:, :]
            ), dim=-2)
            x, Wh, Ww  = self.get_patch_embeddings(x)  # x维度(b, n, c)  # '''wxk修改'''

        elif self.prompt_config.LOCATION == "below":
            # (batch, 3, height, width)
            x = torch.cat((
                    x,
                    self.prompt_norm(
                        self.prompt_embeddings).expand(B, -1, -1, -1),
                ), dim=1)
            x, Wh, Ww  = self.get_patch_embeddings(x)  # '''wxk修改''' x (b, n, c)
        else:
            raise ValueError("Other prompt locations are not supported")

        return x, Wh, Ww  # '''wxk修改'''

    '''wxk修改'''
    def get_patch_embeddings(self, x):
        x = self.patch_embed(x)  # (b, c, h, w)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            x= x.flatten(2).transpose(1, 2)
            x = x + self.absolute_pos_embed
        else:
            # flatten(2)：把空间维度（Wh, Ww）展平成一维，消除空间结构，得到(batch_size, c, Wh * Ww)；
            # transpose(1, 2)：交换通道维和序列维，最终得到(B, Wh * Ww, c)
            x = x.flatten(2).transpose(1, 2)

        x = self.pos_drop(x)  # (B, n, c)
        return x, Wh, Ww

    def init_weights(self, pretrained=None):

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()

            # 加载 checkpoint
            checkpoint = load_checkpoint(self, pretrained, strict=False, logger=logger)

            # VPT 关键逻辑：处理 "below" 模式下的 Patch Embedding 权重
            # 之前原代码是在 load_state_dict 里做的，现在统一放到 init_weights
            if self.prompt_config.LOCATION == "below":
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # 检查权重 key 是否存在
                if "patch_embed.proj.weight" in state_dict:
                    conv_weight = state_dict["patch_embed.proj.weight"]
                    # 将预训练权重与 Prompt 权重拼接
                    # conv_weight shape: [embed_dim, 3, k, k]
                    # self.patch_embed.proj.weight shape: [embed_dim, 3+num_tokens, k, k]
                    # 这里需要将 Prompt 部分 (3:) 保持初始化值，只填充前3通道
                    # 注意：load_checkpoint(strict=False) 已经自动处理了部分加载，
                    # 但对于 shape 变化的层，需要手动赋值覆盖
                    current_weight = self.patch_embed.proj.weight.data
                    current_weight[:, :3, :, :].copy_(conv_weight)
                    if logger:
                        logger.info("Loaded 'patch_embed.proj.weight' for 'below' prompt mode manually.")

        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

        # for name, param in self.named_parameters():
        #     if 'prompt_proj' in name or 'prompt_dropout' in name:
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_vpt, self).train(mode)
        self._freeze_stages()

    # def train(self, mode=True):
    #     # set train status for this class: disable all but the prompt-related modules
    #     if mode:
    #         # training:
    #         # first set all to eval and set the prompt to train later
    #         for module in self.children():
    #             module.train(False)
    #         self.prompt_proj.train()
    #         self.prompt_dropout.train()
    #     else:
    #         # eval:
    #         for module in self.children():
    #             module.train(mode)


    def forward(self, x):
        x, Wh, Ww = self.incorporate_prompt(x)

        if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:
            outs = []
            # 补充i的定义：按层数遍历，同时关联layer和deep_prompt_embd
            for i, (layer, deep_prompt_embd) in enumerate(zip(
                    self.layers, [
                        self.deep_prompt_embeddings_0,
                        self.deep_prompt_embeddings_1,
                        self.deep_prompt_embeddings_2,
                        self.deep_prompt_embeddings_3
                    ]
            )):
                deep_prompt_embd = self.prompt_dropout(deep_prompt_embd)
                x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww, deep_prompt_embd)
                # 现在i已定义，可正常使用
                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    x_out = norm_layer(x_out)


                    '''wxk修改, 在4个特征图输出的时候丢弃VPT'''
                    x_out = x_out[:, self.prompt_config.NUM_TOKENS:, :]
                    out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                    outs.append(out)
        else:
            outs = []
            for i in range(self.num_layers):
                layer = self.layers[i]
                x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
                # x_out, H, W, x, Wh, Ww = layer(x)  # 注释行保留

                if i in self.out_indices:
                    norm_layer = getattr(self, f'norm{i}')
                    x_out = norm_layer(x_out)

                    '''wxk修改, 在4个特征图输出的时候丢弃VPT'''
                    x_out = x_out[:, self.prompt_config.NUM_TOKENS:, :]
                    out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                    outs.append(out)

        return tuple(outs)

    # def load_state_dict(self, state_dict, strict):
    #     if self.prompt_config.LOCATION == "below":
    #         # modify state_dict first   [768, 4, 16, 16]
    #         conv_weight = state_dict["patch_embed.proj.weight"]
    #         conv_weight = torch.cat(
    #             (conv_weight, self.patch_embed.proj.weight[:, 3:, :, :]),
    #             dim=1
    #         )
    #         state_dict["patch_embed.proj.weight"] = conv_weight
    #
    #     super(PromptedSwinTransformer, self).load_state_dict(state_dict, strict)


class PromptedPatchMerging(PatchMerging):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self, num_prompts, prompt_location, deep_prompt, input_resolution,
        dim, norm_layer=nn.LayerNorm
    ):
        super(PromptedPatchMerging, self).__init__(
            input_resolution, dim, norm_layer)
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if prompt_location == "prepend":
            if not deep_prompt:
                self.prompt_upsampling = None
                # self.prompt_upsampling = nn.Linear(dim, 4 * dim, bias=False)
            else:
                self.prompt_upsampling = None

    def upsample_prompt(self, prompt_emb):
        if self.prompt_upsampling is not None:
            prompt_emb = self.prompt_upsampling(prompt_emb)
        else:
            prompt_emb = torch.cat(
                (prompt_emb, prompt_emb, prompt_emb, prompt_emb), dim=-1)
        return prompt_emb

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = x[:, :self.num_prompts, :]
            x = x[:, self.num_prompts:, :]
            L = L - self.num_prompts
            prompt_emb = self.upsample_prompt(prompt_emb)

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H*W, L)
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # add the prompt back:
        if self.prompt_location == "prepend":
            x = torch.cat((prompt_emb, x), dim=1)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PromptedSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
        self, num_prompts, prompt_location, dim, input_resolution,
        num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True,
        qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super(PromptedSwinTransformerBlock, self).__init__(
            dim, input_resolution, num_heads, window_size,
            shift_size, mlp_ratio, qkv_bias, qk_scale, drop,
            attn_drop, drop_path, act_layer, norm_layer)
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if self.prompt_location == "prepend":
            self.attn = PromptedWindowAttention(
                num_prompts, prompt_location,
                dim, window_size=to_2tuple(self.window_size),
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x, mask_matrix):
        # H, W = self.input_resolution

        H, W = self.H, self.W
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)

        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = x[:, :self.num_prompts, :]
            x = x[:, self.num_prompts:, :]
            L = L - self.num_prompts

        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H*W, L)

        x = x.view(B, H, W, C)


        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows --> nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C

        # add back the prompt for attn for parralel-based prompts
        # nW*B, num_prompts + window_size*window_size, C
        num_windows = int(x_windows.shape[0] / B)
        if self.prompt_location == "prepend":
            # expand prompts_embs
            # B, num_prompts, C --> nW*B, num_prompts, C
            prompt_emb = prompt_emb.unsqueeze(0)
            prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
            prompt_emb = prompt_emb.reshape((-1, self.num_prompts, C))
            x_windows = torch.cat((prompt_emb, x_windows), dim=1)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        # seperate prompt embs --> nW*B, num_prompts, C
        if self.prompt_location == "prepend":
            # change input size
            prompt_emb = attn_windows[:, :self.num_prompts, :]
            attn_windows = attn_windows[:, self.num_prompts:, :]
            # change prompt_embs's shape:
            # nW*B, num_prompts, C - B, num_prompts, C
            prompt_emb = prompt_emb.view(-1, B, self.num_prompts, C)
            prompt_emb = prompt_emb.mean(0)

        '''wxk修改'''
        # merge windows
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, Hp, Wp)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        '''wxk修改'''
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # add the prompt back:
        if self.prompt_location == "prepend":
            x = torch.cat((prompt_emb, x), dim=1)
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PromptedWindowAttention(WindowAttention):
    def __init__(
        self, num_prompts, prompt_location, dim, window_size, num_heads,
        qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.
    ):
        super(PromptedWindowAttention, self).__init__(
            dim, window_size, num_heads, qkv_bias, qk_scale,
            attn_drop, proj_drop)
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        # account for prompt nums for relative_position_bias
        # attn: [1920, 6, 649, 649]
        # relative_position_bias: [6, 49, 49])

        if self.prompt_location == "prepend":
            # expand relative_position_bias
            _C, _H, _W = relative_position_bias.shape

            relative_position_bias = torch.cat((
                torch.zeros(_C, self.num_prompts, _W, device=attn.device),
                relative_position_bias
                ), dim=1)
            relative_position_bias = torch.cat((
                torch.zeros(_C, _H + self.num_prompts, self.num_prompts, device=attn.device),
                relative_position_bias
                ), dim=-1)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # incorporate prompt
            # mask: (nW, 49, 49) --> (nW, 49 + n_prompts, 49 + n_prompts)
            nW = mask.shape[0]
            if self.prompt_location == "prepend":
                # expand relative_position_bias
                mask = torch.cat((
                    torch.zeros(nW, self.num_prompts, _W, device=attn.device),
                    mask), dim=1)
                mask = torch.cat((
                    torch.zeros(
                        nW, _H + self.num_prompts, self.num_prompts,
                        device=attn.device),
                    mask), dim=-1)
            # logger.info("before", attn.shape)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # logger.info("after", attn.shape)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
