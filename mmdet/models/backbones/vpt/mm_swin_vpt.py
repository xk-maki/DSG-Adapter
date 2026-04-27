import torch
import torch.nn as nn
from torch.nn import Dropout
from swin_transformer_vpt import PromptedSwinTransformer
from collections import namedtuple
import math
from functools import reduce
from operator import mul
import torchvision.transforms as tv


# 辅助函数：代码库中用到的to_2tuple
def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def prompted_swin_transformer_output_shape():
    # 1. 设置设备：优先使用GPU，无则使用CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    PromptConfig = namedtuple('PromptConfig', [
        'LOCATION',  # prompt位置
        'NUM_TOKENS',  # prompt数量
        'DROPOUT',  # prompt dropout率
        'PROJECT',  # prompt投影维度（-1表示不投影）
        'INITIATION',  # 初始化方式 (注意：请确认底层代码是读 INITIATION 还是 INITIATION)
        'DEEP',  # 是否使用deep prompt
        'NUM_DEEP_LAYERS',  # 【新增】如果设为整数，则进行 partial-deep prompt tuning
        'FORWARD_DEEP_NOEXPAND',  # 【新增】Deep Prompt 前向传播优化
        # 以下为官方配置中默认为 False 或 None 的参数，可视情况添加
        'DEEP_SHARED',  # 是否共享 deep prompt 权重
        'REVERSE_DEEP',  # 是否只更新最后 n 层
    ])

    prompt_config = PromptConfig(
        LOCATION="prepend",  # 对应 _C.MODEL.PROMPT.LOCATION
        NUM_TOKENS=10,  # 对应 _C.MODEL.PROMPT.NUM_TOKENS
        DROPOUT=0.1,  # 对应 _C.MODEL.PROMPT.DROPOUT
        PROJECT=-1,  # 对应 _C.MODEL.PROMPT.PROJECT
        INITIATION="random",  # 对应 _C.MODEL.PROMPT.INITIATION
        DEEP=False,  # 对应 _C.MODEL.PROMPT.DEEP
        NUM_DEEP_LAYERS=None,  # 【新增】对应 _C.MODEL.PROMPT.NUM_DEEP_LAYERS
        FORWARD_DEEP_NOEXPAND=False,  # 【新增】对应 _C.MODEL.PROMPT.FORWARD_DEEP_NOEXPAND
        DEEP_SHARED=False,  # 【新增】
        REVERSE_DEEP=False  # 【新增】
    )
    # 3. 构建PromptedSwinTransformer模型（严格匹配代码库参数）
    model = PromptedSwinTransformer(
        prompt_config=prompt_config,  # 必传的prompt配置
        img_size=384,                 # 代码库默认img_size=224
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,           # 代码库默认drop_path_rate=0.1
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        out_indices=(0, 1, 2, 3),     # wxk新增参数
        frozen_stages=-1
    )
    model = model.to(device)
    model.eval()  # 设置为推理模式

    # 4. 构造测试输入（注意：若LOCATION=pad，img_size会被自动扩充）
    batch_size = 2
    # 原始输入尺寸（pad模式下模型内部会自动加2*NUM_TOKENS）
    test_input = torch.randn(batch_size, 3, 384, 384).to(device)
    print(f"\n输入shape: {test_input.shape}")

    # 5. 前向推理（禁用梯度计算加速）
    with torch.no_grad():
        outputs = model(test_input)

    # 6. 打印各Stage输出的shape
    print("\n各Stage输出shape:")
    for idx, out in enumerate(outputs):
        print(f"Stage {idx}: {out.shape}")

    # 7. 验证输出数量（匹配out_indices长度）
    assert len(outputs) == len(model.out_indices), \
        f"输出Stage数量错误！期望{len(model.out_indices)}个，实际{len(outputs)}个"
    print("\n✅ PromptedSwinTransformer输出验证通过！")


if __name__ == "__main__":
    # 执行测试（已补齐所有必传参数）
    prompted_swin_transformer_output_shape()