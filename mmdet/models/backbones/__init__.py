from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .hourglass import HourglassNet
from .hrnet import HRNet
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .trident_resnet import TridentResNet

from .swin_transformer import SwinTransformer
from .swin_transformer_fixed import SwinTransformer_fixed
from .swin_transformer_partial_1 import SwinTransformer_partial_1
from .swin_transformer_norm_tuning import SwinTransformer_norm_tuning
from .swin_transformer_bitfit import SwinTransformer_bitfit
from .swin_transformer_lora import SwinTransformer_lora
from .swin_transformer_adaptformer import SwinTransformer_adaptformer
from .swin_transformer_mona import SwinTransformer_mona
from .swin_transformer_adapter import SwinTransformer_adapter
from .swin_transformer_lora_imagenet import swin_transformer_lora_imagenet
from .swin_transformer_learnAdaIN import SwinTransformer_LearnableAdaIN
from .swin_transformer_learnAdaIN_adapter import SwinTransformer_LearnableAdaIN_adapter
from .swin_transformer_learnAdaIN_lora import SwinTransformer_LearnableAdaIN_lora
from .swin_transformer_learnAdaIN_partial import SwinTransformer_LearnableAdaIN_partial
from .swin_transformer_learnAdaIN_mona import SwinTransformer_LearnableAdaIN_mona
from .swin_transformer_learnAdaIN_LN import SwinTransformer_LearnableAdaIN_norm_tuning
from .swin_transformer_learnAdaIN_lora_all_blocks import SwinTransformer_LearnableAdaIN_lora_all_blocks

# copy 3 channel and use imagenet z-score
from .swin_transformer_norm_tuning_imagenet import SwinTransformer_norm_tuning_imagenet
from .swin_transformer_adapter_imagenet import SwinTransformer_adapter_imagenet
from .swin_transformer_imagenet import SwinTransformer_imagenet

# add adaptive_channel_head
from .swin_transformer_lora_style_way3 import Swin_transformer_lora_style_way3
# from .swin_transformer_norm_tuning_way3 import SwinTransformer_norm_tuning_way3
# from .swin_transformer_adapter_style_way3 import SwinTransformer_adapter_style_way3
from .swin_transformer_mona_way3 import SwinTransformer_mona_way3
from .swin_transformer_fixed_way3 import SwinTransformer_fixed_way3

# active patch_embed
from .swin_transformer_patchembed import SwinTransformer_patchembed
from .swin_transformer_adapter_patchembed import SwinTransformer_adapter_patchembed
from .swin_transformer_lora_patchembed import SwinTransformer_lora_patchembed
from .swin_transformer_norm_tuning_patchembed import SwinTransformer_norm_tuning_patchembed

from .swin_transformer_adapter_LA_head import SwinTransformer_adapter_LA_head
from .swin_transformer_norm_tuning_LA_head import SwinTransformer_norm_tuning_LA_head

from .swin_transformer_ssf import SwinTransformer_ssf
from .swin_transformer_ssf_no_patch_embed import SwinTransformer_ssf_no_patch_embed

from .vpt.swin_transformer_vpt import SwinTransformer_vpt



from .swin_transformer_tf_mona_attention import SwinTransformer_tf_mona_attention
from .swin_transformer_tf_mona_mlp import SwinTransformer_tf_mona_mlp

from .swin_transformer_tf_mona_mlp_3x3 import SwinTransformer_tf_mona_mlp_3x3
from .swin_transformer_tf_mona_mlp_5x5 import SwinTransformer_tf_mona_mlp_5x5
from .swin_transformer_tf_mona_mlp_9x9 import SwinTransformer_tf_mona_mlp_9x9
from .swin_transformer_tf_mona_mlp_11x11 import SwinTransformer_tf_mona_mlp_11x11
from .swin_transformer_tf_mona_mlp_13x13 import SwinTransformer_tf_mona_mlp_13x13
from .swin_transformer_tf_mona_mlp_15x15 import SwinTransformer_tf_mona_mlp_15x15
from .swin_transformer_tf_mona_mlp_17x17 import SwinTransformer_tf_mona_mlp_17x17


from .swin_transformer_tf_mona_19x19 import SwinTransformer_tf_mona_19x19
from .swin_transformer_tf_mona_attention_19x19 import SwinTransformer_tf_mona_attention_19x19
from .swin_transformer_tf_mona_mlp_19x19 import SwinTransformer_tf_mona_mlp_19x19


from .swin_transformer_tf_mona_ablation_dim_32 import SwinTransformer_tf_mona_dim_32
from .swin_transformer_tf_mona_ablation_dim_128 import SwinTransformer_tf_mona_dim_128
from .swin_transformer_tf_mona_ablation_w_o_freq import SwinTransformer_tf_mona_w_o_use_freq
from .swin_transformer_tf_mona_ablation_w_o_time import SwinTransformer_tf_mona_w_o_use_time
from .swin_transformer_tf_mona_ablation_w_o_gate import SwinTransformer_tf_mona_w_o_use_gate
from .swin_transformer_tf_mona_ablation_w_o_scale import SwinTransformer_tf_mona_w_o_use_scale

from .swin_transformer_tf_mona_attention_19x19_w_o_scale import SwinTransformer_tf_mona_attention_19x19_w_o_scale
from .swin_transformer_tf_mona_mlp_19x19_w_o_scale import SwinTransformer_tf_mona_mlp_19x19_w_o_scale

from .swin_transformer_tf_mona_ablation_dim_32_w_o_scale import SwinTransformer_tf_mona_dim_32_w_o_scale
from .swin_transformer_tf_mona_ablation_dim_128_w_o_scale import SwinTransformer_tf_mona_dim_128_w_o_scale
from .swin_transformer_tf_mona_ablation_w_o_freq_w_o_scale import SwinTransformer_tf_mona_w_o_use_freq_w_o_scale
from .swin_transformer_tf_mona_ablation_w_o_gate_w_o_scale import SwinTransformer_tf_mona_w_o_use_gate_w_o_scale
from .swin_transformer_tf_mona_ablation_w_o_time_w_o_scale import SwinTransformer_tf_mona_w_o_use_time_w_o_scale

from .swin_transformer_tf_mona_ablation_w_o_scale_11x11 import SwinTransformer_tf_mona_w_o_use_scale_11x11
from .swin_transformer_tf_mona_ablation_w_o_scale_3x3 import SwinTransformer_tf_mona_w_o_use_scale_3x3
from .swin_transformer_tf_mona_ablation_w_o_scale_15x15 import SwinTransformer_tf_mona_w_o_use_scale_15x15
from .swin_transformer_tf_mona_ablation_w_o_scale_23x23 import SwinTransformer_tf_mona_w_o_use_scale_23x23

__all__ = [
    "RegNet",
    "ResNet",
    "ResNetV1d",
    "ResNeXt",
    "SSDVGG",
    "HRNet",
    "Res2Net",
    "HourglassNet",
    "DetectoRS_ResNet",
    "DetectoRS_ResNeXt",
    "Darknet",
    "ResNeSt",
    "TridentResNet",
    "SwinTransformer",
    "SwinTransformer_fixed",
    "SwinTransformer_partial_1",
    "SwinTransformer_norm_tuning",
    "SwinTransformer_bitfit",
    "SwinTransformer_lora",
    "SwinTransformer_adaptformer",
    "SwinTransformer_mona",
    "SwinTransformer_adapter",


    'SwinTransformer_imagenet',
    "swin_transformer_lora_imagenet",  # try-1
    'swin_transformer_norm_tuning_imagenet',
    "swin_transformer_adapter_domain",  #

    'SwinTransformer_LearnableAdaIN',     # try-2
    'SwinTransformer_LearnableAdaIN_adapter',
    'SwinTransformer_LearnableAdaIN_lora',
    'SwinTransformer_LearnableAdaIN_partial',
    'SwinTransformer_LearnableAdaIN_mona',
    'SwinTransformer_LearnableAdaIN_norm_tuning',
    'SwinTransformer_LearnableAdaIN_lora_all_blocks',
    'SwinTransformer_adapter_imagenet',

    'Swin_transformer_lora_style_way3',
    # 'SwinTransformer_norm_tuning_way3',
    # 'SwinTransformer_adapter_style_way3',
    'SwinTransformer_mona_way3',
    'SwinTransformer_fixed_way3',


    'SwinTransformer_patchembed',
    'SwinTransformer_adapter_patchembed',
    'SwinTransformer_lora_patchembed',
    'SwinTransformer_norm_tuning_patchembed',


    'SwinTransformer_adapter_LA_head',
    'SwinTransformer_norm_tuning_LA_head',

    'SwinTransformer_ssf',
    'SwinTransformer_ssf_no_patch_embed',

    'SwinTransformer_vpt',

    'SwinTransformer_mufa_serial',
    'SwinTransformer_mufa_parallel',
    'SwinTransformer_smt', 
    'SwinTransformer_mufa_smt',

    'SwinTransformer_mufa_serial_scale_sptial_81632',
    'SwinTransformer_mufa_serial_scale_sptial_163264',

    'SwinTransformer_adapter_freqfit',
    'SwinTransformer_tf_mona',
    'SwinTransformer_adapter_sapt',

    'SwinTransformer_tf_mona_attention',
    'SwinTransformer_tf_mona_mlp',
    'SwinTransformer_tf_mona_mlp_3x3',
    'SwinTransformer_tf_mona_mlp_5x5',
    'SwinTransformer_tf_mona_mlp_9x9',
    'SwinTransformer_tf_mona_mlp_11x11',
    'SwinTransformer_tf_mona_mlp_13x13',
    'SwinTransformer_tf_mona_mlp_15x15',
    'SwinTransformer_tf_mona_mlp_17x17',

    'SwinTransformer_tf_mona_mlp_19x19',
    'SwinTransformer_tf_mona_19x19',
    'SwinTransformer_tf_mona_attention_19x19',

    'SwinTransformer_tf_mona_dim_128',
    'SwinTransformer_tf_mona_dim_32',
    'SwinTransformer_tf_mona_w_o_use_freq',
    'SwinTransformer_tf_mona_w_o_use_time',
    'SwinTransformer_tf_mona_w_o_use_gate',
    'SwinTransformer_tf_mona_w_o_use_scale',

    'SwinTransformer_tf_mona_mlp_19x19_w_o_scale',
    'SwinTransformer_tf_mona_attention_19x19_w_o_scale',


    'SwinTransformer_tf_mona_dim_32_w_o_scale',
    'SwinTransformer_tf_mona_dim_128_w_o_scale',
    'SwinTransformer_tf_mona_w_o_use_time_w_o_scale',
    'SwinTransformer_tf_mona_w_o_use_gate_w_o_scale',
    'SwinTransformer_tf_mona_w_o_use_freq_w_o_scale',

    'SwinTransformer_tf_mona_w_o_use_scale_11x11',
    'SwinTransformer_tf_mona_w_o_use_scale_3x3',
    'SwinTransformer_tf_mona_w_o_use_scale_15x15',
    'SwinTransformer_tf_mona_w_o_use_scale_23x23'


]
