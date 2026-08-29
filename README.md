# DSG-Adapter

[![Paper](https://img.shields.io/badge/Paper-IEEE%20WCL-blue.svg)](https://doi.org/10.1109/LWC.2026.3728104)
[![Dataset](https://img.shields.io/badge/Dataset-BaiduNetdisk-green.svg)](https://pan.baidu.com/s/1qGvRTzVewbVaBqmypSSEjQ)

Official implementation of:

> **DSG-Adapter: Parameter-Efficient Fine-Tuning for Wideband Signal Detection and Recognition**
> Xikang Wang, Hua Xu, Zisen Qi, Qingwei Meng, Yunhao Shi, Wenran Le, and Yu Li
> *IEEE Wireless Communications Letters*, 2026.
> [[Paper](https://doi.org/10.1109/LWC.2026.3728104)]

## Dataset

The dataset used in this work is available at:

* **Baidu Netdisk**: [Download](https://pan.baidu.com/s/1qGvRTzVewbVaBqmypSSEjQ)
* **Extraction Code**: `cg5y`

## Implementation

This repository provides three DSG-Adapter variants corresponding to the adapter insertion strategies introduced in the paper:

* **DSG-Adapter_attention** → **DSG-Adapter†**
* **DSG-Adapter_mlp** → **DSG-Adapter‡**
* **DSG-Adapter** → **DSG-Adapter***

## Training

### DSG-Adapter†

```bash
python tools/train.py \
    DSG_Adapter_configs/torchsig_base_22k_3x/torchsig_retinanet_swin_base_3x_full_rgb_imagenet_DSG-Adapter_attention.py \
    --work-dir exp_DSG-Adapter_attention
```

### DSG-Adapter‡

```bash
python tools/train.py \
    DSG_Adapter_configs/torchsig_base_22k_3x/torchsig_retinanet_swin_base_3x_full_rgb_imagenet_DSG-Adapter_mlp.py \
    --work-dir exp_DSG-Adapter_mlp
```

### DSG-Adapter*

```bash
python tools/train.py \
    DSG_Adapter_configs/torchsig_base_22k_3x/torchsig_retinanet_swin_base_3x_full_rgb_imagenet_DSG-Adapter.py \
    --work-dir exp_DSG-Adapter
```

## Citation

If you find this work useful, please consider citing:

```bibtex
@ARTICLE{wang2026dsgadapter,
  author={Wang, Xikang and Xu, Hua and Qi, Zisen and Meng, Qingwei and Shi, Yunhao and Le, Wenran and Li, Yu},
  journal={IEEE Wireless Communications Letters},
  title={DSG-Adapter: Parameter-Efficient Fine-Tuning for Wideband Signal Detection and Recognition},
  year={2026},
  doi={10.1109/LWC.2026.3728104}
}
```

## Acknowledgement

This codebase is developed based on [Mona](https://github.com/Leiyi-Hu/mona/tree/master). We sincerely thank the authors for their excellent open-source work.
