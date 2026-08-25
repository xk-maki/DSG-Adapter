# DSG-Adapter
[![Dataset](https://img.shields.io/badge/Dataset-BaiduNetdisk-green.svg)](https://pan.baidu.com/s/1qGvRTzVewbVaBqmypSSEjQ)

This repository contains the official implementation for the paper:

> **DSG-Adapter: Parameter-Efficient Fine-Tuning for Wideband Signal Detection and Recognition**, *IEEE Wireless Communications Letters*, **To Appear**.

- **Paper Status**: To be published in IEEE Wireless Communications Letters
- **Dataset Download**: [Baidu Netdisk Link](https://pan.baidu.com/s/1qGvRTzVewbVaBqmypSSEjQ) | Extraction Code: `cg5y`

## Implementation Details

The implementation of DSG-Adapter is developed based on the parameter-efficient fine-tuning framework **Mona**:

> Mona: https://github.com/Leiyi-Hu/mona/tree/master

We sincerely thank the authors of Mona for their open-source contribution.

The repository contains three DSG-Adapter configurations corresponding to different adapter insertion strategies described in the paper:

- **DSG-Adapter_attention**: corresponds to **DSG-Adapter†** in the paper.
- **DSG-Adapter_mlp**: corresponds to **DSG-Adapter‡** in the paper.
- **DSG-Adapter**: corresponds to **DSG-Adapter*** in the paper.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@ARTICLE{wcl2026dsgadapter,
  author={Wang, Xikang and Xu, Hua and Qi, Zisen and Meng, Qingwei and Shi, Yunhao and Le, Wenran and Li, Yu},
  journal={IEEE Wireless Communications Letters},
  title={DSG-Adapter: Parameter-Efficient Fine-Tuning for Wideband Signal Detection and Recognition},
  year={2026},
}