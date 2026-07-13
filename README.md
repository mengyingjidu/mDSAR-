# mDSAR-
Zero-Shot SAR Image Restoration via Dual Statistical Adaptive Regularization
## 预训练模型

本项目采用 ImageNet 256×256 无条件扩散模型作为预训练先验。

请点击[下载预训练模型](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)，并将下载后的模型放入：

```text
exp/logs/imagenet/
```

## 数据集

本项目在 SSDD 和 HRSID 两个 SAR 舰船数据集上进行实验。请从以下链接下载原始数据集，并按照实验设置完成图像预处理。

### SSDD

请从 [SSDD 官方仓库](https://github.com/TianwenZhang0825/Official-SSDD) 下载数据集。

将预处理后的测试图像放入：

```text
SSDD_100/
```

### HRSID

请从 [HRSID 官方仓库](https://github.com/chaozhong2010/HRSID) 下载数据集。

将预处理后的测试图像放入：

```text
HRSID_100/
```

本项目实验中将输入图像统一处理为 256 × 256，并分别选取 100 幅图像进行测试。
