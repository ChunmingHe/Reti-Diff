# Reti-Diff: Illumination Degradation Image Restoration with Retinex-based Latent Diffusion Model

[Chunming He](https://chunminghe.github.io/), [Chenyu Fang](https://chengyufang.tech/), [Yulun Zhang](https://yulunzhang.com), [Kai Li](https://kailigo.github.io), Longxiang Tang, [Chengyu You](http://chenyuyou.me), Fengyang Xiao, Zhenhua Guo and Xiu Li, "Reti-Diff: Illumination Degradation Image Restoration with Retinex-based Latent Diffusion Model", arXiv, 2023 [[arXiv]](https://arxiv.org/abs/2311.11638)

#### 🔥🔥🔥 News

- **2023-11-21:** We release this repository.

> **Abstract:** Illumination degradation image restoration (IDIR) techniques aim to improve the visibility of degraded images and mitigate the adverse effects of deteriorated illumination. Among these algorithms, diffusion model (DM)-based methods have shown promising performance but are often burdened by heavy computational demands and pixel misalignment issues when predicting the image-level distribution. To tackle these problems, we propose to leverage DM within a compact latent space to generate concise guidance priors and introduce a novel solution called Reti-Diff for the IDIR task. Reti-Diff comprises two key components: the Retinex-based latent DM (RLDM) and the Retinex-guided transformer (RGformer). To ensure detailed reconstruction and illumination correction, RLDM is empowered to acquire Retinex knowledge and extract reflectance and illumination priors. These priors are subsequently utilized by RGformer to guide the decomposition of image features into their respective reflectance and illumination components. Following this, RGformer further enhances and consolidates the decomposed features, resulting in the production of refined images with consistent content and robustness to handle complex degradation scenarios. Extensive experiments show that Reti-Diff outperforms existing methods on three IDIR tasks, as well as downstream applications.

![](figs/framework.jpeg)



## 🔧 Todo

- [ ] Complete this repository



## 🔗 Contents

- [ ] Datasets
- [ ] Training
- [ ] Testing
- [x] [Results](https://github.com/ChunmingHe/Reti-Diff/blob/main/README.md#Results)
- [x] [Citation](https://github.com/ChunmingHe/Reti-Diff/blob/main/README.md#Citation)
- [x] [Acknowledgements](https://github.com/ChunmingHe/Reti-Diff/blob/main/README.md#Acknowledgements)



## 🔍 Results

We achieved state-of-the-art performance on *low light image enhancement*, *underwater image enhancement*, *backlit image enhancement* and corresponding downstream tasks. More results can be found in the paper.

<details>
<summary>Quantitative Comparison (click to expan)</summary>

- Results in Table 1 of the main paper
  <p align="center">
  <img width="900" src="figs/table-1.png">
	</p>
- Results in Table 2-3 of the main paper
  <p align="center">
  <img width="900" src="figs/table-2-3.png">
	</p>
- Results in Table 6-9 of the main paper
  <p align="center">
  <img width="900" src="figs/table-6-7-8-9.png">
	</p>
  </details>

<details>
<summary>Visual Comparison (click to expan)</summary>

- Results in Figure 3 of the main paper
  <p align="center">
  <img width="900" src="figs/llie.jpeg">
	</p>
- Results in Figure 4 of the main paper
  <p align="center">
  <img width="900" src="figs/uie.jpeg">
	</p>
- Results in Figure 5 of the main paper
  <p align="center">
  <img width="900" src="figs/backlit.jpeg">
	</p>
  </details>


## 📎 Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).

```
@article{he2023retidiff,
      title={Reti-Diff: Illumination Degradation Image Restoration with Retinex-based Latent Diffusion Model}, 
      author={Chunming He and Chengyu Fang and Yulun Zhang and Kai Li and Longxiang Tang and Chenyu You and Fengyang Xiao and Zhenhua Guo and Xiu Li},
      year={2023},
      eprint={2311.11638},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 💡 Acknowledgements
The codes are based on [BasicSR](https://github.com/XPixelGroup/BasicSR), [Restormer](https://github.com/swz30/Restormer), and [DiffIR](https://github.com/Zj-BinXia/DiffIR). Please also follow their licenses. Thanks for their awesome works.


