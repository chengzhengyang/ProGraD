# Prompt-Guided Relational Reasoning for Social Behavior Understanding

This repository contains the PyTorch implementation for **"Prompt-Guided Relational Reasoning for Social Behavior Understanding with Vision Foundation Models"** [[Paper](https://arxiv.org/abs/2508.07996)]. 

---
<p align="center">
  <img src="figures/teaser.png" alt="teaser" style="max-width:100%;">
</p>

> **Abstract:** Group Activity Detection (GAD) involves recognizing social groups and their collective behaviors in videos. Vision Foundation Models (VFMs), like DinoV2, offer excellent features, but are pretrained primarily on object-centric data and remain underexplored for modeling group dynamics. While they are a promising alternative to highly task-specific GAD architectures that require full fine-tuning, our initial investigation reveals that simply swapping CNN backbones used in these methods with VFMs brings little gain, underscoring the need for structured, group-aware reasoning on top.
We introduce Prompt-driven Group Activity Detection (ProGraD) -- a method that bridges this gap through 1) learnable group prompts to guide the VFM attention toward social configurations, and 2) a lightweight two-layer GroupContext Transformer that infers actor-group associations and collective behavior. We evaluate our approach on two recent GAD benchmarks: Café, which features multiple concurrent social groups, and Social-CAD, which focuses on single-group interactions. While we surpass state-of-the-art in both settings, our method is especially effective in complex multi-group scenarios, where we yield a gain of 6.5\% (Group mAP\@1.0) and 8.2\% (Group mAP\@0.5) using only 10M trainable parameters. Furthermore, our experiments reveal that ProGraD produces interpretable attention maps, offering insights into actor-group reasoning.


---

## Environment setup

This repository targets: 
* **Python:** `3.11.7`
* **CUDA:** `12.4`
* **PyTorch:** `2.6.0`

See `env_setup.sh`

## Experiments
### Dataset preperation

Download the data from the official links below.

* **Café** — [https://dk-kim.github.io/CAFE/](https://dk-kim.github.io/CAFE/)

* **Social-CAD**

  * Original CAD dataset: [https://cvgl.stanford.edu/projects/collective/collectiveActivity.html](https://cvgl.stanford.edu/projects/collective/collectiveActivity.html)
  * Social group formation and activity annotations: [https://github.com/mahsaep/Social-human-activity-understanding-and-grouping](https://github.com/mahsaep/Social-human-activity-understanding-and-grouping)



Place the datasets under your project `datasets/` directory with the following layout:

```
datasets/
├─ cafe/
│  ├─ gt_tracks.pkl 
│  ├─ 1/
│  ├─ 2/
│  └─ ...
└─ social_cad/
   ├─ annotations/      # clone the annotations repo here
   ├─ seq01/
   ├─ seq02/
   └─ ...
```

### Training
* **Café**

```bash
sh train_cafe.sh
```
* **Social-CAD**

```bash
sh train_cad.sh
```

## Citation
If you find this work useful in your own research, please consider citing it: 
```bibtex
@inproceedings{ponbagavathi2025promptguidedrelationalreasoningsocial,
      title={Prompt-Guided Relational Reasoning for Social Behavior Understanding with Vision Foundation Models}, 
      author={Thinesh Thiyakesan Ponbagavathi and Chengzheng Yang and Alina Roitberg},
      year={2025},
}
```
