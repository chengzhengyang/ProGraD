# Prompt-Guided Relational Reasoning for Social Behavior Understanding with Vision Foundation Models

This repository contains the PyTorch implementation for [[https://arxiv.org/abs/2508.07996](https://arxiv.org/abs/2508.07996)]. 

<p align="center">
  <img src="figures/teaser.png" alt="teaser" style="max-width:100%;">
</p>

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
