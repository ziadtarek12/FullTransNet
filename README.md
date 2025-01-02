# FullTransNet: Full Transformer with Local-Global Attention for Video Summarization


## Getting Started

This project is developed on Ubuntu 22.04.1 with CUDA 11.7


```sh
git clone https://github.com/ChiangLu/FullTransNet.git
```


Install python dependencies.

```sh
pip install -r requirements.txt
```

## Datasets and pretraining models Preparation

Download the pre-processed datasets into `datasets/` folder. 

including [TVSum](https://github.com/yalesong/tvsum), [SumMe](https://gyglim.github.io/me/vsum/index.html), [OVP](https://sites.google.com/site/vsummsite/download), and [YouTube](https://sites.google.com/site/vsummsite/download) datasets.


+ (Google Cloud) Link: [datest](https://drive.google.com/file/d/11QRYfmBRVxhVS78AYHEq6ZxAmFb0if5K/view?usp=drive_link)



The datasets and model save structure should look like

```
FullTransFormer
└── datasets/
    ├── eccv16_dataset_ovp_google_pool5.h5
    ├── eccv16_dataset_summe_google_pool5.h5
    ├── eccv16_dataset_tvsum_google_pool5.h5
    ├── eccv16_dataset_youtube_google_pool5.h5
    └── readme.txt
└── model_save/
    ├── summe
    └── tvsum
```


## Training

To train  model on SumMe datasets with canonical settings, run

```sh
python train.py --model-dir ./model_save/summe --splits ./splits/summe.yml

```

## Evaluation

To evaluate  models, run 

```sh
sh evaluate.sh 
```


## Model Zoo
(Google Cloud) Link: [moedel_zoo](https://drive.google.com/file/d/1pnS2ZVi3jTVjIF4hhfdKv99tlrZVDPLe/view?usp=drive_link)



## Acknowledgments

We gratefully thank the below open-source repo, which greatly boost our research.
+ Thank Part of the code is referenced from:
+ Thank [DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) for the pre-processed public datasets.
  + Thank [VASNet](https://github.com/ok1zjf/VASNet) for the training and evaluation pipeline.
  + Thank [Attention is all you ](https://github.com/huggingface/transformers) for the Transformer framework and sparse attention.


## Citation

```
@article{2024fulltransnet
  title={FullTransNet: Full Transformer with Local-Global Attention for Video Summarization},
  author={Libin Lan, Lu Jiang, Tianshu Yu , Xiaojuan Liu , and Zhongshi He},
  year= {2024},
}

```
