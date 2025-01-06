# Realistic Test-Time Adaptation of Vision-Language Models (StatA)
The official implementation of [*Realistic Test-Time Adaptation of Vision-Language Models*]().

Authors:
[Maxime Zanella](https://scholar.google.com/citations?user=FIoE9YIAAAAJ&hl=fr&oi=ao),
[Clément Fuchs](https://scholar.google.com/citations?user=ZXWUJ4QAAAAJ&hl=fr&oi=ao),
[Christophe De Vleeschouwer](https://scholar.google.ca/citations?user=xb3Zc3cAAAAJ&hl=en),
[Ismail Ben Ayed](https://scholar.google.com/citations?user=29vyUccAAAAJ&hl=fr&oi=ao).

We present StatA, a versatile unsupervised transductive method that could handle a wide range of deployment scenarios, including those with a variable number of effective classes at test time. Our approach incorporates a novel regularization term designe specifically for VLMs, which acts as a statistical anchor preserving the initial text-encoder knowledge, particularly in low-data regimes.
The experiments of this paper are divided in two main parts: (1) Batch Adaptation, where test-time adaptation methods are applied on each batch independently and (2) Online Adaptation, where test-time adaptation methods are applied on streams of batch with the possibility of keeping information from one batch to the next one.

## Table of Contents

1. [Installation](#installation) 
2. [Usage](#usage)
3. [Batch adaptation](#batch-adaptation)
4. [Online adaptation](#online-adaptation)
5. [Citation](#citation)
6. [Contact](#contact) 


---

## Installation
This repository requires to install an environment and datasets:
### Environment
Create a Python environment with your favorite environment manager. For example, with `conda`: 
```bash
conda create -y --name my_env python=3.10.0
conda activate my_env
pip3 install -r requirements.txt
```
And install Pytorch according to your configuration:
```bash
pip3 install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
```
### Datasets
Please follow [DATASETS.md](DATASETS.md) to install the datasets.
You will get a structure with the following dataset names:
```
$DATA/
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– stanford_cars/
|–– oxford_flowers/
|–– food-101/
|–– fgvc_aircraft/
|–– sun397/
|–– dtd/
|–– eurosat/
|–– ucf101/
|–– imagenetv2/
|–– imagenet-sketch/
|–– imagenet-adversarial/
|–– imagenet-rendition/
```

## Batch Adaptation
We present the basic usage to get started with our method. You have to pass the datasets folder path.

Here is an example for the imagenet dataset, with the CLIP-ViT-B/16 architecture, a batch size of 64, a variable number of effective classes between 1 and 4. This experiments is run 1000 times.
```bash
python3 main.py --root_path /path/to/datasets/folder --dataset imagenet --method StatA --backbone vit_b16 --batch_size 64 --num_class_eff_min 1 --num_class_eff_max 4 --n_tasks 1000
```

To run the whole experiment of Table 1 in the paper, use the following command:
```bash
bash ./scripts/stata_batch.sh /path/to/datasets/folder vit_b16
```

## Online Adaptation
We present the basic usage to get started with our method. You have to pass the datasets folder path.

Here is an example for the imagenet dataset, with the CLIP-ViT-B/16 architecture, a batch size of 128, a stream correlation factor gamma of 0.1. This experiments is run 100 times.
```bash
python3 main.py --root_path /path/to/datasets/folder --dataset imagenet --method StatA --backbone vit_b16 --batch_size 64 --online --gamma 0.1 --n_tasks 100
```

To run the whole experiment of Table 1 in the paper, use the following command:
```bash
bash ./scripts/stata_online.sh /path/to/datasets/folder vit_b16
```


## Citation

If you find this repository useful, please consider citing our paper:
```

```

You can also cite the TransCLIP paper on which this work is based on:
```
@article{zanella2024boosting,
  title={Boosting Vision-Language Models with Transduction},
  author={Zanella, Maxime and G{\'e}rin, Beno{\^\i}t and Ayed, Ismail Ben},
  journal={arXiv preprint arXiv:2406.01837},
  year={2024}
}
```

## Contact

For any inquiries, please contact us at [maxime.zanella@uclouvain.be](mailto:maxime.zanella@uclouvain.be) and [clement.fuchs@uclouvain.be](mailto:clement.fuchs@uclouvain.be) or feel free to [create an issue](https://github.com/MaxZanella/StatA/issues).


## License
[AGPL-3.0](https://github.com/MaxZanella/StatA/blob/main/LICENSE)
