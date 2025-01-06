# Realistic Test-Time Adaptation of Vision-Language Models (StatA)
The official implementation of [*Realistic Test-Time Adaptation of Vision-Language Models*]().

Authors:
[Maxime Zanella](https://scholar.google.com/citations?user=FIoE9YIAAAAJ&hl=fr&oi=ao),
[Clément Fuchs](https://scholar.google.com/citations?user=ZXWUJ4QAAAAJ&hl=fr&oi=ao),
[Christophe De Vleeschouwer](https://scholar.google.ca/citations?user=xb3Zc3cAAAAJ&hl=en),
[Ismail Ben Ayed](https://scholar.google.com/citations?user=29vyUccAAAAJ&hl=fr&oi=ao).

We introduce **StatA**, a robust and versatile unsupervised transductive method designed to handle diverse deployment scenarios, including those involving a variable number of effective classes during testing. Our approach features a novel anchor-based regularization term specifically crafted for Vision-Language Models (VLMs). This term serves as a statistical anchor, preserving the initial knowledge of the text encoder, especially in low-data settings.

The experiments presented in this paper are organized into two main categories:  

1. **Batch Adaptation**: Test-time adaptation methods are applied independently to each batch with a varying number of effective classes.  

   <div align="center" style="margin-top:20px; margin-bottom:20px;">
      <img src="summary_batch.png" alt="Batch Adaptation" width="500">
      <p style="font-size:90%;"><em>Figure 1: StatA brings consistent improvement when facing Low (between 2 and 10), Medium (between 5 and 25) number of effective classes (Keff) in each batch, or All classes. In comparison, other transductive methods engender significant performance drops in at least one scenario.</em></p>
   </div>

2. **Online Adaptation**: Test-time adaptation methods are applied to a continuous stream of batches with varying correlation in the appearance of each class.  

   <div align="center" style="margin-top:20px; margin-bottom:20px;">
      <img src="summary_online.png" alt="Online Adaptation" width="500">
      <p style="font-size:90%;"><em>Figure 2: StatA shows strong performance when applied on streams of data, with Low or High correlation between batches, and when all the classes are appearing sequentially (Separate).</em></p>
   </div>







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
