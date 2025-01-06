from PIL import Image
import numpy as np
import os

from collections import defaultdict
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm


def _add_channels(img,):
  while len(img.shape) < 3:  # third axis is the channels
    img = np.expand_dims(img, axis=-1)
  while(img.shape[-1]) < 3:
    img = np.concatenate([img, img[:, :, -1:]], axis=-1)
  return img


class TinyImageNetPaths:
  def __init__(self, root_dir, corruption = 'original', level = 5):
    val_path = os.path.join(root_dir, corruption, str(level))

    root_dir_original = root_dir.replace('Tiny-ImageNet-C', 'tiny-imagenet-200')
    wnids_path = os.path.join(root_dir_original, 'wnids.txt')
    words_path = os.path.join(root_dir_original, 'words.txt')

    self._make_paths(val_path, wnids_path, words_path)

  def _make_paths(self, corrupt_path, wnids_path, words_path):
    self.ids = []
    with open(wnids_path, 'r') as idf:
      for nid in idf:
        nid = nid.strip()
        self.ids.append(nid)
    self.nid_to_words = defaultdict(list)
    with open(words_path, 'r') as wf:
      for line in wf:
        nid, labels = line.split('\t')
        labels = list(map(lambda x: x.strip(), labels.split(',')))
        self.nid_to_words[nid].extend(labels)

    self.paths = {
      'corrupt': [],  # [img_path, id, nid, box]
    }

    # Get the corruption paths
    corrupt_nids = os.listdir(corrupt_path)
    for nid in corrupt_nids:
      label_id = self.ids.index(nid)
      path = os.path.join(corrupt_path, nid)
      corrupt_name = os.listdir(path)
      for imgname in corrupt_name:
        fname = os.path.join(path, imgname)
        self.paths['corrupt'].append((fname, label_id, nid))


class TinyImageNetCDataset(Dataset):
  def __init__(self, root_dir, mode='corrupt', transform=None, max_samples=None, corruption = 'snow', level = 5):
    tinp = TinyImageNetPaths(root_dir, corruption=corruption, level=level)
    self.mode = mode
    self.label_idx = 1  # from [image, id, nid, box]
    self.transform = transform
    self.transform_results = dict()

    self.IMAGE_SHAPE = (64, 64, 3)

    self.img_data = []
    self.label_data = []

    self.max_samples = max_samples
    self.samples = tinp.paths[mode]
    self.samples_num = len(self.samples)

    if self.max_samples is not None:
      self.samples_num = min(self.max_samples, self.samples_num)
      self.samples = np.random.permutation(self.samples)[:self.samples_num]

  def __len__(self):
    return self.samples_num

  def __getitem__(self, idx):
    s = self.samples[idx]
    img = Image.open(s[0])
    img_array = np.array(img)
    if img_array.shape[-1] < 3 or len(img_array.shape) < 3:
      img_array = _add_channels(img_array)
      img = Image.fromarray(img_array)
    lbl = None if self.mode == 'test' else s[self.label_idx]

    if self.transform:
      sample = self.transform(img)
    return sample, lbl


# dataroot = "/home/davidoso/Documents/Data/"
# a = TinyImageNetDataset(dataroot + 'tiny-imagenet-200/', preload=False)