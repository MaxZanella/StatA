import torch
import torch.utils.data
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image



common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


def prepare_cifar(root, dataset, preprocess, level=5):
    
    if dataset.startswith('cifar100'):
        root += '/CIFAR/'
        size = 10000
        
        l = dataset.split('-') 
        if len(l)==1:
            corruption = 'original'
        else:
            corruption = l[1]
            
        if corruption == 'original':
            testset = torchvision.datasets.CIFAR100(root=root,
                train=False, download=True, transform=preprocess)
        elif corruption in common_corruptions:
            testset_raw = np.load(root + '/CIFAR-100-C/%s.npy' % (corruption))
            testset_raw = testset_raw[(level - 1) * size: level * size]
            testset = torchvision.datasets.CIFAR100(root=root,
                train=False, download=False, transform=preprocess)

            testset.data = testset_raw
    elif dataset.startswith('cifar10'):
        root += '/CIFAR/'
        size = 10000
        
        l = dataset.split('-') 
        if len(l)==1:
            corruption = 'original'
        else:
            corruption = l[1]
        if corruption == 'original':
            testset = torchvision.datasets.CIFAR10(root=root,
                train=False, download=False, transform=preprocess)
        elif corruption in common_corruptions:
            testset_raw = np.load(root + '/CIFAR-10-C/%s.npy' % (corruption))
            testset_raw = testset_raw[(level - 1) * size: level * size]
            testset = torchvision.datasets.CIFAR10(root=root,
                train=False, download=False, transform=preprocess)
            testset.data = testset_raw

        elif args.corruption == 'cifar_new':
            from utils.cifar_new import CIFAR_New
            teset = CIFAR_New(root=args.dataroot + '/CIFAR-10.1/', transform=te_transforms)
            permute = False
        else:
            raise Exception('Corruption not found!')

    
            
    testset.classnames = testset.classes
    testset.template = ['a photo of a {}.']
    return testset
            
            
"""
    elif args.dataset == 'visda':
        teset = VisdaTest(args.dataroot, transforms=visda_val)

    elif args.dataset == 'tiny-imagenet':
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            teset = TinyImageNetDataset(args.dataroot + '/tiny-imagenet-200/', mode='val', transform=te_transforms)
        elif args.corruption in common_corruptions:
            teset = TinyImageNetCDataset(args.dataroot + '/Tiny-ImageNet-C/', corruption = args.corruption, level = args.level,
                                        transform=te_transforms)
    else:
        raise Exception('Dataset not found!')
"""



def prepare_val_data(args, transform=None):
    if args.dataset == 'visda':
        vset = ImageFolder(root=args.dataroot + 'validation/', transform=transform if transform is not None else visda_val)
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        v_sampler = torch.utils.data.distributed.DistributedSampler(vset)
    else:
        v_sampler = None
    if not hasattr(args, 'workers'):
        args.workers = 1
    vloader = torch.utils.data.DataLoader(vset, batch_size=args.batch_size,
        shuffle=(v_sampler is None), num_workers=args.workers, pin_memory=True, sampler=v_sampler, drop_last=True)
    return vloader, v_sampler

def prepare_train_data(args, transform=None):
    if args.clip :
        tr_transforms = clip_transforms
    if args.dataset == 'cifar10':
        trset = torchvision.datasets.CIFAR10(root=args.dataroot,
            train=True, download=False, transform=tr_transforms)
    elif args.dataset == 'cifar100':
        trset = torchvision.datasets.CIFAR100(root=args.dataroot, train=True, download=False, transform=tr_transforms)
    elif args.dataset == 'visda':
        dataset = ImageFolder(root=args.dataroot + 'train/', transform=visda_train if transform is None else transform)
        trset, _ = random_split(dataset, [106678, 45719], generator=torch.Generator().manual_seed(args.seed))
    elif args.dataset == 'tiny-imagenet':
        trset = TinyImageNetDataset(args.dataroot + '/tiny-imagenet-200/', transform=tinyimagenet_transforms)
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        tr_sampler = torch.utils.data.distributed.DistributedSampler(trset)
    else:
        tr_sampler = None

    if not hasattr(args, 'workers'):
        args.workers = 1
    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
        shuffle=(tr_sampler is None), num_workers=args.workers, pin_memory=True, sampler=tr_sampler)


    return trloader, tr_sampler, trset