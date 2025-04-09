import os
import random
import argparse
import numpy as np
import torch
from datasets import get_all_dataloaders
from utils import *
from sampler import BatchSampler, OnlineSampler
from tqdm import tqdm
from solvers import TransCLIP_solver, StatA_solver, Dirichlet_solver, ZLaP_solver #, TDA_solver, DMN_solver

def get_arguments():
    
    parser = argparse.ArgumentParser()
    
    # General arguments
    parser.add_argument('--dataset', default='dtd', help='dataset name', type=str)
    parser.add_argument('--root_path', default='./datasets', type=str)
    parser.add_argument('--method', default='StatA', type=str, choices=['StatA', 'TransCLIP', 'Dirichlet', 'ZLaP'])
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--backbone', default='vit_b16', type=str, choices=['rn50', 'rn101', 'vit_b32', 'vit_b16', 'vit_l14'], help="CLIP architecture")
    parser.add_argument('--cache_dir', type = str, default = None, help='where to store visual and textual features if not None')
    parser.add_argument('--load', action='store_true', default=False, help="Load features from cache_dir")

    # Experimental arguments
    parser.add_argument('--n_tasks', type=int, default=1, help="number of tasks to run")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--online', action='store_true', default=False, help='online setting or not')
    parser.add_argument('--num_class_eff', type=int, default=None, help='number of effective classes to sample from per batch')
    parser.add_argument('--num_class_eff_min', type=int, default=None, help='number of effective classes per batch minimum')
    parser.add_argument('--num_class_eff_max', type=int, default=None, help='number of effective classes per batch maximum')
    parser.add_argument('--gamma', type = float, default = 1.0, help = 'Dirichlet parameter used for sampling in the online setting.')
    
    # StatA hyperparameters (with default values in all experiments)
    parser.add_argument('--alpha', type=float, default=1.0, help='anchor weighting hyper-parameter')
    parser.add_argument('--lambda_laplacian', type=float, default=1.0, help='Laplacian weighting hyper-parameter')
    parser.add_argument('--soft_beta', action='store_true', default=False, help='use soft beta computation')

    args = parser.parse_args()
    return args

def get_hp(args, method_name):
    if method_name == 'StatA':
        return StatA_solver, {
            'alpha': args.alpha,
            'lambda_y_hat':1,
            'lambda_laplacian': args.lambda_laplacian,
            'n_neighbors':3,
            'soft_beta': args.soft_beta
        }
    elif method_name == 'TransCLIP':
        return TransCLIP_solver, {'lambda_y_hat':1, 'lambda_laplacian': 1, 'n_neighbors':3}
    elif method_name == 'Dirichlet':
        return Dirichlet_solver, {'T':30}
    elif method_name == 'ZLaP':
        return ZLaP_solver, {'k':5, 'gamma':5.0, 'alpha':0.3, 'scale_sim':False}
    else:
        raise NotImplementedError(f"Method {method_name} is not implemented.")



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():

    args = get_arguments()
    
    set_random_seed(args.seed) # for reproducibility
    
    if not args.cache_dir:
        args.cache_dir = os.path.join('./caches', args.dataset)
    os.makedirs(args.cache_dir, exist_ok=True)

    # CLIP model
    backbones = {'rn50': 'RN50',
                 'rn101': 'RN101',
                 'vit_b16': 'ViT-B/16',
                 'vit_b32': 'ViT-B/32',
                 'vit_l14': 'ViT-L/14'}
    clip_model, preprocess = clip.load(backbones[args.backbone])
    clip_model.eval()

    # Prepare dataset
    _, _, test_loader, dataset = get_all_dataloaders(args, preprocess)

    # Load features
    test_features, test_labels, clip_prototypes = get_all_features(args, test_loader, dataset, clip_model)
        
    clip_model = clip_model.to('cpu')  # unload CLIP model from VRAM

    acc_tot = 0
    acc_zs_tot = 0
    
    
    ##############################
    # Batch Test-Time Adaptation #
    ##############################
    
    solver, method_args = get_hp(args, args.method)
    
    if not args.online:
        sampler = BatchSampler(test_features, test_labels, args.batch_size, args.num_class_eff, args.num_class_eff_min, args.num_class_eff_max)
    
        for i in tqdm(range(args.n_tasks)):
    
            indices = sampler.generate_indices()
            if indices == None:
                break
    
            preds_zs, preds = solver(test_features[indices,:], test_labels[indices], clip_prototypes, # visual and textual features  
                                              **method_args)
            
            acc_zs = cls_acc(preds_zs, test_labels[indices])
            acc = cls_acc(preds, test_labels[indices])
            acc_zs_tot += acc_zs
            acc_tot += acc
        
        
        acc_zs_tot /= args.n_tasks
        acc_tot /= args.n_tasks
        
    ###############################
    # Online Test-Time Adaptation #
    ###############################

    if args.online:
   
        for i in tqdm(range(args.n_tasks)):
            
            num_batch = test_features.shape[0]//args.batch_size
            num_slots = min(num_batch, len(torch.unique(test_labels)))
            sampler = OnlineSampler(test_features, test_labels, args.gamma, num_slots, args.batch_size)
            
            indices = sampler.generate_indices()
            all_accs = []
            all_accs_zs = []
            
            while indices is not None:
                
                preds_zs, preds = solver(test_features[indices,:], test_labels[indices], clip_prototypes, # visual and textual features  
                                                  **method_args)
                acc_zs = cls_acc(preds_zs, test_labels[indices])
                acc = cls_acc(preds, test_labels[indices])
                
                all_accs.append(acc)
                all_accs_zs.append(acc_zs)
                indices = sampler.generate_indices()
                
            acc_tot += sum(all_accs)/len(all_accs)
            acc_zs_tot += sum(all_accs_zs)/len(all_accs_zs)

        acc_tot /= args.n_tasks
        acc_zs_tot /= args.n_tasks
      
        
    print("\n============================")
    print("      Final Results         ")
    print("============================")
    print(f"Dataset:         {args.dataset}")
    print(f"Method:          {args.method}")
    print(f"Backbone:        {args.backbone}")
    print(f"Number of Tasks: {args.n_tasks}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Online Setting:  {'Yes' if args.online else 'No'}")
      
    if args.online:
        print(f"Dirichlet Gamma: {args.gamma:.2f}")
    else:
        print(f"Effective Classes Min: {args.num_class_eff_min or 'None'}")
        print(f"Effective Classes Max: {args.num_class_eff_max or 'None'}")
      
    print("----------------------------")
    print(f"ZERO-shot Accuracy: {acc_zs_tot:.4f}")
    print(f"FINAL Accuracy:     {acc_tot:.4f}")
    print("============================\n")


if __name__ == '__main__':
    main()
