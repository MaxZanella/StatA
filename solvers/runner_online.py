import torch
from .TransCLIP import TransCLIP_online_solver, TransCLIP_online_solver_bis
def run_transclip_online(test_features, test_labels, clip_prototypes, num_classes, verbose, kl_weight, lambda_laplacian, covariance_strategy, init_covariance_strategy, hard, hard_soft, sampler, lambda_z):
    K = torch.max(test_labels)+1    
    
    pos_cache = {} # Positives cache of TDA : needs to be stored for online setting
    neg_cache = {} # Negative cache of TDA : needs to be stored for online setting
    indices = sampler.generate_indices()
    all_accs = []
    all_accs_zs = []

    beta_average = 0.
    adapter, init_prototypes, init_covariance, count_soft, count_hard, v, t = None, None, None, None, None, None, None
    
    begin = True
    
    total_count_hard = torch.ones(K).cuda()
    
    count_samples = 0
    while indices is not None:
        
        if begin == False:
            old_t = t
            old_v = v
        """
        acc, acc_zs, adapter, init_prototypes, init_covariance, count_soft, count_hard, v, t = TransCLIP_online_solver(query_features=test_features[indices,:], query_labels=test_labels[indices],
                                                clip_prototypes=clip_prototypes, initial_prototypes=None, initial_predictions=None, verbose=False, num_classes=K, 
                                                kl_weight=kl_weight, lambda_laplacian=lambda_laplacian, covariance_strategy= covariance_strategy, init_covariance_strategy=init_covariance_strategy, hard=hard, hard_soft=hard_soft,
                                                adapter=adapter, init_prototypes=init_prototypes, init_covariance=init_covariance,count_soft=count_soft, count_hard=count_hard, v=v, t=t, lambda_z=lambda_z)

        """
        if adapter == None:
            old_mu = None
        else:
            old_mu = adapter.mu.clone()
            
        acc, acc_zs, adapter, init_prototypes, init_covariance, count_soft, count_hard, v, t = TransCLIP_online_solver_bis(query_features=test_features[indices,:], query_labels=test_labels[indices],
                                                clip_prototypes=clip_prototypes, initial_prototypes=None, initial_predictions=None, verbose=False, num_classes=K, 
                                                kl_weight=kl_weight, lambda_laplacian=lambda_laplacian, covariance_strategy= covariance_strategy, init_covariance_strategy=init_covariance_strategy, hard=hard, hard_soft=hard_soft,
                                                adapter=adapter, init_prototypes=init_prototypes, init_covariance=init_covariance,count_soft=count_soft, count_hard=total_count_hard, v=v, t=t, lambda_z=lambda_z)
        
        
        if begin:
            old_t = init_covariance

            old_v = init_prototypes.squeeze()
            
            old_mu = init_prototypes
            begin=False
        
        
        #print(total_count_hard.shape, count_hard.shape, adapter.mu.shape, v.shape)
        adapter.mu = (total_count_hard.unsqueeze(-1).unsqueeze(-1) * old_mu + count_hard.unsqueeze(-1).unsqueeze(-1) * v.unsqueeze(1))/(total_count_hard.unsqueeze(-1).unsqueeze(-1) + count_hard.unsqueeze(-1).unsqueeze(-1))
        adapter.mu /= adapter.mu.norm(dim=-1, keepdim=True)
        
                     
        total_count_hard += count_hard
                      
        
        
        t = beta_average * old_t + (1-beta_average) * t
        v = beta_average * old_v + (1-beta_average) * v


        
        #print(acc)
        all_accs.append(acc)
        all_accs_zs.append(acc_zs)
        indices = sampler.generate_indices()
        
        #print(sum(all_accs)/len(all_accs))
    avg_accuracy = sum(all_accs)/len(all_accs)
  
    return avg_accuracy