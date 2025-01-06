"""
import time

import torch
import torch.nn.functional as F
import torch.nn as nn



def get_zero_shot_logits(query_features, query_labels, clip_prototypes):

    query_features = query_features.cuda().float()
    query_labels = query_labels.cuda()
    clip_prototypes = clip_prototypes.cuda().float()

    clip_logits = 100 * query_features @ clip_prototypes
       
    return clip_logits



def update_z(gmm_likelihood, y_hat, z, W, lambda_value, n_neighbors, sigma, covariance_strategy, labels=None, lambda_laplacian=1, max_iter=5):
    few_shot = labels is not None
    if few_shot:
        shots_labels = F.one_hot(labels).float()
        z = torch.cat((z.clone(), shots_labels))

    num_samples = gmm_likelihood.size(0)

    for it in range(max_iter):
        
        intermediate = gmm_likelihood.clone()
        
        if lambda_laplacian != 0:
            intermediate += lambda_laplacian*(50 / (n_neighbors * 2)) * (
                    W.T @ z + (W @ z[0:num_samples, :])[0:num_samples, :])
        
        if covariance_strategy == 'class':
            sigma_log_sum = sigma.log().sum(dim=1)
            #print(sigma_log_sum)
            intermediate -= 0.5 * sigma_log_sum.unsqueeze(0)
        
        # For numerical stability
        intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
        intermediate = (y_hat ** lambda_value) * torch.exp(1 / 50 * intermediate)
        z[0:num_samples] = intermediate / torch.sum(intermediate, dim=1, keepdim=True)

    return z


def update_mu(adapter, gamma_value, query_features, z, support_features, labels, init_prototypes, beta):
    affinity_unlabeled = z
    
    
    n_query = affinity_unlabeled.size(0)
    few_shot = support_features is not None
    if few_shot:
        affinity_labeled = torch.nn.functional.one_hot(labels).float()
        n_support = affinity_labeled.size(0)

    weights = (1 / n_query) * affinity_unlabeled

    # Use einsum to compute the new_mu for each class in one pass
    new_mu = torch.einsum('ij,ik->jk', weights, query_features) 

    
    #print(new_mu.shape)
    if few_shot:
        weights = (gamma_value * 50 / n_support) * affinity_labeled
        new_mu += torch.einsum('ij,ik->jk', weights, support_features)

        new_mu /= (1 / n_query * torch.sum(
            affinity_unlabeled, dim=0).unsqueeze(
            -1) + gamma_value * 50 / n_support * torch.sum(
            affinity_labeled, dim=0).unsqueeze(-1))
    else:
        new_mu /= (1 / n_query * torch.sum(
            affinity_unlabeled, dim=0).unsqueeze(-1))
    new_mu = new_mu.unsqueeze(1)

    new_mu /= new_mu.norm(dim=-1, keepdim=True)
    
    new_mu = beta.unsqueeze(-1).unsqueeze(-1) * new_mu + (1-beta).unsqueeze(-1).unsqueeze(-1) * init_prototypes
    
    new_mu /= new_mu.norm(dim=-1, keepdim=True)
    
    adapter.mu = new_mu
    
    return adapter


def update_sigma(mu, gamma_value, query_features, z, support_features, labels, init_covariance, init_prototypes, beta, covariance_strategy):
    affinity_unlabeled = z
    n_query = affinity_unlabeled.size(0)
    few_shot = support_features is not None
    if few_shot:
        affinity_labeled = torch.nn.functional.one_hot(labels).float()
        n_support = affinity_labeled.size(0)

    std = 0

    chunk_size = 2500  # Iterate over query_features in chunks to avoid large memory consumption

    for start_idx in range(0, n_query, chunk_size):
        end_idx = min(start_idx + chunk_size, n_query)
        query_features_chunk = query_features[start_idx:end_idx]
        
        if covariance_strategy == 'shared':
            # Compute the weighted sum of squared differences for the chunk
            chunk_result = (1 / n_query) * torch.einsum(
                'ij,ijk->k',
                affinity_unlabeled[start_idx:end_idx, :],
                # Use a chunk of affinity_unlabeled
                (query_features_chunk[:, None, :] - mu[None, :,
                                                   0, :]) ** 2)
        elif covariance_strategy == 'class':
            # Compute the class-wise variance for the chunk
            chunk_result = (1 / n_query) * torch.einsum(
                'ij,ijk->ijk',
                affinity_unlabeled[start_idx:end_idx, :],
                # Use a chunk of affinity_unlabeled and compute class-wise variance
                (query_features_chunk[:, None, :] - mu[None, :, 0, :]) ** 2
            ).sum(dim=0)  # Sum over the sample dimension to aggregate the variance per class
        
        # If this is the first chunk, initialize std; otherwise, accumulate
        if start_idx == 0:
            std = chunk_result
        else:
            std += chunk_result

    if few_shot and gamma_value > 0:
        # Iterate over query_features in chunks
        for start_idx in range(0, n_support, chunk_size):
            end_idx = min(start_idx + chunk_size, n_support)
            support_features_chunk = support_features[
                                     start_idx:end_idx]

            # Compute the weighted sum of squared differences for the chunk
            chunk_result = (gamma_value * 50 / n_support) * torch.einsum(
                'ij,ijk->k',
                affinity_labeled[start_idx:end_idx, :],
                # Use the relevant part of affinity_unlabeled
                (support_features_chunk[:, None, :] - mu[
                                                      None, :, 0,
                                                      :]) ** 2
            )

            std += chunk_result

        std /= (1 / n_query * torch.sum(
            affinity_unlabeled[:,
            :]) + gamma_value * 50 / n_support * torch.sum(
            affinity_labeled[:, :]))
    else:
        if covariance_strategy == 'shared':
            std /= (1 / n_query * torch.sum(
                affinity_unlabeled[:, :]))
        elif covariance_strategy == 'class':
            std /= (1 / n_query * torch.sum(
                affinity_unlabeled, dim=0)).unsqueeze(-1)
        
            if init_covariance != None:
                
                delta_mu = (init_prototypes - mu).squeeze()
                result = torch.bmm(delta_mu.unsqueeze(2), delta_mu.unsqueeze(1))
                diagonal_result = torch.diagonal(result, dim1=1, dim2=2)  # Shape: [num_classes, num_features]
                #print(diagonal_result)
             
                std = beta.unsqueeze(-1) * std + (1-beta).unsqueeze(-1) * (init_covariance + diagonal_result)
                #print(std.shape)
                
    
    return std





def StatA_solver(query_features, query_labels, clip_prototypes, lambda_y_hat=1, lambda_laplacian=1, n_neighbors=3):

    start_time = time.time()

    ################
    # General init #
    ################

    K = clip_prototypes.size(0)
    d = query_features.size(1)
    num_samples = query_features.size(0)

    zs_logits = get_zero_shot_logits(query_features, query_labels, clip_prototypes)
    
    max_iter = 10  # number of iterations
    
    ##########
    # Z init #
    ##########

    y_hat = F.softmax(zs_logits, dim=1)
    z = y_hat
    
    ###########
    # MU init #
    ###########

    mu = clip_prototypes.permute(2,0,1) 

    ##############
    # SIGMA init #
    ##############
    
    beta = torch.ones(num_classes).cuda()
    gamma_value = None
    init_covariance = update_sigma(mu, gamma_value, query_features, z, support_features, support_labels, None, init_prototypes, beta, init_covariance_strategy)
    
    
    if init_covariance_strategy == 'shared' and covariance_strategy == 'class':
        init_covariance = init_covariance.unsqueeze(0).repeat(num_classes, 1)
    
    adapter = Gaussian(mu=mu, std=init_covariance, covariance_strategy=covariance_strategy).cuda()

    ###################
    # Affinity matrix #
    ###################

    W = build_affinity_matrix(query_features, support_features, num_samples, n_neighbors)
    lambda_value = 1
    for idx, gamma_value in enumerate(gamma_list):
        for k in range(max_iter + 1):
             
            gmm_likelihood = adapter(query_features, no_exp=True)
            

            ############
            # Z update #
            ############

            new_z = update_z(gmm_likelihood, y_hat, z, W, lambda_value, n_neighbors, adapter.std, covariance_strategy, support_labels, lambda_laplacian)[0:num_samples]
            z = new_z
            if k == max_iter:  # STOP
                acc = cls_acc(z, query_labels)
                if support_features is not None:  # Few-shot : validate gamma
                    acc_val = cls_acc(z[neighbor_index, :], val_labels)
                    if acc_val > best_val:
                        best_val = acc_val
                        test_acc_at_best_val = acc

                else:
                    acc = cls_acc(z, query_labels)
                    test_acc_at_best_val = acc
                    if verbose:
                        print("\n**** TransCLIP's test accuracy: {:.2f} ****\n".format(acc))
                break
            
            ###############
            # Beta update #
            ###############
            
            if kl_weight == 0:
                torch.ones(K).cuda()
                
            elif hard_soft:
                
                # Get the predicted class for each sample
                predicted_classes = torch.argmax(z, dim=1)  # [num_samples], returns the index of the max logit for each sample
                
                # Create a mask for the predicted classes
                mask = torch.zeros_like(z, dtype=torch.bool)  # Create a mask with the same shape as z
                mask[torch.arange(z.size(0)), predicted_classes] = True  # Set True for the predicted class of each sample
                
                # Sum the z values only for the most confident (argmax) class
                sum_z = torch.sum(z * mask, dim=0)  # [num_classes]
                
                beta = sum_z / (delta + sum_z + 1e-12)
                
            elif hard:
                # Get the predicted class for each sample
                predicted_classes = torch.argmax(z, dim=1)  # [num_samples], returns the index of the max logit for each sample
                
                # Count the number of predictions for each class
                sum_z = torch.bincount(predicted_classes, minlength=z.size(1))  # [num_classes]
                
                beta = sum_z / (delta + sum_z + 1e-12)
            else:
                sum_z = torch.sum(z, dim=0)  # [num_classes]
                beta = sum_z / (delta + sum_z)

            
            #############
            # MU update #
            #############

            adapter = update_mu(adapter, gamma_value, query_features, z, support_features, support_labels, init_prototypes, beta)

            ################
            # SIGMA update #
            ################
            
            std = update_sigma(adapter.mu, gamma_value, query_features, z, support_features, support_labels, init_covariance, init_prototypes, beta, covariance_strategy)
            adapter.set_std(std)

        if support_features is not None:
            if verbose:
                print("{}/{} TransCLIP's test accuracy: {:.2f} on test set @ best validation accuracy ({:.2f})".format(
                    idx+1, len(gamma_list), test_acc_at_best_val, best_val))
    end_time = time.time()
    elapsed_time = end_time - start_time
    if verbose:
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
    return z, acc_zs, test_acc_at_best_val

"""







import torch
import torch.nn.functional as F
import torch.nn as nn



def get_zero_shot_logits(query_features, query_labels, clip_prototypes):

    clip_logits = 100 * query_features @ clip_prototypes

    return clip_logits.squeeze()


def build_affinity_matrix(query_features, n_neighbors):
    
    device = query_features.device
    num_samples = query_features.size(0)
    affinity = query_features.matmul(query_features.T).cpu()
    num_rows = num_samples
    num_cols = num_samples
        
    knn_index = affinity.topk(n_neighbors + 1, -1, largest=True).indices[:, 1:]
    row_indices = torch.arange(num_rows).unsqueeze(1).repeat(1, n_neighbors).flatten()
    col_indices = knn_index.flatten()
    values = affinity[row_indices, col_indices].to(device)
    W = torch.sparse_coo_tensor(torch.stack([row_indices, col_indices]).to(device), values, size=(num_rows, num_cols),
                                device=device)
    return W




class Gaussian(nn.Module):
    def __init__(self, mu, cov):
        super().__init__()
        self.mu = mu.clone()
        self.cov = cov.clone()

    def forward(self, x, no_exp=False):
        chunk_size = 2500
        N = x.shape[0]
        M, D = self.mu.shape[0], self.cov.shape[0]

        likelihoods = torch.empty((N, M), dtype=x.dtype, device=x.device)
        
        for start_idx in range(0, N, chunk_size):
            end_idx = min(start_idx + chunk_size, N)
            likelihoods[start_idx:end_idx] = -0.5 * (
                    ((x[start_idx:end_idx][:, None, :] - self.mu[None, :, 0, :]) ** 2) *
                    (1 / self.cov[None, :, :])
                ).sum(dim=2)


        if not no_exp:
            likelihoods = torch.exp(likelihoods)
        
        return likelihoods

    def set_cov(self, cov):
        self.cov = cov
        
    def set_mu(self, mu):
        self.mu = mu



def update_z(likelihoods, y_hat, z, W, lambda_y_hat, lambda_laplacian, n_neighbors, max_iter=5):
    for it in range(max_iter):
        intermediate = likelihoods.clone()
        intermediate += lambda_laplacian*(50 / (n_neighbors * 2)) * (
                W.T @ z + (W @ z))
        # For numerical stability
        intermediate -= torch.max(intermediate, dim=1, keepdim=True)[0]
        intermediate = (y_hat ** lambda_y_hat) * torch.exp(1 / 50 * intermediate)
        z = intermediate / torch.sum(intermediate, dim=1, keepdim=True)
    return z


def update_mu(adapter, query_features, z, beta, init_prototypes):

    mu = torch.einsum('ij,ik->jk', z, query_features) 
    mu /= torch.sum(z, dim=0).unsqueeze(-1)
    mu = mu.unsqueeze(1)
    mu /= mu.norm(dim=-1, keepdim=True)
    mu = beta.unsqueeze(-1).unsqueeze(-1) * mu + (1-beta).unsqueeze(-1).unsqueeze(-1) * init_prototypes
    mu /= mu.norm(dim=-1, keepdim=True)
    return mu


def update_cov(adapter, query_features, z, beta, init_prototypes, init_covariance):
    n_query = z.size(0)
    num_classes = z.size(1)  # Assuming z is a one-hot encoded tensor or soft labels of shape [n_query, num_classes]
    chunk_size = 2500  # Iterate over query_features in chunks to avoid large memory consumption

    for start_idx in range(0, n_query, chunk_size):
        end_idx = min(start_idx + chunk_size, n_query)
        query_features_chunk = query_features[start_idx:end_idx]  # Shape: [chunk_size, num_dim]
        
        # Compute diff, squared_diff, and weighted_sum in one line
        weighted_sum = ((query_features_chunk[:, None, :] - adapter.mu[None, :, 0, :]) ** 2 * z[start_idx:end_idx, :, None]).sum(dim=0)  # Shape: [num_classes, num_dim]

        # Accumulate the covariance
        if start_idx == 0:
            cov = weighted_sum  # Initialize the covariance matrix for all classes
        else:
            cov += weighted_sum  # Accumulate per-class contributions

    cov /= z.sum(dim=0)[:, None]  
        
    # Compute delta_mu
    delta_mu = (init_prototypes - adapter.mu).squeeze()  # Shape: [num_classes, num_features]
    result = torch.bmm(delta_mu.unsqueeze(2), delta_mu.unsqueeze(1))  # Shape: [num_classes, num_features, num_features]
    diagonal_result = torch.diagonal(result, dim1=1, dim2=2)  # Extract diagonal, Shape: [num_classes, num_features]
    
    # Final update to std
    cov = beta.unsqueeze(-1) * cov + (1 - beta).unsqueeze(-1) * (init_covariance + diagonal_result)
 
    return cov



def init_cov(clip_prototypes, query_features, z):
    
    n_query = z.size(0)
    chunk_size = 2500  # Iterate over query_features in chunks to avoid large memory consumption
    
    for start_idx in range(0, n_query, chunk_size):
        end_idx = min(start_idx + chunk_size, n_query)
        query_features_chunk = query_features[start_idx:end_idx]
        
        chunk_result = torch.einsum(
            'ij,ijk->k',
            z[start_idx:end_idx, :],
            (query_features_chunk[:, None, :] - clip_prototypes[None, :,
                                               0, :]) ** 2)
        # If this is the first chunk, initialize cov; otherwise, accumulate
        if start_idx == 0:
            cov = chunk_result
        else:
            cov += chunk_result
        cov /= n_query 
    return cov

def update_beta(z, alpha, soft=False):

    if soft:
        sum_z = torch.sum(z, dim=0)  # [num_classes]
        beta = sum_z / (alpha + sum_z)
    else:
        predicted_classes = torch.argmax(z, dim=1) 
        sum_z = torch.bincount(predicted_classes, minlength=z.size(1))  
        beta = sum_z / (alpha + sum_z + 1e-12)
    return beta

    





def StatA_solver(query_features, query_labels, clip_prototypes, alpha=1, soft_beta=False, lambda_y_hat=1, lambda_laplacian=1, n_neighbors=3, max_iter=10):
    
    query_labels = query_labels.cuda().float()
    clip_prototypes = clip_prototypes.cuda().float()
    query_features = query_features.cuda().float()
    
    ##########
    # Z init #
    ##########

    zs_logits = get_zero_shot_logits(query_features, query_labels, clip_prototypes)
    y_hat = F.softmax(zs_logits, dim=1)
    z = y_hat.clone()
    
    ###########
    # MU init #
    ###########

    mu = clip_prototypes.permute(2,0,1) 

    ############
    # COV init #
    ############
     
    cov = init_cov(clip_prototypes.permute(2, 0, 1), query_features, z)
    cov = cov.unsqueeze(0).repeat(y_hat.size(-1), 1) 
    init_covariance = cov

    adapter = Gaussian(mu=mu, cov=cov).cuda()
    
    ###################
    # Affinity matrix #
    ###################
    
    W = build_affinity_matrix(query_features.float(), n_neighbors)
    
    for k in range(max_iter + 1):
        
        likelihoods = adapter(query_features, no_exp=True)
        
        ############
        # Z update #
        ############

        z = update_z(likelihoods, y_hat, z, W, lambda_y_hat, lambda_laplacian, n_neighbors)
        
        if k == max_iter:  # STOP
            break
        
        ###############
        # BETA update #
        ###############
        
        beta = update_beta(z, alpha, soft=soft_beta)
        
        #############
        # MU update #
        #############

        mu = update_mu(adapter, query_features, z, beta, clip_prototypes.permute(2,0,1))
        adapter.set_mu(mu)

        ################
        # SIGMA update #
        ################
        
        cov = update_cov(adapter, query_features, z, beta, clip_prototypes.permute(2,0,1), init_covariance)
        adapter.set_cov(cov)

    return y_hat.cpu(), z.cpu()

