# Source: https://github.com/ThomasMrY/DisDiff
# Original code adapted from the DisDiff repository by ThomasMrY.

import gin
import numpy as np
from sklearn.decomposition import PCA


def eval_func(eval_dataset, data_tensor, seed=0, preflix=''):
    pca_list = []
    for i in range(data_tensor.shape[1]):
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(data_tensor[:,i,:].numpy())
        pca_list.append(pca_result)
    pca_rep = np.concatenate(pca_list, axis=1)

    beta_VAE_score = True
    dci_score = True
    factor_VAE_score = True
    MIG_score = True
    total_results_dict = {}
    def _representation(x):
        return pca_rep[x]
    if beta_VAE_score:
        with gin.unlock_config():
            from evaluation.metrics.beta_vae import compute_beta_vae_sklearn
            gin.bind_parameter('beta_vae_sklearn.batch_size', 64)
            gin.bind_parameter('beta_vae_sklearn.num_train', 10000)
            gin.bind_parameter('beta_vae_sklearn.num_eval', 5000)
        result_dict = compute_beta_vae_sklearn(eval_dataset,_representation,random_state=np.random.RandomState(seed),artifact_dir=None)
        print('beta VAE score:' + str(result_dict))
        total_results_dict['beta_VAE' + preflix] = result_dict
    if dci_score:
        from evaluation.metrics.dci import compute_dci
        with gin.unlock_config():
            gin.bind_parameter('dci.num_train',10000)
            gin.bind_parameter('dci.num_test',5000)
        result_dict = compute_dci(eval_dataset,_representation,random_state=np.random.RandomState(seed),artifact_dir=None)
        print('dci score:' + str(result_dict))
        total_results_dict['dci' + preflix] = result_dict
    if MIG_score:
        with gin.unlock_config():
            from evaluation.metrics.mig import compute_mig
            from evaluation.metrics.utils import _histogram_discretize
            gin.bind_parameter('mig.num_train',10000)
            gin.bind_parameter('discretizer.discretizer_fn',_histogram_discretize)
            gin.bind_parameter('discretizer.num_bins',20)
        result_dict = compute_mig(eval_dataset,_representation,random_state=np.random.RandomState(seed),artifact_dir=None)
        print('MIG score:' + str(result_dict))
        total_results_dict['MIG' + preflix] = result_dict
    if factor_VAE_score:
        with gin.unlock_config():
            from evaluation.metrics.factor_vae import compute_factor_vae
            gin.bind_parameter('factor_vae_score.num_variance_estimate',10000)
            gin.bind_parameter('factor_vae_score.num_train',10000)
            gin.bind_parameter('factor_vae_score.num_eval',5000)
            gin.bind_parameter('factor_vae_score.batch_size',64)
            gin.bind_parameter('prune_dims.threshold',0.05)
        result_dict = compute_factor_vae(eval_dataset,_representation,random_state=np.random.RandomState(seed),artifact_dir=None)
        print('factor VAE score:' + str(result_dict))
        total_results_dict['factor_VAE' + preflix] = result_dict
    return total_results_dict
