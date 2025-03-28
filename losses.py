import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union
from torch.nn.modules.loss import _Loss

class OridinalEntropy(torch.nn.Module):
    def __init__(self, lambda_d_phn=1.0, lambda_t_phn=1.0, margin=1.0, ignore_index=-1):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_d_phn = lambda_d_phn
        self.lambda_t_phn = lambda_t_phn
        self.margin = margin
        self.accum_features = None
        self.accum_labels = None
        self.batch_size = 32
        
        print('margin:',self.margin)
        print('lambda_d_phn:',self.lambda_d_phn)
        print('lambda_t_phn:',self.lambda_t_phn)

    def forward(self, features, label, label_id):
        """
        Features: a certain layer's features
        label: pixel-wise ground truth values, in depth estimation, label.size()= n, h, w
        mask: In case values of some pixels do not exist. For depth estimation, there are some pixels lack the ground truth values
        """

        features = self.accum_features
        label = label_id = self.accum_labels
        
        f_n, f_c = features.size()
        _label = label.view(-1)
        _mask = _label >= 0
        _mask = _mask.to(torch.bool)
        _label = _label[_mask]
        _features = features.reshape(-1, f_c)
        _features = _features[_mask,:]
        _label_id = label_id.view(-1)[_mask]
        
        u_value_phn, u_index_phn, u_counts_phn = torch.unique(_label_id, return_inverse=True, return_counts=True)
        
        # calculate a center for each phn
        center_f_phn = torch.zeros([len(u_value_phn), f_c]).to(_features.device)
        center_f_phn.index_add_(0, u_index_phn, _features)
        u_counts_phn = u_counts_phn.unsqueeze(1)
        center_f_phn = center_f_phn / u_counts_phn
        
        # calculate dist between phn-centers
        p_phn = F.normalize(center_f_phn, dim=1)
        _distance_phn = euclidean_dist(p_phn, p_phn)
        _distance_phn = up_triu(_distance_phn)
        
        # calculate diverse term form phn
        u_value_phn = u_value_phn.unsqueeze(1)
        # assume a margin is 1
        _distance_phn = _distance_phn * self.margin
        ## L_d, diverse term, push away the distence between score-centers
        _entropy_phn = torch.mean(_distance_phn)
        _features = F.normalize(_features, dim=1)
        
        # calculate tightness term from phn
        # find phn-scnter for each features in the batch
        _features_center_phn = p_phn[u_index_phn, :]
        _features_phn = _features - _features_center_phn
        _features_phn = _features_phn.pow(2)
        _tightness_phn = torch.sum(_features_phn, dim=1)
        _mask = _tightness_phn > 0
        
        #come close to center while considering ordinal 
        _tightness_phn = _tightness_phn[_mask] * _label[_mask]
        _tightness_phn = torch.mean(_tightness_phn)
        
        loss_oe = (self.lambda_t_phn * _tightness_phn) - (self.lambda_d_phn * _entropy_phn)
        self.accum_features = None
        self.accum_labels = None

        return loss_oe

def euclidean_dist(x, y):
    """
    Args:
        x: pytorch Variable, with shape [m, d]
        y: pytorch Variable, with shape [n, d]
    Returns:
        dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def up_triu(x):
    # return a flattened view of up triangular elements of a square matrix
    n, m = x.shape
    assert n == m
    _tmp = torch.triu(torch.ones(n, n), diagonal=1).to(torch.bool)
    return x[_tmp]

class OrdinalRegressionLoss(nn.Module):

    def __init__(self, num_class=8, train_cutpoints=False, scale=20.0):
        super().__init__()
        self.num_classes = num_class
        num_cutpoints = self.num_classes - 1
        self.cutpoints = torch.arange(num_cutpoints).float()*scale/(num_class-2) - scale / 2
        self.cutpoints = nn.Parameter(self.cutpoints)
        if not train_cutpoints:
            self.cutpoints.requires_grad_(False)

    def forward(self, pred, label):
        # Note: for slate
        pred = pred.unsqueeze(-1)
        label = label.unsqueeze(-1)
        sigmoids = torch.sigmoid(self.cutpoints - pred)
        link_mat = sigmoids[:, 1:] - sigmoids[:, :-1]
        link_mat = torch.cat((
                sigmoids[:, [0]],
                link_mat,
                (1 - sigmoids[:, [-1]])
            ),
            dim=1
        )

        eps = 1e-15
        likelihoods = torch.clamp(link_mat, eps, 1 - eps)

        neg_log_likelihood = torch.log(likelihoods)
        if label is None:
            loss = 0
        else:
            loss = -torch.gather(neg_log_likelihood, 1, label).mean()
            
        return loss

def _reduction(loss, reduction):
    """
    Reduce loss

    Parameters
    ----------
    loss : torch.Tensor, [batch_size, num_classes]
        Batch losses.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.

    Returns
    -------
    loss : torch.Tensor
        Reduced loss.

    """
    if reduction == 'elementwise_mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f'{reduction} is not a valid reduction')


def cumulative_link_loss(y_pred, y_true,
                         reduction,
                         class_weights
                         ):
    """
    Calculates the negative log likelihood using the logistic cumulative link
    function.

    See "On the consistency of ordinal regression methods", Pedregosa et. al.
    for more details. While this paper is not the first to introduce this, it
    is the only one that I could find that was easily readable outside of
    paywalls.

    Parameters
    ----------
    y_pred : torch.Tensor, [batch_size, num_classes]
        Predicted target class probabilities. float dtype.
    y_true : torch.Tensor, [batch_size, 1]
        True target classes. long dtype.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.

    Returns
    -------
    loss: torch.Tensor

    """
    eps = 1e-15
    likelihoods = torch.clamp(torch.gather(y_pred, 1, y_true), eps, 1 - eps)
    neg_log_likelihood = -torch.log(likelihoods)

    if class_weights is not None:
        # Make sure it's on the same device as neg_log_likelihood
        class_weights = torch.as_tensor(class_weights,
                                        dtype=neg_log_likelihood.dtype,
                                        device=neg_log_likelihood.device)
        neg_log_likelihood *= class_weights[y_true]

    loss = _reduction(neg_log_likelihood, reduction)
    return loss


class CumulativeLinkLoss(nn.Module):
    """
    Module form of cumulative_link_loss() loss function

    Parameters
    ----------
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.

    """

    def __init__(self, reduction = 'elementwise_mean',
                 class_weights = None):
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, y_pred,
                y_true):
        return cumulative_link_loss(y_pred, y_true,
                                    reduction=self.reduction,
                                    class_weights=self.class_weights)

class CDW_CELoss(nn.Module):
    def __init__(self, alpha=5.0, margin=0.05, weight=None):
        """
        alpha: 控制類別距離懲罰的強度，值越大懲罰越嚴重
        margin: 加在預測機率上的 additive margin，鼓勵類別分離
        weight: Tensor，shape 為 (num_classes,)，每個類別的加權值，可用於處理類別不平衡
        """
        super(CDW_CELoss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.weight = weight  # Optional: Tensor of shape (num_classes,)

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        batch_size, num_classes = probs.size()
        device = logits.device

        # 增加 margin，避免機率超過 1
        adjusted_probs = torch.clamp(probs + self.margin, max=1.0)

        # 計算每個樣本對所有類別的距離
        class_indices = torch.arange(num_classes).unsqueeze(0).to(device)  # (1, C)
        target_indices = targets.view(-1, 1)  # (B, 1)
        distances = torch.abs(class_indices - target_indices)  # (B, C)
        penalties = distances.float().pow(self.alpha)  # (B, C)

        # 計算 log(1 - p)
        log_term = torch.log(1.0 - adjusted_probs.clamp(min=1e-6))  # avoid log(0)

        # 加權（若有提供 class weight）
        if self.weight is not None:
            class_weight = self.weight.view(1, -1).to(device)  # (1, C)
            penalties = penalties * class_weight  # (B, C)

        loss = - (log_term * penalties).sum(dim=1).mean()
        return loss                                    

class ComputeLoss(nn.Module):
    def __init__(self, loss_type, model_args):
        super(ComputeLoss, self).__init__()
        self.loss_type = loss_type
        self.problem_type = model_args["problem_type"]
        self.model_args = model_args

    def set_train_vector(self, tr_vector_mean, tr_vector_var):
        self.tr_vector_mean = tr_vector_mean
        self.tr_vector_var = tr_vector_var

    def forward(self, logits, labels, hidden_states):
        if self.problem_type == "regression" and self.loss_type == "mse":
            loss_fct = MSELoss()
            logits = logits.squeeze(-1)
            loss = loss_fct(logits, labels)
        elif self.problem_type == "single_label_classification" and self.loss_type == "ce":
            loss_fct = CrossEntropyLoss(weight=self.class_weight)
            labels = labels - 1
            loss = loss_fct(logits.view(-1, self.model_args.num_labels), labels.view(-1))
        elif self.problem_type == "multi_label_classification" and self.loss_type == "bce":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        elif self.loss_type == "test_time_adaptation":
            logits = logits.squeeze(-1)
            hidden_states_mean = torch.mean(hidden_states, dim=0)
            hidden_states_var = torch.var(hidden_states, dim=0)
                
            self.tr_vector_mean = self.tr_vector_mean.to(logits.device)
            self.tr_vector_var = self.tr_vector_var.to(logits.device)
                
            mean_loss = torch.mean(torch.pow(self.tr_vector_mean - hidden_states_mean, 2))
            var_loss = torch.mean(torch.pow(self.tr_vector_var - hidden_states_var, 2))
            loss = mean_loss + var_loss 
        
        return loss
