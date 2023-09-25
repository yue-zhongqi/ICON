import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        select_mask = (mask.sum(1) != 0)
        mask = mask[select_mask, :]
        log_prob = log_prob[select_mask, :]
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # filter out those with no positive samples
        mean_log_prob_pos=mean_log_prob_pos[mean_log_prob_pos==mean_log_prob_pos]  

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()
        loss = loss.mean()
        return loss

class EqInv(object):
    def __init__(self, num_classes, memory_length_per_class=50, temperature=0.07):
        self.memory_length = memory_length_per_class * num_classes
        self.num_classes = num_classes
        self.temperature = temperature
        self.labels_memory = None
        self.clusters_memory = None
        self.sup_con_loss = SupConLoss(temperature=temperature)

    def __call__(self, x_s, labels_s, clusters_s, x_t, update_memory=True):
        clusters_s = clusters_s.detach().cpu().numpy()
        labels_s = labels_s.detach().cpu().numpy()
        if update_memory:
            self.push_memory(labels_s, clusters_s)
        env1_mask, env2_mask = self.get_env_masks(labels_s, clusters_s)
        sup_con_loss_env1 = self.get_supcon_loss(x_s[env1_mask, :, :], labels_s[env1_mask], x_t)
        sup_con_loss_env2 = self.get_supcon_loss(x_s[env2_mask, :, :], labels_s[env2_mask], x_t)
        irm_loss = torch.var(torch.stack([sup_con_loss_env1, sup_con_loss_env2])) * 2.0
        return sup_con_loss_env1 + sup_con_loss_env2, irm_loss

    def push_memory(self, labels_s, clusters_s):
        clusters_s_hard = np.argmax(clusters_s, axis=1)
        if self.labels_memory is None:
            self.labels_memory = labels_s
            self.clusters_memory = clusters_s_hard
        elif len(self.labels_memory) + len(labels_s) > self.memory_length:
            self.labels_memory = np.append(self.labels_memory, labels_s)
            self.clusters_memory = np.append(self.clusters_memory, clusters_s_hard)
            self.labels_memory = self.labels_memory[-self.memory_length:]
            self.clusters_memory = self.clusters_memory[-self.memory_length:]
        else:
            self.labels_memory = np.append(self.labels_memory, labels_s)
            self.clusters_memory = np.append(self.clusters_memory, clusters_s_hard)

    def get_env_masks(self, labels_s, clusters_s):
        # dominant cluster index for each class
        dom_cluster_indices = [np.argmax(np.bincount(
            self.clusters_memory[self.labels_memory == i], minlength=self.num_classes))
            for i in range(self.num_classes)]
        dom_cluster_indices = np.array(dom_cluster_indices)
        scores = [clusters_s[i][dom_cluster_indices[labels_s[i]]] for i in range(len(labels_s))]
        scores = np.array(scores)

        ranks = np.argsort(scores)
        env1_mask = torch.zeros(len(labels_s)).bool()
        env1_mask[ranks[:len(labels_s) // 2]] = 1
        return env1_mask, ~env1_mask

    def balance_envs(self, env1_mask, env2_mask):
        if env1_mask.sum() > env2_mask.sum():
            larger_mask = env1_mask
        else:
            larger_mask = env2_mask
        num_selected = min(env1_mask.sum(), env2_mask.sum())
        larger_mask[larger_mask > 0][num_selected:] = 0
        return env1_mask, env2_mask

    def get_supcon_loss(self, x_s, labels_s, x_t):
        num_s = x_s.size(0)
        num_all = num_s + x_t.size(0)
        x = torch.cat((x_s, x_t), dim=0)

        # all target as negative samples
        # all target will not act as positive
        mask = torch.zeros((num_all, num_all)).float()
        labels = torch.from_numpy(labels_s).contiguous().view(-1, 1)
        mask_s = torch.eq(labels, labels.T).float()
        mask[:num_s, :num_s] = mask_s
        return self.sup_con_loss(features=F.normalize(x, dim=-1), mask=mask)