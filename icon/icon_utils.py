from tensorboardX import SummaryWriter
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from common.vision.transforms import ResizeImage
import torch.nn.functional as F
from icon.cluster import PairEnum
from icon.randaugment import rand_augment_transform

rgb_mean = (0.485, 0.456, 0.406)
ra_params = dict(translate_const=int(224 * 0.45), img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),)


class Visualizer():
    def __init__(self, root_dir, exp_name):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        log_dir = os.path.join(root_dir, exp_name)
        self.writer = SummaryWriter(log_dir)
        self.step = 0

    def plot_items(self, items):
        for name, value in items.items():
            self.writer.add_scalar(name, value, self.step)
    
    def tick(self):
        self.step += 1


class TwoViewsTrainTransform(object):
    def __init__(self, center_crop):
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        crop = T.CenterCrop(224) if center_crop else T.RandomResizedCrop(224)
        self.weak = T.Compose([
            ResizeImage(256),
            crop,
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
        self.strong = T.Compose([
            ResizeImage(256),
            crop,
            T.RandomHorizontalFlip(),
            T.RandomApply([
                T.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            T.ToTensor(),
            normalize,
        ])
        
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong


sim_list = []
def get_ulb_sim_matrix(mode, sim_matrix_ulb, cluster_preds_t, update_list=True):
    if mode == 'stats':
        return sim_matrix_ulb, 0, 0
    elif mode == 'argmax':
        y_c_t = cluster_preds_t.argmax(dim=1).contiguous().view(-1, 1)
        sim_matrix_ulb_full = torch.eq(y_c_t, y_c_t.T).float().to(cluster_preds_t.device)
        sim_matrix_ulb_full = (sim_matrix_ulb_full - 0.5) * 2
        sim_matrix_ulb_full = sim_matrix_ulb_full.flatten()
        return sim_matrix_ulb_full
    else:
        if mode == 'sim':
            feat_row, feat_col = PairEnum(F.normalize(cluster_preds_t, dim=1))
        elif mode == 'prob':
            feat_row, feat_col = PairEnum(F.softmax(cluster_preds_t, dim=1))
        tmp_distance_ori = torch.bmm(
            feat_row.view(feat_row.size(0), 1, -1),
            feat_col.view(feat_row.size(0), -1, 1)
        )
        sim_threshold = 0.92
        sim_ratio = 0.5 / 12
        diff_ratio = 5.5 / 12
        similarity = tmp_distance_ori.squeeze()
        if update_list:
            global sim_list
            sim_list.append(similarity)
            if len(sim_list) > 30:
                sim_list = sim_list[1:]
        sim_all = torch.cat(sim_list, dim=0)
        sim_all_sorted, _ = torch.sort(sim_all)

        n_diff = min(len(sim_all) * diff_ratio, len(sim_all)-1)
        n_sim = min(len(sim_all) * sim_ratio, len(sim_all))

        low_threshold = sim_all_sorted[int(n_diff)]
        high_threshold = max(sim_threshold, sim_all_sorted[-int(n_sim)])

        sim_matrix_ulb = torch.zeros_like(similarity).float()

        if high_threshold != low_threshold:
            sim_matrix_ulb[similarity >= high_threshold] = 1.0
            sim_matrix_ulb[similarity <= low_threshold] = -1.0
        else:
            sim_matrix_ulb[similarity > high_threshold] = 1.0
            sim_matrix_ulb[similarity < low_threshold] = -1.0
        return sim_matrix_ulb, low_threshold, high_threshold