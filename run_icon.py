import random
import time
import argparse
import shutil
import os
import torch
import numpy as np
from sam import SAM

from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import common.vision.datasets as datasets
import common.vision.models as models
from common.utils.analysis import collect_feature
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from validate import validate_model

from icon.uda_backbone import ImageClassifier
from icon.cluster import ClusterLoss, Normalize, BCE, PairEnum, reduce_dimension
from icon.icon_utils import Visualizer, TwoViewsTrainTransform, get_ulb_sim_matrix
from icon.eqinv import EqInv
from icon.entropy import TsallisEntropy
from icon.transform import TransformFixMatch, get_val_trainsform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    args.log = os.path.join(args.log_root, args.exp_name)
    logger = CompleteLogger(args.log, 'train')
    print(args)

    if args.seed is not None:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(2023)
        random.seed(2023)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # Data loading code
    train_transform = TwoViewsTrainTransform(args.center_crop)
    unlabeled_transform = TransformFixMatch()
    val_transform = get_val_trainsform()

    dataset = datasets.__dict__[args.data]
    train_source_dataset = dataset(root=args.root, task=args.source, download=True, transform=train_transform)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=False)
    train_target_dataset = dataset(root=args.root, task=args.target, download=True, transform=unlabeled_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    source_seq_loader = DataLoader(train_source_dataset, batch_size=args.batch_size * 8,
                                     shuffle=False, num_workers=args.workers * 8, drop_last=False)      # for dim reduction
    target_seq_loader = DataLoader(train_target_dataset, batch_size=args.batch_size * 8,
                                     shuffle=False, num_workers=args.workers * 8, drop_last=False)      # for dim reduction
    test_loader = val_loader
    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model and loss
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    num_classes = train_source_dataset.num_classes
    args.num_cls = num_classes
    classifier = ImageClassifier(
        backbone, num_classes, bottleneck_dim=args.bottleneck_dim
    ).to(device)

    # define optimizer and lr scheduler
    base_optimizer = SGD
    if args.optim == 'sam':
        optimizer = SAM(
            classifier.get_parameters(), base_optimizer, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay,
            adaptive = True, rho = args.rho
        )
    else:
        optimizer = SGD(
            classifier.get_parameters(), lr=args.lr,
            weight_decay=args.weight_decay, momentum=args.momentum
        )
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    ts_loss = TsallisEntropy(temperature=args.temperature, alpha=args.alpha)    # following CST
    cluster_loss = ClusterLoss(device, num_classes, "RK", -1, args.topk)
    disentanglement_loss = EqInv(num_classes, temperature=0.07, memory_length_per_class=500)

    # Visualizer
    visualizer = Visualizer(args.log_root, args.exp_name)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        print("lr:", lr_scheduler.get_last_lr()[0])
        # train for one epoch
        train(
            train_source_iter, train_target_iter, classifier, ts_loss, optimizer,
              lr_scheduler, epoch, args, cluster_loss, disentanglement_loss, visualizer,
              loaders={"s_seq": source_seq_loader, "t_seq": target_seq_loader,
                "s": train_source_loader, "t": train_target_loader}
        )
        # evaluate on validation set
        acc1 = validate_model(val_loader, val_source_loader, classifier, args, device, str(epoch))
        visualizer.plot_items({"target accuracy": acc1,})
        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)
    print("best_acc1 = {:3.1f}".format(best_acc1))
    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate_model(test_loader, val_source_loader, classifier, args, device, "best")
    print("test_acc1 = {:3.1f}".format(acc1))
    logger.close()
    return best_acc1, logger

def train(train_source_iter: ForeverDataIterator, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, ts: TsallisEntropy, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace,
          cluster_loss: ClusterLoss, disentangle_loss: EqInv,
          visualizer:Visualizer, loaders=None):
    batch_time = AverageMeter('Time', ':3.1f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cluster_losses = AverageMeter('Cluster Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    bce = BCE()
    
    w_cluster = 1.0
    w_eqinv = 1.0 if args.eqinv else 0.0
    w_transfer = args.w_transfer
    w_erm = 1.0
    w_con = 1.0 if epoch >= args.con_start_epoch else 0.0
    w_inv = args.w_inv if epoch >= args.inv_start_epoch else 0.0
    w_st = args.w_st
    w_st2 = 0.5
    back_cluster = True if epoch >= args.back_cluster_start_epoch else False

    print("EqInv=%.2f, ERM=%.2f, Self training=%.2f, Others=t%.2f-c%.2f-ba%d"
        % (w_eqinv, w_erm, w_st, w_transfer, w_con, int(back_cluster)))

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cluster_losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # dimensionality reduction
    use_dim_reduce = (args.dim_reduction != 'none')
    if use_dim_reduce:
        s_loader = loaders["s_seq"]
        t_loader = loaders["t_seq"]
        source_feature1, _, labels_s = collect_feature(s_loader, model.backbone, device, None)
        target_feature1, _, labels_t = collect_feature(t_loader, model.backbone, device, None)
        num_s = len(source_feature1)
        features = torch.cat((source_feature1, target_feature1), dim=0).cpu().numpy()
        transformed_features, _ = reduce_dimension(features, args.dim_reduction, args.reduced_dim)
        tf_s = transformed_features[:num_s, :]
        tf_t = transformed_features[num_s:, :]

    # switch to train mode
    model.train()
    l2norm = Normalize(2)
    end = time.time()
    for i in range(args.iters_per_epoch):
        current_iters = epoch * args.iters_per_epoch + i
        (x_s, x_s_u), labels_s, meta_s = next(train_source_iter)
        (x_t, x_t_u), labels_t, meta_t = next(train_target_iter)

        x_s_u = x_s_u.to(device)
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        x_t_u = x_t_u.to(device)
        labels_s = labels_s.to(device)
        labels_t = labels_t.to(device)
        labels_t_scrambled = torch.ones_like(labels_t).to(device) + cluster_loss.num_classes

        # measure data loading time
        data_time.update(time.time() - end)

        ######### 1. compute output
        x = torch.cat((x_s, x_t), dim=0)
        x_u = torch.cat((x_s_u, x_t_u), dim=0)
        o = model(x)
        o_u = model(x_u)

        ######### 2. process output
        y, y_alt, y_nograd, y_alt_nograd =\
            o["y"], o["y_cluster_all"], o["y_nograd"], o["y_cluster_all_nograd"]
        f, bf = o["feature"], o["bottleneck_feature"]
        y_u, y_u_alt, y_u_nograd, y_u_alt_nograd =\
            o_u["y"], o_u["y_cluster_all"], o_u["y_nograd"], o_u["y_cluster_all_nograd"]
        f_u, bf_u = o_u["feature"], o_u["bottleneck_feature"]
        f_s, f_t = f.chunk(2, dim=0)    # Weak Aug: label features, unlabel features
        f_s_u, f_t_u = f_u.chunk(2, dim=0)  # Strong Aug: label features, unlabel features
        y_s, y_t = y.chunk(2, dim=0)        # Weak Aug, cls head: label preds, unlabel preds
        y_s_u, y_t_u = y_u.chunk(2, dim=0)  # Strong Aug, cls head: label preds, unlabel preds
        y_s_alt, y_t_alt = y_alt.chunk(2, dim=0)    # Weak Aug, eqinv head: label preds, unlabel preds
        y_s_u_alt, y_t_u_alt = y_u_alt.chunk(2, dim=0)  # Strong Aug, eqinv head: label preds, unlabel preds
        bf_s, bf_t = bf.chunk(2, dim=0)         # Weak Aug: label bottleneck features, unlabel bottleneck features
        bf_s_u, bf_t_u = bf_u.chunk(2, dim=0)   # Strong Aug: label bottleneck features, unlabel bottleneck features
        # Nograd outputs
        y_s_nograd, y_t_nograd = y_nograd.chunk(2, dim=0)        # Weak Aug, cls head: label preds, unlabel preds (nograd)
        y_s_u_nograd, y_t_u_nograd = y_u_nograd.chunk(2, dim=0)  # Strong Aug, cls head: label preds, unlabel preds (nograd)
        y_s_alt_nograd, y_t_alt_nograd = y_alt_nograd.chunk(2, dim=0)    # Weak Aug, eqinv head: label preds, unlabel preds (nograd)
        y_s_u_alt_nograd, y_t_u_alt_nograd = y_u_alt_nograd.chunk(2, dim=0)  # Strong Aug, eqinv head: label preds, unlabel preds (nograd)
        # dimension reduction
        if use_dim_reduce:
            idx_s = meta_s['index']
            idx_t = meta_t['index']
            f_s_reduce = torch.from_numpy(tf_s[idx_s, :]).to(device)
            f_t_reduce = torch.from_numpy(tf_t[idx_t, :]).to(device)
            f_s_cluster = f_s_reduce
            f_t_cluster = f_t_reduce
        else:
            f_s_cluster = f_s
            f_t_cluster = f_t
        # U only cluster outputs
        y_alt2, y_alt2_nograd = o["y_cluster_u"], o["y_cluster_u_nograd"]
        y_u_alt2, y_u_alt2_nograd = o_u["y_cluster_u"], o_u["y_cluster_u_nograd"]
        y_s_alt2, y_t_alt2 = y_alt2.chunk(2, dim=0)     # Weak Aug, cluster head: label preds, unlabel preds
        y_s_u_alt2, y_t_u_alt2 = y_u_alt2.chunk(2, dim=0)   # Strong Aug, cluster head: label preds, unlabel preds
        y_s_alt2_nograd, _ = y_alt2_nograd.chunk(2, dim=0)  # Weak Aug, cluster head: label preds, unlabel preds (nograd)
        y_s_u_alt2_nograd, _ = y_u_alt2_nograd.chunk(2, dim=0)  # Strong Aug, cluster head: label preds, unlabel preds (nograd)

        ######### 3. generate target pseudo-labels
        max_prob, pseudo_labels = torch.max(F.softmax(y_t, dim=-1), dim=-1)
        max_prob_alt, _ = torch.max(F.softmax(y_t_alt, dim=-1), dim=-1)

        ######### ERM loss
        cls_loss = F.cross_entropy(y_s, labels_s)

        ######### self training loss
        st_loss = (F.cross_entropy(y_t_u, pseudo_labels, reduction='none')
                   * max_prob.ge(args.threshold).float().detach()).mean()

        ########## Entropy Loss
        transfer_loss = ts(y_t)

        ######### EqInv loss
        preds1_u = torch.cat((y_s_alt_nograd, y_t_alt_nograd), dim=0)
        preds2_u = torch.cat((y_s_u_alt_nograd, y_t_u_alt_nograd), dim=0)
        inputs = {
            "x1": torch.cat((f_s_cluster, f_t_cluster), dim=0),
            "preds1_u": preds1_u,
            "preds2_u": preds2_u,
            "labels": torch.cat((labels_s, labels_t_scrambled), dim=0),
        }
        bce_loss, _ = cluster_loss.compute_losses(inputs)   # Cluster loss (for EqInv cluster head if applicable)
        clusters_s_prob = F.softmax(y_s_alt, dim=-1)
        # NOTE: NEED TO SPECIFY update_memory=False IN THE SECOND BACKWARD
        if args.eqinv:
            eqinv_loss, eqinv_loss2 = disentangle_loss(
                x_s=torch.cat((bf_s.unsqueeze(1), bf_s_u.unsqueeze(1)), dim=1),
                labels_s=labels_s,
                clusters_s=clusters_s_prob,
                x_t=torch.cat((bf_t.unsqueeze(1), bf_t_u.unsqueeze(1)), dim=1),
                update_memory=True
            )
        else:
            eqinv_loss, eqinv_loss2 = 0.0, 0.0

        ######### ICON
        p_t_nograd = F.softmax(y_t_nograd, dim=1)
        p_t_u_nograd = F.softmax(y_t_u_nograd, dim=1)
        p_s_alt_nograd = F.softmax(y_s_alt2_nograd, dim=1)
        p_s_u_alt_nograd = F.softmax(y_s_u_alt2_nograd, dim=1)
        inputs = {
            "x1": torch.cat((f_s_cluster, f_t_cluster), dim=0),
            "preds1_u": y_alt2 if back_cluster else y_alt2_nograd,
            "preds2_u": y_u_alt2 if back_cluster else y_u_alt2_nograd,
            "labels": torch.cat((labels_s, labels_t_scrambled), dim=0),
        }
        ######## Cluster loss (for target domain clustering)
        bce_loss_u, sim_matrix_ulb = cluster_loss.compute_losses(inputs)
        bce_loss += bce_loss_u
        max_prob_alt2, pseudo_labels_alt2 = torch.max(F.softmax(y_t_alt2, dim=1), dim=-1)
        st_loss_cluster = (F.cross_entropy(y_t_u_alt2, pseudo_labels_alt2,
                            reduction='none') * max_prob_alt2.ge(args.threshold).float().detach()).mean()
        # Refine unlabel similarity matrix (filter out uncertain pairs)
        cluster_logits = y_t_alt2
        sim_matrix_ulb_refined, low_t, high_t = get_ulb_sim_matrix(
            args.con_mode, sim_matrix_ulb, cluster_logits,
        )
        # classification head consistent with u clusters
        pairs1, _ = PairEnum(p_t_nograd)
        _, pairs2 = PairEnum(p_t_u_nograd)
        con_loss_u = bce(pairs1, pairs2, sim_matrix_ulb_refined)
        # cluster head consistent with s labels (to improve clustering)
        labels_s_view = labels_s.contiguous().view(-1, 1)
        sim_matrix_lb = torch.eq(labels_s_view, labels_s_view.T).float().to(device)
        sim_matrix_lb = (sim_matrix_lb - 0.5) * 2.0  # same label=1.0, diff label=-1.0
        pairs1, _ = PairEnum(p_s_alt_nograd)
        _, pairs2 = PairEnum(p_s_u_alt_nograd)
        con_loss_s = bce(pairs1, pairs2, sim_matrix_lb.flatten())

        ########### Consistency loss
        con_loss = con_loss_u + con_loss_s

        ########### Invariant loss
        sim_matrix_ulb_full, _, _ = get_ulb_sim_matrix(
            'stats', sim_matrix_ulb, cluster_logits, update_list=(args.con_mode=='stats')
        )   # get full ulb pairwise labels for invariant loss
        p_t, p_t_u = F.softmax(y_t, dim=1), F.softmax(y_t_u, dim=1)
        pairs1, _ = PairEnum(p_t)
        _, pairs2 = PairEnum(p_t_u)
        irm_con_t = bce(pairs1, pairs2, sim_matrix_ulb_full)
        p_s, p_s_u = F.softmax(y_s, dim=1), F.softmax(y_s_u, dim=1)
        pairs1, _ = PairEnum(p_s)
        _, pairs2 = PairEnum(p_s_u)
        irm_con_s = bce(pairs1, pairs2, sim_matrix_lb.flatten())
        inv_loss = torch.var(torch.stack([irm_con_t, irm_con_s]))
        
        loss = w_transfer * transfer_loss\
            + w_cluster * bce_loss\
            + w_eqinv * (eqinv_loss + eqinv_loss2)\
            + w_erm * cls_loss\
            + w_con * con_loss\
            + w_st * st_loss\
            + w_st2 * st_loss_cluster\
            + w_inv * inv_loss

        cls_acc = accuracy(y_s, labels_s)[0]

        losses.update(loss.item(), x_s.size(0))
        cluster_losses.update(bce_loss.item(), x_s.size(0))
        cls_accs.update(cls_acc.item(), x_s.size(0))

        loss_dict = {
            "total loss": loss,
            "transfer loss": transfer_loss,
            "bce loss": bce_loss,
            "erm loss": cls_loss,
            "consistency loss": con_loss,
            "self-training loss": st_loss,
            "confident ratio": max_prob.ge(args.threshold).float().mean(),
            "inv loss": inv_loss
        }
        
        visualizer.plot_items(loss_dict)
        visualizer.tick()

        # compute gradient and do the first SGD step
        loss.backward()
        if args.optim == 'sam':
            optimizer.first_step(zero_grad=True)
        else:
            optimizer.step()
            optimizer.zero_grad()

        lr_scheduler.step()

        if args.optim == 'sam':
            # compute gradient and do the second SGD step, code same with 1st step
            o = model(x)
            o_u = model(x_u)
            y, y_alt, y_nograd, y_alt_nograd =\
                o["y"], o["y_cluster_all"], o["y_nograd"], o["y_cluster_all_nograd"]
            f, bf = o["feature"], o["bottleneck_feature"]
            y_u, y_u_alt, y_u_nograd, y_u_alt_nograd =\
                o_u["y"], o_u["y_cluster_all"], o_u["y_nograd"], o_u["y_cluster_all_nograd"]
            f_u, bf_u = o_u["feature"], o_u["bottleneck_feature"]

            f_s, f_t = f.chunk(2, dim=0)    # Weak Aug: label features, unlabel features
            f_s_u, f_t_u = f_u.chunk(2, dim=0)  # Strong Aug: label features, unlabel features
            y_s, y_t = y.chunk(2, dim=0)        # Weak Aug, source head: label preds, unlabel preds
            y_s_u, y_t_u = y_u.chunk(2, dim=0)  # Strong Aug, source head: label preds, unlabel preds
            y_s_alt, y_t_alt = y_alt.chunk(2, dim=0)    # Weak Aug, target head: label preds, unlabel preds
            y_s_u_alt, y_t_u_alt = y_u_alt.chunk(2, dim=0)  # Strong Aug, target head: label preds, unlabel preds
            bf_s, bf_t = bf.chunk(2, dim=0)         # Weak Aug: label bottleneck features, unlabel bottleneck features
            bf_s_u, bf_t_u = bf_u.chunk(2, dim=0)   # Strong Aug: label bottleneck features, unlabel bottleneck features
            # Nograd outputs
            y_s_nograd, y_t_nograd = y_nograd.chunk(2, dim=0)        # Weak Aug, source head: label preds, unlabel preds
            y_s_u_nograd, y_t_u_nograd = y_u_nograd.chunk(2, dim=0)  # Strong Aug, source head: label preds, unlabel preds
            y_s_alt_nograd, y_t_alt_nograd = y_alt_nograd.chunk(2, dim=0)    # Weak Aug, target head: label preds, unlabel preds
            y_s_u_alt_nograd, y_t_u_alt_nograd = y_u_alt_nograd.chunk(2, dim=0)  # Strong Aug, target head: label preds, unlabel preds
            # dimension reduction
            if use_dim_reduce:
                idx_s = meta_s['index']
                idx_t = meta_t['index']
                f_s_reduce = torch.from_numpy(tf_s[idx_s, :]).to(device)
                f_t_reduce = torch.from_numpy(tf_t[idx_t, :]).to(device)
                f_s_cluster = f_s_reduce
                f_t_cluster = f_t_reduce
            else:
                f_s_cluster = f_s
                f_t_cluster = f_t
            # U only cluster outputs
            y_alt2, y_alt2_nograd = o["y_cluster_u"], o["y_cluster_u_nograd"]
            y_u_alt2, y_u_alt2_nograd = o_u["y_cluster_u"], o_u["y_cluster_u_nograd"]
            y_s_alt2, y_t_alt2 = y_alt2.chunk(2, dim=0)
            y_s_u_alt2, y_t_u_alt2 = y_u_alt2.chunk(2, dim=0)
            y_s_alt2_nograd, _ = y_alt2_nograd.chunk(2, dim=0)
            y_s_u_alt2_nograd, _ = y_u_alt2_nograd.chunk(2, dim=0)

            max_prob, pseudo_labels = torch.max(F.softmax(y_t, dim=-1), dim=-1)
            max_prob_alt, _ = torch.max(F.softmax(y_t_alt, dim=-1), dim=-1)
            # CE + self training + entropy
            cls_loss = F.cross_entropy(y_s, labels_s)
            st_loss = (F.cross_entropy(y_t_u, pseudo_labels,
                            reduction='none') * max_prob.ge(args.threshold).float().detach()).mean()
            transfer_loss = ts(y_t)
            # Eqinv
            preds1_u = torch.cat((y_s_alt_nograd, y_t_alt_nograd), dim=0)
            preds2_u = torch.cat((y_s_u_alt_nograd, y_t_u_alt_nograd), dim=0)
            inputs = {
                "x1": torch.cat((f_s_cluster, f_t_cluster), dim=0),
                "preds1_u": preds1_u,
                "preds2_u": preds2_u,
                "labels": torch.cat((labels_s, labels_t_scrambled), dim=0),
            }
            bce_loss, _ = cluster_loss.compute_losses(inputs)
            clusters_s_prob = F.softmax(y_s_alt, dim=-1)
            # NOTE: NEED TO SPECIFY update_memory=False IN THE SECOND BACKWARD
            if args.eqinv:
                eqinv_loss, eqinv_loss2 = disentangle_loss(
                    x_s=torch.cat((bf_s.unsqueeze(1), bf_s_u.unsqueeze(1)), dim=1),
                    labels_s=labels_s,
                    clusters_s=clusters_s_prob,
                    x_t=torch.cat((bf_t.unsqueeze(1), bf_t_u.unsqueeze(1)), dim=1),
                    update_memory=False
                )
            else:
                eqinv_loss, eqinv_loss2 = 0.0, 0.0
            # ICON
            p_t_nograd = F.softmax(y_t_nograd, dim=1)
            p_t_u_nograd = F.softmax(y_t_u_nograd, dim=1)
            p_s_alt_nograd = F.softmax(y_s_alt2_nograd, dim=1)
            p_s_u_alt_nograd = F.softmax(y_s_u_alt2_nograd, dim=1)
            inputs = {
                "x1": torch.cat((f_s_cluster, f_t_cluster), dim=0),
                "preds1_u": y_alt2 if back_cluster else y_alt2_nograd,
                "preds2_u": y_u_alt2 if back_cluster else y_u_alt2_nograd,
                "labels": torch.cat((labels_s, labels_t_scrambled), dim=0),
            }
            ######## Cluster loss (for target domain clustering)
            bce_loss_u, sim_matrix_ulb = cluster_loss.compute_losses(inputs)
            bce_loss += bce_loss_u
            max_prob_alt2, pseudo_labels_alt2 = torch.max(F.softmax(y_t_alt2, dim=1), dim=-1)
            st_loss_cluster = (F.cross_entropy(y_t_u_alt2, pseudo_labels_alt2,
                            reduction='none') * max_prob_alt2.ge(args.threshold).float().detach()).mean()
            # Refine unlabel similarity matrix (filter out uncertain pairs)
            cluster_logits = y_t_alt2
            sim_matrix_ulb_refined, low_t, high_t = get_ulb_sim_matrix(
                args.con_mode, sim_matrix_ulb, cluster_logits,
            )
            # classification head consistent with u clusters
            pairs1, _ = PairEnum(p_t_nograd)
            _, pairs2 = PairEnum(p_t_u_nograd)
            con_loss_u = bce(pairs1, pairs2, sim_matrix_ulb_refined)
            # cluster head consistent with s labels (to improve clustering)
            labels_s_view = labels_s.contiguous().view(-1, 1)
            sim_matrix_lb = torch.eq(labels_s_view, labels_s_view.T).float().to(device)
            sim_matrix_lb = (sim_matrix_lb - 0.5) * 2.0  # same label=1.0, diff label=-1.0
            pairs1, _ = PairEnum(p_s_alt_nograd)
            _, pairs2 = PairEnum(p_s_u_alt_nograd)
            con_loss_s = bce(pairs1, pairs2, sim_matrix_lb.flatten())

            ########### Consistency loss
            con_loss = con_loss_u + con_loss_s

            ########### Invariant loss
            sim_matrix_ulb_full, _, _ = get_ulb_sim_matrix(
                'stats', sim_matrix_ulb, cluster_logits, update_list=(args.con_mode=='stats')
            )   # get full ulb pairwise labels for invariant loss
            p_t, p_t_u = F.softmax(y_t, dim=1), F.softmax(y_t_u, dim=1)
            pairs1, _ = PairEnum(p_t)
            _, pairs2 = PairEnum(p_t_u)
            irm_con_t = bce(pairs1, pairs2, sim_matrix_ulb_full)
            p_s, p_s_u = F.softmax(y_s, dim=1), F.softmax(y_s_u, dim=1)
            pairs1, _ = PairEnum(p_s)
            _, pairs2 = PairEnum(p_s_u)
            irm_con_s = bce(pairs1, pairs2, sim_matrix_lb.flatten())
            inv_loss = torch.var(torch.stack([irm_con_t, irm_con_s]))
            
            loss1 = w_transfer * transfer_loss\
                + w_cluster * bce_loss\
                + w_eqinv * (eqinv_loss + eqinv_loss2)\
                + w_erm * cls_loss\
                + w_con * con_loss\
                + w_st * st_loss\
                + w_st2 * st_loss_cluster\
                + w_inv * inv_loss
            loss1.backward()
            optimizer.second_step(zero_grad=True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='ICON for Unsupervised Domain Adaptation')

    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--center-crop', default=False, action='store_true',
                        help='whether use center crop during training')
    
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    parser.add_argument('--temperature', default=2.0, type=float, help='parameter temperature scaling for entropy')
    parser.add_argument('--alpha', default= 1.9, type=float,
                        help='the entropic index of Tsallis loss')
    parser.add_argument('--threshold', default=0.97, type=float)
    parser.add_argument('--rho', default=0.5, type=float,
                     help='optimizer rho',
                    dest='rho')
    parser.add_argument("--load", type=str, default='best', help="Loading epoch for analysis/test. If none, load nothing (using pre-trained backbone)")
    parser.add_argument("--analysis-model", type=str, default='', help="Loading epoch for analysis/test. If none, load nothing (using pre-trained backbone)")

    # training parameters
    parser.add_argument('-b', '--batch-size', default=24, type=int,
                        metavar='N',
                        help='mini-batch size (default: 28)')
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log-root", type=str, default='.log',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--exp-name", type=str, default='',
                        help="Experiment names.")

    # Self-training
    parser.add_argument('--w-transfer', default=1.0, type=float, help='weight of transfer loss')
    parser.add_argument('--w-st', default=1.0, type=float, help='weight of self training loss')
    
    # Clustering
    parser.add_argument('--back-cluster-start-epoch', default=100, type=int, help='starting epoch to back cluster loss')
    parser.add_argument('--topk', default=5, type=int, help='rank statistics threshold for clustering')
    parser.add_argument("--optim", type=str, default='sam', help="Optimizer type for training and analysis.")
    parser.add_argument('--eqinv', action='store_true', help='Use eqinv')
    parser.add_argument('--dim-reduction', type=str, default='none', help='mode of dimension reduction for feature (used for clustering)')
    parser.add_argument('--reduced-dim', type=int, default=50, help='dim reduction dimension')

    # Consistency
    parser.add_argument('--con-start-epoch', default=100, type=int, help='starting epoch to use compat loss')
    parser.add_argument('--con-mode', type=str, default='stats', help='gt | stats | sim')

    # Invariance
    parser.add_argument('--w-inv', default=0.0, type=float, help='weight of compatibility irm loss')
    parser.add_argument('--inv-start-epoch', default=100, type=int, help='starting epoch to use compat loss')


    args = parser.parse_args()
    print("Running experiment %s." % args.exp_name)
    _, logger = main(args)
    logger.logger.close_terminal()