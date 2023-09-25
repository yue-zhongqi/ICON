import torch
import matplotlib
import os
matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
# import umap

def visualize(source_feature: torch.Tensor, source_labels: torch.Tensor,
              target_feature: torch.Tensor, target_labels: torch.Tensor,
              filename: str, source_color='r', target_color='b'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    print("Transforming features by umap...")
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
    # X_tsne = umap.UMAP(n_components=2, metric='euclidean', n_neighbors=15).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=2)
    plt.savefig(filename)

def visualize_cluster(source_features, source_labels, source_clusters,
                    target_features, target_labels, target_clusters, file_root,
                    num_s=None, num_t=None, umap=True, metric='euclidean'):
    source_features = source_features.numpy()
    target_features = target_features.numpy()
    source_labels = source_labels.numpy()
    target_labels = target_labels.numpy()
    source_clusters = source_clusters.numpy()
    target_clusters = target_clusters.numpy()
    num_classes = len(np.unique(source_labels))
    num_source = source_features.shape[0]
    num_target = target_features.shape[0]

    # select features
    if num_s is not None and num_s < num_source:
        source_tsne_idx = np.random.choice(num_source, num_s, replace=False)
        source_features = source_features[source_tsne_idx, :]
        source_labels = source_labels[source_tsne_idx]
        source_clusters = source_clusters[source_tsne_idx]
        num_source = num_s
    if num_t is not None and num_t < num_target:
        target_tsne_idx = np.random.choice(num_target, num_t, replace=False)
        target_features = target_features[target_tsne_idx, :]
        target_labels = target_labels[target_tsne_idx]
        target_clusters = target_clusters[target_tsne_idx]
        num_target = num_t
    features = np.concatenate([source_features, target_features], axis=0)

    # map features to 2-d using TSNE
    if umap:
        print("Transforming features by umap...")
        assert False
        # X_tsne = umap.UMAP(n_components=2, metric='euclidean', n_neighbors=15).fit_transform(features)
    else:
        print("Transforming features by tsne...")
        X_tsne = TSNE(
            n_components=2, random_state=33, metric=metric
        ).fit_transform(features)

    source_tsne = X_tsne[:num_source, :]
    target_tsne = X_tsne[num_source:, :]
    
    # draw
    plt.figure(figsize=(15, 15))
    colors = cm.rainbow(np.linspace(0, 1, num_classes))
    for i in range(num_classes):
        # draw the anchor class with color
        source_mask = (source_labels == i)
        target_mask = (target_labels == i)
        source_mask_neg = ~source_mask
        target_mask_neg = ~target_mask

        # draw other class as grey
        plt.scatter(
            source_tsne[source_mask_neg][:, 0], source_tsne[source_mask_neg][:, 1],
            c="grey", s=7, marker='o'
        )
        plt.scatter(
            target_tsne[target_mask_neg][:, 0], target_tsne[target_mask_neg][:, 1],
            c="grey", s=12, marker='x'
        )

        # draw anchor class as color
        for j in range(num_classes):
            source_cluster_mask = (source_clusters == j)
            target_cluster_mask = (target_clusters == j)
            source_combined_mask = source_mask & source_cluster_mask
            target_combined_mask = target_mask & target_cluster_mask
            plt.scatter(
                source_tsne[source_combined_mask][:, 0], source_tsne[source_combined_mask][:, 1],
                color=colors[j], s=14, marker='o', alpha=0.7
            )
            plt.scatter(
                target_tsne[target_combined_mask][:, 0], target_tsne[target_combined_mask][:, 1],
                color=colors[j], s=21, marker='x', alpha=0.7
            )

        plt.savefig(os.path.join(file_root, "class_%d.png" % i))
        plt.clf()
    
    # Draw source vs target domain
    plt.scatter(source_tsne[:, 0], source_tsne[:, 1], color='red', s=2)
    plt.scatter(target_tsne[:, 0], target_tsne[:, 1], color='blue', s=2)
    plt.savefig(os.path.join(file_root, "overall.png"))