import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import tqdm


def collect_feature(data_loader: DataLoader, feature_extractor: nn.Module,
                                   device: torch.device, max_num_features=None) -> torch.Tensor:
    """
    Fetch data from `data_loader`, and then use `feature_extractor` to collect features

    Args:
        data_loader (torch.utils.data.DataLoader): Data loader.
        feature_extractor (torch.nn.Module): A feature extractor.
        device (torch.device)
        max_num_features (int): The max number of features to return

    Returns:
        Features in shape (min(len(data_loader), max_num_features), :math:`|\mathcal{F}|`).
    """
    feature_extractor.eval()
    all_features = []
    all_features2 = []
    all_labels = []
    with torch.no_grad():
        for i, ((images, images2), target, _) in enumerate(tqdm.tqdm(data_loader)):
            images = images.to(device)
            images2 = images2.to(device)
            feature = feature_extractor(images).cpu()
            feature2 = feature_extractor(images2).cpu()
            all_features.append(feature)
            all_features2.append(feature2)
            all_labels.append(target)
            if max_num_features is not None and i >= max_num_features:
                break
    return torch.cat(all_features, dim=0), torch.cat(all_features2, dim=0), torch.cat(all_labels, dim=0)