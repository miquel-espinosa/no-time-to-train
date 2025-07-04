
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from copy import deepcopy

import faiss
import faiss.contrib.torch_utils



def array_to_tensor(
    array: np.ndarray, make_array_writeable: bool = True
) -> torch.Tensor:
    # Reference: https://github.com/facebookresearch/foundpose/
    if not array.flags.writeable:
        if make_array_writeable and array.flags.owndata:
            array.setflags(write=True)
        else:
            array = np.array(array)
    return torch.from_numpy(array)


class KNN:
    """K nearest neighbor search.

    References:
    [1] towardsdatascience.com/make-knn-300-times-faster-than-scikit-learns-in-20-lines-5e29d74e76bb
    [2] https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU
    """

    def __init__(
        self,
        k: int = 1,
        metric: str = "l2",
        radius: Optional[float] = None,
        res: Optional[Any] = None,
    ) -> None:
        """
        Args:
            k: The number of nearest neighbors to return.
            metric: The distance metric to use. Can be "l2" or "cosine".
        """

        self.index: Any = None
        self.k: int = k
        self.metric: str = metric
        self.radius: Optional[float] = radius
        self.res: Optional[Any] = res

    def fit(self, data: torch.Tensor) -> None:
        """Creates index from provided vectors.

        Args:
            X: (num_vectors, dimensionality)
        """

        dimensions = data.shape[1]

        if self.metric == "l2":
            self.index = faiss.IndexFlatL2(dimensions)
            if data.is_cuda:
                data = data.cpu()
            self.index.add(data)

        elif self.metric == "cosine":
            self.index = faiss.IndexFlatIP(dimensions)

            # Normalization.
            data = data / torch.linalg.norm(data, dim=1, keepdim=True)

            self.index.train(data)
            self.index.add(data)

        else:
            raise ValueError(f"Metric {self.metric} is not supported.")

    def search(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finds nearest neighbors.

        Args:
            X: (num_vectors, dimensionality)
        Returns:
            Distances and indices of the k nearest neighbors.
        """

        is_cuda = False
        device=None
        if data.is_cuda:
            device = data.get_device()
            is_cuda = True
            data = data.cpu()

        if self.metric == "l2":
            if self.radius is None:
                distances, indices = self.index.search(data, k=self.k)
            else:
                # Convert radii to float type for faster computation.
                distances, indices = self.index.range_search_with_radius(
                    data, float(self.radius)
                )

        elif self.metric == "cosine":

            # Normalize the query vectors.
            data = data / torch.linalg.norm(data, dim=1, keepdim=True)

            similarity, indices = self.index.search(data, k=self.k)

            # Cosine similarity to cosine distance.
            distances = 1.0 - similarity
        else:
            raise ValueError(f"Metric {self.metric} is not supported.")

        if is_cuda:
            distances = distances.to(device)
            indices = indices.to(device)

        return distances, indices

    def serialize_index(self) -> None:
        self.index = faiss.serialize_index(self.index)

    def deserialize_index(self) -> None:
        self.index = faiss.deserialize_index(self.index)



def compute_foundpose(feats: torch.Tensor, masks: torch.Tensor, k_kmeans: int, n_pca: int, k_assign: int):
    # Reference: https://github.com/facebookresearch/foundpose/

    device = feats.device
    n_class, n_shot = feats.shape[:2]
    mem_dim = feats.shape[-1]

    feats = feats.reshape(n_class * n_shot, -1, mem_dim)
    masks = masks.reshape(n_class * n_shot, -1)

    fore_feats = []
    class_inds = []
    for i in range(feats.shape[0]):
        _feats = feats[i][masks[i] > 0]
        fore_feats.append(_feats)
        class_inds.append(
            torch.zeros((_feats.shape[0],), dtype=torch.float32, device=device) + float(int(i // n_shot))
        )
    fore_feats = torch.cat(fore_feats, dim=0)
    class_inds = torch.cat(class_inds, dim=0)

    # PCA
    fore_feats_np = fore_feats.cpu().numpy()
    pca = PCA(n_components=n_pca)
    pca.fit(fore_feats_np)
    pca_mean = torch.from_numpy(pca.mean_).to(device=device)
    pca_components = torch.from_numpy(pca.components_).to(device=device)
    fore_feats = (fore_feats - pca_mean.reshape(1, -1)) @ pca_components.t()

    # K-means
    fore_feats_cpu = fore_feats.cpu()
    kmeans = faiss.Kmeans(
        n_pca,
        k_kmeans,
        niter=100,
        gpu=False,
        verbose=True,
        seed=0,
        spherical=False,
    )
    kmeans.train(fore_feats_cpu)

    feats_centroids = array_to_tensor(kmeans.centroids).to(device)
    centroid_distances, cluster_ids = kmeans.index.search(fore_feats_cpu, 1)
    centroid_distances = centroid_distances.squeeze(axis=-1).to(device=device)
    cluster_ids = cluster_ids.squeeze(axis=-1).to(device=device)

    # TF-IDF
    word_occurances = torch.zeros(k_kmeans, dtype=torch.float32, device=device)
    for i in range(n_class):
        unique_word_ids = torch.unique(cluster_ids[class_inds == i])
        word_occurances[unique_word_ids] += 1.0
    idfs = torch.log(torch.as_tensor(float(n_class)) / word_occurances)

    feat_knn_index = KNN(k=k_assign, metric="l2")
    feat_knn_index.fit(feats_centroids.cpu())

    tfidf_descs = []
    for i in range(n_class):
        word_dists, word_ids = feat_knn_index.search(fore_feats[class_inds == i])






        tfidf = calc_tfidf(
            feature_word_ids=word_ids,
            feature_word_dists=word_dists,
            word_idfs=word_idfs,
            soft_assignment=tfidf_soft_assign,
            soft_sigma_squared=tfidf_soft_sigma_squared,
        )
        tfidf_descs.append(tfidf)
    tfidf_descs = torch.stack(tfidf_descs, dim=0)