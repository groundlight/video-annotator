# Quick clustering of images using a small CNN and k-means

from typing import List

from sklearn.cluster import KMeans
import numpy as np
import torch
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights


class QCluster:
    """Quick clustering of images using a small CNN and k-means."""

    def __init__(self):
        self.model = self._load_model()
        self.images = {}

    def _load_model(self):
        model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
        # Remove the last layer so we get a feature vector instead of a classification
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        return model

    def just_embed(self, image: np.ndarray) -> np.ndarray:
        """Embed images into a feature space."""
        # Convert image to tensor
        tensor = torch.from_numpy(image)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        tensor = tensor / 255.0  # Normalize to [0, 1]

        # Embed the image
        with torch.no_grad():
            embedding = self.model(tensor)
            # now embedding has a shape like [1, 1024, 19, 25]
            # we want to flatten it to [1024, 19*25]
            embedding = embedding.squeeze()
            embedding = embedding.view(embedding.shape[0], -1)
            # Now we want to average it over the 19x25
            embedding = embedding.mean(axis=1)
            # now it's just a 1024-dimensional vector
        return embedding.numpy()

    def add_image(self, image: np.ndarray, id: int) -> None:
        """Add an image to the set of images to be clustered."""
        embedding = self.just_embed(image)
        self.images[id] = embedding

    def cluster(self, k: int) -> List[List[int]]:
        """Cluster the images into k clusters."""
        features = np.array(list(self.images.values()))
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(features)
        label_ids = kmeans.labels_.tolist()
        # label_ids is numpy array with the cluster ids for each image.
        # we want to convert that to a list of lists of int ids
        clusters = [[] for _ in range(k)]
        for id, label in zip(self.images.keys(), label_ids):
            clusters[label].append(id)
        return clusters

    def diversity_order(self) -> List[int]:
        """Returns the images in an order designed to maximize diversity."""
        # First calculate a useful k
        N = len(self.images)
        k = int(N**0.5)
        clusters = self.cluster(k)
        # Now just go through the clusters round robin
        order = []
        while len(order) < N:
            found_any = False
            for i in range(k):
                if len(clusters[i]) > 0:
                    order.append(clusters[i].pop(0))
                    found_any = True
            if not found_any:
                print("Warning: didn't find any images to add to order. This shouldn't happen!")
                break

        return order
