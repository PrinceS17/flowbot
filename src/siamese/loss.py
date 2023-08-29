import numpy as np
import torch
import torch.linalg as LA
import unittest


def clustering_loss(output, label, th_inter=1):
    """Clustering loss based on Silhouette coefficient.
    
    Given output (feature vector of flows), and the ground truth label,
    return the clustering loss. Here n_feature_out is just n_out.

    Args:
            output (Tensor): feature vectors of shape [n_flow, n_feature_out], or [batch_size, n_flow, n_feature_out]
            label (Tensor): ground truth labels of shape [n_flow] or [batch_size, n_flow]
    
    Returns:
            Tensor: clustering loss value
    """
    def _clustering_loss(output, label): 
        groups = {}     # cluster No -> centroid vector
        for i in range(len(label)):
            groups.setdefault(label[i], []).append(output[i])

        # centroid = {k: torch.mean(groups[k], dim=0) for k in groups.keys()}

        centroid = []
        for i in range(len(label)):
            group_value = [output[j] for j in range(len(label)) if label[j] == label[i]]
            group_tensor = torch.stack(group_value)
            centroid.append(torch.mean(group_tensor, dim=0))
        centroid = torch.stack(centroid)

        # min intra-cluster distance & max inter-cluster distance
        #   doesn't work well for now
        # w = 10
        # intra_dist = [LA.norm(output[i] - centroid[i]) ** 2
        #   for i in range(len(label))]
        # # TODO: the next line is wrong...
        # inter_dist = [LA.norm(centroid[i] - centroid[j]) ** 2
        #   for i in range(len(label)) for j in range(i + 1, len(label))]
        # loss = torch.mean(torch.stack(intra_dist)) - w * torch.mean(torch.stack(inter_dist))
        
        # silhouette coefficient
        # a: mean distance from point X to others in its own cluster
        # b: mean distance from point X to others in the closest cluster
        def sil(i, output, centroid, th_inter):
            intra_dist = [LA.norm(output[i] - output[j]) ** 2
                                                for j in range(len(label)) if label[j] == label[i] and i != j]
            inter_dist = [LA.norm(output[i] - centroid[j]) ** 2
                                                for j in range(len(label)) if label[j] != label[i]]
            # Special case 1 is no inter cluster ([0, 0]), & we expect a <= th_inter by
            # setting b = th_inter.
            # Special case 2 is no intra cluster ([0, 1]), & we expect b >= th_inter by
            # setting a = th_inter.
            a = torch.mean(torch.stack(intra_dist)) if intra_dist else torch.tensor(th_inter)
            b = torch.min(torch.stack(inter_dist)) if inter_dist else torch.tensor(th_inter)      

            return (b - a) / torch.max(torch.stack([a, b]))
        
        # the loss per sample is converted to [0, 1]
        loss = (1 - torch.mean(torch.stack([sil(i, output, centroid, th_inter)
                                                                        for i in range(len(label))]))) / 2

        return loss

    if len(output.shape) == 2:
        return _clustering_loss(output, label)

    loss = torch.tensor(0.0)
    for i in range(output.shape[0]):
        tmp_loss = _clustering_loss(output[i], label[i])
        loss += tmp_loss
    return loss

def cluster_triplet_loss(cluster1, cluster2, n_sample=None,
                                                 margin=1, lambd=1):
    """Cluster-wise triplet loss from ShapeNet (AAAI'21).

    We use cluster pair as the unit, and the loss function takes cluster pair
    as input, randomly samples anchors from either of the two clusters, and then
    computes the aggregated loss of sampled triplets.

    TODO: determine the sampling ratio that achieves best balance of speed and
                accuracy

    Args:
        cluster1 (Tensor): shape [n_flow, n_feature_out] 
        cluster2 (Tensor): shape [n_flow, n_feature_out] 
        n_sample (int): number of anchors to sample
        margin (float): margin for triplet loss. Default: 0.2
        lambd (float): weight for intra-cluster loss. Default: 1

    Returns:
        Tensor: aggregated loss of all sampled triplets
    """
    loss = 0.0
    if n_sample is None:
        n_sample = max(1, int((len(cluster1) + len(cluster2)) // 3))
        # n_sample = 1
    # print('n_sample', n_sample)

    for i in range(n_sample):
        flow = np.random.choice(range(len(cluster1) + len(cluster2)))
        # print(' flow', flow)
        if flow < len(cluster1):
            anchor = cluster1[flow]
            positive, negative = cluster1, cluster2
        else:
            flow -= len(cluster1)
            anchor = cluster2[flow]
            positive, negative = cluster2, cluster1

        d_ap = torch.mean(torch.stack([LA.norm(anchor - positive[j]) ** 2
                     for j in range(len(positive)) if j != flow]))
        d_an = torch.mean(torch.stack([LA.norm(anchor - negative[j]) ** 2
                     for j in range(len(negative))]))
        # adjustment: max to mean
        d_pos = torch.mean(torch.stack([LA.norm(positive[j] - positive[k]) ** 2
                     for j in range(len(positive) - 1) for k in range(j + 1, len(positive))]))
        d_neg = torch.mean(torch.stack([LA.norm(negative[j] - negative[k]) ** 2
                     for j in range(len(negative) - 1) for k in range(j + 1, len(negative))]))
        loss += torch.max(torch.stack([torch.tensor(0.0), torch.log((d_ap + margin) / d_an)]))
        loss += lambd * (d_pos + d_neg)

    return loss / n_sample


class LossTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()
    
    def test_clustering_loss(self):
        x1 = torch.tensor([[1, 1], [12, 2]], dtype=torch.float)
        y1 = torch.tensor([0, 1], dtype=torch.float)
        loss = clustering_loss(x1, y1, th_inter=5)
        self.assertLess(loss.item(), 0.2)
        x2 = torch.rand(2, 10, 2)
        y2 = torch.rand(2, 10)
        clustering_loss(x2, y2, th_inter=2)

    def test_cluster_triplet_loss(self):
        cluster1 = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float,
                   requires_grad=True)
        cluster2 = torch.tensor([[10, 20, 30], [15, 20, 30]], dtype=torch.float,
                   requires_grad=True)
        loss = cluster_triplet_loss(cluster1, cluster2)
        loss.backward()
        self.assertAlmostEqual(loss.item(), 25.0)
        self.assertAlmostEqual(cluster1.grad[0][0], 0.0)
        self.assertAlmostEqual(cluster1.grad[1][1], 0.0)


if __name__ == '__main__':
    unittest.main()