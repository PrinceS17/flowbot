import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import torch
import torch.nn as nn
import torch.linalg as LA

from torch import optim
from .loss import clustering_loss, cluster_triplet_loss

from sklearn.manifold import MDS


class AbsTripletLoss(nn.Module):
    """Absolute Triplet Loss, referred to the third triplet loss function
    proposed in Learning Embeddings for Image Clustering An Empirical Study
    of Triplet Loss Approaches as below:

        L(a, p, n) = max (alpha - d(a_i, n_i)^2, 0)
                    + max (d(a_i, p_i)^2 - beta, 0)
    
    To ease the argument passing, denote alpha = a^2, beta = b^2, and
    we take a, b as the arguments.

    Online default: a^2 = 3, b^2 = 1, thus if a = 1, then b = sqrt(1/3) ~ 0.58.

    Compared to TripletMarginLoss in pytorch, this loss function uses square
    to penalize the outliers harder, and optimize both the intra-cluster and
    inter-cluster distances.
    """
    def __init__(self, a=1.0, b=0.2, w=0.5):
        super().__init__()
        self.a = a
        self.b = b
        self.w = w
    
    def euclidean_sq(self, x1, x2):
        """Note that dim 0 needs to be preserved."""
        return (x1 - x2).pow(2).sum(1)
    
    def forward(self, anchor, pos, neg):
        """Modify the vannila triplet loss by using the strict pos-neg loss."""
        d_an = self.euclidean_sq(anchor, neg)
        d_ap = self.euclidean_sq(anchor, pos)
        d_pn = self.euclidean_sq(pos, neg)
        neg_loss = torch.max(torch.relu(self.a ** 2 - d_an), torch.relu(self.a ** 2 - d_pn))
        pos_loss = torch.relu(d_ap - self.b ** 2)
        losses = self.w * neg_loss + (1 - self.w) * pos_loss
        return losses.mean(), neg_loss.mean(), pos_loss.mean()


class NaiveModel(pl.LightningModule):
    def __init__(self, c_in, l_in, n_out, learning_rate=1e-3,
                 weight_decay=0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c_in * l_in, n_out),
            # nn.Sigmoid(),
        )
        self.n_out = n_out
        self.l_in = l_in
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.model(x)
        return y * 10 / LA.norm(y, dim=1, keepdim=True)
    
    def _step(self, batch, idx, tag, margin=1):
        x1, x2, x3 = batch
        y1, y2, y3 = self(x1), self(x2), self(x3)
        loss = nn.TripletMarginLoss(margin=margin)(y1, y2, y3)
        # self.log(f'{tag}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{tag}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate,
                                 weight_decay=self.weight_decay)
        return optimizer


class NaiveCNN(NaiveModel):
    def __init__(self, c_in, l_in, c_outs, n_out, learning_rate=1e-3,
                 weight_decay=0, use_global_avg_pooling=True):
        super().__init__(c_in, l_in, n_out, learning_rate, weight_decay)
        kernel_size = 7
        l_out = l_in
        layers = []
        for i in range(len(c_outs)):
            padding = int((kernel_size - 1) / 2)
            conv = nn.Conv1d(c_in, c_outs[i], kernel_size, padding=padding)
            relu = nn.ReLU()
            c_in = c_outs[i]
            l_out = l_out + 2 * padding - kernel_size + 1
            layers.extend([conv, relu])

        if use_global_avg_pooling:
            layers.extend([
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(c_outs[-1], n_out),
            ])
        else:
            layers.extend([
                nn.Flatten(),
                nn.Linear(l_out * c_outs[-1], n_out),
            ])
        self.model = nn.Sequential(*layers)
        self.n_out = n_out
        self.l_in = l_in
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay


class MySiameseModel(pl.LightningModule):
    """
    Siamese network with triplet loss. Typical network architectures include:

        FCN: use CNN w/o local pooling, add global avg pooling, no BN, possible dropout
        Temporal Conv Network: k=3, d=1,2,4,8, no local pooling, no BN, use weight norm &
                                dropout
    """
    def __init__(self, c_in, l_in, n_out, c_outs=[64, 64], kernel_size=13, 
                dilation=[1,2,4,8], pool_size=2, dropout=0, learning_rate=1e-4,
                weight_decay=0, lr_decay=0.5, lr_decay_step_size=2,
                use_weight_norm=False, use_batch_norm=False, use_local_pooling=False,
                use_global_avg_pooling=True, loss_mode='triplet', remove_mean=True,
                w_loss=0.5):
        super().__init__()
        layers = []
        l_out = l_in
        if type(kernel_size) != list:
            kernel_size = [kernel_size for _ in range(len(c_outs))]
        pool_size = 0 if not use_local_pooling else pool_size
        assert len(kernel_size) == len(c_outs) <= len(dilation)
        for i in range(len(c_outs)):
            padding = int((kernel_size[i] - 1) / 2)
            block, l_out = self._conv_block(c_in, c_outs[i], l_out, kernel_size[i],
                                            dilation[i], padding, dropout,
                                            use_weight_norm, use_batch_norm, pool_size)
            layers.extend(block)
            c_in = c_outs[i]
        
        if use_global_avg_pooling:
            layers.extend([
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten()
            ])
            n_linear_in = c_outs[-1]
        else:
            layers.append(nn.Flatten())
            n_linear_in = l_out * c_outs[-1]

        layers.extend([
            nn.Linear(n_linear_in, n_out),
            # nn.Sigmoid(),
        ])

        self.model = nn.Sequential(*layers)
        if loss_mode == 'cluster':
            self.f_loss = clustering_loss
        elif loss_mode == 'cluster_triplet':
            self.f_loss = cluster_triplet_loss
        elif 'triplet' not in loss_mode:
            raise NotImplementedError
        self.n_out = n_out
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.lr_decay_step_size = lr_decay_step_size
        self.loss_mode = loss_mode
        self.remove_mean = remove_mean
        self.l_in = l_in
        self.w_loss = w_loss
        self.recursive = nn.Sequential(
            nn.Linear(n_out * 2, n_out),    # seems just weighted avg
            # nn.Sigmoid(),                 # not sure
        )
        self.init_weights()
    
    def _conv_block(self, c_in, c_out, l_out, k, d, padding, dropout=0,
                    use_weight_norm=False,
                    use_batch_norm=False,
                    pool_size=0):
        """Returns one conv1d block.
        
        Settings:
            FCN's block: [conv, relu, dropout]
            Dilated conv block: [conv, wn, relu, dropout], k=3, d=2^n
        """
        conv = nn.Conv1d(c_in, c_out, k, dilation=d,
                         padding=padding)
        if use_weight_norm:
            conv = nn.utils.weight_norm(conv)
        relu = nn.ReLU()
        l_out = l_out + 2 * padding - d * (k - 1)
        block = [conv, relu]
        if pool_size > 0:
            max_pool = nn.MaxPool1d(pool_size, stride=pool_size)
            block.append(max_pool)
            l_out = int((l_out - pool_size) / pool_size  + 1)
        if use_batch_norm:
            block.append(nn.BatchNorm1d(c_out))
        if dropout > 0:
            block.append(nn.Dropout(dropout))
        return block, l_out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
    
    def forward(self, x, last_y=None):
        """Implicit Siamese model, as each flow traverses the same model, and the loss is computed
        at the run level.
        
        Args:
            x (Tensor): input of shape [flow_num, seq_len, n_feature_in]
            last_y (Tensor): last output of shape [flow_num, n_feature_out]
        
        Returns:
            Tensor: output of shape [flow_num, n_feature_out].
        """
        x = x.permute(0, 2, 1)
        if self.remove_mean:        # tested
            x_avg = x.mean(dim=-1)
            x_avg = x_avg.reshape(x.shape[0], x.shape[1], 1).repeat(1, 1, x.shape[2])
            x[:, 0:2, :] -= x_avg[:, 0:2, :]    # only OWD and RTT
        y = self.model(x)
        
        # normalize the the output to a vector of length 1
        y = y * 1 / LA.norm(y, dim=1, keepdim=True)
        
        if last_y is not None:
            y = torch.cat([y, last_y], dim=-1)
            y = self.recursive(y)
        return y
    
    def _step(self, batch, idx, tag):
        """Internal step function for training, validation and testing. Loss mode
        takes effect here to decompose the input batch respectively for different
        loss calculation. 'tag' is used solely for logging.
        """
        if self.loss_mode == 'triplet':
            x1, x2, x3 = batch    # anchor, positive, negative
            y1, y2, y3 = self(x1), self(x2), self(x3)
            loss = nn.TripletMarginLoss(margin=1.0)(y1, y2, y3)
        elif self.loss_mode == 'abs_triplet':
            x1, x2, x3 = batch
            y1, y2, y3 = self(x1), self(x2), self(x3)
            loss, neg_loss, pos_loss = AbsTripletLoss(w=self.w_loss)(y1, y2, y3)
        elif self.loss_mode == 'flow_triplet':
            # TODO: need tests
            anchor, pos, neg, _ = batch
            assert anchor.shape[1] == pos.shape[1] == neg.shape[1]
            last_y = torch.zeros(anchor.shape[0], self.n_out).to(anchor.device)
            losses = []
            for i in range(0, anchor.shape[1], self.l_in):
                ys = []
                for data in [anchor, pos, neg]:
                    x = data[:, i:i+self.l_in, :]
                    y = self(x, last_y)
                    last_y = y
                    ys.append(y)
                loss = nn.TripletMarginLoss(margin=1.0)(ys[0], ys[1], ys[2])
                losses.append(loss)
            loss = torch.stack(losses).mean(axis=0)     # average over time
        elif self.loss_mode == 'cluster_triplet':
            x1, x2, _ = batch           # positive and negative samples
            ys = []
            for x in [x1, x2]:
                batch_size, n_flow, seq_len, n_in = x.shape
                x = x.reshape(batch_size * n_flow, seq_len, n_in)
                # batch_size can only be 1, so the reshape below works
                y_hat = self(x).reshape(n_flow, self.n_out)
                ys.append(y_hat)
            loss = self.f_loss(ys[0], ys[1])
        elif self.loss_mode == 'cluster':
            x, y = batch
            batch_size, n_flow, seq_len, n_in = x.shape
            x = x.reshape(batch_size * n_flow, seq_len, n_in)
            y_hat = self(x).reshape(batch_size, n_flow, self.n_out)
            loss = self.f_loss(y_hat, y)
        # self.log(f'{tag}_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
        #          logger=True)
        # self.log(f'{tag}_neg_loss', neg_loss, on_step=True, on_epoch=True,
        #          prog_bar=True)
        # self.log(f'{tag}_pos_loss', pos_loss, on_step=True, on_epoch=True,
        #          prog_bar=True)

        # or a logdict style?
        self.log_dict({f'{tag}_loss': loss, f'{tag}_neg': neg_loss,
                       f'{tag}_pos': pos_loss}, on_step=tag == 'train', on_epoch=True,
                       prog_bar=True, logger=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        assert not self.training
        val = self._step(batch, batch_idx, 'val')
        return val
    
    def test_step(self, batch, batch_idx):
        assert not self.training
        val = self._step(batch, batch_idx, 'test')
        return val
    
    def predict_step(self, batch, idx):
        return self(batch[0])
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                                lr=self.learning_rate,
                                weight_decay=self.weight_decay)

        return optimizer

        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
        #                                         step_size=self.lr_decay_step_size,
        #                                         gamma=self.lr_decay)

        # return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}


class TemporalBlock(nn.Module):
    """Conv block w/ residual connection.
    """
    def __init__(self, c_in, c_out, kernel_size, dilation, padding,
        dropout=0, use_weight_norm=False, use_batch_norm=False):
        super().__init__()
        conv = nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation,
                         padding=padding)
        if use_weight_norm:
            conv = nn.utils.weight_norm(conv)
        relu = nn.ReLU()
        block = [conv, relu]
        if use_batch_norm:
            block.append(nn.BatchNorm1d(c_out))
        if dropout > 0:
            block.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*block)
        self.downsample = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.block(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class ResSiameseModel(MySiameseModel):
    """Implementation based on TCN except only including one conv1d in
    each block."""
    def __init__(self, c_in, l_in, n_out, c_outs=[64, 64], kernel_size=13, 
                dilation=[1,2,4,8], pool_size=2, dropout=0, learning_rate=1e-4,
                weight_decay=0, lr_decay=0.5, lr_decay_step_size=2,
                use_weight_norm=False, use_batch_norm=False, use_local_pooling=False,
                use_global_avg_pooling=True, loss_mode='triplet', remove_mean=True):
        super().__init__(c_in, l_in, n_out)
        print('Warning: pooling deprecated in TCN.')
        layers = []
        if type(kernel_size) != list:
            kernel_size = [kernel_size] * len(c_outs)
        assert len(kernel_size) == len(c_outs) <= len(dilation)
        for i, (c_out, k) in enumerate(zip(c_outs, kernel_size)):
            padding = int((k - 1) * dilation[i] // 2)
            block = TemporalBlock(c_in, c_out, k, dilation[i], padding,
                dropout, use_weight_norm, use_batch_norm)
            layers.append(block)
            c_in = c_out
        if use_global_avg_pooling:
            layers.extend([
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(c_outs[-1], n_out)
            ])
        else:
            layers.extend([
                nn.Flatten(),
                nn.Linear(c_outs[-1] * l_in, n_out)
            ])
        self.model = nn.Sequential(*layers)
        if loss_mode == 'cluster':
            self.f_loss = clustering_loss
        elif loss_mode == 'cluster_triplet':
            self.f_loss = cluster_triplet_loss
        elif 'triplet' not in loss_mode:
            raise NotImplementedError
        self.n_out = n_out
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.lr_decay_step_size = lr_decay_step_size
        self.loss_mode = loss_mode
        self.remove_mean = remove_mean
        self.l_in = l_in
        self.recursive = nn.Sequential(
            nn.Linear(n_out * 2, n_out),    # seems just weighted avg
        )
        self.init_weights()
