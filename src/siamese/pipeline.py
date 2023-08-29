import numpy as np
import pandas as pd
# import modin.pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import unittest

from collections import Counter
from matplotlib import pyplot as plt
import sklearn
from sklearn.manifold import MDS
# from sklearn import metrics
from scipy.optimize import linear_sum_assignment

from .preprocess import DataPreprocessor
from .dataset import SiameseDataset, SiameseNetDataLoader, CleanDataLoader
from .model import MySiameseModel
from .cluster import cluster


def get_method_to_model_th(methods, model_path, model_th, model_th_list):
    """Get a dict of {method: [(model_tag, model_path, model_th)]}
    from model_th_list or model_path and model_th."""
    if model_th_list is None:
        method_to_model_th = {m: [(m, None, model_th)] for m in methods}
        method_to_model_th['ml'] = [('ml', model_path, model_th)]
    else:
        method_to_model_th = {}
        for model_tag, model_path, model_th in model_th_list:
            # currently used to support DCW + th & ML + model + th
            model_path = None if 'nan' in str(model_path) else model_path 
            added = False
            for method in methods:
                if method in model_tag:
                    method_to_model_th.setdefault(method, []).append(
                        (model_tag, model_path, model_th))
                    added = True
                    break
            if not added:
                print(f'Warning: model_tag {model_tag} not added to any method')
        for method in methods:
            if method not in method_to_model_th:
                method_to_model_th[method] = [(method, None, model_th)]
    return method_to_model_th


def get_detection_delay(ys, y_hats, flows, interval, n=2, n1=2):
    """Get the first detecting time of each flow given truth
    ys and predictions y_hats, i.e. time difference of ys / y_hats
    that first match the stable truth for n consecutive ticks.
    Here we use DataFrame to handle the case of dynamic flows, which
    cannot be removed as it's intended in the experiment.

    Args:
        ys (list): Truth labels along time
        y_hats (list): Predicted labels along time
        flows (list): list of flows at the time tick
        interval (float): time unit
        n (int, optional): number of consecutive ticks to consider
                            a good match
        n1 (int, optional): number of flows for determine the detection
    
    Returns:
        list: t_detect [n_flow]
        DataFrame: t_detect_df ['flow', 't_detect']
    """
    y_hats1 = []
    for i in range(len(ys)):
        assert len(ys[i]) == len(y_hats[i]) == len(flows[i])
        y_hats1.append(permutate_labels(ys[i], y_hats[i]))

    y_df = None
    for i in range(len(ys)):
        cur = pd.DataFrame({'time': [i * interval] * len(ys[i]),
                            'flow': flows[i], 'y': ys[i], 'y_hat': y_hats1[i]})
        y_df = cur if y_df is None else pd.concat([y_df, cur])

    t_detect = []
    for flow in y_df.flow.unique():
        flow_ydf = y_df[y_df.flow == flow]
        y_stable = flow_ydf.y.value_counts().index[0]
        ts = []
        for ylist in [flow_ydf.y, flow_ydf.y_hat]:
            for ti in range(len(ylist)):
                y_local = ylist.iloc[ti: ti + n].tolist()
                # if n1 / n samples are correct, and 1st one is correct, then good
                # this is intended to loose the condition for SBD
                if y_local.count(y_stable) >= n1 and \
                    y_local[0] == y_stable or ti == len(ylist) - 1:
                # if y_local.tolist() == [y_stable] * n or ti == len(ylist) - 1:
                    ts.append(ti)
                    break
        assert len(ts) == 2, f'ts len: {len(ts)} < 2'
        t_detect.append([flow, (ts[1] - ts[0]) * interval])
    t_detect_df = pd.DataFrame(t_detect, columns=['flow', 't_detect'])
    return t_detect_df



# @measure_time()
def update_non_btnk_flows(classifier, segment, flow_labels, skip=False):
    """Updates the y_hat by classifying the non-btnk flows first.
    
    Args:
        segment (dataframe): flows' data of a period
        flow_labels (dict): {flow: labels}
        skip (bool): if skip this process
    """
    if skip:
        return flow_labels
    if 'run' in segment.columns:
        assert segment.run.nunique() == 1
    for col in ['flow']:
        assert col in segment.columns, f'{col} not in segment!'

    # convert df to the feature numpy array for prediction
    # this might be slow in our current setting
    stat_field_func = [
        ('owd_std', 'owd', lambda x: x.std()),
        ('rtt_std', 'rtt', lambda x: x.std()),
        ('slr_avg', 'slr', lambda x: x.mean()),
        ('slr_std', 'slr', lambda x: x.std()),
        ('cwnd_avg', 'cwnd', lambda x: x.mean()),
        ('cwnd_std', 'cwnd', lambda x: x.std()),
    ]
    stats = [stat for stat, _, _ in stat_field_func]
    stat_df = segment.copy()
    for stat, field, func in stat_field_func:
        stat_df[stat] = stat_df.groupby(['flow'])[field].transform(func)
    stat_df = stat_df.sort_values(['flow']).drop_duplicates(
        subset=['flow'], ignore_index=True)
    # TODO: discontinuous flows in stat_df?
    X_stat = stat_df[stats].to_numpy()
    y_nonbtnk = classifier.predict(X_stat)
    assert len(y_nonbtnk) == len(flow_labels.keys())
    new_y = max(flow_labels.keys()) + 1
    j = 1
    for i in range(len(y_nonbtnk)):
        if y_nonbtnk[i] == 1:
            continue
        flow_labels[list(flow_labels.keys())[i]] = new_y + j
        j += 1
    return flow_labels



def set_df_metadata(df, run, method, tag):
    """Set df's metadata."""
    df['run'] = run
    df['method'] = method
    df['tag'] = tag



def predict_segment(model, x, max_iter=20, n_voter=9, th=0.4, flow_labels=None):
    """Predict the labels given segment x.

    Args:
        model (MySiameseModel): model used for prediction
        x (array): segment of shape [n_flow, seq_len, n_feature]
        max_iter (int, optional): max number of iterations
            for the clustering algorithm. Defaults to 20.
        th (float, optional): distance threshold
        flow_labels (dict, optional): {flow: label}, -1 for btnk flows to be assigned.

    Returns:
        Tensor: [n_flow] of the cluster No. for each flow
        Tensor: [n_flow, n_out], the latent vectors
        Dict: {flow: label} after completion
    """
    latent = model(torch.tensor(x))
    latent = latent.detach().numpy()
    assert latent.shape == (x.shape[0], model.n_out)
    # y_hat, flow_labels, _ = cluster(latent, max_iter=max_iter, n_voter=0, th=th,
    #                                 flow_labels=flow_labels)
    y_hat, flow_labels, _ = cluster(latent, max_iter=max_iter, n_voter=n_voter,
                                    th=th, flow_labels=flow_labels)
    return y_hat, latent, flow_labels

def compare_segment(y, y_hat):
    """Compare the predicted labels with the ground truth labels.

    Args:
        y (Tensor): [n_flow] of the ground truth labels
        y_hat (Tensor): [n_flow] of the predicted labels
        i (int): the index of the segment

    Returns:
        list: contains accuracy metrics like precision, recall, f1, adj_rand_score
        list: the name list of these metrics
    """
    assert len(y) == len(y_hat), f'Error: len(y) = {len(y)} != len(y_hat) = {len(y_hat)}'
    precision, recall, f1 = precision_recall_f1_from_raw(y, y_hat)
    adj_rand_score = sklearn.metrics.cluster.adjusted_rand_score(y, y_hat)
    pair_precision, pair_recall = pairwise_precision_recall(y, y_hat)
    rel_precision, rel_recall, rel_f1 = relative_precision_recall_f1(y, y_hat)
    metrics = [precision, recall, f1, adj_rand_score, pair_precision, pair_recall,
        rel_precision, rel_recall, rel_f1]
    names = ['precision', 'recall', 'f1', 'adj_rand_score', 'pair_precision',
        'pair_recall', 'rel_precision', 'rel_recall', 'rel_f1']
    return list(map(lambda x: round(x, 4), metrics)), names

def plot_accuracy(res_df, fields=['precision', 'recall']):
    fields = res_df.columns[1:] if fields is None else fields
    _, ax = plt.subplots(len(fields), 1, figsize=(8, 2.5 * len(fields)))
    for i in range(len(fields)):
        if fields[i] in ['time']:
            continue
        if 'run' not in res_df.columns:
            sns.lineplot(x='time', y=fields[i], data=res_df, ax=ax[i])
        else:
            sns.lineplot(x='time', y=fields[i], hue='run', data=res_df, ax=ax[i])
    plt.show()
    plt.close()

def relative_precision_recall_f1(a, b):
    """Relative flowwise precision, recall and f1. Similar to the standard metrics from
    raw calculation, but the difference is: we follow relativity for the predictions
    of each clusters. In other words, here we first iterate over the clusters,
    then permutate the flows per cluster, e.g. [1,1,2,2] -> [1,1,1,1], the standard
    recall is (0.5 + 0) / 2 = 0.25, while here it's (0.5 + 0.5) / 2 = 0.5.
    We use this to evaluate the labels in a way that is more close to the cobtnk detection
    nature.
    """
    metrics = []
    for src, dst in zip([b, a], [a, b]):
        info = {}
        for label in set(src):
            src1 = [src[i] for i in range(len(src)) if src[i] == label]            
            dst1 = [dst[i] for i in range(len(src)) if src[i] == label]
            n_most_common = max(map(dst1.count, dst1))
            accuracy = n_most_common / len(src1)        # precision then recall
            # here n is the src length, instead of always the true label length
            # the reason is I think it's reasonable to use prediction length
            # when computing recall, as the basis is the predicted samples,
            # and there won't be any NaN using this method
            info[label] = {'n': len(src1), 'accuracy': accuracy}
        res = sum([info[i]['accuracy'] * info[i]['n'] for i in info]) \
            / sum([info[i]['n'] for i in info])
        metrics.append(res)
    f1 = 2 * metrics[0] * metrics[1] / (metrics[0] + metrics[1])
    return metrics[0], metrics[1], f1        
            
def pairwise_precision_recall(a, b):
    """Pairwise precision and recall, i.e. for any flow pair, if the prediction
    is correct, then positive, otherwise it's negative.

    Args:
        a (list): actual labels
        b (list): predicted
    
    Returns:
        pairwise precision, recall
    """
    assert len(a) == len(b)
    n = len(a)
    tp, tn, fp, fn = 0, 0, 0, 0     # true/false positive/negative
    for i in range(n - 1):
        for j in range(i + 1, n):
            if b[i] == b[j]:    # positive
                if a[i] == a[j]:
                    tp += 1
                else:
                    fp += 1
            else:               # negative
                if a[i] != a[j]:
                    tn += 1
                else:
                    fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    return precision, recall


def match_two_labels(a, b):
    """Match a & b to some same basis of the labels. Here we use the
    contingency matrix and linear sum assignment to find the best mapping.
    a & b are list-like."""
    a1, b1 = list(set(a)), list(set(b))
    a2 = [a1.index(i) for i in a]
    b2 = [b1.index(i) for i in b]
    # contingency_matrix m:
    #     y_true x y_pred, records the number of samples w/ (y_true, y_pred)
    # The problem can be then converted into linear sum assignment (w/ max), 
    # which finds the max sum of a trajectory (= max number of matches) where
    # each row and column is only used once. row_idx is sorted, col_idx[i]
    # maps true label i to pred j.
    m = sklearn.metrics.cluster.contingency_matrix(a2, b2)
    row_idx, col_idx = linear_sum_assignment(m, maximize=True)

    # convert a to b if all a's elements are present in row_idx, or b to a
    a_to_b = {r: c for r, c in zip(row_idx, col_idx)}
    b_to_a = {c: r for r, c in zip(row_idx, col_idx)}
    a3, b3 = a2, b2
    if len(set(b2)) <= len(set(a2)):
        b3 = [b_to_a[i] for i in b2]
    else:
        a3 = [a_to_b[i] for i in a2]

    return a3, b3


def precision_recall_f1_from_raw(a, b, average='weighted',
    print_array=False):
    """Compute precision, recall, and F1 score from raw values,
    i.e. a mapping needs to be done.
    
    In the case that y_true, y_pred have different number of labels,
    it's possible that a wrong mapping is found, due to the fact that no
    non-zero term can be found for an extra class. For instance, in
    [0, 1, 2, 3, 4] -> [0, 1, 2, 2, 5], 3 will be mapped to 5.

    Args:
        a (list like): list of expected labels
        b (list like): list of predicted labels
        average (str, optional): 'macro' or 'micro'. Defaults to 'weighted'.
            As the cluster sizes are quite differenty, but each flow should
            be treated equally.
        print_array (bool, optional): print the final array or not for
            debugging. Defaults to False.

    Returns:
        tuple: average (precision, recall, f1) values for all classes
    """
    a = a if type(a) != torch.Tensor else a.tolist()
    b = b if type(b) != torch.Tensor else b.tolist()
    a3, b3 = match_two_labels(a, b)
    if print_array:
        data_tags = ['y_final', 'y_hat_final']
        for i, data in enumerate([a3, b3]):
            print('\n', data_tags[i], ':\n    ', end='')
            for j in range(len(data)):
                print(data[j], end=' ')
                if j % 16 == 16 - 1:
                    print('\n    ', end='')
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        a3, b3, average=average, zero_division=0)
    return precision, recall, f1


def permutate_labels(a, b):
    """Given truth a and prediction b, permutate b to match a.

    Args:
        a (array-list): truth labels
        b (array-like): prediction labels

    Returns:
        list: permutated b
    """
    assert len(a) == len(b)
    j = 1
    for x in [a, b]:
        j = 1
        for i in range(len(x)):
            if x[i] < 0:
                x[i], j = max(x) + j, j + 1
    assert min(a) >= 0 and min(b) >= 0

    # we want to find the mapping from b to a, so we need to find the
    # mapping for every element in b.
    # 1) we have the mapping b_to_b1 & a1_to_a, which is one-to-one mapping
    # 2) a1, b1 is connected by their values, but it's possible that it's not
    #   an one-to-one mapping
    a1, b1 = match_two_labels(a, b)
    a_max = max(a)
    b_to_b1 = {b[i] : b1[i] for i in range(len(b))}
    a1_to_a = {a1[i] : a[i] for i in range(len(a))}

    # v: value in matched space, i.e. a1 & b1
    # if v appears in a1, then it can be mapped to a directly
    # otherwise, it's some label only occuring in b1, so we add a new label
    b_to_a = {k : a1_to_a[v] if v in a1_to_a else a_max + k + 1
                for k, v in b_to_b1.items()}
    return [b_to_a[k] for k in b]


def get_consensus_labels(ys):
    """Get consensus labels given labels of a few time steps.

    Args:
        ys (list): list of y in the past intervals

    Returns:
        list: the y after consensus
    """
    for y in ys:
        assert len(y) == len(ys[0])
    return [Counter([ys[t][flow] for t in range(len(ys))]).most_common(1)[0][0]
                for flow in range(len(ys[0]))]


class SiamesePipeline:
    """Pipeline for Siamese model, including data loading, model setup, training,
    test, prediction, output visualization and evaluation.
    """
    def __init__(self, seq_len, fields, dataset_type) -> None:
        self.seq_len = seq_len
        self.fields = fields
        self.dataset_type = dataset_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loader = SiameseNetDataLoader(seq_len, fields)

    def load_data(self, root, folder_to_ids, labels, ts, batch_size,
                  no_dats=False, caches=[], f_smooth=None):
        self.loader.load(root, folder_to_ids, labels, no_dats, caches)
        self.loader.smooth(f_smooth)
        self.loader.split(ts, self.dataset_type, batch_size)
    
    def set_model(self, **params):
        default_params = {
            'c_in': len(self.fields),
            'l_in': self.seq_len,
            'n_out': 8,
            'c_outs': [64, 64],
            'dropout': 0.0,
        }
        default_params.update(params)
        self.model = MySiameseModel(**default_params)
    
    def train(self, patience, max_epochs, check_val_every_n_epoch=1, dry_run=False):
        early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',
            min_delta=1e-4, patience=patience, verbose=False, mode='min')
        self.trainer = pl.Trainer(max_epochs=max_epochs,
                                  check_val_every_n_epoch=check_val_every_n_epoch,
                                  callbacks=[early_stopping],
                                  log_every_n_steps=6,
                                  accelerator=self.device,
                                  devices=1)
        if not dry_run:
            assert self.model and self.loader
            self.trainer.fit(self.model, self.loader.get_train_loader(),
                             self.loader.get_val_loader())

    def test(self):
        assert self.trainer and self.loader
        self.trainer.test(self.model, self.loader.get_test_loader())
    
    def save_model(self, path='siamese_pipeline_model.pt'):
        torch.save(self.model, path)
    
    def load_model(self, path='siamese_pipeline_model.pt'):
        self.model = torch.load(path)
    
    def save_prediction(self, prediction, path):
        torch.save(prediction, path)
    
    def load_prediction(self, path):
        return torch.load(path)

    def predict(self, run_to_idx, max_iter=100, run_id_to_save={},
                f_cache=None, rewrite=False):
        """Given run and index of the segments, predict the labels.
        """
        df, truth_df = self.loader.get_dataframes()
        res = {}
        for run, indices in run_to_idx.items():
            df_in = df[df.run == run].reset_index(drop=True)
            label_df = truth_df[truth_df.run == run].reset_index(drop=True)
            dataset = SiameseDataset(df_in, label_df, self.seq_len, self.fields)
            for i in indices:
                x, y = dataset[i]       # input x: [n_flow, seq_len, n_feature_in]
                y_hat, latent, clusters = predict_segment(self.model, x, max_iter)
                res[(run, i)] = (y, y_hat, latent, clusters)
                if (run, i) in run_id_to_save.items():
                    torch.save(latent, f'latent_{run}_{i}.pt')
        return res

    def compute_accuracy(self, predictions, times=None, f_latent_loss=None,
                         t_unit=1.5):
        """Computes accuracy metrics (precision, recall, f1, adjusted rand score
        from predictions.
        Args:
            predictions (dict): dict {(run, idx): (y, y_hat, latent, clusters)}
            times (list, optional): df.times to label time on i's sequence.
                                    Defaults to None.
            f_latent_loss (function, optional): function to compute latent loss.
                                                Defaults to None.
            t_unit (float, optional): duration of a segment. Defaults to 1.5 (s).
        
        Returns:
            DataFrame: accuracy metrics
        """
        assert False, "Deprecated function here."
        cols = ['run', 'time', 'loss', 'precision', 'recall', 'f1', 'adj_rand_score']
        res_df = pd.DataFrame(columns=cols)
        for (run, i), (y, y_hat, latent, clusters) in predictions.items():
            loss = f_latent_loss(latent, clusters).item() if f_latent_loss else -1
            precision, recall, f1, adj_rand_score = compare_segment(y, y_hat)
            time = times[i * self.seq_len] if times else i * t_unit
            tmp_df = pd.DataFrame([[run, time, loss, precision, recall,
                                    f1, adj_rand_score]], columns=cols)
            res_df = tmp_df if res_df.empty else pd.concat([res_df, tmp_df],
                                                                ignore_index=True)

        return res_df


class CleanPipeline(SiamesePipeline):
    def __init__(self, seq_len, fields, n_run_debug=None) -> None:
        self.seq_len = seq_len
        self.fields = fields
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessor = DataPreprocessor('pre_cache', n_run_debug=n_run_debug)

    def load_data(self, folder, boundary, batch_size, prefix, to_save,
                  split_on='run', train_or_predict='train'):
        # TODO: multiple folder support: decision? as we can also move all
        #       data files into one folder
        if to_save:
            self.preprocessor.read(folder)
            self.preprocessor.preprocess(self.seq_len)
            self.preprocessor.save()
        else:
            self.preprocessor.load()

        self.preprocessor.check()

        assert train_or_predict in ['train', 'predict']
        if train_or_predict == 'train':
            self.loader = CleanDataLoader(self.seq_len, self.fields,
                                        self.preprocessor.xdf,
                                        self.preprocessor.tick_truth_df)
            self.loader.split(boundary, batch_size, split_on, prefix, to_save)
        
        if to_save:
            self.preprocessor.preprocess_for_prediction(self.seq_len, append=True)
            self.preprocessor.save_prediction()
        else:
            self.preprocessor.load_prediction()
        self.preprocessor.check_for_prediction(self.seq_len)

    def set_model(self, **params):
        default_params = {
            'c_in': len(self.fields),
            'l_in': self.seq_len,
            'n_out': 8,
            'c_outs': [64, 64],
            'dropout': 0.0,
        }
        default_params.update(params)
        self.model = MySiameseModel(**default_params)

    def train(self, patience, max_epochs, check_val_every_n_epoch=1, dry_run=False):
        early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',
            min_delta=1e-4, patience=patience, verbose=False, mode='min')
        self.trainer = pl.Trainer(max_epochs=max_epochs,
                                  check_val_every_n_epoch=check_val_every_n_epoch,
                                  callbacks=[early_stopping],
                                  log_every_n_steps=6,
                                  accelerator=self.device,
                                  devices=1)
        if not dry_run:
            assert self.model and self.loader
            self.trainer.fit(self.model, self.loader.get_train_loader(),
                             self.loader.get_val_loader())

    def test(self):
        assert self.trainer and self.loader
        self.trainer.test(self.model, self.loader.get_test_loader())
    
    def save_model(self, path='clean_pipeline_model.pt'):
        torch.save(self.model, path)
    
    def load_model(self, path='clean_pipeline_model.pt'):
        self.model = torch.load(path)
    
    def save_prediction(self, prediction, path):
        torch.save(prediction, path)
    
    def load_prediction(self, path):
        return torch.load(path)

    def predict(self, run_to_idx, max_iter=100, run_id_to_save={}):
        df, truth_df = self.preprocessor.xdf, self.preprocessor.truth_df
        res = {}
        for run, indices in run_to_idx.items():
            df_in = df[df.run == run].reset_index(drop=True)
            label_df = truth_df[truth_df.run == run].reset_index(drop=True)
            dataset = SiameseDataset(df_in, label_df, self.seq_len, self.fields)
            for i in indices:
                x, y = dataset[i]       # input x: [n_flow, seq_len, n_feature_in]
                y_hat, latent, clusters = predict_segment(self.model, x, max_iter)
                res[(run, i)] = (y, y_hat, latent, clusters)
                if (run, i) in run_id_to_save.items():
                    torch.save(latent, f'latent_{run}_{i}.pt')
        return res


class SiamesePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        folder_to_ids = {'db_pie_f5_0905': [49917 + i for i in range(3)]}
        root = '~/dumbbell_datasets_0908'
        labels = pd.DataFrame({
            'run': [49917] * 5 + [49918] * 5 + [49919] * 5, 
            'flow': list(range(5)) * 3,
            'label': ([0] * 3 + [1] * 2) * 3
        })
        ts = [2, 4, 6, 8]
        batch_size = 2

        self.sp = SiamesePipeline(10, ['owd', 'drop'], 'triplet')
        self.sp.load_data(root, folder_to_ids, labels, ts, batch_size)
        params = {'n_out': 1, 'loss_mode': 'triplet'}
        self.sp.set_model(**params)
        for item in params.items():
            self.assertIn(item, self.sp.model.__dict__.items())
    
    def test_load(self):
        df, truth_df = self.sp.loader.get_dataframes()
        self.assertEqual(len(df), 200 * 30 * 5 * 3)
        self.assertEqual(truth_df.shape, (15, 3))
    
    def test_train_test(self):
        self.sp.train(1, 1)
        self.sp.test()
        self.assertIsNotNone(self.sp.trainer)
    
    def test_predict_accuracy(self):
        self.sp.train(1, 1)
        run_to_idx = {49917: [0], 49918: [1]}
        predictions = self.sp.predict(run_to_idx)
        self.assertEqual(len(predictions), 2)
        for run, indices in run_to_idx.items():
            for i in indices:
                y, y_hat, latent, clusters = predictions[(run, i)]
                self.assertEqual(y.shape, (5,))
                self.assertEqual(y_hat.shape, (5,))
                self.assertEqual(latent.shape, (5, 1))

        res_df = self.sp.compute_accuracy(predictions)
        self.assertEqual(res_df.shape, (2, 7))


if __name__ == '__main__':
    unittest.main()