import numpy as np
import pandas as pd
import pickle
import time
import torch
import unittest

from denoise import DeNoise
from models import Pipeline

from siamese.cluster import cluster
from sbd import SBDAlgorithm
from siamese.dataset import *
from siamese.model import *
from siamese.pipeline import predict_segment, compare_segment, plot_accuracy, SiamesePipeline
from siamese.pipeline import permutate_labels, update_non_btnk_flows, set_df_metadata, get_method_to_model_th, get_detection_delay
from siamese.preprocess import DataModifier, measure_time


def get_detector_accuracy(df, runs, fields, truth_df, model_name,
                          methods, corr_interval=1.5, sbd_interval=0.35,
                          t_unit=0.005, ts=None, model_th=0.2,
                          converge_nonbtnk=False, model_th_list=None,
                          max_iter=10):
    """Aggregates the detector's accuracy results of different runs.
    
    Here we hardcode the SBD's window duration to be 0.35s.
    
    Now we support detecting multiple model/th combinations for the same data by
    using model_th_list, which is a list of (model_tag, model_name, model_th) tuples.
    For example, model_th_list=[('ml', 'models/ml_model_1.pt', 0.2)].

    ts behavior: some caveat for dynamic flows as well.

    Keys: 1) run_df from xdf_pred is already preprocessed to have n * tick start & end;
          2) if t_start > run_df.time.min, and < run_df.flow.time.min, i.e. still earlier
              than some flow, then the length of 1st segment won't be correct for that
              flow unless t_start is set to be some tick, i.e. n * interval.

    Returns:
        res_df: df['run', 'method', 'time', 'f1', 'precision', 'recall', 'adj_rand_score']
        predict_infos: {run: {method: predict_info} }, predict_info is
                    list of ys, y_hats (2d list of labels), and latent_map, i.e.
                    {t: latent}, latent = [flow_1_latent, flow_2_latent, ...]
        time_df: df['run', 'method', 't_classify']
        converge_nonbtnk: if True, converge non-btnk flow label to -1. Deprecated.
        model_th_list: list of (model_tag, model_path, model_th) tuples.
    """
    res_df = None
    predict_infos = {}
    time_df = pd.DataFrame(columns=['run', 'method', 'tag', 't_classify'])
    # hack for local detection
    if len(runs) == df.run.nunique() == 1:
        runs = [df.run.unique()[0]]
    corr_interval = round(corr_interval, 3)
    sbd_interval = round(sbd_interval, 3)
    for run in runs:
        run_df = df[df.run == run].copy()
        if run_df.empty:
            print(f'Warning: skipping empty run {run}, probably producing res_df=None')
            continue
        run_labels = truth_df[truth_df.run == run].copy()
        if ts:
            assert len(ts) == 2
            for t in ts:
                assert np.isclose(t % corr_interval, 0), \
                    f"{t} not on tick for ML, corr_interval = {corr_interval}!"
            t_start = max(ts[0], run_df.time.min())
            t_end = min(ts[1], run_df.time.max())
            run_df = run_df[(run_df.time >= t_start) & (run_df.time < t_end)]
            if run_df.empty:
                print(f'Warning: no data in given time range, time range in run_df: '
                    f'{run_df.time.min()} ~ {run_df.time.max()} s, probably producing '
                    'res_df=None')
                continue
            if 'time' in run_labels.columns:
                run_labels = run_labels[(run_labels.time >= t_start) &
                                        (run_labels.time < t_end)]

        predict_infos[run] = {}
        detector = CobottleneckDetector(run_df, fields, corr_interval,
                                        labels=run_labels, t_unit=t_unit,
                                        max_iter=max_iter)
        if '/' not in model_name:
            model_path = os.path.join('models', model_name)
        else:
            model_path = model_name

        method_to_model_th = get_method_to_model_th(methods, model_path,
                                                    model_th, model_th_list)
        for method in methods:
            t1 = time.time()
            print(f'run {run}, method {method}, start time: {t1:.3f} s')
            itv = sbd_interval if 'sbd' in method else 1.5 if 'dcw' in method \
                else corr_interval
            # use_old_sbd = True if last_sbd and 'sbd' in method else False
            use_old_sbd = False

            last_sbd = True if 'sbd' in method else False
            # detector_labels = run_labels2 if 'sbd' in method else run_labels0 \
            #     if 'dcw' in method else run_labels1
            for model_tag, model_path, model_th in method_to_model_th[method]:
                assert model_path is not None or method != 'ml'
                detector.set_model_th_field(model_path, model_th, fields)
                tmp_df, predict_info, t_classify, t_detect_df = detector.quick_run(method, plot=False,
                                                interval=itv, use_old_sbd=use_old_sbd,
                                                converge_nonbtnk=converge_nonbtnk)
                tag_parts = [str(s) for s in [model_tag, model_th]
                            if str(s) not in ['nan', 'None']]
                tag = '_'.join(tag_parts)
                set_df_metadata(tmp_df, run, method, tag)
                res_df = tmp_df if res_df is None else pd.concat([res_df, tmp_df])
                predict_infos[run][tag] = predict_info
                time_df = pd.concat([time_df, pd.DataFrame({
                    'run': [run], 'method': [method], 'tag': [tag],
                    't_classify': [t_classify]})])
                # set_df_metadata(t_detect_df, run, method, tag)
                t2 = time.time()
                print(f'   end time: {t2:.3f} s, duration: {t2 - t1:.3f} s')
        return res_df, predict_infos, time_df, t_detect_df



# TODO: maybe need another separate file for this
class CobottleneckDetector:
    """This is the cobottleneck detector with different methods like
    DCW, SBD, & proposed ML-based models. It should include the whole
    procedures of cobottleneck detection with the help of Pipeline or SBD,
    specially, data getter, prediction features i.e. queue, skewness, etc,
    clustering, and accuracy and speed evaluation with ground truth. The
    features include:
        
        1. Supports different detection methods for clustering.
        2. Supports online prediction with real-time data communication.
        3. Supports evaluation of accuracy and speed with ground truth.
        4. * Exports different types of data for official plotting.
            ? determine figure type before coding this
    """

    def __init__(self, flow_df, fields, interval,
        t_unit=0.005, label_path=None, labels=None,
        bin_classifier_path='bin_cls/random_forest_93_92.pkl',
        max_iter=10, num_threads=1):
        """Initialize the detector.
        Note that fields take effect in the conversion from df to array.
        """
        # labels is a dataframe w/ columns [run, flow, label] or
        # [run, flow, time, label]
        self.flow_df = flow_df
        assert self.flow_df[['run', 'flow', 'time']].duplicated().sum() == 0

        self.fields = fields
        self.interval = interval
        self.bin_classifier_path = bin_classifier_path
        self.t_unit = t_unit
        self.labels = None
        if labels is not None:
            self.labels = labels
        elif label_path is not None:
            self.labels = pd.read_csv(label_path, index_col=False)
        if self.bin_classifier_path is not None:
            with open(self.bin_classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
        self.sbd_df = None
        self.latent_map = {}
        self.flat_flows = []        # list of flat_flows of all times
        self.max_iter = max_iter
        self.flow_df = self._join_data_labels(self.flow_df, self.labels)
        # force it to use one thread to for inferecne for a correct time measurement
        if num_threads > 0:
            torch.set_num_threads(num_threads)

    def set_model_th_field(self, model_path, model_th,
                           fields=['owd', 'rtt', 'slr', 'cwnd']):
        """Set the model path, th, fields for current quick_run to support the
        config from model_th.csv"""
        self.model_path = model_path
        self.model_th = model_th
        if self.model_path is not None:
            model_info = torch.load(self.model_path)
            self.model = MySiameseModel(**model_info['params'])
            self.model.load_state_dict(model_info['state_dict'])
            self.model.eval()
            print(f' - model: {self.model_path} th: {self.model_th};')
            self.fields = fields
            if 'fields' in model_info:
                self.fields = model_info['fields']
            print(f'   fields: {self.fields}')
            assert len(self.fields) == model_info['params']['c_in']


    def quick_run(self, method, plot=True, interval=None, use_old_sbd=False,
                  converge_nonbtnk=False):
        """Quick run for convenience. SBD needs interval to be 
        specified, i.e. window duration ~ 0.35s.
        
        Args:
            method (str): within ['dcw', 'rmcat_sbd', 'dc_sbd', 'ml',
                                'neg_naive', 'pos_naive'].
                        naive_neg means labelling all flows as non-btnk, while
                        naive_pos means giving them the same label.
            plot (bool): if True, plot the accuracy
            interval (float): interval of the time series.
                        If None, use self.interval.
            use_old_sbd (bool): if True, use the old SBD df from stream_process().
        
        Returns:
            res_df: a dataframe ['time', 'precision', 'recall', 'f1', 'adj_rand_score'].
            predict_info: list of ys, y_hats (2d list of labels), and latent_map,
                    i.e. {t: latent}, latent = [flow_1_latent, flow_2_latent, ...]
                    flows (list of flows per time), and flat flows (for debug only)
            t_classify: the average processing time per second in data.
        """
        # TODO: SBDs need to run twice, but only difference is the clustering
        self.flat_flows = []
        self.interval = interval if interval is not None else self.interval
        seq_len = int(self.interval / self.t_unit)
        selected_dfs = self._get_selected_dfs(0, max(self.flow_df.time), seq_len)

        # use time.time() to measure the latency when torch using multiple threads
        t1 = time.process_time()
        # t1 = time.time()
        self.y_hats = self.classify(0, max(self.flow_df.time), selected_dfs,
                                    method=method, use_old_sbd=use_old_sbd,
                                    seq_len=seq_len)
        t2 = time.process_time()
        # t2 = time.time()
        # TODO: check the effect of t_classify below
        # self.t_classify = (t2 - t1) / (max(self.ticks) - min(self.ticks) + interval)
        self.t_classify = (t2 - t1) / len(self.ticks) / self.interval
        print(f'    t_classify: {self.t_classify:.3f} s, # ticks: {len(self.ticks)}, delta t: {t2 - t1:.3f} s')
        self.ys, flows = self.classify_truth(selected_dfs, method=method,
                                             seq_len=seq_len)

        assert abs(len(self.ys) - len(self.y_hats)) <= 2
        for i, (y, y_hat) in enumerate(zip(self.ys, self.y_hats)):
            assert len(y) == len(y_hat), \
                f'len(y) = {len(y)} != len(y_hat) = {len(y_hat)} at {i}th interval'
        res_df = self.evaluate(self.ys, self.y_hats, interval)
        if 'sbd' in method:
            n, n1 = 6, 4
        else:
            n, n1 = 2, 2
        # t_detect_df = get_detection_delay(self.ys, self.y_hats, flows, self.interval,
        #                                  n=n, n1=n1)
        t_detect_df = None
        if plot:
            plot_accuracy(res_df)
        predict_info = [self.ys, self.y_hats, self.latent_map, flows, self.flat_flows]
        return res_df, predict_info, self.t_classify, t_detect_df
 
    def classify(self, start, end, selected_dfs,
                method='dcw', use_old_sbd=False, seq_len=None):
        """Given flows' data of a period, output their group No. time series.
        Returns a list of list.
        """
        self.latent_map = {}
        if method == 'rmcat_sbd':
            return self._sbd_classify(start, end, 'rmcat', use_old=use_old_sbd)
        if method == 'dc_sbd':
            return self._sbd_classify(start, end, 'dc', use_old=use_old_sbd)

        # For DCW and ML, we want the segment to be exactly the same length of
        # interval due to the needs to utilize ML model. Thus, the 1st line below
        # is chosen and the incomplete segment in the end is discarded.
        y_hats = []
        for tick, selected in zip(self.ticks, selected_dfs):
            t0 = round(tick, 3)
            if 'label' in selected.columns:
                tmp_df = selected.drop(columns=['label'])
            y_hat, latent = self._classify_segment(tmp_df, method)
            y_hats.append(y_hat)
            self.latent_map[t0] = latent
        return y_hats

    @measure_time()
    def classify_truth(self, selected_dfs, method='dcw', seq_len=None):
        """Similar to classify, get ys from the flows' data."""
        ys, flows = [], []
        if 'sbd' in method:
            self.sbd_df.sort_values(by=['time', 'flow'], inplace=True)
            for t in self.sbd_df.time.unique():
                sbd_seg_df = self.sbd_df[self.sbd_df.time == t]
                ys.append(sbd_seg_df.label.tolist())
                flows.append(sbd_seg_df.flow.tolist())
                assert sbd_seg_df.flow.nunique() == len(sbd_seg_df.flow) \
                    == len(sbd_seg_df.label)
        else:
            for selected in selected_dfs:
                y = selected.groupby('flow')['label'].agg(lambda x: x.mode()[0])
                y = list(y)
                y_max = max(y)
                y = [y[i] if y[i] != -1 else y_max + 1 + i for i in range(len(y))]
                ys.append(list(y))
                flows.append(sorted(selected.flow.unique().tolist()))
        return ys, flows

    @measure_time()
    def evaluate(self, ys, y_hats, interval=None):
        """Evaluates accuracy by comparing the results inoutputs and
        labels.

        Args:
            ys (list): list of actual labels.
            y_hats (list): list of predicted labels.
            interval (float): interval of the time series. If None, use
                              self.interval.

        Returns:
            DataFrame: a dataframe of accuracy metrics by time
        """
        res_df = None
        interval = interval if interval is not None else self.interval
        for i, (y, y_hat) in enumerate(zip(ys, y_hats)):
            time = i * interval
            accuracy_metrics, cols = compare_segment(y, y_hat)
            tmp_df = pd.DataFrame([[time] + accuracy_metrics],
                                columns=['time'] + cols)
            res_df = tmp_df if res_df is None else pd.concat([res_df, tmp_df],
                                                           ignore_index=True)
        return res_df


    def _sbd_classify(self, start, end, cluster_alg='dc', use_old=False):
        """SBD method from start to end.

        Since SBD has defined window internally, we call it as a whole
        to produce the full group No. time series. Cluster can be rmcat
        or dc (Chinese Whisper).
        """
        start = max(start, self.flow_df.time.min())
        end = min(end, self.flow_df.time.max())
        flow_df = self.flow_df[(self.flow_df.time >= start) & (self.flow_df.time < end - 1e-6)]
        sbd = SBDAlgorithm()
        if use_old:
            sbd_df = self.sbd_df
        else:
            sbd_df = sbd.stream_process(flow_df, t_unit=self.t_unit)

        sbd_df1 = sbd_df.drop(columns=['label'])
        if cluster_alg == 'rmcat':
            y_hats = self._rmcat_sbd_classify(sbd, sbd_df1)
        elif cluster_alg == 'dc':
            y_hats = self._dc_sbd_classify(sbd, sbd_df1)
        self.sbd_df = sbd_df
        self.ticks = sbd_df.time.unique()
        return y_hats

    def _ml_classify_segment(self, segment):
        flow_labels = {flow: -1 for flow in segment.flow.unique()}
        # TODO: skip now to bypass the SLR issue
        flow_labels = update_non_btnk_flows(self.classifier, segment, flow_labels)
        x = self._get_array_from_segment(segment)
        y_hat, latent, flow_labels = predict_segment(self.model, x, max_iter=self.max_iter,
                                           th=self.model_th, flow_labels=flow_labels)
        return y_hat.tolist(), latent.tolist()

    def _rmcat_sbd_classify(self, sbd: SBDAlgorithm, sbd_df: pd.DataFrame):
        group_df = sbd.rmcat_clustering(sbd_df)
        group_df = group_df.sort_values(['time', 'flow']).reset_index(
                                        drop=True)
        # res = [group_df[group_df.time == t].group.tolist()
        #         for t in group_df.time.unique()]
        y_hats = []
        for i, t in enumerate(sorted(group_df.time.unique())):
            tmp_df = group_df[group_df.time == t]
            y_hat = tmp_df.group.tolist()
            self.latent_map[t] = None
            y_hats.append(y_hat.copy())
        return y_hats

    def _dc_sbd_classify(self, sbd: SBDAlgorithm, sbd_df: pd.DataFrame):
        # sbd_df: [time, flow, & skewness, variability, etc]
        y_hats = []
        sbd_df.set_index(['time'], inplace=True)
        sbd_df.sort_index(inplace=True)
        last_labels = None
        prev_btnk = {flow: False for flow in sbd_df.flow.unique()}
        for i, t in enumerate(sbd_df.index.unique()):
            seg_df = sbd_df.loc[t].copy()
            if type(seg_df) == pd.Series:   # Series is returned if only one flow
                seg_df = pd.DataFrame([seg_df])
            seg_df.sort_values(by='flow', inplace=True)
            seg_df.reset_index(inplace=True)
            if 'time' not in seg_df.columns:
                seg_df.rename(columns={'index': 'time'}, inplace=True)

            # use SBD btnk condition to first separate the non-btnk flows
            flow_labels = {flow: -1 for flow in seg_df.flow.unique()}
            f_max = max(seg_df.flow.unique())
            for ri, row in seg_df.iterrows():
                btnked = sbd.btnk_condition(row, prev_btnk[row.flow])
                prev_btnk[row.flow] = btnked
                if not btnked:
                    flow_labels[row.flow] = f_max + ri + 1

            # cluster the btnk flows
            seg_df = seg_df.drop(columns=['time', 'flow', 'd_avg'])
            # segment = torch.tensor(seg_df.to_numpy(dtype=np.float32))
            segment = seg_df.to_numpy(dtype=np.float32)
            self.latent_map[t] = segment.tolist()
            y_hat, flow_labels, _ = cluster(segment, max_iter=self.max_iter,
                                            n_voter=1, weighting='inv_sq',
                                            last_labels=last_labels,
                                            flow_labels=flow_labels)
            y_hat = y_hat.tolist()
            last_labels = flow_labels.copy()
            y_hats.append(y_hat.copy())

        sbd_df.reset_index(inplace=True)
        return y_hats

    def _get_selected_dfs(self, start, end, seq_len):
        """Get selected dfs for later usage in classify and classify_truth.
        Abstracted to maintain the consistency of the segments."""
        selected_dfs = []
        self.ticks = self._set_ticks(start, end, None)
        for tick in sorted(self.ticks):
            t0 = round(tick, 3)     # necessary for precision issue avoidance
            if t0 == self.flow_df.time.max():
                break
            selected = self._get_seg_df(self.flow_df, t0, seq_len)
            selected_dfs.append(selected)
        return selected_dfs

    def _set_ticks(self, start, end, ticks):
        start = max(start, self.flow_df.time.min())
        end = min(end, self.flow_df.time.max())
        ticks = ticks if ticks is not None else \
            [start + i * self.interval for i in range(int((end - start) / self.interval))]
        return ticks

    def _get_seg_df(self, df, t0, seq_len):
        """Ensure getting seq_len samples for each flow from t0.
        TODO: ensure head_enough is not slow for overhead test
        """
        def head_enough(df, n):
            if len(df) >= n:
                return df.head(n)
        interval = seq_len * self.t_unit
        seg_df = df[(df.time >= t0) &
                    (df.time < t0 + interval - 1e-6)].sort_values(by='time')\
                .groupby('flow', group_keys=False).apply(head_enough, seq_len)\
                .reset_index(drop=True)\
                .sort_values(by=['time', 'flow'])
        seg_df['cnt'] = seg_df.groupby('flow')['time'].transform('count')
        seg_df['t_start'] = seg_df.groupby('flow')['time'].transform('min')
        assert  seg_df.cnt.nunique() == 1, \
            f'Multiple flow count: {seg_df.cnt.unique()}'
        assert  seg_df.t_start.max() - seg_df.t_start.min() <= 0.005, \
            f'Multiple flow start time: {seg_df.t_start.unique()}'
        return seg_df
        # selected = df[(df.time >= t0) & (df.time < t0 + self.interval - 1e-6)]
        # assert selected.time.nunique() == seq_len, \
        #     f'selected.time.nunique() = {selected.time.nunique()} != seq_len = {seq_len}'
        # return selected

    def _classify_segment(self, segment, method):
        """Classify segments, i.e. flow df of a small period, into groups.
        Returns a list of group No. for each flow in the segment.
        """
        latent = None
        if method == 'dcw':
            res = self._dcw_classify_segment(segment)
        elif method == 'ml':
            res, latent = self._ml_classify_segment(segment)
        elif method == 'neg_naive':
            res = list(range(segment.flow.nunique()))
        elif method == 'pos_naive':
            res = [1] * segment.flow.nunique()
        else:
            raise NotImplementedError
        return res, latent

    def _dcw_classify_segment(self, segment):
        # TODO(Bowen): use pairwise flow owd correlations
        # TODO(Bowen): when using all flat signals, out is nan

        level = 6
        n_flow = segment.flow.nunique()
        th = self.model_th or 0.5

        d1 = DeNoise(segment)
        d2 = DeNoise(segment)
        flow_list = [-1 for i in range(n_flow)]
        group_number = 0
        flow_list[0] = 0

        for flow1 in range(n_flow): # how many flow we need to deal with?
            # flow 1/2 is the index, while flow 1/2_seg is the real flow number
            flow1_seg = segment.flow.unique()[flow1]
            d1.load_flow(flow1_seg)
            if d1.flow.nunique() == 1:
                # print(f' Warning: flow {flow1} has only one value, set group to NaN!')
                flow_list[flow1] = -1
                continue
            coe1 = d1.dwt(level)
            d1.threshold(coe1)
            coe1 = d1.soft_threshold(coe1)
            out1 = d1.idwt(coe1)
            for flow2 in range(flow1 + 1, n_flow):
                flow2_seg = segment.flow.unique()[flow2]
                d2.load_flow(flow2_seg)

                if d2.flow.nunique() == 1:
                    # print(f' Warning: flow {flow2} has only one value, set group to NaN!')
                    flow_list[flow2] = -1
                    continue
                coe2 = d2.dwt(level)
                d2.threshold(coe2)
                coe2 = d2.soft_threshold(coe2)
                out2 = d2.idwt(coe2)
                std1 = np.std(out1) if np.std(out1) else 1
                std2 = np.std(out2) if np.std(out2) else 1
                # assert np.std(out1) != 0 and np.std(out2) != 0, \
                #     f'flow {flow1} and {flow2} has std 0 in out!'
                assert not (np.isnan(std1) or np.isnan(std2)), \
                    f'flow {flow1} and {flow2} has NaN in out!'
                out1 = (out1 - np.mean(out1)) / (std1 * len(out1))
                out2 = (out2 - np.mean(out2)) / (std2)
                result = np.correlate(out1, out2)
                if result[0] > th:
                    if flow_list[flow2] == -1:
                        flow_list[flow2] = flow_list[flow1]
                    else:
                        flow_list[flow1] = min(flow_list[flow1], flow_list[flow2])
                        flow_list[flow2] = min(flow_list[flow1], flow_list[flow2])
                elif flow_list[flow2] == -1:
                    group_number = group_number + 1
                    flow_list[flow2] = group_number
        for flow1 in range(n_flow):
            if flow_list[flow1] == -1:
                group_number += 1
                flow_list[flow1] = group_number

        return flow_list

    # def _update_non_btnk_flows(self, sample, slr_sample, y_hat, skip=True):
    #     """Updates the y_hat by considering the flat signals in the sample.
    #     Assume the first feature in sample is the OWD.

    #     Args:
    #         sample (list): 2-D list (n_flow, seq_len)
    #         slr_sample (list): 2-D list (n_flow, seq_len)
    #         y_hat (list): list of the labels
    #         skip (bool): if skip this process
        
    #     TODO: skip set to True to use -1 as label for now

    #     Returns:
    #         list: the updated y_hat for given sample
    #     """
    #     assert len(y_hat) == len(sample)
    #     if skip:
    #         return y_hat
    #     seq_len = len(sample)

    #     _non_btnk = lambda i: DataGetter.f_non_btnk(i, sample, slr_sample)
    #     non_btnk_flows = filter(_non_btnk, range(len(sample)))
    #     new_y = max(y_hat)
    #     for i, flow in enumerate(non_btnk_flows):
    #         y_hat[flow] = new_y + i
    #     self.flat_flows.append(list(non_btnk_flows))
    #     return y_hat

    def _get_array_from_segment(self, segment):
        """Converts segment into 3D array for model & y_hat update.
        Assuming the segment has exactly seq_len time steps for each
        flow, but NOT necessarily the same time steps for all flows.

        Returns x as an array [n_flow, seq_len, n_features].
        """
        seq_len = int(self.interval / self.t_unit)
        sample = segment.sort_values(['flow', 'time']).reset_index(drop=True)
        x = sample[self.fields].to_numpy(dtype=np.float32)
        return x.reshape(sample.flow.nunique(), seq_len, len(self.fields))

    # [TODO: deprecate]
    # def _match_labels_flow(self, df, labels, i):
    #     """Match the flows in df with labels at time i."""
    #     if i >= labels.time.nunique():
    #         print(f'match_labels_flow: i={i}')
    #         print('labels.time.nunique()', labels.time.nunique())
    #         print('labels.time', labels.time.unique())
    #     t = sorted(labels.time.unique())[i]
    #     flows1 = set(df.flow.unique())
    #     flows2 = set(labels[labels.time == t].flow.unique())
    #     if flows1 == flows2:
    #         return df
    #     common_flows = flows1 & flows2
    #     df = df[df.flow.isin(common_flows)]
    #     labels.drop(labels[(labels.time == t) & (~labels.flow.isin(common_flows))].index,
    #         inplace=True)
    #     return df

    def get_sbd_df(self):
        """None if method is not SBD family."""
        return self.sbd_df
    
    def get_flat_flows(self):
        return self.flat_flows

    def _join_data_labels(self, data, labels):
        if 'time' in labels:
            assert len(set(data.time.unique()) & set(labels.time.unique())) > \
                0.95 * labels.time.nunique(), 'data and labels have different time'
            data = data.merge(labels, on=['run', 'time', 'flow'], how='inner')
        else:
            data = data.merge(labels, on=['run', 'flow'], how='inner')
        return data


if __name__ == '__main__':
    # runner = unittest.TextTestRunner()
    # runner.run(suite())
    pass
