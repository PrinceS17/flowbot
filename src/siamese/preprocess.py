import argparse
import os
import psutil
import numpy as np
import pandas as pd
# import modin.pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess as sp
import time
import torch

from collections import Counter

os.environ['MODIN_ENGINE'] = 'dask'

from siamese.indexed_sample_io import IndexedSampleIO
from siamese.reader import ConfigReader, DataReader

def measure_time(tag=None):
    """This decorator is used to measure the time of a function.
    tag is given to avoid sometimes the function name is another wrapper.
    """
    def time_decorator(func):
        def time_wrapper(*args, **kwargs):
            start = time.time()
            ftag = tag if tag else func.__name__
            print(f" {start:.3f} s: {ftag} starts ...")
            res = func(*args, **kwargs)
            end = time.time()
            print(f" {end:.3f} s: {ftag} ends, time elapsed: {end - start:.6f} s")
            return res
        return time_wrapper
    return time_decorator


def measure_mem(tag=None):
    """This decorator is used to measure the memory of a function.
    tag is given to avoid sometimes the function name is another wrapper.
    """
    def mem_decorator(func):
        def mem_wrapper(*args, **kwargs):
            mem1 = psutil.Process(os.getpid()).memory_info().rss / 1e6
            ftag = tag if tag else func.__name__
            res = func(*args, **kwargs)
            pid = os.getpid()
            mem2 = psutil.Process(pid).memory_info().rss / 1e6
            print(f"    pid {pid}: {ftag} memory: {mem1:.2f} -> {mem2:.2f} MB, change {mem2 - mem1:.2f} MB")
            return res
        return mem_wrapper
    return mem_decorator


def multi_run_handler(n_in, n_out):
    """This decorator is used to allow multi-run processing for the
    functions that only allow one run in the dataframe.

    Args:
            n_in (int): number of input args that want to split runs.
                                    Note that n_in can be less than # of all args.
            n_out (int): number of output args that need merging runs.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            results = [None] * n_out
            runs = args[0].run.unique()
            for run in runs:
                new_args = []
                for i in range(len(args)):
                    if i < n_in:
                        df = args[i]
                        assert 'DataFrame' in str(type(df))
                        new_args.append(df[df.run == run].copy())
                    else:
                        new_args.append(args[i])
                res = func(*new_args, **kwargs)
                if n_out == 1:
                    res = [res]
                for i in range(n_out):
                    results[i] = res[i] if results[i] is None else pd.concat(
                        [results[i], res[i]], ignore_index=True, copy=False)
            return tuple(results) if n_out > 1 else results[0] if n_out == 1\
                else None
        return wrapper
    return decorator


class DataChecker:
    """
    Checker those that only check if the dataframe for a specific run
    pass some validicity check, and return a boolean.

    The following dataframes are assumed as the inputs.

    Dataframe from the data files: 
        xdf: [(run), flow, time, owd, rtt, ..., qid1, qid2]
        qdf: [(run), qid, time, packet_in_queue, max_q_size]

    Dataframe from the config files:
        btnk_df (link + cross): [(run), src, dst, qid, num, bw_avail ...]
        flow_df: [(run), src, dst, num, rate, ...]

    Run will be one of the original columns, but here we assume run is
    either the same for all rows or not in the dataframe.
    """

    @staticmethod
    @multi_run_handler(1, 0)
    def check_tick_number(df):
        assert not df.empty
        assert df.time.nunique() >= 1

    @staticmethod
    @measure_time('check_flow_number')
    @multi_run_handler(1, 0)
    def check_flow_number(xdf, flow_df):
        # check the flow number for each path in the data and config
        assert xdf.run.nunique() == flow_df.run.nunique() == 1
        assert xdf.flow.nunique() == sum(flow_df.num)

    @staticmethod
    @measure_time()
    def check_queue_size(qdf):
        # check if the queue size is valid, should be used after merge_qsize
        for col in ['qid', 'packet_in_queue', 'max_q_size']:
            assert col in qdf.columns
        assert (qdf.packet_in_queue <= qdf.max_q_size).all()

    @staticmethod
    # @measure_time()
    def check_multi_btnk(qdf, btnk_df, q_size_dev=100):
        # check if the current run contains multiple btnk links
        # TODO: here use queue to infer, but actually not strictly follow the
        # definition as the link bw inference is harder to implement
        agg_qdf = qdf.groupby(['qid'])['packet_in_queue'].mean()
        agg_qdf = agg_qdf.reset_index()
        assert agg_qdf.shape[0] == btnk_df.shape[0]
        btnks = agg_qdf.packet_in_queue / (btnk_df.q_size + q_size_dev) > 0.9
        return btnks.sum() > 1

    @staticmethod
    @measure_time()
    def check_label_change(y, start, end):
        """Checks if the label changes in given interval.
        Truth_df: [(run), flow, time, label]
        Returns how many label changes are detected in the interval.
        """
        y = y[(y.time >= start) & (y.time < end)]
        y = y.groupby('flow').agg(lambda x: x.nunique()).reset_index()
        return len(y[y.label > 1])

    @staticmethod
    @measure_time('check_time_alignment')
    @multi_run_handler(2, 0)
    def check_time_alignment(df1, df2):
        """For all the flows that are in both dfs, check if the time
        stamps are aligned."""
        for col in ['run', 'flow', 'time']:
            assert col in df1.columns and col in df2.columns
        assert df1.run.nunique() == 1 and df2.run.nunique() == 1
        common_flows = set(df1.flow.unique()) & set(df2.flow.unique())
        df1 = df1[df1.flow.isin(common_flows)].reset_index(drop=True)
        df2 = df2[df2.flow.isin(common_flows)].reset_index(drop=True)
        tmp1 = df1.groupby(['run', 'flow']).count().reset_index()
        tmp2 = df2.groupby(['run', 'flow']).count().reset_index()
        assert (tmp1[['run', 'flow', 'time']] == \
                tmp2[['run', 'flow', 'time']]).all().all()

    @staticmethod
    @measure_time()
    def check_time_steps(df, step=0.005):
        """Check there's at least one flow containing all time steps.
        TODO: what to do about this? Reject the whole data?
        """
        assert np.isclose((df.time.nunique() - 1) * step, df.time.max() - df.time.min()), \
            f'# time steps: {df.time.nunique()}, start-end: {df.time.min()}-{df.time.max()}'

    @staticmethod
    @measure_time('check_segment_alignment')
    @multi_run_handler(1, 0)
    def check_segment_alignment(df, seq_len):
        assert df.time.nunique() % seq_len == 0
        n_segment = df.time.nunique() / seq_len
        df1 = df.groupby(['run', 'flow'], as_index=False).count()
        df1['cnt'] = df1['time']
        assert (df1.cnt % seq_len == 0).all()
        # TODO: check the count of [flow, time] within the single interval
        #       is the multiple of seq_len


class DataGetter:
    # nonbtnk condition group, collected here for consistent change
    
    Q_SLR = 0.7
    OWD_TH = 1e-3
    SLR_LOW_TH = 1e-3
    SLR_HIGH_TH = 2e-3

    # query used in process_invalid_entry, the meaning is to remove
    #   1) flat owd w/ relative small slr;
    #   2) close to zero slr, whether the owd is
    NONBTNK_QUERY = f'(owd_std <= {OWD_TH} and high_slr < {SLR_HIGH_TH}) ' \
                    f'or high_slr <= {SLR_LOW_TH}' 

    @staticmethod
    def converge_nonbtnk_label(y):
        """Given a label list y, convert those that only occurs once to -1 to
        represent non-btnk. Note that this function only works for 3d-scan
        scenario, where we know for sure that those flows are non-btnk but
        not self-btnked."""
        cnter = Counter(y)
        y1 = [x if cnter[x] > 1 else -1 for x in y]
        return y1

    @staticmethod
    def inflate_nonbtnk_label(y):
        """Inflate the -1 label in y to unique flows."""
        new_y = max(y) + 1
        for i in range(len(y)):
            if y[i] == -1:
                y[i], new_y = new_y, new_y + 1
        return y

    @staticmethod
    def f_non_btnk(i, owd, slr):
        """Given i, owd, slr, return if i-th flow is non-btnk flow. Used in filters."""
        high_slr = np.quantile(slr[i], DataGetter.Q_SLR)
        return np.std(owd[i]) < DataGetter.OWD_TH and high_slr < DataGetter.SLR_HIGH_TH \
            or high_slr <= DataGetter.SLR_LOW_TH

    @staticmethod
    @measure_time()
    def find_qid_to_flow(qid_df):
        """Find qid to flow mapping by scanning qid_df.
        
        Returns:
            dict: {run: {qid: [flows]}}
        """
        # qid_df: [run, flow, qid1, qid2]
        qid_to_flow = {}
        for run in qid_df.run.unique():
            qid_to_flow[run] = {}
            qid_df_run = qid_df[qid_df.run == run]
            for _, row in qid_df_run.iterrows():
                for qid in [row.qid1, row.qid2]:
                    if qid not in qid_to_flow[run]:
                        qid_to_flow[run][qid] = []
                    qid_to_flow[run][qid].append(row.flow)
        return qid_to_flow
    
    @staticmethod
    def find_flow_to_qid(qid_df):
        """Find flow to qid mapping.
        
        Returns:
            dict: {run: {flow: [qid1, qid2]}}
        """
        flow_to_qid = {}
        for run in qid_df.run.unique():
            flow_to_qid[run] = {}
            qid_df_run = qid_df[qid_df.run == run]
            for _, row in qid_df_run.iterrows():
                flow_to_qid[run].setdefault(row.flow, [row.qid1, row.qid2])
        return flow_to_qid

    @staticmethod
    # @measure_time()
    def get_double_queues(qdf, qid1, qid2):
        """Get queue 1 and queue 2's packets data, rolling window average,
        and label. Intended to be used for getting two queues' data for
        the same flow.
        
        Returns:
            DataFrame: [run, time, q_win1, q_win2,
                        packet_in_queue, packet_in_queue2, label]
        """

        def _truth(row, qid1, qid2, th=50):
            if row.q_win1 < th and row.q_win2 < th:
                # return 'none'
                return '-1'
            return qid1 if row.q_win1 > row.q_win2 else qid2

        assert qdf.run.nunique() == 1
        df = qdf[qdf.qid == qid1].reset_index(drop=True)
        queue2 = qdf[qdf.qid == qid2].reset_index(drop=True)
        assert df.shape[0] == queue2.shape[0], \
            f'run: {df.run.unique()}, qid1: {qid1}, qid2: {qid2}, df.shape: {df.shape}, queue2.shape: {queue2.shape}'
        df['q_win1'] = df.packet_in_queue.rolling(10, min_periods=1).mean()
        df['q_win2'] = queue2.packet_in_queue.rolling(10, min_periods=1).mean()
        # df['packet_in_queue2'] = queue2.packet_in_queue
        df['label'] = df.apply(_truth, args=(qid1, qid2), axis=1)
        df['qid1'] = qid1
        df['qid2'] = qid2
        return df

    @staticmethod
    @measure_time('get_truth_from_queue')
    @multi_run_handler(2, 1)
    def get_truth_from_queue(qid_df, qdf):
        """[Critical] Update the truth value by comparing the relative queue size.

        Args:
            qid_df: [(run), flow, qid1, qid2]
            qdf: [(run), qid, time, packet_in_queue, max_q_size]

        Returns:
            truth_df: [(run), flow, time, label],
                                label is one of (qid1, qid2, none).
        """
        assert 'max_q_size' in qdf.columns, 'q_size not merged into qdf yet.'
        qpairs = qid_df.drop_duplicates(['run', 'qid1', 'qid2'])
        qid_to_label_df = None
        for _, row in qpairs.iterrows():
            qdf_run = qdf[qdf.run == row.run]
            df = DataGetter.get_double_queues(qdf_run, row.qid1, row.qid2)
            df = df[['run', 'qid1', 'qid2', 'time', 'label']]
            qid_to_label_df = df.copy() if qid_to_label_df is None else \
                              pd.concat([qid_to_label_df, df], copy=False)
        truth_df = qid_df.merge(qid_to_label_df, on=['run', 'qid1', 'qid2'],
                                copy=False)
        return truth_df[['run', 'flow', 'time', 'label']].reset_index(drop=True)


    # @staticmethod
    # @measure_time('get_truth_from_queue')
    # @multi_run_handler(2, 1)
    # def get_truth_from_queue(qid_df, qdf):
    #     """Update the truth value by comparing the relative queue size.

    #     In real, we should leave it in doubt if no one is dominating,
    #     But here for simplicity we just choose the larger one.

    #     Args:
    #         qid_df: [(run), flow, qid1, qid2]
    #         qdf: [(run), qid, time, packet_in_queue, max_q_size]

    #     Returns:
    #         truth_df: [(run), flow, time, label],
    #                             label is one of (qid1, qid2, none).
    #     """
    #     truth_df = None
    #     assert 'max_q_size' in qdf.columns, 'q_size not merged into qdf yet.'
    #     assert qid_df.run.nunique() == qdf.run.nunique() == 1, 'Only 1 run supported.'
    #     qpair_to_label = {}     # {(qid1, qid2): label}
    #     qpairs = qid_df.drop_duplicates(['run', 'qid1', 'qid2'])
    #     for _, row in qpairs.iterrows():
    #         df = DataGetter.get_double_queues(qdf, row.qid1, row.qid2)

    #         qpair_to_label[(row.qid1, row.qid2)] = df[['time', 'label']].copy()
    #     for _, row in qid_df.iterrows():
    #         df = qpair_to_label[(row.qid1, row.qid2)]
    #         df['run'] = row.run
    #         df['flow'] = row.flow
    #         # df = df[['run', 'flow', 'time', 'label']]
    #         truth_df = df.copy() if truth_df is None else pd.concat([truth_df, df],
    #                                                          ignore_index=True)
    #     return truth_df

    @staticmethod
    # @measure_time()
    def fetch_truth_for_segment(y, start, end):
        """Fetch the truth of df [run, flow, label] from [run, flow, time, label].

        Args:
            y (DataFrame): df of [run, flow, time, label] w/ only one run.
            start (float): start time
            end (float): end time

        Returns:
            DataFrame: dataframe w/ [run, flow, label]
        """
        assert not y.empty
        assert y.run.nunique() == 1, 'Only one run is allowed.'
        y = y[(y.time >= start) & (y.time < end)]
        y1 = y.groupby('flow').agg(lambda x: x.mode()[0])
        y1 = y1.drop(columns='time').reset_index()    # keep [run, flow, label]
        y1 = y1[['run', 'flow', 'label']]
        return y1
    
    @staticmethod
    def mark_invalid_owd(df):
        """Given xdf, returns the invalid flows as a list.
        Note that process_invalid_entry targets for the whole sequence instead
        of one segment.
        """
        assert 'owd' in df.columns
        def _check_owd_jump(xdf_flow):
            # check if there's invalid jump in owd time series
            jump = list(xdf_flow.diff().abs().nlargest(4))[-1]
            return jump > 0.5 * xdf_flow.mean() or jump > 0.03

        # remove owd w/ invalid large jump, or too small variation
        df.sort_values(by=['run', 'flow', 'time'], inplace=True)
        df['owd_std'] = df.groupby(['run', 'flow'])['owd'].transform('std')
        df['mid_owd'] = df.groupby(['run', 'flow'])['owd'].transform('median')

        # remove the flows w/ invalid owd jump
        df['jump'] = df.groupby(['run', 'flow'])['owd'].transform(_check_owd_jump)
        res = df.loc[(df.owd_std <= 6e-4) | (df.mid_owd > 0.5) | df.jump,
            ['run', 'flow']].drop_duplicates()
        return res

    @staticmethod
    @measure_time('get_non_btnk_flows')
    def get_non_btnk_flows(df_seg):
        """Return the non-btnk flows for segment. Note that the condition
        of owd th and slr th must be satisfied at the same time, as it is
        possible that highly congested link also has flat OWD!
        """
        df_seg = df_seg.copy()
        df_seg.set_index(['flow'], inplace=True)
        df_seg.sort_index(inplace=True)
        df_seg.sort_values(by=['time'], inplace=True)
        all_flows = sorted(df_seg.index.unique())
        x_owd = [df_seg.loc[flow, 'owd'].tolist() for flow in all_flows] 
        x_slr = [df_seg.loc[flow, 'slr'].tolist() for flow in all_flows]
        _non_btnk = lambda i: DataGetter.f_non_btnk(i, x_owd, x_slr)
        non_btnk_flows = filter(_non_btnk, range(len(x_owd)))
        return list(map(lambda i: all_flows[i], non_btnk_flows))


class DataModifier:
    @staticmethod
    def get_updated_nonbtnk_labels(y):
        # update the flow label to -1 if it only occurs once
        assert type(y) == list
        cnter = Counter(y)
        one_time_flows = [k for k, v in cnter.items() if v == 1]
        y1 = [x if x not in set(one_time_flows) else -1 for x in y] 
        return y1

    @staticmethod
    def update_nonbtnk_labels_for_df(seg_df):
        # update the label to -1 if it only occurs once
        seg_df['cnt'] = seg_df.groupby(['run', 'label']).flow.transform('nunique')
        seg_df.loc[seg_df.cnt == 1, 'label'] = -1
        seg_df.drop(columns=['cnt'], inplace=True)

    @staticmethod
    def translate_runs(btnk_df, flow_df, xdf):
        # translate the relative run in btnk_df/flow_df into
        # absolute runs
        if not btnk_df.run.nunique() == flow_df.run.nunique() == \
            xdf.run.nunique():
            print('btnk_df runs:', btnk_df.run.unique())
            print('flow_df runs:', flow_df.run.unique())
            print('xdf runs:', xdf.run.unique())
            raise ValueError('btnk_df/flow_df/xdf should have the same runs.')
        run_map = {r1 : r2 for r1, r2 in zip(sorted(btnk_df.run.unique()),
                                             sorted(xdf.run.unique()))}
        btnk_df['run'] = btnk_df.apply(lambda x: run_map[x.run], axis=1)
        flow_df['run'] = flow_df.apply(lambda x: run_map[x.run], axis=1)
        return btnk_df, flow_df
    
    @staticmethod
    def add_drop_from_slr(xdf):
        if 'drop' in xdf.columns:
            return
        xdf['drop'] = xdf.groupby(['run', 'flow'])['slr'].transform(
            lambda s: s.diff().gt(0).astype(int))

    @staticmethod
    def convert_delay_to_s(xdf):
        for col in ['owd', 'rtt']:
            if xdf[col].median() >= 1000:
                xdf[col] = xdf[col] / 1e6

    @staticmethod
    def convert_ms_to_s(df):
        assert 'time' in df.columns
        t_median = df.time.diff().median()
        if t_median >= 1:
            assert np.isclose(t_median, 5)
            df['time'] = df.time / 1000

    @staticmethod
    def add_slr(xdf, win=1000):
        # add slr to xdf
        if 'slr' in xdf.columns:
            return
        xdf['slr'] = xdf.groupby(['run', 'flow'])['drop'].transform(
            lambda s: s.rolling(win, min_periods=1).sum() / win)

    @staticmethod
    def convert_flow_to_int(df):
        # convert the flow field from str to int
        flow_to_idx = {}
        for i, flow in enumerate(sorted(df.flow.unique())):
            flow_to_idx[flow] = i
        df['flow'] = df.flow.map(flow_to_idx)

    @staticmethod
    @measure_time()
    def merge_qsize(qdf, btnk_df, q_size_dev=100):
        # merge the queue size info from btnk_df to qdf
        if 'q_size' in qdf.columns or 'max_q_size' in qdf.columns:
            return qdf
        btnk_qdf = btnk_df[['run', 'qid', 'q_size']].copy()
        btnk_qdf['max_q_size'] = btnk_qdf['q_size'] + q_size_dev
        qdf = qdf.merge(btnk_qdf, on=['run', 'qid'], copy=False)
        return qdf

    # @staticmethod
    # @measure_time('match_time')
    # @multi_run_handler(2, 2)
    # def match_time(xdf, qdf, HZ=200):
    #     """Match the time sequence of xdf and qdf when timestamps are
    #     not the same by flow.
    #     """
    #     assert xdf.run.nunique() == qdf.run.nunique() == 1, \
    #         'Only 1 run supported.'
    #     xdf, qdf = xdf.copy(), qdf.copy()
    #     xdf['time'] = round(xdf.time * HZ) / HZ   # round to 5ms
    #     qdf['time'] = round(qdf.time * HZ) / HZ

    #     for flow in xdf.flow.unique():
    #         xdf_flow = xdf[xdf.flow == flow]
    #         qdf_flow = qdf[qdf.flow == flow]
    #         t_overlap = set(xdf_flow.time) & set(qdf_flow.time)
    #         xdf = xdf.drop(xdf[(xdf.flow == flow) & (
    #             ~xdf.time.isin(t_overlap))].index)
    #         qdf = qdf.drop(qdf[(qdf.flow == flow) & (
    #             ~qdf.time.isin(t_overlap))].index)
    #     return xdf, qdf
    
    @staticmethod
    @measure_time('match_time')
    # @multi_run_handler(2, 2)
    def match_time(xdf, qdf, HZ=200):
        """Match the time sequence of xdf and qdf when timestamps are
        not the same by flow.
        
        """
        xdf.loc[:, 'time'] = round(xdf.time * HZ) / HZ   # round to 5ms
        qdf.loc[:, 'time'] = round(qdf.time * HZ) / HZ
        DataModifier.match_run_flow_time_inplace(xdf, qdf)


    @staticmethod
    @measure_time()
    def match_run_flow_time_inplace(xdf, qdf):
        xdf.set_index(['run', 'flow', 'time'], inplace=True)
        qdf.set_index(['run', 'flow', 'time'], inplace=True)

        x_extra = xdf.index.difference(qdf.index)
        q_extra = qdf.index.difference(xdf.index)
        xdf.drop(x_extra, inplace=True)
        xidx = set(xdf.index)
        xdf.reset_index(inplace=True)
        qdf.drop(q_extra, inplace=True)
        qidx = set(qdf.index)
        qdf.reset_index(inplace=True)
        assert xidx == qidx


    @staticmethod
    @measure_time()
    def crop_time_sequence(xdf, seq_len):
        """Crop the per-flow time sequence to the multiple of given seq_len.
        Pandas: Number rows of each flow using cumcount, and select only
        the multiple of seq_len rows.
        """
        xdf = xdf.sort_values(
            by=['run', 'flow', 'time']).reset_index(drop=True)
        xdf['num'] = xdf.groupby(['run', 'flow']).cumcount()
        xdf['cnt'] = xdf.groupby(['run', 'flow'])['time'].transform('count')
        xdf = xdf[xdf.num < xdf.cnt - (xdf.cnt % seq_len)]
        return xdf.drop(columns=['num', 'cnt'])

    @staticmethod
    @measure_time('crop_time_sequence_by_flow')
    @multi_run_handler(1, 1)
    def crop_time_sequence_by_flow(xdf, interval):
        """[Critical] Crop every flow to make it aligned with interval, i.e.
        flows data only start at the interval boundary, and for each given
        interval, all flows' data are aligned.

        Faster, also assume 1) xdf is continuous in time; 2) t_unit are consistent.
        """
        df = xdf.copy()
        df['f_start_tick'] = df.groupby(['run', 'flow'])['time'].transform(
            lambda ts: np.ceil(min(ts) / interval) * interval)
        df['f_end_tick'] = df.groupby(['run', 'flow'])['time'].transform(
            lambda ts: np.floor(max(ts) / interval) * interval)
        df = df[(df.time >= df.f_start_tick) & (df.time < df.f_end_tick)]
        return df.drop(columns=['f_start_tick', 'f_end_tick']).reset_index(drop=True)


    @staticmethod
    @measure_time('crop_time_for_prediction')
    @multi_run_handler(1, 1)
    def crop_time_for_prediction(df, seq_len):
        """Crop time only, regardless of each flow, for later flow padding.
        """
        assert df.run.nunique() == 1
        times = sorted(df.time.unique())
        times = set(times[:len(times) - (len(times) % seq_len)])
        return df[df.time.isin(times)]

    @staticmethod
    @measure_time()
    def crop_sample_for_flow_change(sample):
        """For given sample, check if it contains different flows. If so, we
        discard the flows that change in this interval to make sure the flow
        number keeps consistent.

        Possible issue: if multiple flows start just after the first time step,
        or ends just before the last time step, then they are all discarded,
        and the detecting latency will be very high.
        """
        for col in ['flow', 'time']:
            assert col in sample.columns, f'{col} not in sample.'
        assert sample.run.nunique() == 1, 'Only 1 run supported.'
        consistent_flows = None
        n_flows = []
        for t in sample.time.unique():
            flows = set(sample[sample.time == t].flow.unique())
            n_flows.append(len(flows))
            if consistent_flows is None:
                consistent_flows = flows
            else:
                consistent_flows = consistent_flows & flows
        assert min(n_flows) == len(consistent_flows)
        if np.percentile(n_flows, 50) > len(consistent_flows):
            print(f'WARNING: # consistent flows {len(consistent_flows)} < 50% of n_flows'
                  f' {np.percentile(n_flows, 50)}.')
        sample = sample[sample.flow.isin(consistent_flows)]
        return sample

    # @staticmethod
    # @measure_time('process_invalid_entry')
    # @multi_run_handler(1, 1)
    # def process_invalid_entry(df):
    #     """Remove outliers, NaN, and invalid flow data with owd = 0."""
    #     df = df.dropna()
    #     if 'label' in df.columns:
    #         assert 'flow' in df.columns
    #         df.loc[df.label == 'none', 'label'] = df[df.label == 'none'].apply(
    #             lambda r: str(r.flow), axis=1)
    #     elif 'owd' in df.columns:
    #         for flow in df.flow.unique():
    #             tmp_df = df[df.flow == flow]
    #             if tmp_df.owd.max() == 0:
    #                 df = df.drop(tmp_df.index)
    #     return df.reset_index(drop=True)


    @staticmethod
    @measure_time()
    def remove_zero_owds(xdf):
        def _first_non_zero_time(xdf_flow):
            # check the std of the non-zero part of owd
            # note that i_start should be the idx relative to xdf_flow
            i_starts = xdf_flow.owd.to_numpy().nonzero()[0]
            if len(i_starts) > 0:
                return xdf_flow.iloc[i_starts[0]].time
            return xdf_flow.iloc[-1].time + 1
        
        xdf.set_index(['run', 'flow'], inplace=True)
        xdf['t_non_zero'] = xdf.groupby(['run', 'flow']).apply(
            _first_non_zero_time)
        xdf.reset_index(inplace=True)
        xdf.drop(xdf[xdf.time < xdf.t_non_zero].index, inplace=True)
        xdf.drop(columns=['t_non_zero'], inplace=True)

    @staticmethod
    @measure_time('process_invalid_entry')
    # @multi_run_handler(1, 1)
    def process_invalid_entry(df):
        """Remove outliers, NaN, and invalid flow data with owd = 0.
        Modified to include the run inside to speed up. Should be
        run after remove_zero_owds, as 0s interfere the std."""
        # df = df.dropna()
        # if 'label' in df.columns:
        #     assert 'flow' in df.columns
        #     df.loc[df.label == 'none', 'label'] = df[df.label == 'none'].apply(
        #         lambda r: str(r.flow), axis=1)
        # elif 'owd' in df.columns:
        #     # for simulated data, shouldn't remove any
        #     df['max_owd'] = df.groupby(['run', 'flow'])['owd'].transform('max')
        #     df = df[df.max_owd > 0]
        # return df.reset_index(drop=True)

        df.dropna(inplace=True)
        keys = set(['run', 'flow', 'time']) & set(df.columns)
        df.drop_duplicates(subset=keys, inplace=True)
        # TODO: deprecated, now just use a unified label for non-btnk flows
        # if 'label' in df.columns:
        #     assert 'flow' in df.columns
        #     df.loc[df.label == 'none', 'label'] = df[df.label == 'none'].apply(
        #         lambda r: str(r.flow), axis=1)

        # TODO: deprecated, now non-btnk flows shouldn't be dropped
        # if 'owd' in df.columns:
        #     def _check_owd_jump(xdf_flow):
        #         # check if there's invalid jump in owd time series
        #         if len(xdf_flow) <= 1:      # only 1 data point
        #             return True
        #         jump = list(xdf_flow.diff().abs().nlargest(4))[-1]
        #         return jump > 0.5 * xdf_flow.mean() or jump > 0.05

            # # remove owd w/ invalid large jump, or too small variation
            # df.sort_values(by=['run', 'flow', 'time'], inplace=True)
            # df['owd_std'] = df.groupby(['run', 'flow'])['owd'].transform('std')
            # df['high_slr'] = df.groupby(['run', 'flow'])['slr'].transform(
            #     lambda x: x.quantile(DataGetter.Q_SLR))
            # df['mid_owd'] = df.groupby(['run', 'flow'])['owd'].transform('median')

            # # remove the flows w/ invalid owd jump
            # df['jump'] = df.groupby(['run', 'flow'])['owd'].transform(_check_owd_jump)
            # # due to the slr precision, < 1e-3 is just 0 in our implementation
            # query_str = DataGetter.NONBTNK_QUERY + ' or mid_owd > 0.5 or jump == True'
            # df.drop(df.query(query_str).index, inplace=True)
            # df.drop(columns=['owd_std', 'mid_owd', 'jump', 'high_slr'],
            #         inplace=True)

    @staticmethod
    @measure_time()
    def match_flow(df1, df2):
        flows = set(df1.flow.unique()) & set(df2.flow.unique())
        df1.drop(df1[~df1.flow.isin(flows)].index, inplace=True)
        df2.drop(df2[~df2.flow.isin(flows)].index, inplace=True)

    @staticmethod
    @measure_time('match_flow_of_data')
    @multi_run_handler(2, 1)
    def match_flow_of_data(xdf, qid_df):
        """Match the flow sequence of xdf and qid_df when flows are not the same,
        mainly due to the removement of owd = 0 flows in xdf.
        """
        # assert xdf.run.nunique() == qid_df.run.nunique() == 1
        # # assume qid_df has more flows than xdf
        # for flow in qid_df.flow.unique():
        #     if flow not in xdf.flow.unique():
        #         qid_df = qid_df.drop(qid_df[qid_df.flow == flow].index)
        # return qid_df.reset_index(drop=True)

        # new workflow: 1) set index; 2) index difference; 3) drop in place
        assert xdf.run.nunique() == qid_df.run.nunique() == 1
        xdf = xdf.set_index(['run', 'flow'])
        qid_df = qid_df.set_index(['run', 'flow'])
        qid_df = qid_df.drop(qid_df.index.difference(xdf.index))
        return qid_df.reset_index()

    @staticmethod
    @measure_time('match_flow_along_time')
    @multi_run_handler(2, 2)
    def match_flow_along_time(xdf, truth_df):
        """For each interval, make sure the flow No. matches for easy calculation
        of the accuracy later.

        TODO: seems unnecessary, as each flows' times are already matched?
        """
        # first round, rm flow in truth but not in xdf
        tdf = truth_df.copy()
        assert xdf.run.nunique() == tdf.run.nunique() == 1
        for time in tdf.time.unique():
            x_flows = set(xdf[xdf.time == time].flow.unique())
            truth_flows = set(tdf[tdf.time == time].flow.unique())
            tdf.drop(tdf[(tdf.time == time) &
                (tdf.flow.isin(truth_flows - x_flows))].index, inplace=True)
        
        # second round, check if any flow in xdf but not in truth
        for time in xdf.time.unique():
            x_flows = set(xdf[xdf.time == time].flow.unique())
            truth_flows = set(tdf[tdf.time == time].flow.unique())
            assert len(x_flows - truth_flows) == 0
        return xdf,tdf 

    @staticmethod
    @measure_time()
    def fetch_truth_for_segments(y, interval):
        """[Critical] Fetch the truth of df [run, flow, time(tick), label]
        from [run, flow, time, label].

        Args:
            y (DataFrame): df of [run, flow, time, label]

        Returns:
            DataFrame: dataframe w/ [run, flow, time, label]
        """
        y_tick = y.copy()
        y_tick['time'] = (y_tick.time - y_tick.time.min()) // interval * interval + \
                         y_tick.time.min()
        DataChecker.check_tick_number(y_tick)
        y_tick = y_tick.groupby(['run', 'flow', 'time'], as_index=False).agg(
            lambda x: x.mode()[0])
        y_tick = y_tick[['run', 'flow', 'time', 'label']]
        return y_tick.reset_index(drop=True)

    @staticmethod
    @measure_time('pad_flows')
    @multi_run_handler(1, 1)
    def pad_flows(df, seq_len, pre_padding_mode='zero', mid_padding_mode='last',
                  post_padding_mode='self'):
        """
        Pad the flows with incomplete data at some time interval of a
        specific run. Note that this only fills the flow at interval level,
        and does not guarantee all flows have same number of times in total!
        This operations change the traffic pattern, and should
        only be used in prediction but not training.

        [Design]
        1. divide the sequence into segments by time, here we assume the
             at least one flow's times contains no gaps. Checker of the time
             step is needed to guarantee this.
        2. pad the flows in each segment with the padding mode.

        Args:
                df (DataFrame): input df
                padding (str, optional): Padding mode among [self, zero, last].
                                                                 Defaults to 'self'.

        Returns:
                DataFrame: the padded df.
        """
        for i, mode in enumerate([pre_padding_mode, mid_padding_mode, post_padding_mode]):
            assert mode in ['self', 'zero', 'last']

        def _get_zero_row(columns, run, flow, t):
            row = pd.DataFrame([[0] * len(columns)], columns=columns)
            row['run'] = run
            row['flow'] = flow
            row['time'] = t
            return row

        def _get_last_row(row, run, flow, t):
            row = row.copy()
            row['run'] = run
            row['flow'] = flow
            row['time'] = t
            return row

        def _fill_blanks(df, row, blank_times):
            cur_df = pd.concat([row] * len(blank_times),
                               ignore_index=True, copy=False)
            cur_df['time'] = pd.Series(blank_times)
            return pd.concat([df, cur_df], ignore_index=True, copy=False)

        def _fill_with_self_seq(df, flow_df, blank_times):
            """Use the samples in the middle """
            while len(blank_times) > len(flow_df):
                flow_df = pd.concat([flow_df, flow_df],
                                    ignore_index=True, copy=False)
            cur_df = flow_df.iloc[:len(blank_times)].reset_index(drop=True)
            cur_df['time'] = pd.Series(blank_times)
            return pd.concat([df, cur_df], ignore_index=True, copy=False)

        assert df.run.nunique() == 1
        run = df.run.unique()[0]
        df = df.sort_values(['run', 'flow', 'time'])
        # t_step = round(df.sort_values('time').drop_duplicates('time').diff()
        #                ['time'].dropna().min(), 3)
        t_step = pd.Series(df.time.unique()).diff().median()
        assert t_step > 0
        columns = df.columns
        df_times = sorted(df.time.unique())
        df_times = df_times[:len(df_times) - len(df_times) % seq_len]
        df = df[df.time.isin(df_times)]
        for i in range(0, df.time.nunique(), seq_len):
            t_start = df_times[i]
            t_end = df_times[i + seq_len - 1]
            times = df_times[i : i + seq_len]
            tmp_df = df[(df.time >= t_start) & (df.time <= t_end)]
            # print('t_start', t_start, 't_end', t_end, 'tmp_df.time.shape', tmp_df.time.nunique())
            assert tmp_df.time.nunique() == seq_len

            for flow in tmp_df.flow.unique():
                flow_df = tmp_df[tmp_df.flow == flow]
                if len(flow_df) == seq_len:
                    continue
                t_flow_start, t_flow_end = flow_df.time.min(), flow_df.time.max()
                i_flow_start, i_flow_end = times.index(t_flow_start), times.index(t_flow_end)
                # several cases, mode only works for case 1) and 2), for 3)
                # we generally interpolate it with the last value

                # 1. blank(s) in the middle: need to be first handled, as 'self'
                # mode for the two below needs the middle values
                last_row = None
                
                # n_step_in_flow = i_flow_end - i_flow_start
                if flow_df.time.nunique() == i_flow_end - i_flow_start + 1:
                    # no blank in the middle
                    continue
                else:
                    print(f' - Warning: segment {i} flow {flow} is filling middle'
                          'blanks! Shouldn\'t happen with simulated data!')
                    new_rows = pd.DataFrame(columns=columns)
                    for i in range(i_flow_start, i_flow_end):
                        # TODO: assume that not all flows miss this time step!!!
                        t = times[i]
                        if t not in flow_df.time.unique():
                            if mid_padding_mode == 'zero':
                                last_row = _get_zero_row(columns, run, flow, t)
                            else:
                                assert mid_padding_mode == 'last'
                                last_row = _get_last_row(last_row, run, flow, t)
                            new_rows = pd.concat([new_rows, last_row])
                        else:
                            # TODO: possibly a bug below, as t may not be exactly the
                            # same as the time in flow_df
                            last_row = flow_df[flow_df.time == t]
                    df = pd.concat([df, new_rows], ignore_index=True, copy=False)
                
                # assert df[(df.time >= t_start) & (df.time <= t_end) & (df.flow == flow)].time.nunique() == n_step_in_flow + 1, \
                #     f'run {run} flow {flow} has {df[df.flow == flow].time.nunique()} times, '

                # 2. blanks in the beginning or the end
                modes = [pre_padding_mode, post_padding_mode]
                for j in range(2):
                    if j == 0:      # pre-padding
                        blank_times = times[ : i_flow_start]
                        if len(blank_times) == 0:
                            continue
                        last_row = flow_df[flow_df.time == t_flow_start]
                    else:           # post-padding
                        blank_times = times[i_flow_end + 1 : ]
                        if len(blank_times) == 0:
                            continue
                        last_row = flow_df[flow_df.time == t_flow_end]

                    if modes[j] == 'zero':
                        row = _get_zero_row(columns, run, flow, None)
                        df = _fill_blanks(df, row, blank_times)
                    elif modes[j] == 'last':
                        row = _get_last_row(last_row, run, flow, None)
                        df = _fill_blanks(df, row, blank_times)
                    elif modes[j] == 'self':
                        df = _fill_with_self_seq(df, flow_df, blank_times)
                
                df2 = df[(df.time >= t_start) & (df.time <= t_end) & (df.flow == flow)]
                assert len(df2) % seq_len == 0
                assert df2.time.nunique() == seq_len

        assert not df.duplicated().any()
        assert df.time.nunique() % seq_len == 0
        # for flow in df.flow.unique():
        #     assert df[df.flow == flow].time.nunique() % seq_len == 0
        return df        


def sample_from_segment(df_seg, tdf, isio, run, ti, pos_factor=2, neg_factor=0.7,
                        neg_to_pos_ratio=1):
    assert type(isio) == IndexedSampleIO
    df_flow = df_seg.set_index('flow')
    for pos_label in tdf.label.unique():
        pos_candidates = sorted(tdf[(tdf.label == pos_label)].flow.unique())
        neg_candidates = sorted(tdf[(tdf.label != pos_label)].flow.unique())
        if len(pos_candidates) < 2 or len(neg_candidates) < 1:
            continue
        # for each cluster, scan all pos pairs / neg pairs at least once
        n_per_cluster = int(max(pos_factor * len(pos_candidates),
                            neg_factor * len(neg_candidates)))
        for j in range(n_per_cluster):
            res = np.random.choice(pos_candidates, 2, replace=False)
            anchor, pos = res[0], res[1]
            for _ in range(neg_to_pos_ratio):
                neg = np.random.choice(neg_candidates)
                neg_label = tdf[tdf.flow == neg].label.unique()[0]
                for flow in [anchor, pos, neg]:
                    segment = df_flow.loc[flow].copy()
                    label = pos_label if flow != neg else neg_label
                    isio.add_sample_data(run, ti, label, flow, segment)
                isio.add_triplet(run, ti, anchor, pos, neg)


def build_triplets(df, truth_df, df_tag, seq_len, folder=None,
                   fields=None, for_test=False):
    """Build triplets from df for dataset, and save them to folder
    if specified. If for_test is True, then remove the non-btnk flows
    that will get removed in detection & cluster the rest of them.
    
    Flow sampling is done by first sampling positive pairs, then negative
    samples. More details can be found in sample_from_segment().
    """
    print(f'Building triplets for {df_tag}...')
    basic_fields = ['run', 'time', 'flow']
    fields = fields if fields is not None else ['owd', 'rtt', 'slr', 'cwnd']
    df = df.sort_values(basic_fields).reset_index(drop=True)
    isio = IndexedSampleIO(fields, seq_len)
    interval = pd.Series(sorted(truth_df.time.unique())).diff().median()

    # To save time, fetch [run, time] segment first, then scan flow
    #  thus query only needs to be done once for each segment!
    for run in df.run.unique():
        print(' [build triplet] run ', run, ' started...')
        df_run = df[df.run == run]
        run_times = sorted(df_run.time.unique())
        assert df.time.nunique() % seq_len == 0
        for i in range(0, df.time.nunique() - seq_len, seq_len):
            ti = int(i // seq_len)
            t_start = run_times[i]
            t_end = run_times[i + seq_len - 1]
            df_seg = df_run[(df_run.time >= t_start) & (df_run.time <= t_end)]
            assert np.isclose(t_start % interval, 0.0)
            tdf = truth_df.query(f'run == {run} and abs(time - {t_start}) < 1e-6')
            # TODO: deprecate due to label -1 for non-btnk
            # if for_test:
            #     non_btnk_flows = set(DataGetter.get_non_btnk_flows(df_seg))
            #     df_seg = df_seg[~df_seg.flow.isin(non_btnk_flows)].copy()
            #     tdf = tdf[~tdf.flow.isin(non_btnk_flows)].copy()
            assert df_seg.flow.nunique() == tdf.flow.nunique()
            # pos_factor / neg_factor control how many times we want to
            # repeat the sampling of pos pairs / neg samples, can be tuned
            # later
            sample_from_segment(df_seg, tdf, isio, run, ti)
            df.drop(df_seg.index, inplace=True)
            truth_df.drop(tdf.index, inplace=True)

    if folder is not None:
        isio.save(folder, df_tag)


# TODO: deprecate
# def build_triplets_for_test(df, truth_df, df_tag, seq_len, folder=None,
#                             fields=None):
#     """Build triplets for test and dump them to samples for validation/test.
#     This is the similar to build_triplets but iterate over all possible triplets
#     instead of sampling some of them.
    
#     TODO: need test for overall correctness & the filtering of
#         non-btnk flows & candidates
#     """
#     print(f'Building triplets for test for {df_tag}...')
#     fields = fields if fields is not None else ['owd', 'rtt', 'slr', 'cwnd']
#     df = df.sort_values(['run', 'time', 'flow']).reset_index(drop=True)

#     samples = {}       # {run: [ [anchor, pos, neg] ]}
#     i_triplet = 0
#     interval = pd.Series(sorted(truth_df.time.unique())).diff().median()
#     isio = IndexedSampleIO()

#     for run in df.run.unique():
#         print(' [build triplet for test] run ', run, ' started...')
#         df_run = df[df.run == run]
#         run_times = sorted(df_run.time.unique())
#         assert df.time.nunique() % seq_len == 0
#         samples[run] = []
#         for i in range(0, df.time.nunique() - seq_len, seq_len):
#             ti = int(i // seq_len)
#             t_start = run_times[i]
#             t_end = run_times[i + seq_len - 1]
#             df_seg = df_run[(df_run.time >= t_start) &
#                             (df_run.time <= t_end)]
#             assert np.isclose(t_start % interval, 0.0)
#             tdf = truth_df.query(f'run == {run} and abs(time - {t_start}) < 1e-6')
#             non_btnk_flows = set(DataGetter.get_non_btnk_flows(df_seg))
#             # print(f'  {t_start} ~ {t_end} s, interval {i} ')
#             # print(f'  non-btnk flows: {len(non_btnk_flows)} / {df_seg.flow.nunique()}')

#             df_seg = df_seg[~df_seg.flow.isin(non_btnk_flows)].copy()
#             tdf = tdf[~tdf.flow.isin(non_btnk_flows)].copy()
#             assert df_seg.flow.nunique() == tdf.flow.nunique()
#             i_df_seg = df_seg.index
#             df_seg.set_index('flow', inplace=True)
#             visited_samples = set()

#             for pos_label in tdf.label.unique():
#                 pos_candidates = sorted(tdf[(tdf.label == pos_label)].flow.unique())
#                 neg_candidates = sorted(tdf[(tdf.label != pos_label)].flow.unique())
#                 # print(f'   Group {pos_label}: # pos: {len(pos_candidates)}, # neg: {len(neg_candidates)}')
#                 if len(pos_candidates) < 2 or len(neg_candidates) < 1:
#                     continue
                
#                 # sample pos pairs, but ensure all neg samples are visited
#                 # TODO: seems wrong, below is even stricter than all pos_pair * neg...
#                 #       below is O(n_pos * n_neg), what I want is O(max(2*n_pos, n_neg))


#                 n_pos_samples = 2 * len(pos_candidates)     # tunable
#                 for j in range(n_pos_samples):
#                     res = np.random.choice(pos_candidates, 2, replace=False)
#                     anchor, pos = res[0], res[1]
#                     # emphasize on negative samples
#                     for neg in neg_candidates:
#                         tmp = []
#                         for flow in [anchor, pos, neg]:
#                             if flow not in visited_samples:
#                                 segment = df_seg.loc[flow].copy()
#                                 sample = convert_segment_to_array(segment, fields, seq_len)
#                                 visited_samples.add(flow)
#                                 isio.add_sample_data(run, ti, flow, sample)
#                         isio.add_triplet(run, ti, anchor, pos, neg)
#                         i_triplet += 1
#             df.drop(i_df_seg, inplace=True)
#             truth_df.drop(tdf.index, inplace=True)
    
#     if folder is not None:
#         isio.save(folder, df_tag)


class DataVisualizer:
    @staticmethod
    def plot_label_and_queue_vs_time(qid_df, qdf, run, flow,
                                     do_save=False, figsize=(10, 4)):
        """For each run, flow, plot the label along time vs the queue size
        along time.
        TODO: only plot subplots within the function, return the handlers,
        and use a decorator to customize the layout outside?
        """
        row = qid_df[(qid_df.run == run) & (qid_df.flow == flow)].iloc[0]
        qdf_run = qdf[qdf.run == run]
        df = DataGetter.get_double_queues(qdf_run, row.qid1, row.qid2)
        labels = sorted(df.label.unique())
        df['nlabel'] = df.apply(lambda r: labels.index(r.label), axis=1)

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        sns.lineplot(x='time', y='q_win1', data=df, ax=ax[0], label=row.qid1)
        sns.lineplot(x='time', y='q_win2', data=df, ax=ax[0], label=row.qid2)
        sns.lineplot(x='time', y='nlabel', data=df, ax=ax[1], label='label').set(
            yticks=range(len(labels)), yticklabels=labels)
        plt.tight_layout()
        if do_save:
            plt.savefig(f'label_vs_queue_{run}_{flow}.pdf')
        else:
            plt.show()
        plt.close()


class DataPreprocessor:
    """
    As mentioned before, we want to generate the data frames below for
    later processing:

    Dataframe from the data files: 
        xdf: [(run), flow, time, owd, rtt, ...]
        qid_df: [(run), flow, qid1, qid2]
        qdf: [(run), qid, time, packet_in_queue, q_size]

    Dataframe from the config files:
        btnk_df (link + cross): [(run), src, dst, qid, num, bw_avail ...]
        flow_df: [(run), src, dst, num, rate, ...]

    Input is folder and ids, output is the dataframes used by dataset / 
    ml data loader.

    Note that here the runs are controlled by the given args, so typically
    the runs in the link / btnk configs do not need match the runs in the
    log and dats directories. In current implementations, the runs are
    determined by the create_arg_list_for_preprocess() which uses the 'all-data'
    files to determine the runs.
    """

    def __init__(self, cache_folder, runs=None, config_runs=None):
        """Initialize the preprocessor.
        
        Args:
            cache_folder (str): the folder to store the preprocessed data
            runs (list): [run0, run1), if None, process all runs
            config_runs (list): [run0, run1), if None, process all runs
        """
        self.cache_folder = cache_folder
        self.runs = runs
        self.config_runs = config_runs
        self.qtype_dict = {'run': float, 'qid': str,
            'time': float, 'packet_in_queue': float}
        self.qid_type_dict = {'run': float, 'flow': float,
            'qid1': str, 'qid2': str}
        self.btype_dict = {'run': float, 'qid': str, 'q_size': float}

    def read(self, folder):
        print(f' - preprocess: read() from {folder}')
        cr = ConfigReader(folder, runs=self.config_runs)
        # self.btnk_df = cr.get_btnk_info().apply(pd.to_numeric,
        #                errors='ignore', downcast='float')
        self.btnk_df = cr.get_btnk_info().astype(self.btype_dict)
        self.flow_df = cr.get_agg_flow_info().apply(pd.to_numeric,
                       errors='ignore', downcast='float')
        dr = DataReader(folder, order_check=False, runs=self.runs)
        if dr.get_data_df() is None or dr.get_data_df().empty:
            print('WARNING: data is empty, exitting.')
            exit(1)
        self.xdf = dr.get_data_df().astype(float)
        self.qdf = dr.get_queue_df().astype(self.qtype_dict)
        self.qid_df = dr.get_qid_df().astype(self.qid_type_dict)

        print('     qdf qids:', self.qdf.qid.unique())
        print('     btnk_df qids:', self.btnk_df.qid.unique())
        # TODO: the relative run issue for dummy test folders

        assert set(self.qdf.qid.unique()) == set(self.btnk_df.qid.unique()), \
            'qid in queue df and btnk df are not the same, check if the runs ' \
            'of btnk_df is loaded as expected, the most probable reason is ' \
            ' log based id is wrong (rare in general). \n' \
            f'qdf qid (from log_id): \n     {self.qdf.qid.unique()}, \n' \
            f'btnk_df\'s qid (from link_inflated w/ config_run): \n      {self.btnk_df.qid.unique()}'

        # handle the missing runs, now run is relative id
        missing_run_no, missing_runs = dr.get_missing_runs()
        if len(missing_runs) > 0:
            self.btnk_df = self.btnk_df[~self.btnk_df.run.isin(missing_runs)]
            self.flow_df = self.flow_df[~self.flow_df.run.isin(missing_runs)]
            all_runs = list(self.btnk_df.run.unique())
            self.btnk_df['run'] = self.btnk_df.apply(lambda r: all_runs.index(r.run), axis=1)
            self.flow_df['run'] = self.flow_df.apply(lambda r: all_runs.index(r.run), axis=1)
            print(f'WARNING: missing runs (runs in log but not in dats): {missing_run_no})')

    def read_real(self, folder):
        print(f' - preprocess: read_real() from {folder}')
        dr = DataReader(folder, order_check=False, runs=self.runs)
        xtypes = {'run': int, 'flow': str, 'time': float, 'owd': float,
                     'rtt': float}
        # truth_types = {'flow': str, 'label': str}
        self.xdf = dr.get_data_df(subdir=None).astype(xtypes)
        self.truth_df = dr.get_truth_df().astype(str)

    def read_cache(self):
        print(f' - preprocess: read_cache() from {self.cache_folder}')
        self.xdf, self.truth_df, self.tick_truth_df = None, None, None
        if self.runs is None:
            files = sp.getoutput(f'ls {self.cache_folder}/xdf_pred_*.csv').split('\n')
            files = [os.path.basename(f) for f in files]
            print(files)
            runs = list(map(lambda s: int(s.split('_')[2][:-4].split('-')[0]),
                    files))
        else:
            runs = range(self.runs[0], self.runs[1])
        for run in runs:
            for typ in ['xdf', 'truth_df']:
                csv = f'{typ}_pred_{run}-{run+1}.csv'
                csv_path = os.path.join(self.cache_folder, csv)
                df = pd.read_csv(csv_path, index_col=False)
                exec(f'self.{typ} = df if self.{typ} is None else '
                    f'pd.concat([self.{typ}, df], ignore_index=True)')

    def filter_run(self, runs):
        self.xdf = self.xdf[self.xdf.run.isin(runs)].reset_index(drop=True)
        self.qdf = self.qdf[self.qdf.run.isin(runs)].reset_index(drop=True)
        self.qid_df = self.qid_df[self.qid_df.run.isin(runs)].reset_index(drop=True)
        self.btnk_df = self.btnk_df[self.btnk_df.run.isin(runs)].reset_index(drop=True)
        self.flow_df = self.flow_df[self.flow_df.run.isin(runs)].reset_index(drop=True)

    def preprocess(self, seq_len, fields, for_test=False):
        """Preprocess the dfs to get clean df for data loader.
        Note that this preprocessing is comprehensive except that it does not
        contain any intra-sample processing, such as flow padding or crop.
        
        Critical steps include:
            1. remove invalid flows with all owd = 0 (process_invalid_entry), mainly
                 caused by simulation issues, and then match the flows of xdf and
                 qid_if (match_flow_of_data);
            2. infer the truth label from queue size (get_truth_from_queue);
            3. match and crop the time of xdf and truth_df (match_time,
                 crop_time_sequence), especially for the flow change and label change
                 within each time interval.

        If for test is True, then it is used to generate the test dataset,
        which
            1) does not remove invalid entries;
            2) loops over all triplets for each sagment instead of sampling,
            3) remove the non-btnk flows when building triplets to be consistent
                with later detection.

        Check each modifiers for more details.
        """
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        if for_test:
            print('Preprocessing for test ...')

        self.btnk_df, self.flow_df = DataModifier.translate_runs(self.btnk_df,
            self.flow_df, self.xdf)

        self.qdf = DataModifier.merge_qsize(self.qdf, self.btnk_df)
        DataModifier.remove_zero_owds(self.xdf)
        # DataModifier.add_slr(self.xdf)
        # TODO: deprecate, currently just dropna within process_invalid_entry
        DataModifier.process_invalid_entry(self.xdf)
        self.qid_df = DataModifier.match_flow_of_data(self.xdf, self.qid_df)
        self.truth_df = DataGetter.get_truth_from_queue(self.qid_df, self.qdf)
        DataModifier.match_time(self.xdf, self.truth_df)
        DataChecker.check_time_alignment(self.xdf, self.truth_df)

        t_unit = self.xdf.time.diff().median()
        interval = seq_len * t_unit
        self.xdf = DataModifier.crop_time_sequence_by_flow(self.xdf, interval)
        self.truth_df = DataModifier.crop_time_sequence_by_flow(self.truth_df, interval)

        # self.xdf = DataModifier.crop_time_sequence(self.xdf, seq_len)
        # self.truth_df = DataModifier.crop_time_sequence(self.truth_df, seq_len)
        self.tick_truth_df = DataModifier.fetch_truth_for_segments(self.truth_df, interval)

        build_triplets(self.xdf, self.tick_truth_df, f'{self.runs[0]}-{self.runs[1]}',
                       seq_len, self.cache_folder, fields=fields, for_test=for_test)

    def preprocess_for_prediction(self, seq_len):
        """Preprocess the dfs to get clean df for prediction."""
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        self.btnk_df, self.flow_df = DataModifier.translate_runs(self.btnk_df,
            self.flow_df, self.xdf)

        self.qdf = DataModifier.merge_qsize(self.qdf, self.btnk_df)
        # DataModifier.remove_zero_owds(self.xdf)
        DataModifier.process_invalid_entry(self.xdf)
        # DataModifier.add_slr(self.xdf)
        self.qid_df = DataModifier.match_flow_of_data(self.xdf, self.qid_df)
        self.truth_df = DataGetter.get_truth_from_queue(self.qid_df, self.qdf)
        # TODO: deprecate
        # DataModifier.process_invalid_entry(self.truth_df)
        DataModifier.match_time(self.xdf, self.truth_df)
        DataChecker.check_time_alignment(self.xdf, self.truth_df)

        t_unit = self.xdf.time.diff().median()
        interval = seq_len * t_unit
        print('# xdf times: ', self.xdf.time.nunique())
        self.xdf = DataModifier.crop_time_sequence_by_flow(self.xdf, interval)
        self.truth_df = DataModifier.crop_time_sequence_by_flow(self.truth_df, interval)
        # self.tick_truth_df = DataModifier.fetch_truth_for_segments(self.truth_df, interval)
        print('# xdf times: ', self.xdf.time.nunique())
        self.xdf = DataModifier.crop_time_for_prediction(self.xdf, seq_len)
        self.truth_df = DataModifier.crop_time_for_prediction(
            self.truth_df, seq_len)
        print('xdf times: ', self.xdf.time.nunique())
        self.xdf = DataModifier.pad_flows(self.xdf, seq_len)
        self.truth_df = DataModifier.pad_flows(self.truth_df, seq_len)
        # seems no need, as all flow times are already matched
        # self.xdf, self.truth_df = DataModifier.match_flow_along_time(self.xdf,
        #                                                              self.truth_df)
        return self.xdf, self.truth_df

    def preprocess_for_real(self, seq_len):
        """Preprocess for real data, which has some specific features like:
            - No metadata like btnk_df & flow_df
            - SLR processing: cannot solve
            - Truth labels w/ no time axis: supported in detector
            - No queue needed for truth.
        """
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
        print('Preprocessing for real data ...')

        DataModifier.convert_delay_to_s(self.xdf)
        DataModifier.convert_ms_to_s(self.xdf)

        # DataModifier.remove_zero_owds(self.xdf)
        DataModifier.process_invalid_entry(self.xdf)
        DataModifier.process_invalid_entry(self.truth_df)
        if 'drop' not in self.xdf.columns:
            DataModifier.add_drop_from_slr(self.xdf)

        xdf_outstanding = set(self.xdf.flow.unique()) - set(self.truth_df.flow.unique())
        tdf_outstanding = set(self.truth_df.flow.unique()) - set(self.xdf.flow.unique())
        if len(xdf_outstanding) > 0 or len(tdf_outstanding) > 0:
            print('xdf outstanding flows: ', xdf_outstanding)
            print('tdf outstanding flows: ', tdf_outstanding)
            DataModifier.match_flow(self.xdf, self.truth_df)
        DataModifier.convert_flow_to_int(self.xdf)
        DataModifier.convert_flow_to_int(self.truth_df)

        # DataModifier.add_slr(self.xdf)        # TODO: this won't work...
        # t_unit = self.xdf.time.diff().median()
        t_unit = 0.005
        interval = seq_len * t_unit
        print('# xdf times: ', self.xdf.time.nunique())
        self.xdf = DataModifier.crop_time_sequence_by_flow(self.xdf, interval)
        self.xdf = DataModifier.crop_time_for_prediction(self.xdf, seq_len)
        # helpful when there's some small holes in the middle
        self.xdf = DataModifier.pad_flows(self.xdf, seq_len)
        return self.xdf, self.truth_df

    def save(self):
        run_str = ''
        if self.runs is not None:
            run_str = '-'.join([str(r) for r in self.runs])
        self.xdf.to_csv(os.path.join(
            self.cache_folder, f'xdf_{run_str}.csv'), index=False)
        self.qid_df.to_csv(os.path.join(
            self.cache_folder, f'qid_df_{run_str}.csv'), index=False)
        self.qdf.to_csv(os.path.join(
            self.cache_folder, f'qdf_{run_str}.csv'), index=False)
        self.truth_df.to_csv(os.path.join(
            self.cache_folder, f'truth_df_{run_str}.csv'), index=False)
        self.btnk_df.to_csv(os.path.join(
            self.cache_folder, f'btnk_df_{run_str}.csv'), index=False)
        self.flow_df.to_csv(os.path.join(
            self.cache_folder, f'flow_df_{run_str}.csv'), index=False)
        self.tick_truth_df.to_csv(os.path.join(
            self.cache_folder, f'tick_truth_df_{run_str}.csv'), index=False)
        print('Preprocessor dfs saved to cache_folder: ', self.cache_folder)

    def save_prediction(self, tag=''):
        self.xdf.to_csv(os.path.join(
            self.cache_folder, f'xdf_pred_{tag}.csv'), index=False)
        self.truth_df.to_csv(os.path.join(
            self.cache_folder, f'truth_df_pred_{tag}.csv'), index=False)
        # self.tick_truth_df.to_csv(os.path.join(
        #     self.cache_folder, f'tick_truth_df_pred_{tag}.csv'), index=False)
        print('Prediction dfs saved to cache_folder: ', self.cache_folder)

    def load(self):
        run_str = ''
        if self.runs is not None:
            run_str = '-'.join([str(r) for r in self.runs])
        self.xdf = pd.read_csv(os.path.join(self.cache_folder, f'xdf_{run_str}.csv'),
                               index_col=False).astype(float)
        self.qid_df = pd.read_csv(os.path.join(self.cache_folder, f'qid_df_{run_str}.csv'),
                                  index_col=False).astype(self.qid_type_dict)
        self.qdf = pd.read_csv(os.path.join(self.cache_folder, f'qdf_{run_str}.csv'),
                               index_col=False).astype(self.qtype_dict)
        self.truth_df = pd.read_csv(os.path.join(self.cache_folder, f'truth_df_{run_str}.csv'),
                                    index_col=False).astype(float)
        self.btnk_df = pd.read_csv(os.path.join(self.cache_folder, f'btnk_df_{run_str}.csv'),
                                   index_col=False)
        self.flow_df = pd.read_csv(os.path.join(self.cache_folder, f'flow_df_{run_str}.csv'),
                                   index_col=False)
        # # one time hack
        # interval = 1.5
        # self.tick_truth_df = DataModifier.fetch_truth_for_segments(self.truth_df, interval)
        # self.tick_truth_df.to_csv(os.path.join(
        #     self.cache_folder, 'tick_truth_df.csv'), index=False)
        self.tick_truth_df = pd.read_csv(os.path.join(self.cache_folder, f'tick_truth_df_{run_str}.csv'),
                                         index_col=False).astype(float)
        print('Preprocessor dfs loaded from cache_folder: ', self.cache_folder)
    
    def load_prediction(self):
        self.xdf = pd.read_csv(os.path.join(self.cache_folder, 'xdf_pred.csv'),
                               index_col=False).astype(float)
        self.truth_df = pd.read_csv(os.path.join(self.cache_folder, 'truth_df_pred.csv'),
            index_col=False).astype(float)
        # self.tick_truth_df = pd.read_csv(os.path.join(self.cache_folder, 'tick_truth_df_pred.csv'),
        #     index_col=False).astype(float)
        print('Prediction dfs loaded from cache_folder: ', self.cache_folder)

    def global_load_prediction(self, cache_folder,
                               runs=None, tag=None):
        cwd = os.getcwd()
        os.chdir(cache_folder)
        tag = '' if tag is None else tag
        dirs = sp.getoutput(f'ls -d *{tag}*/').split('\n')
        dirs = [d[:-1] for d in dirs]
        self.xdf = None
        self.truth_df = None
        self.tick_truth_df = None

        if runs is not None:
            suffix = [f'{i}-{i+1}' for i in range(runs[0], runs[1])]

        for d in dirs:
            os.chdir(d)
            print('Loading from ', d)
            for typ in ['xdf', 'truth_df']:
                if runs is None:
                    csvs = sp.getoutput(f'ls {typ}*.csv').split('\n')
                else:
                    csvs = [f'{typ}_pred_{s}.csv' for s in suffix]

                for csv in csvs:
                    print(' ', csv)
                    df = pd.read_csv(csv, index_col=False)
                    exec(f'self.{typ} = df if self.{typ} is None else '
                         f'pd.concat([self.{typ}, df], ignore_index=True)')
            print(f'Loaded prediction df from {d}')
            os.chdir('..')
        os.chdir(cwd)

    def check(self):
        DataChecker.check_time_steps(self.xdf)
        DataChecker.check_time_steps(self.qdf)
        # DataChecker.check_flow_number(self.xdf, self.flow_df)
        DataChecker.check_queue_size(self.qdf)
        for run in self.btnk_df.run.unique():
            qdf_run = self.qdf[self.qdf.run == run]
            btnk_run = self.btnk_df[self.btnk_df.run == run]
            if DataChecker.check_multi_btnk(qdf_run, btnk_run):
                print(f'Warning: multiple btnks exist in run {run}.')
        DataChecker.check_time_alignment(self.xdf, self.truth_df)
        # TODO: check the tick_df's time step alignes with xdf

    def check_for_prediction(self, seq_len):
        DataChecker.check_segment_alignment(self.xdf, seq_len)
        DataChecker.check_segment_alignment(self.truth_df, seq_len)
    
    def check_for_real(self, seq_len):
        DataChecker.check_segment_alignment(self.xdf, seq_len)

    def get_metadata(self):
        return self.xdf[['run', 'flow', 'time']].drop_duplicates()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preprocessor')
    parser.add_argument('--config_folder', '-cf', type=str, required=True,
                        help='folder of the config files')
    parser.add_argument('--data_folder', '-df', type=str, required=True,
                        help='folder of the data files')
    args = parser.parse_args()
