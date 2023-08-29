"""
Copyright (c) 2022, IETF Trust and the persons identified as authors of the code.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os, sys
from math import ceil
# from typing import OrderedDict
import numpy as np
import pandas as pd
from collections import deque, Counter

from matplotlib import pyplot as plt
import seaborn as sns

from siamese.cluster import rmcat

from siamese.preprocess import measure_time

"""
This is an implementation of the Shared Bottleneck Detection (SBD) algorithm
presented in Online Identification of Groups of Flows Sharing a Network Bottleneck
(IEEE ToN'20).
"""

"""
System design overview

1.  Input: OWD arrays of flows with a specified time unit;
    Output 1: the estimate of the summary statistics per flow, i.e. 
        skew_est, var_est, freq_est;
    Output 2: flow grouping result, format TBD.
"""

class SBDAlgorithm:
    def __init__(self):
        """Initializes with the default parameter values given by RFC 8382, Sec 2.2.
        """
        self.T = 0.35           # interval duration
        self.N = 50             # number of intervals to look over for freq_est
        self.M = 30             # number of intervals for skew_est and var_est
        self.c_s = 0.1
        self.c_h = 0.3
        self.p_s = 0.15         # grouping threshold for skew_est
        self.p_v = 0.7          # threashold used in freq_est
        self.p_f = 0.1          # grouping threshold for freq_est
        self.p_mad = 0.1        # grouping threshold for var_est
        self.p_d = 0.1          # grouping threshold for pkt_loss
        self.p_l = 0.05         # TODO: pkt_loss threashold for flow bottleneck 
        self.F = 20             # number of sample in the flat portion used in the moving avg

    def _summarize_owd(self, owds, K):
        """Summarizes OWD from data, i.e. averages them over K raw samples.
        Returns the averaged OWD.
        """
        avg_owds, window = [], []
        for i, owd in enumerate(owds):
            window.append(owd)
            if i % K == K - 1 or i == len(owds) - 1:
                avg_owds.append(np.mean(window))
                window.clear()
        # self.d_base = np.mean(window[-self.M:])     # should be used in stream
        return avg_owds
    
    def _piecewise_linear_weighted_moving_average(self, samples, Ks):
        """Given a sample series, return the piecewise linear weighted moving average
        defined in RFC 8382, Sec 4.1.

        TODO: test len(samples) < = > self.F
        """
        if not len(samples):
            return 0
        samples, Ks = np.array(samples), np.array(Ks)
        mf = max(len(samples) - self.F + 1, 1)      # support samples less than F too
        dec_f = np.linspace(mf - 1, 1, mf - 1)
        s = mf * sum(samples[:self.F]) + sum(dec_f * samples[self.F:])
        # n = mf * (len(samples) - mf + 1) + (len(samples) - mf) * (mf - 1)
        n = mf * sum(Ks[:self.F]) + sum(dec_f * Ks[self.F:])    # in case different K in the end
        return s / n

    def _owd_process(self, flow_owds, K):
        """Processes OWDs of a single flow.

        Returns:
            dict: { 'd_avg': [...], 'skew_est': [...],
                    'var_est': [...], 'freq_est': [...]}
        """
        # avg_owds = self._summarize_owd(flow_owds, K)
        window = []
        d_avg_queue = deque(maxlen=self.M)        # for average over M avg_owds
        skew_queue = deque(maxlen=self.M)
        var_queue = deque(maxlen=self.M)
        cross_queue = deque(maxlen=self.N)
        Ks = deque(maxlen=self.M)
        d_avg, x_h = 0, None
        res = {'d_avg': [], 'skew_est': [], 'var_est': [], 'freq_est': []}
        for i, owd in enumerate(flow_owds[::1]):
            # TODO: the reverse traversal of owd? correct or not?
            #       should be like only keeping K from the end, but still the normal order?

            window.append(owd)
            if not (i % K == K - 1 or i == len(flow_owds) - 1):
                continue

            d_avg = np.mean(window)
            d_avg_queue.append(d_avg)
            d_base = np.mean(d_avg_queue)
            Ks.append(len(window))
            res['d_avg'].append(d_avg)

            # calculates skew_est
            skew_base = sum(map(lambda d: 1 if d < d_base else -1 if d > d_base else 0, window))
            skew_queue.append(skew_base)
            skew_est = self._piecewise_linear_weighted_moving_average(skew_queue, Ks)
            res['skew_est'].append(skew_est)

            # calculates var_est
            var_base = sum(map(lambda d: abs(d - d_avg), window))
            var_queue.append(var_base)
            var_est = self._piecewise_linear_weighted_moving_average(var_queue, Ks)
            res['var_est'].append(var_est)

            # calculates freq_est
            # TODO: freq_est seems need an initial outstanding x_h, as defined below
            var_thd = self.p_v * var_est
            if d_avg < d_base - var_thd and (x_h is None or x_h == 1):
                x, x_h = -1, -1
            elif d_avg > d_base + var_thd and (x_h is None or x_h == -1):
                x, x_h = 1, 1
            else:
                x = 0
            cross_queue.append(x)
            freq_est = np.mean(list(map(abs, cross_queue)))
            res['freq_est'].append(freq_est)
            window.clear()

        return res
    
    def _loss_process(self, flow_drops, K):
        """Processes the losses of given flow. K is the number of samples within 
        an interval T.

        Returns:
            dict: { 'pkt_loss': [...] }
        """
        K = int(K)
        res = {'pkt_loss': []}
        loss_queue = deque(maxlen=self.N)
        Ks = deque(maxlen=self.N)
        for i in range(ceil(len(flow_drops) / K)):
            lo, hi = i * K, min((i + 1) * K, len(flow_drops))
            samples = flow_drops.iloc[len(flow_drops)-hi:len(flow_drops)-lo]     # first the closest
            loss_queue.append(sum(samples))         # assuming loss is 1
            Ks.append(hi - lo)

            pkt_loss = sum(loss_queue) / sum(Ks)    # loss is not using piecewise linear
            # pkt_loss = self._piecewise_linear_weighted_moving_average(loss_queue, Ks)
            res['pkt_loss'].append(pkt_loss)
        res['pkt_loss'].reverse()
        return res

    def _label_process(self, flow_labels, K):
        """Processes the labels of given flow.
        
        Returns:
            dict: { 'label': [...] }
        """
        K = int(K)
        res = {'label': []}
        for i in range(ceil(len(flow_labels) / K)):
            lo, hi = i * K, min((i + 1) * K, len(flow_labels))
            samples = flow_labels.iloc[len(flow_labels)-hi:len(flow_labels)-lo]     # first the closest
            label = Counter(samples).most_common(1)[0][0]
            res['label'].append(label)
        res['label'].reverse()
        return res

    @measure_time()
    def stream_process(self, flows, t_unit):
        """Processes the flows inputs and obtains the results.

        Args:
            flows (DataFrame): DataFrame of flows including OWD & loss data.
            t_unit (float): time unit used in the flow data.

        Returns:
            DataFrame: result with skew_est, var_est, freq_est, pkt_loss.
        """
        K = round(self.T / t_unit)
        for col in ['flow', 'owd', 'drop']:
            assert col in flows.columns, f'Column {col} not found in flows.'
        res_df = pd.DataFrame(columns=['time', 'flow', 'skew_est', 'var_est', 'freq_est', 'pkt_loss'])
        # intervals are all aligned to t_base for all flows
        t_start, t_end = flows.time.min(), flows.time.max()
        times = [t_start + i * self.T
                for i in range(int((t_end - t_start) // self.T) + 1)]

        for flow in flows['flow'].unique():
            # TODO: parallelization here for multiflow
            flow_df = flows[flows.flow == flow]
            res1 = self._owd_process(flow_df['owd'], K)
            res2 = self._loss_process(flow_df['drop'], K)
            res3 = self._label_process(flow_df['label'], K)
            for k in res1:
                assert len(res2['pkt_loss']) == len(res1[k]) == len(res3['label']) \
                    == ceil(len(flow_df['owd']) / K) > 0
            res1.update(res2)
            res1.update(res3)
            metric_df = pd.DataFrame(res1)            
            metric_df['flow'] = flow

            # need to use the dynamic time of each flow
            # TODO: debug the part below
            i_flow_start = int((flow_df.time.min() - t_start) // self.T)
            flow_times = times[i_flow_start : i_flow_start + len(metric_df)]

            # OWD ticks can be 1 step less than t_start based ticks due to
            # it starts from each flow's start tick but not the t_start,
            # so we need to append the last sample if they are not equal
            i_flow_end = int((flow_df.time.max() - t_start) // self.T)
            n_flow_tick = i_flow_end - i_flow_start + 1
            assert n_flow_tick - 1 <= len(flow_times) <= n_flow_tick
            if len(flow_times) < n_flow_tick:
                flow_times.append(times[i_flow_start + len(metric_df)])
                metric_df = pd.concat([metric_df, metric_df.iloc[-1:]],
                                       ignore_index=True)

            # flow_df.time.max() will fall in the next tick that aren't used
            # so flows_times[-1] + self.T should be that tick
            # since the K in owd_process is not aligned with t_start, the check
            # below is not always true, so we just check length
            # assert np.isclose(flow_times[-1] + self.T,
            #     (flow_df.time.max() - t_start) // self.T * self.T + t_start)
            assert len(flow_times) == len(metric_df)
            metric_df['time'] = flow_times
            res_df = pd.concat([res_df, metric_df], ignore_index=True)
        return res_df

    def btnk_condition(self, row, is_prev_btnk):
        btnked = row.skew_est < self.c_s or (row.skew_est < self.c_h and is_prev_btnk) \
            or row.pkt_loss > self.p_l
        return btnked

    def rmcat_clustering(self, res_df):
        """Based on RFC 8382 Section 3.3.
        TODO: test definitely needed!

        Input df has columns ['time', 'flow', 'skew_est', 'var_est', 'freq_est', 'pkt_loss'],
        i.e. the features are provided for each interval T. Thus, we will cluster the flow
        for each interval T, i.e. each row. The output is added as a new column for the df,
        with 0 for not traversing bottleneck, and 1~n for the group number.

        Returns:
            DataFrame: result with the 'group' column added.
        """
        flows = res_df.flow.unique()
        df_groups = res_df.groupby(by='time')
        btnk_group = pd.DataFrame(columns=['time', 'flow', 'group'])
        prev_btnk = {}          # flow: btnk or not

        def update_group(row, grp, btnk_group):
            brow = pd.DataFrame([[row.time, row.flow, grp]], columns=['time', 'flow', 'group'])
            btnk_group = pd.concat([btnk_group, brow], ignore_index=True)
            return btnk_group        


        for flow in flows:
            prev_btnk[flow] = False
        for time, group in df_groups:
            # each group is a df at the time
            # TODO: maybe some better pandas way than iteration
            # assert (group.flow.unique() == flows).all()
            rest = []
            # unused_grp_no = 1


            # 1. check the flow traversing a bottleneck
            for _, row in group.iterrows():
                btnked = self.btnk_condition(row, prev_btnk[row.flow])
                prev_btnk[row.flow] = btnked
                if not btnked:
                    btnk_group = update_group(row, row.flow, btnk_group)
                else:
                    rest.append(row.flow)

            # 2. group by all metrics
            if not rest:
                # print(' Warning: all the flows are not bottlenecked!')
                continue

            cols = ['skew_est', 'var_est', 'freq_est', 'pkt_loss']
            ths = [self.p_s, self.p_mad, self.p_f, self.p_d]
            relatives = [False, True, False, True]
            df = group.query(f'flow in {rest}').sort_values(by=cols).copy()
            n_col = len(cols) if df.pkt_loss.min() > self.p_l else len(cols) - 1
            df = df[['flow'] + cols[:n_col]].reset_index(drop=True)
            df = rmcat(df, cols[:n_col], ths[:n_col], relatives[:n_col])
            df['time'] = time
            for _, row in df.iterrows():
                btnk_group = update_group(row, row.group, btnk_group)

        return btnk_group


def sbd_process(csv, t_unit=0.005, folders=['.', '../BBR_test/ns3.27/MboxStatistics']):
    """Given csv file, runs SBD algorithm to get the features.
    """
    for folder in folders:
        path = os.path.join(folder, csv)
        if os.path.exists(path):
            csv = path
            break
    df = pd.read_csv(csv, index_col=False)
    sbd = SBDAlgorithm()
    res_df = sbd.stream_process(df, t_unit=t_unit)

    fields = ['skew_est', 'var_est', 'freq_est', 'pkt_loss']
    fig, axs = plt.subplots(len(fields), 1, figsize=(10, 2*len(fields)))
    for field, ax in zip(fields, axs):
        sns.lineplot(x='time', y=field, hue='flow', data=res_df, ax=ax)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 sbd.py CSV_NAME')
        exit(0)
    sbd_process(sys.argv[1])
