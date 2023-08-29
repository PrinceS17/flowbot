import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from siamese.plotter import DataPlotter
from siamese.preprocess import DataModifier

class DataInspector:
    """
    """
    def __init__(self, xdf, real_flows, ts,
                 out_dir='figures',
                 do_show=False,
                 tag='d-insp'):
        """
        Data Inspector, a tool used to quickly inspect the flow time series from raw flow data
        and their labels. It includes the following features:

            1. Faster flow data processing through downsampling;
            2. Intra-cluster flow range plot in detail using avg flow and range;
            3. Convenient inter-cluster comparison;
            4. Correlation between intra-cluster flows and avg flows of each cluster;
            5. Easy comparison w/ clustering plots in loss_debugger to analyze it.

        Data inspector takes the xdf (raw data), ys (labels), real_flows (y to x mapping)
        as inputs, computes avg flows of each clusters, intra and inter cluster
        correlations, and plots flow range plots for intra / inter cluster flows.
        Label and flow are the two main filters when using data inspector, and typical
        usage is to compare the raw time series with the clustering plots to figure out
        if the time series is abnormal or the model works poorly.    

        Args:
            xdf (DataFrame): df containing the raw time series [run, flow, time, owd, rtt, slr, cwnd]
            ys (list): list containing the labels of chosen flows
            real_flows (list): the flow No. in xdf for ys
            ts (list): [t_start, t_end]
        """
        self.real_flows = real_flows
        assert len(set(xdf.flow.unique()) & set(real_flows)) > 0, \
            f'Error: no flow intersection: {set(xdf.flow.unique()) & set(real_flows)}'
        self.xdf = xdf[(xdf.flow.isin(set(real_flows))) & (xdf.time.between(ts[0], ts[1]))].copy()
        self.seg_df = None
        assert self.xdf.run.nunique() == 1, f'Error: # run in xdf: {self.xdf.run.nunique()}'
        # for flow in real_flows:       # contradicts w/ the previous filtering
        #     assert flow in self.xdf.flow.unique()
        out_dir = os.path.join(out_dir, tag)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        self.plotter = DataPlotter(out_dir, do_show, tag)
        self.ts = None
        self.y = None

    def general_process(self):
        self._downsample()

    def segment_process(self, ts, y, labels=None, field='owd', plot=False,
                        compute_corr=False):
        # avg flows for each time interval, as labels / ys change over time
        self.avg_flows = None
        self._set_seg_df(ts, y)
        if compute_corr:
            self._compute_intra_cluster_corr(ts, field)
            self._compute_inter_cluster_corr(ts, field)
        if not plot:
            return
        if len(labels) > 20:
            print(f'Warning: {len(labels)} clusters are selected, too many to plot.')
        self.plotter.plot_separate_clusters(self.seg_df, labels, field)
        self.plotter.plot_repeated_clusters(self.seg_df, labels, field)
        # cluster w far members is the active functionality, listed here for reference
        self.plotter.plot_cluster_w_members(self.seg_df, labels, 
                                            self.intra_cluster_corrs, field,
                                            n_member=100, ascending=False)

    def _downsample(self, ds_interval=0.05):
        """Downsample xdf to reduce the size. """
        assert ds_interval >= 0.005
        self.xdf['time'] = self.xdf.time // ds_interval * ds_interval
        self.xdf = self.xdf.groupby(['run', 'flow', 'time']).mean().reset_index()
    
    def get_n_flow_per_cluster(self):
        """Get the number of ts, and general number of each clusters.
        TODO: currently flows are labeled only at seg_df but not xdf
        """
        if self.seg_df is None:
            print('Warning: seg_df is None, cannot get n_flows per cluster.')
            return None
        t0, t1 = self.seg_df.time.min(), self.seg_df.time.max()
        cluster_df = self.seg_df[['run', 'label', 'flow']].drop_duplicates()
        cluster_df = cluster_df.groupby(['run', 'label']).nunique().reset_index()
        cluster_df = cluster_df.rename(columns={'flow': 'n_flows'}).sort_values('n_flows')
        return cluster_df, t0, t1
    
    def _set_seg_df(self, ts, y=[]):
        """Prepare seg_df w/ labels."""
        # increase efficiency, but affect debugging
        # if self.ts== ts and self.y == list(y):
        #     return
        seg_df = self.xdf[self.xdf.time.between(ts[0], ts[1])].copy()
        if len(y) == 0:
            self.seg_df = seg_df
            self.ts = ts
            self.y = []
            return
        assert len(y) == len(self.real_flows), \
            f'Error: length of y ({len(y)}) and real flows ({len(self.real_flows)}) '\
            f'are not equal, mapping cannot be established (occurs mostly at the '\
            f'beginning)'
        labeled_flows = [self.real_flows[i] for i in range(len(y))]
        seg_df = seg_df[seg_df.flow.isin(set(labeled_flows))]
        seg_df['label'] = seg_df.flow.apply(lambda x: y[self.real_flows.index(x)])
        DataModifier.update_nonbtnk_labels_for_df(seg_df)

        # get avg flows
        self.avg_flows = seg_df.groupby(['run', 'time', 'label']).mean().reset_index()
        self.avg_flows.drop(columns=['flow'], inplace=True)
        self.seg_df = seg_df
        self.ts = ts
        self.y = list(y)

    def _compute_intra_cluster_corr(self, ts, field):
        """Computes intra cluster correlations among all the flow pairs.
        The default field to compute is OWD, ts is for interval filtering.
        Returns df [run, label, flow1, flow2, {field}_corr]. Note that
        if flow2 == -1, it means the correlation is between flow1 and the
        avg flow of the cluster.

        Note that corr returns NaN if one of s1, s2 is constant.
        """
        intra_cluster_corrs = []
        run = self.xdf.run.unique()[0]
        seg_df = self.seg_df.set_index('flow')
        seg_avg_flows = self.avg_flows[self.avg_flows.time.between(ts[0], ts[1])]
        seg_avg_flows = seg_avg_flows.set_index('label')

        for label in seg_df.label.unique():
            all_flows = seg_df[seg_df.label == label].index.unique()
            for i, flow1 in enumerate(all_flows):
                s1 = seg_df.loc[flow1, field].reset_index(drop=True)
                for flow2 in all_flows[i + 1:]:
                    s2 = seg_df.loc[flow2, field].reset_index(drop=True)
                    intra_cluster_corrs.append([run, label, flow1, flow2, s1.corr(s2)]) 
                s3 = seg_avg_flows.loc[label, field].reset_index(drop=True)
                intra_cluster_corrs.append([run, label, flow1, -1, s1.corr(s3)])
        self.intra_cluster_corrs = pd.DataFrame(intra_cluster_corrs,
            columns=['run', 'label', 'flow1', 'flow2', f'{field}_corr'])
        self.intra_cluster_corrs.dropna(inplace=True)

    def _compute_inter_cluster_corr(self, ts, field):
        """Computes inter cluster correlations, i.e. the correlations between
        the avg flows of each cluster to save computation time.
        Returns df [run, label1, label2, {field}_corr].
        """
        assert self.avg_flows is not None
        inter_cluster_corrs = []
        run = self.xdf.run.unique()[0]
        seg_avg_df = self.avg_flows[self.avg_flows.time.between(ts[0], ts[1])]
        seg_avg_df = seg_avg_df.set_index('label')
        for i, label1 in enumerate(self.seg_df.label.unique()):
            for label2 in self.seg_df.label.unique()[i + 1:]:
                s1 = seg_avg_df.loc[label1, field].reset_index(drop=True)
                s2 = seg_avg_df.loc[label2, field].reset_index(drop=True)
                inter_cluster_corrs.append([run, label1, label2, s1.corr(s2)])
        self.inter_cluster_corrs = pd.DataFrame(inter_cluster_corrs,
            columns=['run', 'label1', 'label2', f'{field}_corr'])
        self.inter_cluster_corrs.dropna(inplace=True)

    def get_corr_dfs(self):
        return self.intra_cluster_corrs, self.inter_cluster_corrs

    def dataplot_flows(self, name, t1, t2, flows, hue, fields, axes, n_flow_legend=15):
        """Plot the flows' fields in the given axies.
            hue can be chosen from 'flow' and 'label'.
        """
        assert len(fields) == len(axes)
        for i, (field, ax) in enumerate(zip(fields, axes)):
            if hue == 'label':
                kwargs = {'hue': 'label', 'units': 'flow', 'legend': 'brief',
                            'style': 'flow', 'estimator': None}
            elif hue == 'flow':
                kwargs = {'hue': 'flow', 'legend': 'brief'}
            else:
                raise ValueError('flow_hue must be label or flow')
            self.plotter.axplot_flows(self.seg_df, flows, t1, t2, field, ax, **kwargs)
            h, l = ax.get_legend_handles_labels()
            nl = l.index('flow') if hue == 'label' else n_flow_legend
            self.plotter.cut_legend(ax, nl)
            ax.set_title(f'{name} {field} at {t1} - {t2} s')

    def keep_large_owd_flow(self):
        """Find those flows that avg owd > rtt for later visualization.
        """
        df = self.seg_df.copy()
        df['avg_owd'] = df.groupby(['run', 'flow']).owd.transform('mean')
        df['avg_rtt'] = df.groupby(['run', 'flow']).rtt.transform('mean')
        self.seg_df = df[df.avg_owd >= df.avg_rtt].reset_index(drop=True)
        assert len(self.seg_df) > 0, 'Nothing to keep!'