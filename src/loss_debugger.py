from datetime import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

from siamese.pipeline import predict_segment
from siamese.preprocess import DataGetter, DataModifier
from siamese.cluster import cluster
import torch
from siamese.plotter import DataPlotter
from siamese.pipeline import precision_recall_f1_from_raw, pairwise_precision_recall
from data_visualizer import DataVisualizer
from data_inspector import DataInspector

from threading import Thread, Lock, active_count

class LossEstimator:
    def __init__(self, a=1.0, b=0.3, w=0.5):
        self.a = a
        self.b = b
        self.w = w
    
    def euclidean_sq(self, x1, x2):
        return np.sum((x1 - x2) ** 2)
    
    def get_loss(self, anchor, pos, neg):
        anchor, pos, neg = np.array(anchor), np.array(pos), np.array(neg)
        d_an = self.euclidean_sq(anchor, neg)
        d_ap = self.euclidean_sq(anchor, pos)
        neg_loss = max(0, self.a ** 2 - d_an)
        pos_loss = max(0, d_ap - self.b ** 2)
        return neg_loss, pos_loss, self.w * neg_loss + (1 - self.w) * pos_loss


class LossDebugger:
    """Loss Debugger.
    
    Given the predict_info, xdf_pred and truth_df_pred of a certain runs,
    compute the losses of all triplet along all time intervals, as well as
    the all the pairwise distances & MDS plot.

    The idea is to quickly connect the model losses w/ the clustering details,
    and thus understand if the training procedure / loss function is good.
    The procedure is quite similar to DataVisualizer, as they both include
    reading various data & show the results, but LossDebugger is abstracted
    here to focus on the losses & distances analysis.

    Specifications:
        1. load: read the xdf, truth_df from raw data folder and the
            predict info [ys, y_hats, latents] using data visualizer.
        2. compute losses:
            a) compute the losses from latents and ys for a 
              given segment.
            b) compute aggregated losses along time.
        3. get pairwise distances:
            a) compute the pairwise distances
            b) show them in MDS plot.
        4. compare ys and final y_hats
        5. identify the time and flows that introduces large losses.

    Key data structues:
        1. loss_df: df [time, label, anchor, pos, neg, neg_loss, pos_loss, loss]
        2. distance_df: df [time, i, j, yi, yj, dist]
        3. y_hats_direct, y_hats
    
    Direct use cases:
        1. remove the zero OWD from the 3d testset, and check the losses
            vs time, to validate the losses on 3d testset.
        2. show the losses vs time & clustering for training set to validate
            if the model works for ALL flows in training set.
    
    The typical workflow is:

        - Load: load the predict_infos and raw data of an interesting run
        - Plot: agg loss vs t, losses for segment, clustering plot
        - Pinpoint: based on the clustering plot, use distance df & loss_df to find the bad flows
        - Debug: check raw time series of the bad flows at given time
        - Conclude: understand the reasons. 
    """
    def __init__(self, folder,
                data_root='round2_small',
                cache_root='/home/sapphire/neuralforecast/my_src/pre_cache',
                cache_folder=None):
        self.vis = DataVisualizer(folder, data_root=data_root,
                   cache_root=cache_root, cache_folder=cache_folder)
        self.folder = folder
        self.ins = None

    def load(self, run=None, n_flow=None, read_queue=False,
             config_run_base=0, read_raw_only=False, ts=None, interval=1.5,
             method='ml', no_cached_data=True, read_real=False):
        """Load raw data and predict info of a given run.
        Since it is probable that the raw data is read using caches like
        xdf_pred, truth_df_pred, we should mark the invalid flows to match
        the training process.

        Data below will be loaded into self:
            xdf, truth_df, y_mask, ys, y_hats, latent_map, ts
        
        The structure of latent_map is {t: latent}, latent is (n_flow, n_out).

        Args:
            run: the run to load
            n_flow: the number of flows to fetch entry from predict info
            check_cache: if check cache first, else read qid_df, qdf as well
                    this is required for queue plot for flows
            config_run_base: base of the relative config run for btnk_df
            no_cached_data: read data from data dir instead of cache dir.
                        Added to bypass the dir dependency on cache dir for data.
            read_real: read the real data instead of simulated dataset
        """
        if read_raw_only:
            assert ts is not None
            self.ts = list(np.arange(ts[0], ts[1], interval))
        self.vis.read_raw_across_runs(runs=[run], read_queue=read_queue,
                                      config_run_base=config_run_base,
                                      read_raw_only=read_raw_only,
                                      no_cached_data=no_cached_data,
                                      read_real=read_real)
        if read_raw_only:
            self.real_flows = list(self.vis.xdf.flow.unique())
            self.run = run
            self.run_abs = self.vis.xdf.run.unique()[0]
            self.n_flow = n_flow
            return
        self.vis.read_predict_infos(runs=[run, run + 1])
        invalid_flow_df = DataGetter.mark_invalid_owd(self.vis.xdf)
        assert invalid_flow_df.run.nunique() <= 1
        self.y_mask = invalid_flow_df.flow.unique()     # masked flows real No.
        self.keys = list(self.vis.all_predict_infos.keys())
        self.run = self.keys[0][0] if run is None else run
        self.n_flow = self.keys[0][1] if n_flow is None else n_flow
        key = (self.run, self.n_flow)
        run_abs = list(self.vis.all_predict_infos[key]['infos'].keys())[0]
        ml_predict_info = self.vis.all_predict_infos[key]['infos'][run_abs][method]
        self.real_flows = sorted(self.vis.all_predict_infos[key]['flows'])
        self.run_abs = run_abs
        self.ys = ml_predict_info[0]
        self.y_hats = ml_predict_info[1]
        self.latent_map = ml_predict_info[2]
        self.ts = sorted(list(self.latent_map.keys()))
 
    def get_non_btnk_flows(self, t, interval=1.5):
        """Get the segment non-btnk mask."""
        tmp = self.vis.xdf[(self.vis.xdf.time >= t) & (self.vis.xdf.time < t + interval)].copy()
        return DataGetter.get_non_btnk_flows(tmp)

    def compute_loss_for_segment(self, latent, y, y_mask=[]):
        """Compute the losses from latent and y. If the flow is in y_mask,
        ignore it. Here, we consider all the possible triplets, and compute
        their losses.
        
        Returns:
            loss_df: a dataframe with columns ['label', 'anchor', 'pos', 'neg',
                'neg_loss', 'pos_loss', 'loss']
        """
        # get dict {label: set[flow indices]}
        label_to_flows = {}
        valid_flows = set()
        for i, label in enumerate(y):
            if self.real_flows[i] in y_mask:
                continue
            if label not in label_to_flows:
                label_to_flows[label] = []
            label_to_flows[label].append(i)
            valid_flows.add(i)
        
        # compute losses
        le = LossEstimator()
        losses = []
        for label, flows in label_to_flows.items():
            for i, anchor in enumerate(flows[:-1]):
                for j, pos in enumerate(flows[i + 1:]):
                    neg_flows = valid_flows - set(flows)
                    for neg in neg_flows:
                        neg_loss, pos_loss, loss = le.get_loss(
                            latent[anchor], latent[pos], latent[neg])
                        losses.append([label, anchor, pos, neg, neg_loss, pos_loss, loss])
        loss_df = pd.DataFrame(losses,
            columns=['label', 'anchor', 'pos', 'neg', 'neg_loss', 'pos_loss', 'loss'])
        return loss_df

    def compute_loss_w_time(self, ts=None):
        """Computes the losses along time. Returns a df with columns
        ['time', 'label', 'anchor', 'pos', 'neg', 'neg_loss', 'pos_loss', 'loss'].
        """
        tloss_df = None
        ts = self.ts if ts is None else ts
        for i, t in enumerate(ts):
            latent = self.latent_map[t]
            loss_df = self.compute_loss_for_segment(latent, self.ys[i], self.y_mask)
            loss_df['time'] = t
            tloss_df = loss_df if tloss_df is None else pd.concat([tloss_df, loss_df],
                                                               ignore_index=True)
        return tloss_df

    def compute_distance(self, t):
        """Return a distance df of the latent of each flows, i.e.
        df with columns ['i', 'j', 'yi', 'yj', 'dist']."""
        latent = self.latent_map[t]
        y = self.ys[self.ts.index(t)]
        n = len(latent)
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(np.array(latent[i]) - np.array(latent[j]))
                res.extend([[i, j, y[i], y[j], d], [j, i, y[j], y[i], d]])
        distance_df = pd.DataFrame(res, columns=['i', 'j', 'yi', 'yj', 'dist'])
        return distance_df.sort_values(by=['i', 'j'])

    def update_y_hat_raw(self, max_iter=20, th=0.3):
        """Cluster the latents directly to get y_hats_raw w/o non-btnk update.
        y_hats_raw is loaded into self."""
        self.y_hats_raw = []
        for t in self.ts:
            latent = torch.tensor(self.latent_map[t])
            y_hat_raw, _ = cluster(latent, max_iter=max_iter, th=th)
            self.y_hats_raw.append(y_hat_raw)

    def print_y(self, y, tag):
        print('\n', tag)
        for j in range(len(y)):
            print(f'{int(y[j]):8d}', end=' ')
            if (j + 1) % 8 == 0:
                print()

    def get_distance_of_label(self, distance_df, label1, label2):
        """Get distances of label1 to label2. Abstratacted as it's one step of
        the loss debugging, and more flexible filtering can be developed in
        notebook directly.
        """
        df = distance_df[(distance_df.yi == label1) & (distance_df.yj == label2)]
        return df.sort_values(by='dist', ascending=False)
    
    def get_large_loss_per_group(self, loss_df, loss_type='loss', n=3):
        df = loss_df.copy()
        idx = df.groupby(['label'])[loss_type].nlargest(n).reset_index()['level_1']
        return df.loc[idx].sort_values(by=['label', loss_type], ascending=False)

    def visualize_flow_segment(self, t1, t2, flows, remove_mask=False, interval=1.5):
        real_flows = [self.real_flows[i] for i in flows]
        if remove_mask:
            real_flows = list(set(real_flows) - set(self.y_mask))
        self.vis.plotter.plot_flows(self.vis.xdf, self.run_abs, t1, t2, flows=real_flows)

    def visualize_agg_loss(self, tloss_df):
        """Visualize the pos/neg/overall losses along time."""
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        for i, loss_type in enumerate(['neg_loss', 'pos_loss', 'loss']):
            g = sns.boxplot(x='time', y=loss_type, data=tloss_df, ax=ax[i])
            g.set_xticklabels(g.get_xticklabels(), rotation=45, fontsize=6)
        plt.tight_layout()
        plt.show()

    def _get_masked_var(self, var, mask):
        return [var[i] for i in range(len(var)) if self.real_flows[i] not in mask]
    
    def _get_masked_vars(self, *args, mask=None):
        """Note: this cannot be used twice, as the yi changes!!!"""
        if mask is None:
            return tuple(args)
        return tuple(map(lambda x: self._get_masked_var(x, mask), args))

    def visualize_latent(self, ts, use_mask=False, remove_nonbtnk=False):
        """A quick plot of latents of some times."""
        fig, ax = plt.subplots(len(ts), 1, figsize=(5, 3.5 * len(ts)))
        if len(ts) == 1:
            ax = [ax]
        markers = {}
        for i, t in enumerate(ts):
            ti = self.ts.index(t)
            y, y_hat = self.ys[ti], self.y_hats[ti]
            latent = self.latent_map[t]
            nonbtnk_flows = self.get_non_btnk_flows(t)
            if not remove_nonbtnk and use_mask:
                y, y_hat, latent = self._get_masked_vars(y, y_hat, latent, mask=self.y_mask)
            elif remove_nonbtnk and not use_mask:
                y, y_hat, latent = self._get_masked_vars(y, y_hat, latent, mask=nonbtnk_flows)
            elif remove_nonbtnk and use_mask:
                mask = set(self.y_mask) | set(nonbtnk_flows)
                y, y_hat, latent = self._get_masked_vars(y, y_hat, latent, mask=mask)
            for label in y:
                markers[label] = 'o' if label < 500 else 'X'
            self.vis.plotter.axplot_latent(y, y_hat, latent, hue='truth',
                style='truth', markers=markers, palette='tab10', ax=ax[i])
            ax[i].legend(bbox_to_anchor=(1.1, 0), loc='lower right', fontsize=5)
        plt.tight_layout()
        plt.show()

    def init_inspector(self, out_dir='figures', tag='d-insp',
                      do_show=False, down_sample=True):
        """This is the usage of the data inspector.
        Note that out_dir, tag and do_show are passed to the inspector,
        and plot and labels are passed to the segment process.
        """
        ts_ins = [self.ts[0], self.ts[-1]]
        # patch for the real data
        if type(self.vis.xdf.flow[0]) == str:
            DataModifier.convert_flow_to_int(self.vis.xdf)
        self.ins = DataInspector(self.vis.xdf, list(self.real_flows), ts_ins,
                            out_dir=out_dir, tag=tag, do_show=do_show)
        if down_sample:
            self.ins.general_process()

    def print_summary(self, file=None, lock=None):
        """Print the summary of this run."""
        assert self.ins is not None
        ti = int(len(self.ts) // 2) - 1
        t0, t1 = self.ts[ti], self.ts[ti + 1]
        if hasattr(self, 'ys'):
            self.ins._set_seg_df([t0, t1], self.ys[ti])
        else:
            self.ins._set_seg_df([t0, t1], [])
        self.cluster_df, t0, t1 = self.ins.get_n_flow_per_cluster()

        print('- Print summary of', self.folder, self.run, '...')
        if lock is not None:
            lock.acquire()
        print(f'\n=== Summary of {self.folder}, run {self.run} ({self.run_abs}) ===\n', file=file)
        print(f'  n_flow: {self.n_flow}, n_ts: {len(self.ts)} ({self.ts[0]} - {self.ts[-1]}s)', file=file)
        if self.cluster_df is not None:
            print(f'\n n_flow_per_cluster (sampled at {t0}-{t1}s)\n',
                  self.cluster_df, file=file)
        print('\n=== End of summary ===', file=file)
        if lock is not None:
            lock.release()

    def plot_separate_cluster(self, labels, ts, field):
        self.ins.plotter.plot_separate_clusters(self.ins.xdf, labels, ts, field)
 
    def plot_repeated_cluster(self, labels, ts, field):
        self.ins.plotter.plot_repeated_clusters(self.ins.xdf, labels, ts, field)

    def plot_cluster_w_members(self, labels, ts, field, n_member=3):
        # use large n_member to plot all members
        self.ins.plotter.plot_cluster_w_members(self.ins.xdf, labels, ts,
                                                    self.ins.intra_cluster_corrs, field,
                                                    n_member=n_member)

    def visualize_flows_and_latents(self, ti, flows=None, labels=None,
                                    flow_hue='label', latent_hue='label',
                                    n_flow_legend=15, fields=['owd', 'rtt'],
                                    plot_queue=False,
                                    xlim=None, ylim=None, ax_lim=None):
        """Visualize the raw flow time series and the latents clustering plot
        side by side for the given flows / labels, w/ the same cluster having
        the same color and legend label.
        
        The plot layout is n x 2, where n rows are len(fields) + 1 (for latent), and
        the two columns are for raw and masked figures.
        Note that the hue of left and right plots can be chosen from label and flow,
        as label hue is good for clustering observation, while flow hue provides the
        No. for detailed debug later.
        
        TODO: abstract the flow and latent plot to dataplot in either inspector
        or visualizer, not directly done yet as current version is well tested.

        Args:
            ti (int): the index of the time segment
            flows (list): the real flows' No. to be plotted
            labels (list): the labels of chosen clusters
            flow_hue (str): the hue of the left columns, either 'label' or 'flow'
            latent_hue (str): the hue of the right columns, either 'label' or 'flow'
            fields (list): the fields to be plotted in the left columns
            plot_queue (bool): whether to plot queue for given labels
            xlim (tuple): xlim for specified axis
            ylim (tuple): ylim for specified axis
            ax_lim (list): list of tuples (i, j) of ax[i, j] to apply lim
        
        Returns:
            folder, run, ti, labels, flows
        """
        # try:
        t1, t2 = self.ts[ti], self.ts[ti + 1]
        y0, y_hat0 = self.ys[ti], self.y_hats[ti]
        self.ins.segment_process([t1, t2], y0, labels=labels) 
        # except Exception as e:
        #     print('loss_debugger: Error in segment_process, skip this time segment.')
        #     print(e)
        #     return None, None, None, None, None
        if not flows and not labels:
            flows, labels = self.real_flows, self.ins.seg_df.label.unique()
        elif labels and not flows:
            flows = self.ins.seg_df[self.ins.seg_df.label.isin(labels)].flow.tolist()
        latent0 = self.latent_map[t1]
        unwanted = set(self.real_flows) - set(flows)
        nonbtnk_flows = self.get_non_btnk_flows(t1)
        data_state = ['Raw', 'Masked / non-btnk removed']
        latent_hue = 'truth' if latent_hue == 'label' else latent_hue
        n_row = len(fields) + 2 if plot_queue else len(fields) + 1
        fig, ax = plt.subplots(n_row, 2, figsize=(9, 3.0 * n_row))

        j0 = 0
        if plot_queue:
            assert hasattr(self.vis, 'qid_df') and hasattr(self.vis, 'qdf'), \
                'No qdf/qid_df found, use check_cache=True in load() to load them'
            labels_list = [labels, labels]  # assume labels don't change
            axes = [ax[0, 0], ax[0, 1]]
            self.vis.dataplot_queue_for_label(self.run_abs, t1, t2, labels_list, axes)
            j0 = 1

        for i in range(2):
            # prepare and mask the data
            assert 'label' in self.ins.seg_df.columns
            if i == 0:
                mask = set(unwanted)
            else:
                # careful: real flows must be filter as well to allow y->flow mapping
                mask = set(self.y_mask) | set(nonbtnk_flows) | set(unwanted)
            y, y_hat, latent = self._get_masked_vars(y0, y_hat0, latent0,
                                                     mask=mask)
            real_flows = self._get_masked_var(self.real_flows, mask=mask)
            flows1 = set(flows) - set(mask)      # expect still flows
            for var in [y, y_hat, latent]:
                if var is None or len(var) == 0:
                    print(' - Error: empty y, y_hat or latent!')
                    return None, None, None, None, None
 
            # plot flows
            axes = [ax[j0 + j, i] for j in range(len(fields))]
            self.ins.dataplot_flows(data_state[i], t1, t2, flows1, flow_hue, fields,
                                    axes, n_flow_legend)
            
            # plot latents
            y1 = DataModifier.get_updated_nonbtnk_labels(y)
            # do this for y_hat or not?
            y_hat1 = DataModifier.get_updated_nonbtnk_labels(y_hat)
            markers = {}
            for label in y1:
                markers[label] = 'o' if label < 500 else 'X'
            self.vis.plotter.axplot_latent(y1, y_hat1, latent, hue=latent_hue,
                                           style='truth',
                                           markers=markers, palette='tab10',
                                           ax=ax[-1, i],
                                           real_flows=real_flows)
            ax[-1, i].set_title(data_state[i] + ' latents')
            nl = n_flow_legend if latent_hue == 'flow' else None
            self.vis.plotter.cut_legend(ax[-1, i], nl)
            # self.ins.plotter.cut_legend(ax[len(fields), i], nl)

        if ax_lim is not None:
            for i, j in ax_lim:
                ax[i, j].set_xlim(xlim)       
                ax[i, j].set_ylim(ylim)

        ftag = self.folder[8:]
        ftag = ftag[:-5] if 'run' in ftag[-5:] else ftag
        fname = f'flow_vs_latent_{ftag}_r{self.run}_f{len(flows)}_ti{ti}_{t1}-{t2}s.pdf'
        print(f'  - folder {self.folder} run {self.run} ti {ti}: {fname}')
        self.ins.plotter.show_or_save(fname, tight_layout=False)
        return self.folder, self.run, ti, labels, flows

    def visualize_raw_flows(self, ti, flows=None,
                            flow_hue='label', tj=None,
                            n_flow_legend=15, fields=['owd', 'rtt'],
                            plot_queue=False,
                            keep_large_owd=False,
                            xlim=None, ylim=None, ax_lim=None,
                            ylims=None, axes_ylim=None):
        """Visualize the raw flow time series only to quick debug queue and flows.

        Args:
            ti (int): the index of the time segment
            flows (list): the real flows' No. to be plotted
            labels (list): the labels of chosen clusters
            flow_hue (str): the hue of the left columns, either 'label' or 'flow'
            n_flow_legend (int): max number for flow legend to show on plot
            fields (list): the fields to be plotted in the left columns
            plot_queue (bool): whether to plot queue for given labels
            keep_large_owd (bool): if only keep the flows w/ owd > rtt
            xlim (tuple): xlim for specified axis
            ylim (tuple): ylim for specified axis
            ax_lim (list): list of int i to apply lim
            ylims (list): list of ylim for different ax
            axes_ylim (list): list of axes to apply ylims

        Returns:
            folder, run, ti, labels, flows
        """
        tj = ti + 1 if tj is None else tj
        t1, t2 = self.ts[ti], self.ts[tj]
        y = []
        self.ins.segment_process([t1, t2], y) 
        if keep_large_owd:
            self.ins.keep_large_owd_flow()
        flows = flows if flows else self.real_flows
        n_row = len(fields) + 1 if plot_queue else len(fields)
        fig, ax = plt.subplots(n_row, 1, figsize=(5, 2.5 * n_row))

        j0 = 0
        if plot_queue:
            assert hasattr(self.vis, 'qid_df') and hasattr(self.vis, 'qdf'), \
                'No qdf/qid_df found, use check_cache=True in load() to load them'
            labels_list = [None]  # assume labels don't change
            self.vis.dataplot_queue_for_label(self.run_abs, t1, t2, labels_list, [ax[0]])
            j0 = 1

        # plot flows
        axes = [ax[j0 + j] for j in range(len(fields))]
        self.ins.dataplot_flows('rawflows', t1, t2, flows, flow_hue, fields,
                                axes, n_flow_legend)
            
        if axes_ylim is not None:
            for i, ylim in zip(axes_ylim, ylims):
                ax[i].set_ylim(ylim)
        elif ax_lim is not None:
            for i in ax_lim:
                ax[i].set_xlim(xlim)       
                ax[i].set_ylim(ylim)

        ftag = self.folder[8:]
        ftag = ftag[:-5] if 'run' in ftag[-5:] else ftag
        fname = f'flow_{ftag}_r{self.run}_f{len(flows)}_ti{ti}_{t1}-{t2}s.pdf'
        print(f'  - folder {self.folder} run {self.run} ti {ti}: {fname}')
        self.ins.plotter.show_or_save(fname)
        return self.folder, self.run, ti, flows




"""
Goal: plot the figures for all the runs / intervals in given folders. 

Requirements:

1) Reduce the size to read.
    Considering there are too many runs / intervals in total, we should do some
    sampling for them, e.g. sample 1 run / interval in two, and draw all of them for some
    interesting intervals.

    Combine figures for quicker reading? Not for now, as complicated for the visualization.

2) Increase the speed.
    Should be okay, but remember to use the general process only once, and then just reuse
    the segment process.
    multi threading to reduce the I/O time

3) subdir
    Use out_dir to directly input 'figures/folder', and use tag to represent the run.
    Maybe just show them in notebook.

4) Fine plotting after round 1
    To enable this, should take (folder, run, ti, labels, flows) as input, and the api
    should support both first and this round. 
    And to ease the input of fine plot config, the api should output a csv containing
    current config w/ all labels for current ti, but flows is [].

Data architecutre:

- input: df
- folder, run:
    Init LossDebugger & DataInspector
    general_process
- ti, labels/flows:
    visualize_flows_and_latents
    collect the output configs for later df output
    
"""

def _visualize_flows_and_latents_for_run(config_df, folder, run, t_sample_step,
                                         res_config, out_dir, lock, file):
    """Visualize in a thread, and append the configs in res_config.
    This should be called to create a thread. Since we have enough runs for
    the processes, here we do not add the parallelism for plotting.

    Note that if ti is None in config_df, here it'll complete all the ti from ts.
    File: is a file object used to write the summary to.
    """
    for col in ['folder', 'run', 'ti', 'labels', 'flows']:
        assert col in config_df.columns
    print(f'  - Start visualize folder {folder} run {run} ...')
    ldbg = LossDebugger(folder)
    ldbg.load(run)
    config_df = config_df[(config_df.folder == folder) & (config_df.run == run)].copy()
    config_df.sort_values(by=['ti'], inplace=True)
    if len(config_df) == 1 and config_df.ti.values[0] is None:
        rows = [[folder, run, i, None, None]
                for i in range(0, len(ldbg.ts), t_sample_step)]
        rows_df = pd.DataFrame(rows, columns=config_df.columns)
        config_df.dropna(subset=['ti'], inplace=True)
        config_df = pd.concat([config_df, rows_df], ignore_index=True)

    do_show = out_dir is None
    out_dir = os.path.join(out_dir, folder) if out_dir is not None else None
    for i, r in config_df.iterrows():
        if i % t_sample_step == 1:
            print(f'    - Skip ti {r.ti} ...')
            continue
        print(f'    - Inspect ti {r.ti}')
        ldbg.init_inspector(out_dir=out_dir, do_show=do_show, tag=f'run{run}') 
        _, _, _, labels, flows = ldbg.visualize_flows_and_latents(
            r.ti, labels=r.labels, flows=r.flows,
            flow_hue='label', latent_hue='label')
        lock.acquire()
        flows = flows if flows is None else flows[:10]
        res_config.append([folder, run, r.ti, labels, flows])
        lock.release()
    ldbg.print_summary(file=file, lock=lock)


def visualize_flows_and_latents_for_all(config_df, run_sample_step=1, t_sample_step=2,
                                        out_dir=None, n_thd=4):
    """Visualize flows and latents for all folders & runs.

    Args:
        config_df (DataFrame): df [folder, run, ti, labels, flows]
        run_sample_step (int): in how many entries sample one run
        t_sample_step (int): in how many entries sample one interval
        do_show (bool): if show them directly or save the figures in dir
        out_dir (str or None): the figure output dir. None for show them.
    
    Returns:
        DataFrame: the config info of actual plotted figures
    """
    for col in ['folder', 'run', 'ti', 'labels', 'flows']:
        assert col in config_df.columns
    # config_df.sort_values(['folder', 'run', 'ti'], inplace=True)
    threads, res_config = [], []
    lock = Lock()
    time_str = datetime.now().strftime('%b%d-%H:%M:%S')
    file = open(os.path.join(out_dir, f'flow_vs_latent_summary_{time_str}.txt'), 'w')
    folder_run = config_df[['folder', 'run']].drop_duplicates()
    for i, (folder, run) in enumerate(folder_run.values):
        if i % run_sample_step == 1:
            continue
        thread = Thread(target=_visualize_flows_and_latents_for_run,
                        args=(config_df, folder, run, t_sample_step, res_config,
                              out_dir, lock, file))
        threads.append(thread)
    base_thd_cnt = active_count()
    for i, thread in enumerate(threads):
        while active_count() - base_thd_cnt >= n_thd:
            time.sleep(3)
        thread.start()
    for thread in threads:
        thread.join()
    file.close()
    return pd.DataFrame(res_config, columns=['folder', 'run', 'ti', 'labels', 'flows'])


if __name__ == '__main__':
    plot_all = True
    
    if not plot_all:
        # raw test here
        folder = 'results_cb-tleft-small_4run'
        run, n_flow = 0, 300
        t = 21
        labels = [203, 506]

        ldbg = LossDebugger(folder)
        ldbg.load(run)
        ti = ldbg.ts.index(t)
        ldbg.init_inspector(do_show=False)
        ldbg.visualize_flows_and_latents(ti, labels=labels,
                                        flow_hue='flow', latent_hue='flow')
        flows = [272, 545]
        ldbg.visualize_flows_and_latents(ti, flows=flows)
    else:
        # use visualzie for all
        folder_run = {
            'results_dummy-left-small_2run': [0, 1],
            'results_dummy-right-small_2run': [0, 1],
            'results_dummy-left-middle_2run': [0, 1],
            'results_dummy-right-middle_2run': [0, 1],
            'results_dummy-large_2run': [0, 1],
        }
        rows = [] 
        for folder, runs in folder_run.items():
            for run in runs:
                rows.append([folder, run, None, None, None])
        config_df = pd.DataFrame(rows, columns=['folder', 'run', 'ti', 'labels', 'flows'])
        res_df = visualize_flows_and_latents_for_all(config_df, run_sample_step=1,
                                                    t_sample_step=1, out_dir='figures',
                                                    n_thd=8)
        time_str = datetime.now().strftime('%b%d-%H:%M:%S')
        fconfig = f'figures/flow_latent_config_{time_str}.csv'
        res_df.to_csv(fconfig, index=False)
        print(f'Config saved to {fconfig}.')
