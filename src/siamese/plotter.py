import bisect
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from siamese.preprocess import DataGetter
from sklearn.manifold import MDS
from matplotlib.legend_handler import HandlerTuple

MTC_NAME = {
    'rel_f1': 'F1',
    'rel_precision': 'Precision',
    'rel_recall': 'Recall',
    't_classify': 'Detecting\nTime (s)',     # Processing Time (s)?
    'para_btnk': 'Bottleneck Count',
    'n_flow': 'Flow Count',
    'btnk_ratio': 'Bottleneck Count\n/ Link Count',
    'distance': 'Distance'
}

METHOD_NAME = {
    'dcw_0.8': 'DCW',
    'rmcat_sbd': 'rmcatSBD',
    'dc_sbd': 'dcSBD',
    'ml_02_0.2': 'FlowBot',
    'ml_012_0.2': 'FlowBot',
    'ml_v9_th0.40': 'FlowBot_v9',
    'ml_v9_0.4': 'FlowBot',
    'neg_naive_0.2': 'Neg Naive',
    'pos_naive_0.2': 'Pos Naive',
    'intra_dist': 'Intra-cluster neighbor',
    'inter_dist': 'Inter-cluster neighbor',
}


def get_method_name(m):
    return METHOD_NAME[m] if m in METHOD_NAME else m


def group_df_in_runs(res_df, n_run=None, run_boundary=None):
    """Group the runs based on n_run or run_boundary.
    If a value is on the boundary, then assign to the right group,
    And  Last boundary is not needed.
    e.g. run_boundary = [0, 4, 8], then run 0, 1, 2, 3 will be in group 0,
    and run 4, 5, 6, 7 will be in group 1, run 8, 9, 10, 11 will be in group 2.
    """
    n_run = 4 if n_run is None else n_run
    if run_boundary is None:
        run_boundary = range(0, res_df.run.nunique(), n_run)
    run_boundary = list(map(lambda i: sorted(res_df.run.unique())[i], run_boundary))
    res_df = res_df.copy()
    res_df['group'] = res_df.run.transform(lambda r: bisect.bisect(run_boundary, r))
    return res_df


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


class DataPlotter:
    """
    Plotter: all plotters.

    The core motivation is to utilize the flexibility of axis to customize
    the ax attribiute xlabel, title, etc and the layout outside the function
    for easy finalization.

    All functions with prefix 'axplot' is the atomic axis plot functions,
    which returns the axis object for further customization. All other plot
    functions are high level functions intended for the final figure.
    """
    def __init__(self, out_dir=None, do_show=True, tag=None) -> None:
        self.out_dir = out_dir
        self.do_show = do_show
        self.tag = tag
        sns.set_context('paper', font_scale=0.8)
        # sns.set_style('white',
        #               {'font.family':'serif', 'font.serif':'Times New Roman'})
        plt.rc(('xtick', 'ytick'), labelsize=7)
        plt.rc('xtick.major', size=2, pad=2)
        plt.rc('ytick.major', size=2, pad=2)
        plt.rc('ytick.minor', size=0, pad=2)
        plt.rc('axes', labelweight='bold')
        self.hatches = ['////', '||||', '\\\\\\\\', '....',
                        '++++', 'xxxx', '1111', '2222']
        # single plot height slight larger to compensate for legend & xlabel
        self.height, self.single_height = 0.85, 0.9

    # TODO: replace all plt.show() with this function & fname
    def show_or_save(self, fname, tight_layout=True, wspace=None, merge_axis='y',
                     whspace=None):
        assert merge_axis in ['x', 'y']
        if whspace:
            plt.subplots_adjust(wspace=whspace[0], hspace=whspace[1])
        elif tight_layout:
            plt.tight_layout()
        elif wspace:
            plt.subplots_adjust(wspace=wspace)
        elif merge_axis == 'y':
            plt.subplots_adjust(hspace=0)
        elif merge_axis == 'x':
            plt.subplots_adjust(wspace=0)
        # plt.rcParams.update({
        #         'xtick.labelsize': 7,
        #         'ytick.labelsize': 7,
        #         'xtick.major.size': 2,
        #         'xtick.major.pad': 2,
        #         'ytick.major.size': 2,
        #         'ytick.major.pad': 2,
        #         'ytick.minor.size': 0,
        #         'ytick.minor.pad': 2,
        #         'axes.labelweight': 'bold',
        #     })
        if self.do_show:
            plt.show()
            plt.close()
        else:
            if self.tag is not None:
                fname = f'{self.tag}_{fname}'
            if self.out_dir is not None:
                fname = os.path.join(self.out_dir, fname)
            # bbox_inches & pad_inches control the white margin around the figure
            plt.savefig(fname, bbox_inches='tight', pad_inches=0.01)
            plt.close()
            print(f'Figure saved to {fname}')

    def _set_miscellaneous(self, ax):
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    def _set_y_for_t_classify(self, ax, keep_ticks=True):
        ax.set(yscale='log')
        if not keep_ticks:
            ax.set(yticks=[], yticklabels=[])
        else:
            ax.set(yticks=[0.1, 1, 10, 100])
        return 'point'        # for plt_type

    def add_legend(self, ax, title, loc='lower left', fontsize=6, bbox_to_anchor=None):
        legend = ax.legend(labels=[], title=title, loc=loc, frameon=True,
                           title_fontsize=fontsize, fontsize=4, borderpad=0.3,
                           bbox_to_anchor=bbox_to_anchor)
        ax.add_artist(legend)

    def _set_hatches(self, ax, n_hatch, n_grp, plt_type='box'):
        # print(f'Number of hatches: {n_hatch}, number of groups: {n_grp}')
        patch_type = matplotlib.patches.PathPatch if plt_type == 'box' else \
                matplotlib.patches.Rectangle
        patches = [patch for patch in ax.patches if type(patch) == patch_type]
        assert len(patches) == n_hatch * n_grp, \
            f'Number of patches {len(patches)} != {n_hatch} * {n_grp}'
        for i, patch in enumerate(patches):
            hatch = self.hatches[i % n_hatch] if plt_type == 'box' else \
                self.hatches[i // n_grp]
            patch.set_hatch(hatch)
            fc = patch.get_facecolor()
            patch.set_edgecolor(fc)
            patch.set_facecolor('none')

    def _set_legend(self, ax, title=None, labels=None, keep=True, loc='lower center',
                    bbox_to_anchor=(0.5, 1.01), fontsize=6, ncol=4, plt_type='box'):
        if not keep:
            ax.legend().remove()
            # ax.legend().set_visible(False)
            return
        lg = ax.legend(title=title, loc=loc, bbox_to_anchor=bbox_to_anchor,
                       fontsize=fontsize, ncol=ncol, frameon=False)
        labels = labels or [get_method_name(m.get_text()) for m in ax.get_legend().texts]
        if labels is not None:
            for t, l in zip(ax.get_legend().texts, labels):
                t.set_text(l)
        for lp, hatch in zip(lg.get_patches(), self.hatches):
            lp.set_hatch(hatch)
            # barplot doesn't need the lines below somehow
            if plt_type == 'box':
                fc = lp.get_facecolor()
                lp.set_edgecolor(fc)
                lp.set_facecolor('none')

    def _combine_legends(self, ax1, ax2, title=None, labels=None, loc='lower center',
                    bbox_to_anchor=(0.5, 1.01), fontsize=6, ncol=4):
        h1, l1 = ax1.get_legend_handles_labels()
        h2, _ = ax2.get_legend_handles_labels()
        lg = ax1.legend(zip(h1, h2), l1, title=title, loc=loc,
                        bbox_to_anchor=bbox_to_anchor,
                        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
                        fontsize=fontsize, ncol=ncol, frameon=False, handlelength=2.5)
        labels = labels or [get_method_name(m.get_text()) for m in ax1.get_legend().texts]
        if labels is not None:
            for t, l in zip(ax1.get_legend().texts, labels):
                t.set_text(l)
        for lp, hatch in zip(lg.get_patches(), self.hatches):
            lp.set_hatch(hatch)
            fc = lp.get_facecolor()
            lp.set_edgecolor(fc)
            lp.set_facecolor('none')
 
    def axplot_flows(self, xdf_run, flows, t1, t2, field, ax=None, hue='flow',
                     palette='tab10', **kwargs):
        """Plot multiple flows vs time on the same axis, works for both
        xdf and truth_df. Returns the plot axis."""
        ax = ax if ax is not None else plt.gca()
        xdf_run = xdf_run[(xdf_run.flow.isin(flows))]
        df = xdf_run[(xdf_run.time >= t1) & (xdf_run.time <= t2)].copy()
        if field == 'owd':
            df['owd_mean'] = df.groupby('flow')['owd'].transform('mean')
            # df['owd'] = df['owd'] - df['owd_mean']
            df['owd'] = df.apply(lambda r: r.owd % 0.1 if r.owd > 0.12 else r.owd,
                                 axis=1)
        g = sns.lineplot(x='time', y=field, hue=hue, data=df, ax=ax,
                             palette=palette, **kwargs)
        g.legend(fontsize=6)
        return g
    
    def axplot_queues(self, qdf_run, qids, field, t1, t2, ax=None,  palette='tab10',
                      **kwargs):
        """Plot multiple queues vs time on the same axis. Returns the plot axis."""
        ax = ax if ax is not None else plt.gca()
        df = qdf_run[(qdf_run.time >= t1) & (qdf_run.time <= t2)]
        if qids is None:
            qids = df.qid.unique()
        elif type(qids[0]) != str:
            qids = [str(int(q)).zfill(4) for q in qids]
        df = df.loc[df.qid.isin(qids)]
        return sns.lineplot(x='time', y=field, hue='qid', data=df, ax=ax,
                            palette='tab10', **kwargs)

    def axplot_segment(self, xdf_run, flows, i, field, ax=None, **kwargs):
        """Plot a segment of a flow vs time on the same axis. Returns the plot axis."""
        ax = ax if ax is not None else plt.gca()
        times = sorted(xdf_run.time.unique())
        t1, t2 = times[i], times[i + 1]
        return self.axplot_flows(xdf_run, flows, t1, t2, field, ax, **kwargs)
    
    def axplot_segment_queues(self, qdf_run, qids, i, field, ax=None, **kwargs):
        """Plot a segment of a queue vs time on the same axis.
        Returns the plot axis."""
        ax = ax if ax is not None else plt.gca()
        times = sorted(qdf.time.unique())
        t1, t2 = times[i], times[i + 1]
        return self.axplot_queues(qdf_run, qids, field, t1, t2, ax, **kwargs)
    
    def axplot_latent_distances(self, distance_df, xtag='run', ax=None, **kwargs):
        """Plot latent distances vs time. xtag can group or run."""
        ax = ax if ax is not None else plt.gca()
        df = distance_df.copy()
        df = df.melt(id_vars=['run', 'grp', 'time', 'i'], var_name='type', value_name='distance')
        kwargs.update({'flierprops': {'marker': '.', 'markersize': 1.5}})
        g = sns.boxplot(x=xtag, y='distance', hue='type', data=df, ax=ax,
                           **kwargs)
        g.set_ylabel(MTC_NAME['distance'])
        self._set_miscellaneous(g)
        # print('[axplot accuracy]:', res_df[[x_field, hue]].drop_duplicates())
        n_hatch, n_grp = 2, df.grp.nunique()
        self._set_hatches(g, n_hatch, n_grp, plt_type='box')

    def plot_queues(self, qdf, run, t1, t2, qids=None):
        """Plot all queues vs time."""
        df = qdf[qdf.run == run]
        qids = qids if qids is not None else df.qid.unique()
        plt.figure(figsize=(12, 4))
        ax = self.axplot_queues(df, qids, 'packet_in_queue', t1, t2,
            linestyle='-.')
        ax.set(title=f'All queues in run {run}')
        self.show_or_save(f'queues_run{run}_t{t1}-{t2}.pdf')
    
    def plot_flows(self, xdf, run, t1, t2, fields=['owd', 'rtt', 'drop', 'cwnd'],
                   flows=None, n_col=2):
        """Plot all flows vs time, need flow group."""
        df = xdf[xdf.run == run]
        flows = flows if flows is not None else df.flow.unique()
        n_row = int(np.ceil(len(fields) / n_col))
        fig, ax = plt.subplots(n_row, n_col, figsize=(5 * n_col, 3 * n_row))
        for i, field in enumerate(fields):
            self.axplot_flows(df, flows, t1, t2, field, ax.flatten()[i])
            ax.flatten()[i].set(title=f'Field {field} in run {run}')
        self.show_or_save(f'flows_run{run}_t{t1}-{t2}.pdf')

    def plot_flows_by_queue(self, xdf, qid_df, qdf, run, t1, t2,
                            btnk_qids=None, fields=['owd', 'rtt', 'drop', 'cwnd']):
        """Plot flows and queue time series by given bottleneck queue/qid.
        
        For each row, plot the flows and queue time series with the
        same qid.
        """
        xdf_run, qdf_run = xdf[xdf.run == run], qdf[qdf.run == run]
        qid_to_flow = DataGetter.find_qid_to_flow(qid_df)[run]
        if btnk_qids:
            qid_to_flow = {qid: qid_to_flow[qid] for qid in btnk_qids}
        n_row, n_col = len(fields) + 1, len(qid_to_flow)
        fig, ax = plt.subplots(n_row, n_col, figsize=(4.5 * n_col, 3 * n_row))
        for i, (qid, flows) in enumerate(qid_to_flow.items()):
            self.axplot_queues(qdf_run, [qid], 'packet_in_queue', t1, t2,
                ax[0, i])
            ax[0, i].set(title=f'Queue {qid} in run {run}')
            for j, field in enumerate(fields):
                self.axplot_flows(xdf_run, flows, t1, t2, field, ax[j + 1, i])
                ax[j + 1, i].set(title=f'Field {field} in run {run}')
        self.show_or_save(f'flows_by_queue_run{run}_t{t1}-{t2}.pdf')

    # TODO: extract the axplot later?
    def plot_queue_vs_label(self, qid_df, qdf, run):
        qdf_run = qdf[qdf.run == run]
        n_row = qid_df.qid1.nunique() * qid_df.qid2.nunique()
        fig, ax = plt.subplots(n_row, 2, figsize=(8, 2 * n_row))
        i_row = 0
        for qid1 in qid_df.qid1.unique():
            for qid2 in qid_df.qid2.unique():
                df = DataGetter.get_double_queues(qdf_run, qid1, qid2)
                labels = sorted(df.label.unique())
                df['nlabel'] = df.apply(lambda r: labels.index(r.label), axis=1)
                sns.lineplot(x='time', y='q_win1', data=df, ax=ax[i_row, 0], label=qid1)
                sns.lineplot(x='time', y='q_win2', data=df, ax=ax[i_row, 0], label=qid2)
                sns.lineplot(x='time', y='nlabel', data=df, ax=ax[i_row, 1],
                    label='label').set(yticks=range(len(labels)), yticklabels=labels)
                i_row += 1
        self.show_or_save(f'queue_vs_label_run{run}.pdf')

    def axplot_precision_w_run(self, res_df, metric='rel_precision', x_field='run',
                               hue='tag', ax=None, **kwargs):
        """Plot line plot for precision.
        
        Note: lineplot doesn't directly work as boxplot starts from position 0, while
        lineplot starts from 1."""
        df = res_df.groupby([x_field, hue]).mean(numeric_only=True).reset_index()
        ax = ax or plt.gca()
        g = sns.pointplot(x=x_field, y=metric, data=res_df, hue=hue, ax=ax, 
                          errorbar=None, style=hue,
                          markers=['s', 'v', '^', 'o'],
                        #   linestyles=[':', '--', '-', '-.'],
                          scale=0.65,
                        #   lw=0.3,
                          linestyles=':',
                          **kwargs)
        # TODO: legend?
        self._set_miscellaneous(g)
        return g

    def axplot_time_point(self, res_df, x_field='flow', hue='tag', ax=None, **kwargs):
        """Use pointplot to plot time w/ x_field."""
        ax = ax or plt.gca()
        g = sns.pointplot(x=x_field, y='t_classify', data=res_df, hue=hue, ax=ax,
                        #   errorbar='sd',
                          ci='sd',
                          scale=0.55,
                          style=hue,
                          dodge=0.4,
                          markers=['s', 'v', '^', 'o'],
                          linestyles='--', **kwargs)
        self._set_miscellaneous(g)
        return g


    def axplot_accuracy_w_run(self, res_df, metric='f1', x_field='run', hue='tag',
                              ax=None, **kwargs):
        """Given res_df, plot metric (f1, precision, recall, t_classify) vs given
        x field (run, etc) w/ hue tag, i.e. a brief of method. If metric is not
        t_classify, boxplot is used, otherwise barplot is used.
        """
        ax = ax or plt.gca()
        if metric != 't_classify':
            f_sns_plot = sns.boxplot
            kwargs.update({'flierprops': {'marker': '.', 'markersize': 1.5}})
            plt_type = 'box'
        else:
            f_sns_plot = sns.barplot
            plt_type = 'bar'
        g = f_sns_plot(x=x_field, y=metric, data=res_df, hue=hue, ax=ax, **kwargs)
        self._set_miscellaneous(g)
        # print('[axplot accuracy]:', res_df[[x_field, hue]].drop_duplicates())
        n_hatch, n_grp = res_df[hue].nunique(), res_df[x_field].nunique()
        self._set_hatches(g, n_hatch, n_grp, plt_type=plt_type)
        return g

    def axplot_heatmap(self, df, row_field, col_field, metric, ax=None, **kwargs):
        """Plot heatmap of metric vs x_field and y_field.
        Default settings: annotate the avg value, no colorbar, use reverse color.
        """
        ax = ax or plt.gca()
        df = df.copy()
        # df[metric] = df[metric].apply(lambda x: f'{x:.2f}'.lstrip('0'))
        df = df.pivot(index=row_field, columns=col_field, values=metric)
        return sns.heatmap(data=df, annot=True, cbar=False,
                        cmap=sns.cm.rocket_r,
                        vmin=0.45, vmax=1.1, ax=ax, **kwargs)

    def plot_all_detection(self, res_df, metric='f1', fsize=None):
        fsize = fsize or (4, 2)
        plt.figure(figsize=fsize)
        g = self.axplot_accuracy_w_run(res_df, metric)
        g.set(title=f'{metric} for all runs')
        self.show_or_save(f'{metric}_vs_runs.pdf')
    
    def plot_all_detection_vs_metrics(self, res_df, metrics, fsize=None,
                                      rotation=0):
        fsize = fsize or (8, 2 * len(metrics) * self.height)
        fig, ax = plt.subplots(len(metrics), 1, figsize=fsize)
        for i, metric in enumerate(metrics):
            g = self.axplot_accuracy_w_run(res_df, metric, ax=ax[i])
            # ax[i].set(title=f'{metric} for all runs')
            self._set_legend(ax[i], keep=i == 0)
            g.set_xticklabels(g.get_xticklabels(), rotation=rotation)
            # TODO: below is temporary hack!
            # g.set(xticklabels=['1200 (original)', '1200 x 2', '1200 x 5'],
            #       xlabel='# flows x # runs for synthesis')
            g.set(xticklabels=['1000', '2000', '4000', '8000'],
                  xlabel='# flow#  #for synthesis')
            # g.set(xticklabels=['250 x 1', '250 x 2', '250 x 3', '250 x 4'],
                #   xlabel='# flows x # runs for synthesis')
        self.show_or_save(f'all_metrics_vs_runs.pdf')
    
    def plot_all_detection_by_field(self, res_df, x_field, xs, y_field='f1', fsize=(6, 3),
                             bbox_to_anchor=(1.4, 1)):
        """Plots f1 scores by a given field.

        Args:
            res_df (DataFrame): Original accuracy df
            x_field (str): Name of the x variable
            xs (Series like): Values of the x variable
            y_field (str, optional): Name of y variable. Defaults to 'f1'.
            fsize (tuple, optional): Size of the figure. Defaults to (8, 3).
        """
        df_plot = res_df.copy()
        plt.figure(figsize=fsize)
        g = self.axplot_accuracy_w_run(df_plot, y_field)
        g.set_xticks(range(len(xs)), xs)
        g.set_xlabel(x_field)
        # TODO: put a name here
        labels = ['DCW', 'rmcatSBD', 'dcSBD', 'Proposed-TBD']
        g.legend(loc='upper right', bbox_to_anchor=bbox_to_anchor)
        for t, l in zip(g.get_legend().texts, labels):
            t.set_text(l)
        self.show_or_save(f'{y_field}_vs_{x_field}.png')

    def plot_group_detection_by_field(self, res_df, metric, xlabels=None, xfield=None,
        n_run=None, run_boundary=None, hue='tag', fsize=(8, 3), bbox_to_anchor=(1.25, 1)):
        if metric == 't':
            metric = 't_classify'
            res_df = res_df[['run', hue, 't_classify']].drop_duplicates()
        if xlabels is not None:
            gres_df = group_df_in_runs(res_df, n_run=n_run, run_boundary=run_boundary)
            gres_df['run'] = gres_df['group']
            self.plot_all_detection_by_field(gres_df, xfield, xlabels,
                y_field=metric, fsize=fsize, bbox_to_anchor=bbox_to_anchor)
        elif type(metric) == list:
            self.plot_all_detection_vs_metrics(res_df, metric, fsize=fsize)
        else:
            assert type(metric) == str
            self.plot_all_detection(res_df, metric=metric, fsize=fsize)

    def _check_equal_nrun(self, res_df, expected_nrun):
        if res_df.run.nunique() == expected_nrun:
            return True
        res_runs = [r - res_df.run.min() for r in res_df.run.unique()]
        missing_runs = set(range(expected_nrun)) - set(res_runs)
        print(f'    Warning: res_df # runs: {res_df.run.nunique()} != total runs: {expected_nrun}')
        print(f'    Missing runs: {missing_runs}')
        return False

    # data dimension: 4 para btnk * 3 load * 2 run * < 8 flows
    # plot:  1 figure per para btnk: subplots of 3 load * 8 flows, all other aggregated 
    def plot_3d_scan(self, res_df, para_btnks=[4, 8, 12, 16], loads=[0.9, 0.95, 1.0],
                     n_run_per_load=2, metric='precision'):
        # highly customized, so no parameters
        expected_nrun = len(para_btnks) * len(loads) * n_run_per_load
        self._check_equal_nrun(res_df, expected_nrun)
        abs_runs = sorted(res_df.run.unique())
        for i, para_btnk in enumerate(para_btnks):
            fig, ax = plt.subplots(len(loads), 1, figsize=(9, 3 * len(loads)))
            for j, load in enumerate(loads):
                run0 = abs_runs[i * len(loads) * n_run_per_load + j * n_run_per_load]
                run1 = run0 + n_run_per_load
                df = res_df[(res_df.run >= run0) & (res_df.run < run1)]
                g = self.axplot_accuracy_w_run(df, metric, x_field='n_flow', ax=ax[j])
                g.set(title=f'n para btnk: {para_btnk}; load: {load}')
                self._set_legend(ax[j])
            self.show_or_save(f'3d_scan_para_btnk_{para_btnk}_{metric}.pdf')

    def plot_metrics_vs_btnk(self, res_df, metrics, hue_labels=None, fsize=None):
        """Plot metrics vs para_btnk for each metric. Mainly used for the scanning
        of the different ML seq_len."""
        fsize = fsize or (8, 3 * len(metrics))
        fig, ax = plt.subplots(len(metrics), 1, figsize=fsize)
        for i, metric in enumerate(metrics):
            self.axplot_accuracy_w_run(res_df, metric, x_field='para_btnk', ax=ax[i])
            self._set_legend(ax[i], keep=i == 0, labels=hue_labels)
        self.show_or_save(f'metrics_vs_btnk.pdf')

    def plot_var_scan(self, res_df, var, label, labels, n_run=4, fsize=None,
                      method_tags=None, use_twinx=False):
        """Visualize var scan using twin axis to plot boxplot of f1 and lineplot of
        precision to save space when precisions are naively close to 1 and deemphasize
        the small scale results."""
        if method_tags:
            res_df = res_df[res_df.tag.isin(method_tags)]
            print(f'  Filtering methods: {method_tags}')

        expected_nrun = len(labels) * n_run
        print(res_df.run.unique())
        if res_df.run.nunique() != expected_nrun:
            print(f'    Warning: res_df # runs: {res_df.run.nunique()} != total runs: {expected_nrun}')
            # return
        abs_runs = sorted(res_df.run.unique())
        gres_df = group_df_in_runs(res_df, n_run=n_run)
        gres_df['run'] = gres_df['group']
        # metrics = ['f1', 'precision', 'recall'] 
        # metrics = ['rel_f1', 'rel_precision', 'rel_recall']
        metrics = ['rel_precision', 'rel_f1']

        if not use_twinx:
            fsize = fsize or (4, self.height * len(metrics))
            fig, ax = plt.subplots(len(metrics), 1, figsize=fsize)
            # for output the data
            # for tag in gres_df.tag.unique():
            #     for grp in gres_df.run.unique():
            #         print('Tag:', tag, 'grp:', grp)
            #         print(gres_df.loc[(gres_df.tag == tag) & (gres_df.run == grp), metrics].median())
            for i, metric in enumerate(metrics):
                self.axplot_accuracy_w_run(gres_df, metric, x_field='run', ax=ax[i])
                ax[i].set(ylabel=MTC_NAME[metric])
                if i == len(metrics) - 1:
                    ax[i].set(xticks=range(len(labels)), xticklabels=labels)
                xlabel = label if i == len(metrics) - 1 else None
                ax[i].set_xlabel(xlabel)
                self._set_legend(ax[i], keep=i == 0)
        else:
            fsize = fsize or (4, self.height + 0.5)
            plt.figure(figsize=fsize)
            ax1 = plt.gca()
            g = self.axplot_accuracy_w_run(gres_df, 'rel_f1', x_field='run', ax=ax1)
            g.set(ylabel=MTC_NAME['rel_f1'])
                # , xticks=range(len(labels)), xticklabels=labels)
            self._set_legend(ax1, keep=True)
            ax2 = g.twinx()
            # ax1.get_shared_y_axes().join(ax1, ax2)
            # ax1.sharey(ax2)
            g2 = self.axplot_precision_w_run(gres_df, 'rel_precision', x_field='run',
                                             ax=ax2)
            g2.set(ylabel=MTC_NAME['rel_precision'])
            # ax1.set_xlabel(label)
            # ax1.set_xticks(range(len(labels)), labels)
            ax1.set_ylim(top=1.3)
            ax1.set(xlabel=label, xticks=range(len(labels)), xticklabels=labels,
                    yticks=[0.25, 0.50, 0.75, 1.0])
            ax2.set_ylim(top=1.05)
            ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            g2.legend().remove()
            self._combine_legends(ax1, ax2)
        self.show_or_save(f'{var}.pdf', tight_layout=False)

    def plot_segments(self, xdf, truth_df, run_to_idx,
                      btnk_qids, fields=['owd', 'rtt', 'drop', 'cwnd']):
        """Plot segments of flows and queues by given bottleneck queue/qid.
        run_to_idx is used to find the segment index of each run.
        TODO: use custom flow groups for each run to plot later
        """
        
        n_row = sum([len(indices) for indices in run_to_idx.values()])
        n_col = len(fields) + 1
        fig, ax = plt.subplots(n_row, n_col, figsize=(4.5 * n_col, 3 * n_row))
        
        i = 0
        for _, (qid, flows) in enumerate(qid_to_flow.items()):
            for run, indices in run_to_idx.items():
                xdf_run, truth_df_run = xdf[xdf.run == run], truth_df[truth_df.run == run]
                qid_to_flow = DataGetter.find_qid_to_flow(qid_df)[run]
                # TODO: want to use different color to group flows in 
                #     segment plot, so the grouping is clear
                #       => need to merge xdf and truth_df


        
        self.show_or_save(f'segments_run{run}_t{t1}-{t2}.pdf')

    def axplot_latent(self, y, y_hat, latent, hue='truth', ax=None, real_flows=None,
                      **kwargs):
        """Plot latent space of a specific segment w/ actual & predictedd
        labels.
        Use hue = 'truth' for labeling ground truth, and 'pred' for predictions.
        """
        assert real_flows is not None or hue != 'flow'
        ax = ax if ax is not None else plt.gca()
        mds = MDS(n_components=2)
        latent_2d = mds.fit_transform(np.array(latent))
        df = pd.DataFrame(latent_2d, columns=['x', 'y'])
        df['labels'] = [f'{y[j]}->({y_hat[j]})' for j in range(len(y))]
        df['truth'] = y
        df['pred'] = y_hat
        if real_flows is not None:
            df['flow'] = [f'{real_flows[j]}' for j in range(len(y))]
        assert hue in ['labels', 'truth', 'pred', 'flow']
        return sns.scatterplot(x='x', y='y', hue=hue, data=df, ax=ax,
                               legend='full', **kwargs)

    # TODO: test it w/ axplot_latent
    def plot_latent_space(self, predictions, n_col=2, indices=[]):
        """Given run and index of the segments, plot the latent space.

        Args:
            predictions (dict): dict {(run, idx): (y, y_hat, latent, clusters)}
            n_col (int, optional): number of subplot columns. Defaults to 2.
            indices (list, optional): list of indices to plot. Defaults to [].
        """
        n_row = int(np.ceil(len(predictions) / n_col))
        i = 0
        for ((run, idx), (y, y_hat, latent, _)) in predictions.items():
            if indices and idx not in indices:
                continue
            if i % n_col == 0:
                if i > 0:
                    plt.tight_layout()
                    plt.show()
                _, axs = plt.subplots(1, n_col, figsize=(n_col*3, 2.5))
            cur_ax = axs[i % n_col]
            i += 1
            self.axplot_latent(y, y_hat, latent, ax=cur_ax)
            cur_ax.set_title(f'run {run}, segment {idx}')
        self.show_or_save(f'latent_space.pdf')
    
    def plot_all_latents(self, xdf, all_predict_infos, keys,
                        ts, interval=1.5, fields=['owd', 'rtt', 'slr'],
                        methods=['dc_sbd', 'ml'], hue='truth', remove_legend=False,
                        xdf_query=None):
        """Plot all latent spaces of runs.
        
        Each figure in time by methods for a given key.
        
        Args:
            xdf (pd.DataFrame): the signal data
            all_predict_infos (dict): dict {(run, n_flow): {run: {method: predict_info }}
                predict_info: [ys, y_hats, latent_map]
            keys (int): [(run, n_flow)] to plot
            ts (list): the time intervals to plot
            methods (list, optional): the methods to compare. DCW won't
                collect the latents, and rmcat has the same latents as dcSBD.
            hue (str, optional): the field to use for hue. Choices include 'truth',
                'labels', 'pred'. Defaults to 'truth' for ground truth.
            xdf_query (str, optional): the query to filter xdf. Defaults to None.
        """
        abs_runs = sorted(xdf.run.unique())
        for run, n_flow in keys:
            abs_run = xdf.run.unique()[0] if xdf.run.nunique() == 1 else abs_runs[run]
            predict_infos = all_predict_infos[(run, n_flow)]['infos']
            flows = all_predict_infos[(run, n_flow)]['flows']
            fig, ax = plt.subplots(len(fields) + len(methods), len(ts),
                                   figsize=( len(ts) * 3.5, (len(methods) + 1) * 3.5 ))
            xdf_run = xdf[xdf.run == abs_run]
            if len(ts) == 1:
                ax = ax.reshape(-1, 1)
            if xdf_query:
                xdf_run = xdf_run.query(xdf_query)

            for i, t1 in enumerate(ts):
                t2 = t1 + interval
                for k, field in enumerate(fields):
                    g = self.axplot_flows(xdf_run, flows, t1, t2, field, ax=ax[k, i])
                    self._set_legend(g, keep=False)
                for j, method in enumerate(methods):
                    # print(predict_infos.keys())
                    predict_info = predict_infos[abs_run][method]
                    latent_map, cur_flows = predict_info[2], predict_info[3]        # {t: latent}
                    times = sorted(latent_map.keys())
                    idx = bisect.bisect_left(times, t1)
                    assert idx < len(times)
                    y, y_hat = predict_info[0][idx], predict_info[1][idx]
                    t = times[idx]
                    latent = latent_map[t]
                    g = self.axplot_latent(y, y_hat, latent, hue=hue, ax=ax[len(fields) + j, i])
                    # ax[i, len(fields) + j].legend(loc='right', bbox_to_anchor=(1.05, 1), fontsize=7)
                    self._set_legend(g, keep=False)
                    g.set_title(f'run {run}, n_flow {n_flow}, t {t:.2f}s, {method}', fontsize=6)
            self.show_or_save(f'latent_map_run{run}_nflow{n_flow}.pdf', tight_layout=True)

    def axplot_detection_vs_time(self, res_df, metric='f1', ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        assert res_df.run.nunique() == 1
        return sns.lineplot(x='time', y=metric, data=res_df, hue='tag', ax=ax, **kwargs)

    def plot_detection_vs_time(self, res_df, metric, fsize=(6, 3)):
        run = res_df.run.unique()[0]
        plt.figure(figsize=fsize)
        self.axplot_detection_vs_time(res_df, metric)
        self.show_or_save(f'detection_vs_time_run{run}.pdf')

    def plot_detections_vs_time(self, res_df, run, metrics, fsize=(10, 9)):
        data = res_df[res_df.run == run]
        n_row, n_col = len(metrics), 1
        fig, ax = plt.subplots(n_row, n_col, figsize=fsize)
        for i, metric in enumerate(metrics):
            self.axplot_detection_vs_time(data, metric=metric, ax=ax[i])
        self.show_or_save(f'detections_vs_time_run{run}.pdf')
    
    def plot_detection_by_metric_family(self, res_df, families):
        """Row: old, pair, relative; col: f1, precision, metric."""
        n_row, n_col = len(families), 3
        fig, ax = plt.subplots(n_row, n_col, figsize=(3 * n_col, 3 * n_row))
        metric_matrix = {
            'old': ['f1', 'precision', 'recall'],
            'pair': ['pair_f1', 'pair_precision', 'pair_recall'],
            'rel': ['rel_f1', 'rel_precision', 'rel_recall']
        }
        for i in range(n_row):
            for j in range(n_col):
                metric = metric_matrix[families[i]][j]
                g = self.axplot_accuracy_w_run(res_df, metric=metric, x_field='method',
                                           hue=None, ax=ax[i, j])
                g.set_xticklabels(g.get_xticklabels(), rotation=45)
        self.show_or_save(f'detection_by_metric_family.pdf')
    
    def plot_th_scan(self, res_df, metric, fsize=(6, 3)):
        plt.figure(figsize=fsize)
        self.axplot_accuracy_w_run(res_df, metric, 'th')
        self.show_or_save(f'th_scan_{metric}.pdf')
    
    def plot_grp_th_scan(self, res_df, runs):
        metrics = ['rel_f1', 'rel_precision', 'rel_recall'
                   'pair_f1', 'pair_precision', 'pair_recall']
        fig, ax = plt.subplots(2, 3, figsize=(9, 6))
        df = res_df[res_df.run >= runs[0] & res_df.run < runs[-1]]
        for i, metric in enumerate(metrics):
            self.axplot_accuracy_w_run(df, metric, 'th', ax=ax[i // 3, i % 3])
        self.show_or_save(f'grouped_th_scan_{runs[0]}-{runs[1]}.pdf')

    # def axplot_loss(self, tloss_df, metric, ax=None, **kwargs):
    #     if ax is None:
    #         ax = plt.gca()
    #     return sns.boxplot(x='time', y=metric, data=tloss_df, ax=ax, **kwargs)

    # functions below support data inspector
    def axplot_cluster_flow_range(self, seg_df, flow_label, ax, field='owd', **kwargs):
        """Plot the flow range plot, i.e avg flow + range of all flows, for
        given cluster in the given time. Line plot w/o hue gives us such plot.
        """
        seg_df = seg_df[seg_df.label == flow_label].copy()
        # TODO: debugging, remove hue later
        sns.lineplot(data=seg_df, x='time', y=field,
                     ci='sd', ax=ax, **kwargs)
        return ax

    def plot_separate_clusters(self, seg_df, labels, field):
        """Plot cluster flow range in separate subplots. This is the default
        plot we get using data inspector."""
        n_row = int(np.ceil(len(labels) / 2))
        fig, ax = plt.subplots(n_row, 2, figsize=(8, n_row * 3))
        if n_row == 1:
            ax = ax.reshape(1, 2)
        for i, label in enumerate(labels):
            self.axplot_cluster_flow_range(seg_df, label, ax[i // 2, i % 2], field=field) 
            ax[i // 2, i % 2].set_title(f'Cluster {label}')
        self.show_or_save('separate_clusters.pdf')

    def plot_repeated_clusters(self, seg_df, labels, field):
        """Plot the flow range of given clusters together to compare the inter
        cluster flow ranges."""
        ax = plt.gca()
        for flow_label in labels:
            self.axplot_cluster_flow_range(seg_df, flow_label, ax, field=field,
                                           label=flow_label)
        ax.legend()
        self.show_or_save('repeated_clusters.pdf')
    
    def plot_cluster_w_members(self, seg_df, labels, intra_cluster_corr, field,
                               n_member=3, ascending=True):
        fig, ax = plt.subplots(len(labels), 1, figsize=(10, 3.5 * len(labels)))
        # get the three farthest members for each cluster by default
        for i, label in enumerate(labels):
            self.axplot_cluster_flow_range(seg_df, label, ax[i], field=field)
            label_corr_df = intra_cluster_corr.query(f'label == {label} and flow2 == -1')
            members = label_corr_df.sort_values(by=f'{field}_corr', ascending=ascending)
            members = members.flow1.reset_index(drop=True)
            members = np.random.choice(members, min(n_member, len(members)),
                                       replace=False)
            # members = members.iloc[:n_member]
            label_seg_df = seg_df[seg_df.label == label]
            print(f' Cluster: {label}, draw {n_member} / {label_seg_df.flow.nunique()}')

            flow_label_df = label_seg_df[label_seg_df.flow.isin(members)]
            sns.lineplot(data=flow_label_df, x='time', y=field, hue='flow',
                         ax=ax[i], legend='brief') 
        self.show_or_save('cluster_w_far_members.pdf')
    
    def cut_legend(self, ax, n=None, loc='right'):
        """Given plot g, cut legend to only show the first n entries."""
        h, l = ax.get_legend_handles_labels()
        if n is not None:
            h, l = h[:n], l[:n]
        ax.legend(h, l, loc=loc, bbox_to_anchor=(1.2, 0.5), fontsize=5)
    
    def axplot_stats(self, stat_df, ax=None, **kwargs):
        """Plot stat df [run, flow, time, stat, value, isbtnk.]"""
        if ax is None:
            ax = plt.gca()
        g = sns.boxplot(x='stat', y='value', hue='isbtnk', data=stat_df,
                        ax=ax, **kwargs)
        g.set(xlabel='Stats of flow segments', ylabel='Value')
        return g
    
    def plot_stats(self, stat_df):
        field_list = [
            ['owd_std', 'rtt_std'],
            ['slr_avg', 'slr_std'],
            # ['cwnd_avg', 'cwnd_std']
        ]
        n_col = len(field_list)
        fig, axes = plt.subplots(1, n_col, figsize=(1.5 * n_col, 2))
        for ax, field in zip(axes, field_list):
            # showfliers = not 'cwnd' in field[0]
            showfliers = 'slr' in field[0]
            self.axplot_stats(stat_df[stat_df.stat.isin(field)], ax=ax,
                              showfliers=showfliers)
        self.show_or_save('stats.pdf')
    
    def plot_detect_time(self, df, labels=None, n_run=None, run_boundary=None,
                         xlabel=None, fsize=None):
        for col in ['run', 'flow', 't_detect']:
            assert col in df
        gres_df = group_df_in_runs(df, n_run=n_run, run_boundary=run_boundary)
        gres_df['run'] = gres_df['group']
        fsize = fsize or (4, self.single_height)
        plt.figure(figsize=fsize)
        g = sns.boxplot(data=gres_df, x='run', y='t_detect', hue='tag',
                        flierprops={'marker': '.', 'markersize': 1.5})
        g.set(xticks=range(len(labels)), xticklabels=labels,
              xlabel=xlabel or 'Run', ylabel='Detection Delay (s)')
        # self._set_miscellaneous(g)
        n_hatch, n_grp = gres_df.tag.nunique(), gres_df.run.nunique()
        self._set_hatches(g, n_hatch, n_grp)
        self._set_legend(g)
        self.show_or_save('detect_time.pdf', tight_layout=False)



class Plotter2d(DataPlotter):
    """Plotter2d: plotter for para btnk * n_flow 2d-scan.
    This class uses only detection data, and typical use cases include:
        1. (Multi-fig) Metric vs n_flow for metric of para-btnk;
        2. (3x1 fig) F1 (or others) boxplot vs n_flow for para-btnk;
        3. F1 (or others) heatmap vs para-btnk x n_flow.
    """
    def __init__(self, res_df, para_btnks=None, n_flows=None,
                 out_dir=None, do_show=True, tag=None, n_run=2,
                 condition=None) -> None:
        # define the member related to the data here, as these won't change
        # when we plot different figures.
        super().__init__(out_dir, do_show, tag)
        self.res_df = res_df
        # print('[Plotter2d]', self.res_df[['n_flow']].drop_duplicates())
        self.para_btnks = para_btnks
        self.n_flows = n_flows
        self.n_run = n_run
        expected_nrun = len(para_btnks) * len(n_flows) * n_run
        self._check_equal_nrun(res_df, expected_nrun)
        self.abs_runs = sorted(res_df.run.unique())
        self.n_run_per_btnk = len(self.n_flows) * self.n_run
        run_base = self.abs_runs[0]
        self._add_rel_run(self.res_df)
        self._update_para_btnk(para_btnks)
        self._update_nflow(n_flows)
        self._apply_condition(condition)

    def _update_para_btnk(self, para_btnks):
        if 'para_btnk' not in self.res_df.columns:
            f_get_btnk = lambda r: para_btnks[int(r.rel_run // self.n_run_per_btnk)]
            self.res_df['para_btnk'] = self.res_df.apply(f_get_btnk, axis=1)

    def _update_nflow(self, n_flows):
        # the simulation probably doesn't give exactly the same n_flow, affecting
        # heatmap plot, so we merge them
        f_get_nflow = lambda r: min(n_flows, key=lambda x: abs(x - r.n_flow))
        self.res_df['n_flow'] = self.res_df.apply(f_get_nflow, axis=1)
    
    def _apply_condition(self, condition):
        if condition is None:
            return
        if type(condition) == dict:
            for k, v in condition.items():
                self.res_df = self.res_df[self.res_df[k] == v]
        elif type(condition) == str:
            self.res_df = self.res_df.query(condition)
        print(f'  Applied condition: {condition}')
        print(self.res_df[['para_btnk', 'n_flow']].drop_duplicates())

    def filter_methods(self, method_tags=None):
        if method_tags:
            self.res_df = self.res_df[self.res_df.tag.isin(method_tags)]
            print(f'  Filtering methods: {method_tags}')

    def _add_rel_run(self, df):
        """Given a df, update the run field to be relative to the first run,
        and consecutive."""
        assert 'run' in df.columns
        abs_to_rel = {r: i for i, r in enumerate(sorted(df.run.unique()))}
        df['rel_run'] = df.run.map(abs_to_rel)

    def plot_metrics_vs_flow(self, metrics=['precision', 'recall', 'f1'], fsize=None,
                             yscale='linear', plt_type='box', ytick_list=None):
        """Given para_btnks & n_flows, plot metric vs n_flow for metric
        of each para_btnk."""
        height = self.height if len(metrics) > 1 else self.single_height
        fsize = fsize or (4, height * len(metrics))
        ytick_list = ytick_list or [[0.5, 0.75, 1.0]] * len(metrics)
        for i, para_btnk in enumerate(self.para_btnks):
            # for each para btnk, plot a metrics * 1 figure
            fig, ax = plt.subplots(len(metrics), 1, figsize=fsize)
            if len(metrics) == 1:
                ax = [ax]
            df = self.res_df[self.res_df.para_btnk == para_btnk]
            for j, metric in enumerate(metrics):
                g = self.axplot_accuracy_w_run(df, metric, x_field='n_flow', ax=ax[j])
                g.set(ylabel=MTC_NAME[metric])
                if metric == 't_classify':
                    plt_type = self._set_y_for_t_classify(g)
                else:
                    g.set(yticks=ytick_list[j])
                g.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                xlabel = 'Flow count' if j == len(metrics) - 1 else None
                g.set(xlabel=xlabel)
                self._set_legend(ax[j], keep=j == 0, plt_type=plt_type)
            # plt.title(f'n para btnk: {para_btnk}')
            metric_str = '_'.join(metrics)
            self.show_or_save(f'{metric_str}_vs_flow_btnk_p{para_btnk}.pdf',
                              tight_layout=False)

    def axplot_overhead_vs_x(self, ax, para_btnk=None, n_flow=None, **kwargs):
        """Plot overhead vs flow to support 1x2 layout outside.
        Set para_btnk to plot overhead vs n_flow, and vice versa."""
        if para_btnk:
            df = self.res_df[self.res_df.para_btnk == para_btnk]
            x_field, xlabel = 'n_flow', 'Flow count, 16 Bottlenecks'
        elif n_flow:
            df = self.res_df[self.res_df.n_flow == n_flow]
            x_field, xlabel = 'para_btnk', 'Bottleneck count'
        else:
            n_btnk = self.res_df.para_btnk.nunique()
            n_flow = self.res_df.n_flow.nunique()
            assert n_btnk == 1 or n_flow == 1, f'Error: n_btnk = {n_btnk}, n_flow = {n_flow}'
        g = self.axplot_time_point(df, x_field=x_field, ax=ax, **kwargs)
        # g.set(xlabel=xlabel)
        g.set_xlabel(xlabel, weight='bold', fontsize=8)
        return g

    def plot_metric_vs_btnk_and_flow(self, metric='rel_f1', fsize=None):
        """Given para_btnks & n_flows, plot metric vs para_btnk and n_flow."""
        fsize = fsize or (4, self.height * len(self.para_btnks))
        fig, ax = plt.subplots(len(self.para_btnks), 1, figsize=fsize)
        for i, para_btnk in enumerate(self.para_btnks):
            df = self.res_df[self.res_df.para_btnk == para_btnk]
            g = self.axplot_accuracy_w_run(df, metric, x_field='n_flow', ax=ax[i])
            if i != len(self.para_btnks) - 1:
                g.set(xlabel=None)
            # g.set(title=f'n para btnk: {para_btnk}')
            self.add_legend(ax[i], f'# btnk: {para_btnk}')
            self._set_legend(ax[i], keep=i == 0)
            xlabel = 'Flow count' if i == len(self.para_btnks) - 1 else None
            g.set(xlabel=xlabel, ylabel=MTC_NAME[metric])
        self.show_or_save(f'{metric}_vs_btnk_and_flow.pdf')

    def plot_metric_heatmap(self, metric='rel_f1', tag=None, fsize=None):
        """Plot metric heatmap of given method (tag) for para_btnk * n_flow."""

        def func(x, pos):
            return f"{x:.2f}".replace("0.", ".").replace("1.00", "")

        # annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)

        fsize = fsize or (0.4 * len(self.n_flows), 0.4 * len(self.para_btnks))
        res_df = self.res_df.copy()
        if tag is not None:
            res_df = res_df[res_df.tag == tag]
        res_df = res_df[['para_btnk', 'n_flow', metric]]\
                .groupby(['para_btnk', 'n_flow']).mean().reset_index()
        plt.figure(figsize=fsize)
        g = self.axplot_heatmap(res_df, 'para_btnk', 'n_flow', metric)
        g.set(xlabel='Flow count', ylabel='Bottleneck count')
        # plt.title(f'{metric} for {tag} heatmap')
        tag_str = tag[:tag.find('.') - 2] if '.' in tag else tag
        self.show_or_save(f'{metric}_{tag_str}_heatmap.pdf')

    def plot_metrics_vs_btnk(self, metrics, n_flow=None, fsize=None, plt_type='box'):
        """Plot metrics vs para_btnk for each metric."""
        height = self.height if len(metrics) > 1 else self.single_height
        fsize = fsize or (4, height * len(metrics))
        res_df = self.res_df
        if n_flow:
            res_df = res_df[res_df.n_flow == n_flow]
        fig, ax = plt.subplots(len(metrics), 1, figsize=fsize)
        if len(metrics) == 1:
            ax = [ax]
        for i, metric in enumerate(metrics):
            g = self.axplot_accuracy_w_run(res_df, metric, x_field='para_btnk', ax=ax[i])
            xlabel = None if i != len(metrics) - 1 else 'Bottleneck count'
            g.set(xlabel=xlabel, ylabel=MTC_NAME[metric])
            if metric == 't_classify':
                plt_type = self._set_y_for_t_classify(g)
            self._set_legend(ax[i], keep=i == 0, plt_type=plt_type)
        metric_str = '_'.join(metrics)
        self.show_or_save(f'{metric_str}_vs_btnk.pdf', tight_layout=False)

class Plotter3d(DataPlotter):
    def __init__(self, res_df, para_btnks=[4, 8, 12, 16], loads=[0.9, 0.95, 1.0],
                 out_dir=None, do_show=True, tag=None, n_run=2):
        super().__init__(out_dir, do_show, tag)
        self.res_df = res_df
        self.para_btnks = para_btnks
        self.loads = loads
        expected_nrun = len(para_btnks) * len(loads) * n_run
        self._check_equal_nrun(res_df, expected_nrun)
        self.abs_runs = sorted(res_df.run.unique())
        run_base = self.abs_runs[0]
        self.n_run_per_btnk = len(loads) * n_run
        self.res_df['para_btnk'] = 4 * ((res_df.run - run_base) //
                                        self.n_run_per_btnk + 1).astype(int)
        f_get_load = lambda r: loads[int((r.run - run_base) % self.n_run_per_btnk // n_run)]
        self.res_df['load'] = self.res_df.apply(f_get_load, axis=1)

    # data dimension: 4 para btnk * 3 load * 2 run * < 8 flows
    # plot:  1 figure per para btnk: subplots of 3 load * 8 flows, all other aggregated 
    def plot_3d_scan(self, metric='precision'):
        # highly customized, so no parameters
        for i, para_btnk in enumerate(self.para_btnks):
            fig, ax = plt.subplots(len(self.loads), 1, figsize=(6, 2 * len(self.loads)))
            for j, load in enumerate(self.loads):
                df = self.res_df[(self.res_df.para_btnk == para_btnk) &
                                 (self.res_df.load == load)]
                g = self.axplot_accuracy_w_run(df, metric, x_field='n_flow', ax=ax[j])
                g.set(title=f'n para btnk: {para_btnk}; load: {load}')
                self._set_legend(ax[j])
            self.show_or_save(f'3d_scan_para_btnk_{para_btnk}_{metric}.pdf')

    def plot_metrics_vs_btnk(self, metrics, hue_labels=None, fsize=None,
                             use_hmap=True):
        """Plot metrics vs para_btnk for each metric. Mainly used for the scanning
        of the different ML seq_len."""
        df = self.res_df.copy()
        df = df[df.n_flow == 64]
        hue_labels = ['0.35s', '0.7s', '1.0s', '1.5s']
        intervals = [0.35, 0.7, 1.0, 1.5]
        hue_order = ['ml_s70_012_0.2', 'ml_s140_012_0.2', 'ml_s200_012_0.2',
                     'ml_owd-rtt-slr_0.2']

        if not use_hmap:
            fsize = fsize or (8, 2 * len(metrics))
            fig, ax = plt.subplots(len(metrics), 1, figsize=fsize)
            for i, metric in enumerate(metrics):
                # print('df', df, 'metric', metric, 'ax', ax[i])
                self.axplot_accuracy_w_run(df, metric, x_field='para_btnk', ax=ax[i],
                                        hue_order=hue_order, hue='tag')
                self._set_legend(ax[i], keep=i == 0, title='interval', labels=hue_labels,
                                bbox_to_anchor=(1.15, 0.5))
        else:
            # for each metric, plot a heatmap, each has a size ~ 1/3 column size
            # no subplot here as title needs to be set in latex
            fsize = fsize or (0.4 * len(self.para_btnks), 0.5 * len(hue_labels))
            tag_to_interval = {k: v for k, v in zip(hue_order, intervals)}
            df['interval'] = df.tag.map(tag_to_interval)
            for metric in metrics:
                plt.figure(figsize=fsize)
                df1 = df[['para_btnk', 'interval', metric]].groupby(
                    ['para_btnk', 'interval']).mean().reset_index()
                g = self.axplot_heatmap(df1, 'interval', 'para_btnk', metric) 
                g.set(xlabel='Number of\nbottlenecks', ylabel='Detecting interval (s)')
                self.show_or_save(f'{metric}_vs_btnk_itv_hmap.pdf')

