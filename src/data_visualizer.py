import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from bisect import bisect, bisect_left

from siamese.plotter import DataPlotter, Plotter2d, Plotter3d
from siamese.preprocess import measure_time
from siamese.cluster import cluster
from siamese.pipeline import relative_precision_recall_f1
from visualizer import DataVisualizer, AdvPlotter, filter_eb_res_df
from visualizer import adv_vis_sim_metrics, adv_vis_real_metrics, adv_vis_sim_btnk, \
    adv_vis_sim_tdetect, adv_vis_nonbtnk, adv_vis_large_ovaerhead, \
    adv_vis_dash_final, adv_vis_dash_ms, adv_vis_syn


g_data_root = None
g_cache_root = None
g_detect_root = None

# DEFAULT_ML_TAG = 'ml_v9_th0.40'
# DEFAULT_ML_TAG = 'ml_02_0.2'
DEFAULT_ML_TAG = 'ml_v9_0.4'


# TODO: merge into plotters
def plot_triplet(sample_df, i_triplet):
    x1 = sample_df[(sample_df.i_triplet == i_triplet)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    sns.lineplot(x='t_in_sample', y='owd', hue='flow', data=x1, ax=ax[0])
    sns.lineplot(x='t_in_sample', y='rtt', hue='flow', data=x1, ax=ax[1])
    plt.show()
    plt.close()


def test_plot_queue_flow_detection():
    folder = 'results_cbtest-large-flow_Dec-28-18:05:5486' 
    runs = [0, 1]
    visualizer = DataVisualizer(folder, out_dir='test_fig')
    visualizer.read_raw(runs)
    visualizer.read_detection(runs)

    run = visualizer.get_absolute_run(0)
    t1, t2 = 15, 35
    visualizer.plot_queues(run, t1, t2)
    qid_df = visualizer.prep.qid_df
    run_abs = qid_df.run.unique()[0]
    qids = qid_df[(qid_df.run == run_abs)].qid1.unique()
    qids = np.hstack([qids, qid_df[(qid_df.run == run_abs)].qid2.unique()])
    visualizer.plot_flows_for_queue(run, qids[0], 10, t1, t2)
    visualizer.plot_flows_for_queue(run, qids[-1], 10, t1, t2)
    visualizer.plot_queue_vs_label(run)

    xlabels = ['2', '3', '4']
    xfield = 'Number of right middle links'
    visualizer.plot_group_detection_by_field('f1', xlabels, xfield, n_run=4)
    visualizer.plot_group_detection_by_field('t', xlabels, xfield, n_run=4)


def visualize_detections_vs_time(folder, out_dir, tag, run, metrics):
    """Plots detection accuracies along time mainly for debugging."""
    assert run is not None
    visualizer = DataVisualizer(folder, out_dir=out_dir, tag=tag,
                                data_root=g_data_root, cache_root=g_cache_root)
    visualizer.read_detection()
    if run not in visualizer.res_df.run.unique():
        run = visualizer.res_df.run.unique()[run]
    visualizer.plotter.plot_detections_vs_time(visualizer.res_df, run, metrics)


def visualize_detections(folder, out_dir, tag, run_boundary=None,
                         metrics=['rel_f1', 'rel_precision', 'rel_recall', 't']):
    """This function is provided for customized label, field, & run boundary."""
    visualizer = DataVisualizer(folder, out_dir=out_dir, tag=tag,
                                data_root=g_data_root, cache_root=g_cache_root)
    visualizer.read_detection()

    xlabels, xfield = None, None
    if 'large-flow' in folder:
        xlabels = [
            '2+2, 1.0', '2+2, 1.5', '2+2, 2.0',
            '2+6, 1.0', '2+6, 1.5', '2+6, 2.0'
        ]
        xfield = 'Left+right bottlenecks, user ratio'
    elif 'para-btnk' in folder:
        xlabels = ['4', '8', '12', '16']
        xfield = 'Number of right bottlenecks'
    
    fsize = (4, 4.5)
    visualizer.plot_group_detection_by_field(metrics[:-1], xlabels, xfield,
                                            run_boundary=run_boundary, fsize=fsize)
    visualizer.plot_group_detection_by_field('t', xlabels, xfield,
                                            run_boundary=run_boundary)
    

def visualize_detections_by_metric(out_dir, detect_folders, tag,
                                   families=['rel']):
    """Visualize detections by metric family, i.e. old, pair,
    relative. Each row contains one family, and columns are f1, precesion,
    recall. Used for the testset from training data, as here the data
    groups are not needed, and metric should be focused on."""
    visualizer = DataVisualizer(out_dir=out_dir, tag=tag,
                                data_root=g_data_root, cache_root=g_cache_root)
    visualizer.read_detection(folders=detect_folders)
    visualizer.plotter.plot_detection_by_metric_family(visualizer.res_df, families)


def visualize_th_scan(folder, out_dir, tag, metric):
    visualizer = DataVisualizer(folder, out_dir=out_dir, tag=tag,
                                data_root=g_data_root, cache_root=g_cache_root)
    visualizer.read_detection()
    visualizer.plotter.plot_th_scan(visualizer.res_df, metric)


def visualize_grp_th_scan(folder, out_dir, tag, runs):
    visualizer = DataVisualizer(folder, out_dir=out_dir, tag=tag,
                                data_root=g_data_root, cache_root=g_cache_root)
    visualizer.read_detection()
    visualizer.plotter.plot_grp_th_scan(visualizer.res_df, runs)


def visualize_flow_latent(folder, out_dir, cache_folder, detect_folder, tag, run, ts,
                          methods, xdf_query):
    # keys & indices should be customized
    vis = DataVisualizer(folder, out_dir=out_dir, tag=tag,
                         data_root=g_data_root, cache_root=g_cache_root)
    vis.read_predict_infos(runs=[run, run + 1], detect_folder=detect_folder)
    # print('keys', vis.all_predict_infos.keys())
    keys = list(vis.all_predict_infos.keys())[:2]
    # vis.read_raw_across_runs(runs_to_read, no_cached_data=False)
    vis.read_raw_from_cache(cache_folder, [run, run + 1])
    vis.plotter.plot_all_latents(vis.xdf, vis.all_predict_infos, keys, ts,
                                 methods=methods, xdf_query=xdf_query)


def visualize_flow_latent_official(folder, out_dir, cache_folder, detect_folder, tag,
                                   run, ts, methods, xdf_query=None,
                                   fields=['owd'], fsize=None, interval=1.5):
    ml_tag = [m for m in methods if 'ml' in m][0]
    pvis = PredictInfoDebugger(folder, out_dir, tag, run, detect_folder)
    pvis.read_raw_from_cache(cache_folder, [run, run + 1])

    fsize = fsize or (4, 1.0)
    fsize = (6, len(ts) * 1.6)       # manual hack
    fig, ax = plt.subplots(len(ts), len(fields) + 2, figsize=fsize)
    ax = ax.reshape(len(ts), -1)
    for i, t in enumerate(ts):
        # 1. get the labels for flows
        pvis.decode(ml_tag)
        y, y_hat, latent, flows = pvis.get_snapshot(t)
        flow_to_y = {f: y[i] for i, f in enumerate(flows)}
        assert pvis.xdf.run.nunique() == 1
        xdf_plot = pvis.xdf[pvis.xdf.flow.isin(flows)].copy()
        if xdf_query:
            xdf_plot = xdf_plot.query(xdf_query)
        xdf_plot['label'] = xdf_plot.flow.map(flow_to_y).astype(int)

        # 2. plot the flows
        # kwargs = {'hue': 'label', 'units': 'flow', 'legend': 'brief',
        #                     'style': 'flow', 'estimator': None}
        t2 = t + interval
        for k, field in enumerate(fields):
            # print('flatent flows', flows, 'xdf flows', xdf_plot.flow.unique())
            g = pvis.plotter.axplot_flows(xdf_plot, flows, t, t2, field, ax=ax[i, k],
                                          hue='label', palette=None)
            g.set(xlabel='Time (s)', ylabel=None)
            g.legend(loc='center left', fontsize=5, markerscale=0.3, title=field.upper(),
                     title_fontsize=5, handlelength=0.2)
            g.tick_params(axis='both', which='major', labelsize=5)
            for tx in g.get_legend().texts:
                tx.set_text('G' + tx.get_text())

        for k, method in enumerate(methods):
            pvis.decode(method)
            y, y_hat, latent, flows = pvis.get_snapshot(t)
            y = [int(yi) for yi in y]
            g = pvis.plotter.axplot_latent(y, y_hat, latent, hue='truth',
                                           ax=ax[i, len(fields) + k], s=4**2)
            g.legend(loc='best', fontsize=5, markerscale=0.3, handlelength=0.2)
            for tx in g.get_legend().texts:
                tx.set_text('G' + tx.get_text())
            g.tick_params(axis='both', which='major', labelsize=5)
            xlabel = 'FlowBot Latents' if 'ml' in method else 'SBD Latents'
            g.set(xlabel=xlabel, ylabel=None)
        pvis.plotter.show_or_save(f'flatent_run{int(run)}_t{int(t)}.pdf', tight_layout=False,
                                  wspace=0.4)


def visualize_raw_flow(folder, out_dir, tag, runs=None, ts=[10, 19],
                       fields=['owd', 'rtt', 'slr']):
    """Visualize raw flow only from folder to check the simulated data."""
    vis = DataVisualizer('sim', out_dir=out_dir, tag=tag,
                         data_root=f'/home/sapphire/hdd/detector_pipeline/train_data/{folder}',
                         cache_root=g_cache_root)
    vis.read_raw_across_runs(runs, read_queue=True, read_raw_only=True,
                             no_cached_data=True)
    fig, ax = plt.subplots(len(runs), 1 + len(fields),
                           figsize=(3 * (1 + len(fields)), 3 * len(runs)))
    for i, run in enumerate(runs):
        labels_list = [None]
        run_abs = vis.xdf.run.min() - min(runs) + run
        print(f' Plotting run {run} queue')
        vis.dataplot_queue_for_label(run_abs, ts[0], ts[1], labels_list, [ax[i, 0]])
        flows = vis.xdf[(vis.xdf.run == run_abs)].flow.unique()
        assert len(flows) > 0, f'No flow for run {run_abs}'
        flows = np.random.choice(flows, 30, replace=False)
        print(f'            flows ... ')
        for j, field in enumerate(fields):
            g = vis.plotter.axplot_flows(vis.xdf[vis.xdf.run == run_abs], flows, ts[0], ts[1], field, ax[i, j + 1])
            g.legend().remove()
    vis.plotter.show_or_save('raw_flow.pdf')


# deprecate by PredictInfoDebugger
# def compare_latent_inference(folder, out_dir, detect_folder, tag, run, t, method_tag):
#     """Given the latents, cluster them and compare w/ existing y / y_hats.
#     """
#     # TODO: 1. cluster the latents; 2. recall w/ y, compared w/ original recall
#     vis = DataVisualizer(folder, out_dir=out_dir, tag=tag,
#                          data_root=g_data_root, cache_root=g_cache_root)
#     vis.read_predict_infos(runs=[run, run + 1], detect_folder=detect_folder)
#     ys, y_hats, latent_map, flows = vis.decode_predict_infos(vis.all_predict_infos,
#                                                              method_tag)

#     # 1. cluster the latents
#     assert t in latent_map
#     ti = list(latent_map.keys()).index(t)
#     latent = latent_map[t]
#     y, y_hat0 = ys[ti], y_hats[ti]
#     flow_labels = {f: -1 for f in flows[ti]}
#     y_hat1, flow_labels, _ = cluster(latent, max_iter=20, n_voter=9, th=0.4,
#                                      flow_labels=flow_labels)
#     assert len(y) == len(y_hat0) == len(y_hat1) == len(flow_labels)
#     print('y\n  ', y)
#     print('y_hat0 (from predict_info)\n  ', y_hat0)
#     print('y_hat1 (from latent clustering, no bin classification)\n  ', y_hat1)
#     print('flow_labels', flow_labels)

#     # 2. recall w/ y, compared w/ original recall
#     precision0, recall0, f1_0 = relative_precision_recall_f1(y, y_hat0)
#     precision1, recall1, f1_1 = relative_precision_recall_f1(y, y_hat1)
#     print('\n                  precision    recall  f1')
#     print(f' y vs y_hat 0:  {precision0:.3f}, {recall0:.3f}, {f1_0:.3f}')
#     print(f' y vs y_hat 1:  {precision1:.3f}, {recall1:.3f}, {f1_1:.3f}')
#     print()

#     # 3. calculate the latents distances
#     dists = []
#     latent = np.array(latent)
#     for i in range(len(latent) - 1):
#         for j in range(i + 1, len(latent)):
#             a, b = latent[i], latent[j]
#             dist = np.linalg.norm(a - b)
#             dists.append([i, j, dist])
#     dist_df = pd.DataFrame(dists, columns=['i', 'j', 'dist'])
#     print(dist_df.describe())

#     # 4. sample SBD latents and print
#     ys, y_hats, latent_map, flows = vis.decode_predict_infos(vis.all_predict_infos,
#                                                              'dc_sbd')
#     ti = bisect(list(latent_map.keys()), t)
#     t1 = list(latent_map.keys())[ti]
#     # ti = list(latent_map.keys()).index(t)
#     y = ys[ti]
#     latent = latent_map[t1]
#     vis.sample_sbd_vectors(y, latent)


class PredictInfoDebugger(DataVisualizer):
    """Load and debug for predict infos, e.g. latent, ys, y_hats.
    It focuses on a single run of the predict infos, and usually
    a single time step to do the clustering analysis.

    Design: 1) get_snapshot() and data at each t is the core of all APIs;
        2) reuse cluster & calculate_latent_distances parts for ML and dcSBD.
        3) Ensure an easy interface for steps.
    
    TODO: usage of get_nearest_intra_inter_cluster_distances should be combined
    with the plotter.

    Caller:
        ndf = None
        for run in range(run0, run1):
            pdbg = PredictInfoDebugger(folder, out_dir, tag, run, detect_folder)
            df = pdbg.get_nearest_intra_inter_cluster_distances(t)
            ndf = df if ndf is None else pd.concat([ndf, df])
        
        then use a plotter to plot ndf
            e.g. plot: dist vs. group w/ hue=intra/inter-cluster)
                left/right for two chosen scenarios

    """
    def __init__(self, folder, out_dir, tag, run, detect_folder):
        super().__init__(folder, out_dir=out_dir, tag=tag,
                         data_root=g_data_root, cache_root=g_cache_root)
        self.read_predict_infos(runs=[run, run + 1], detect_folder=detect_folder)
        self.run = run

    def run_all(self, t, method_tag, steps=['cluster', 'calc_dist', 'sample_sbd',
                                            'sbd_dist']):
        """Run all the possible functionalities for a given method_tag."""
        self.decode(method_tag)
        if 'cluster' in steps:
            self.cluster_latent(t)
        if 'calc_dist' in steps:
            self.calculate_latent_distances(t)
        if method_tag == 'dc_sbd':
            if 'sample_sbd' in steps:
                self.sample_sbd_vectors(t)
            if 'sbd_dist' in steps:
                self.print_sbd_latent_distribution(t)

    def decode(self, method_tag):
        self.ys, self.y_hats, self.latent_map, self.flows = \
            self.decode_predict_infos(self.all_predict_infos, method_tag)
        self.ts = list(self.latent_map.keys())

    def get_snapshot(self, t):
        ti = bisect_left(list(self.latent_map.keys()), t)
        # print(f'Getting snapshot at t={t}, ti={ti}')
        # print(f'keys', list(self.latent_map.keys()))
        t1 = list(self.latent_map.keys())[ti]
        return self.ys[ti], self.y_hats[ti], self.latent_map[t1], self.flows[ti]

    def cluster_latent(self, t, max_iter=50, n_voter=9, th=0.4, do_print=True):
        """Replay clustering for debugging here."""
        y, y_hat0, latent, flows = self.get_snapshot(t)
        latent = np.array(latent)
        flow_labels = {f: -1 for f in flows}
        y_hat1, flow_labels, _ = cluster(latent, max_iter=max_iter, n_voter=n_voter, th=th,
                                         flow_labels=flow_labels)
        assert len(y) == len(y_hat0) == len(y_hat1) == len(flow_labels)
        if do_print:
            print('y\n  ', y)
            print('y_hat0 (from predict_info)\n  ', y_hat0)
            print('y_hat1 (from pure clustering)\n  ', y_hat1)
            # print('flow_labels', flow_labels)

        precision0, recall0, f1_0 = relative_precision_recall_f1(y, y_hat0)
        precision1, recall1, f1_1 = relative_precision_recall_f1(y, y_hat1)
        print('\n                  precision    recall  f1')
        print(f' y vs y_hat 0:  {precision0:.3f}, {recall0:.3f}, {f1_0:.3f}')
        print(f' y vs y_hat 1:  {precision1:.3f}, {recall1:.3f}, {f1_1:.3f}')
        print()
        return y, y_hat0, y_hat1, flow_labels

    @measure_time()
    def calculate_latent_distances(self, t, do_print=True):
        _, _, latent, _ = self.get_snapshot(t)
        dists = []
        dist_dict = {}
        latent = np.array(latent)
        for i in range(len(latent) - 1):
            for j in range(i + 1, len(latent)):
                a, b = latent[i], latent[j]
                dist = np.linalg.norm(a - b)
                dists.append([i, j, dist])
                dist_dict[(i, j)] = dist
        dist_df = pd.DataFrame(dists, columns=['i', 'j', 'dist'])
        if do_print:
            print(dist_df.describe())
        return dist_df, dist_dict


    def _compute_centroid_distances(self, latent, y):
        """Compute the distances between centroids of different clusters."""
        centroids = {}
        for i in range(len(y)):
            if y[i] not in centroids:
                centroids[y[i]] = []
            centroids[y[i]].append(i)
        for label, flows in centroids.items():
            centroid = np.mean([latent[i] for i in flows], axis=0)
            centroids[label] = centroid
        centroid_dists = {}
        closest_cluster = {}
        min_dist = {}
        for i in range(len(centroids) - 1):
            for j in range(i + 1, len(centroids)):
                a, b = centroids[i], centroids[j]
                dist = np.linalg.norm(a - b)
                centroid_dists[(i, j)] = dist
                for k in [i, j]:
                    if k not in min_dist or dist < min_dist[k]:
                        min_dist[k] = dist
                        closest_cluster[k] = j if k == i else i
        return centroids, closest_cluster

    # @measure_time()
    def _get_nearest_neighbor_distance(self, i, cluster_no, dist_dict, y,
                                      top_k=9):
        """Get the distance between flow i and its nearest neighbor of the same
        cluster or different clusters."""
        if type(cluster_no) == bool:
            if cluster_no:
                candidates = [j for j, yj in enumerate(y) if yj == y[i] and j != i]
            else:
                candidates = [j for j, yj in enumerate(y) if yj != y[i]]
        else:
            assert type(cluster_no) == int
            candidates = [j for j, yj in enumerate(y) if yj == cluster_no and j != i]
        k_neighbors, neighbors = [], []
        k_max, dist_max = -1, 1e10
        for j in candidates:
            # find the smallest top_k distances, i.e., replace the largest one
            # and update dist_max and k_max
            dj = dist_dict[(min(i,j), max(i,j))]
            # dj = dist_df[(dist_df['i'] == min(i,j)) & (dist_df['j'] == max(i,j))]['dist'].values
            # assert len(dj) == 1, f'Error: {j} not found in dist_df!'
            if len(k_neighbors) < top_k:
                k_neighbors.append(dj)
                neighbors.append(j)
                k_max = np.argmax(k_neighbors)
                dist_max = k_neighbors[k_max]
            elif dj < dist_max:
                k_neighbors[k_max] = dj
                neighbors[k_max] = j
                k_max = np.argmax(k_neighbors)
                dist_max = k_neighbors[k_max]
        return np.mean(k_neighbors), neighbors

    @measure_time()
    def get_nearest_intra_inter_cluster_distances(self, t, mode, top_k=9):
        """Given the snapshot, calculate the df of the distances between
        flow i and its closest intra-cluster and inter-cluster neighbors.
        mode can be 'sc' for same cluster or None for default.
        """
        print(f'  Calculating intra/inter-cluster distances: '
              f'mode={mode}, run={self.run}, t={t}')
        y, _, latent, _ = self.get_snapshot(t)
        dist_df, dist_dict = self.calculate_latent_distances(t, do_print=False)
        if mode == 'sc':
            centroids, closest_cluster = self._compute_centroid_distances(latent, y)
        distances = []      # [i, intra_dist, inter_dist]
        for i in range(len(latent)):
            intra_dist, _ = self._get_nearest_neighbor_distance(i,
                    True, dist_dict, y, top_k)
            if mode == 'sc':
                cluster = closest_cluster[y[i]]
                inter_dist, _ = self._get_nearest_neighbor_distance(i,
                    cluster, dist_dict, y, top_k)
            else:
                inter_dist, _ = self._get_nearest_neighbor_distance(i,
                    False, dist_dict, y, top_k)
            distances.append([i, intra_dist, inter_dist])
        df = pd.DataFrame(distances, columns=['i', 'intra_dist', 'inter_dist'])
        df['time'] = t
        df['run'] = self.run
        return df

    def sample_sbd_vectors(self, t, N=5):
        """Sample N flows' SBD vector for each cluster."""
        y, _, latent, _ = self.get_snapshot(t)
        label_to_flow = {label: [i for i, l in enumerate(y) if l == label]
                         for label in set(y)}
        label_to_latent = {}
        for label, flows in label_to_flow.items():
            samples = np.random.choice(flows, N, replace=False)
            print(f'  Label {label}:')
            for sample in samples:
                # assert len(latent[sample]) == 4
                latent_str = ' '.join([f'{x:.3f}' for x in latent[sample]])
                print(f'    Flow {sample}: [{latent_str}]')
            label_to_latent[label] = [latent[sample] for sample in samples]
        return label_to_latent
    
    def print_sbd_latent_distribution(self, t):
        y, _, latent, _ = self.get_snapshot(t)
        sbd_cols = ['skew_est', 'var_est', 'freq_est', 'pkt_loss']
        latent_df = pd.DataFrame(latent, columns=sbd_cols)
        latent_df['label'] = y
        desc_df = latent_df.groupby('label').describe()
        # pd.options.display.float_format = '{:.2e}'.format
        for label in desc_df.index:
            tmp_df = desc_df.loc[label].reset_index()
            print(f'Label {label}:')
            tmp_df = tmp_df[tmp_df.level_1.isin(['mean', 'std'])]
            tmp_df = tmp_df.pivot(index='level_1', columns='level_0', values=label)
            print(tmp_df[sbd_cols])
        return latent_df


def visualize_3d_scan(folder, out_dir, tag, metric, method=None):
    vis = DataVisualizer(folder, out_dir=out_dir, tag=tag,
                         data_root=g_data_root, cache_root=g_cache_root)
    vis.read_detection()
    if method:
        vis.res_df = vis.res_df[vis.res_df.method == method]
    plotter = Plotter3d(vis.res_df, out_dir=out_dir, do_show=out_dir==None, tag=tag)
    plotter.plot_3d_scan(metric=metric)


def visualize_seq_len_comparison(folder, out_dir, tag, metrics, method=None, use_hmap=True):
    vis = DataVisualizer(folder, out_dir=out_dir, tag=tag,
                         data_root=g_data_root, cache_root=g_cache_root)
    vis.read_detection()
    if method:
        vis.res_df = vis.res_df[vis.res_df.method == method]
    res_df = vis.res_df.copy()
    plotter = Plotter3d(vis.res_df, out_dir=out_dir, do_show=out_dir==None, tag=tag)
    plotter.plot_metrics_vs_btnk(metrics, use_hmap=use_hmap)


# here begins the visualizers for paper

def visualize_vars(folder, out_dir, tag, mode):
    """Figure: 3x1, relative precision / recall / F1 vs time lags w/ methods."""
    vis = DataVisualizer(folder, out_dir=out_dir, tag=tag,
                         data_root=g_data_root, cache_root=g_cache_root)
    vis.read_detection()
    ml_tags = [tag for tag in vis.res_df.tag.unique() if 'ml' in tag]
    ml_tag = ml_tags[0] if len(ml_tags) == 1 else DEFAULT_ML_TAG
    method_tags = ['dcw_0.8', 'rmcat_sbd', 'dc_sbd', ml_tag]
    if mode == 'path-lag':
        label = 'Path Lag (ms)'
        # lags = ['10', '30', '50', '70', '90', '110']
        # lags = ['50', '100', '150', '200', '250', '300']
        lags = ['40', '80', '120', '160', '200']
        vis.plotter.plot_var_scan(vis.res_df, 'path-lag', label, lags, n_run=4,
                                  method_tags=method_tags)
    elif mode == 'cross-load':
        label = 'Ratio of Cross Traffic to User Traffic'
        # loads = ['40', '80', '120', '160']      # cross bw
        loads = ['0.3', '0.6', '0.9', '1.2', '1.5']      # cross bw
        vis.plotter.plot_var_scan(vis.res_df, 'cross-load', label, loads, n_run=4,
                                  method_tags=method_tags)
    elif mode == 'overall-load':
        # label = 'Overall link bandwidth (Mbps)'
        # loads = ['130', '150', '170', '190']    # total link bw
        label = 'Ratio of Link Capacity to Total Traffic Rate'
        method_tags = ['dcw_0.8', 'rmcat_sbd', 'dc_sbd', ml_tag,
                       'neg_naive', 'pos_naive']
        loads = ['0.4', '0.6', '0.8', '1.0', '1.2']

        # inverse the runs in res_df
        # label = 'Congestion level' 
        # loads = ['0.8', '1.0', '1.3', '1.7', '2.5']
        # runs = sorted(vis.res_df.run.unique())
        # vis.res_df.run = vis.res_df.run.apply(lambda x: runs[-1] - x)

        vis.plotter.plot_var_scan(vis.res_df, 'overall-load', label, loads, n_run=4,
                                  method_tags=method_tags)
    elif mode == 'real-load':
        label = 'Number of cross traffic flows'
        loads = ['5', '15', '20', '25', '35', '45']
        vis.plotter.plot_var_scan(vis.res_df, 'real-load', label, loads, n_run=15)


MODE_TO_BTNK_FLOW = {
    'l': {'n_btnks': [1, 2, 3, 4],
        #   'n_flows': [32, 64, 128, 256, 512],
          'n_flows': [50, 100, 150, 200, 250],
          'n_btnk': None, 'n_flow': 150},
    # 'r': {'n_btnks': [4, 8, 12, 16],
    #       'n_flows': [100, 200, 300, 400, 500],
    #       'n_btnk': None, 'n_flow': 300},
    'r': {'n_btnks': [4, 8, 12, 16],
          'n_flows': [50, 100, 200, 500, 1000],
          'n_btnk': None, 'n_flow': 200},
    'eb': {'n_btnks': [4, 8, 12, 16], 'n_flows': [30],
           'n_btnk': None, 'n_flow': 30},
    'ebv': {'n_btnks': [4, 8, 12, 16], 'n_flows': [30],
            'n_btnk': None, 'n_flow': 30},
    'ef': {'n_btnks': [16], 'n_flows': [320, 400, 480, 560],
           'n_btnk': 16, 'n_flow': None},
}

# TODO: currently written for left-btnk 2d scan, generalize it later
def visualize_2d_metrics(folder, out_dir, tag, mode, metrics, method_tag,
                         label_mode, n_run, n_flows0, condition=None):
    vis = DataVisualizer(folder, out_dir=out_dir, tag=tag,
                         data_root=g_data_root, cache_root=g_cache_root)
    vis.read_detection()
    btnk_flow = MODE_TO_BTNK_FLOW[label_mode]
    para_btnks, n_flows = btnk_flow['n_btnks'], btnk_flow['n_flows']
    if 'e' in label_mode:       # real test
        assert n_run == 15
    n_flows = n_flows0 or n_flows

    ml_tags = [tag for tag in vis.res_df.tag.unique() if 'ml' in tag]
    ml_tag = method_tag or (ml_tags[0] if len(ml_tags) == 1 else DEFAULT_ML_TAG)
    assert ml_tag in ml_tags
    method_tags = ['dcw_0.8', 'rmcat_sbd', 'dc_sbd', ml_tag]

    if label_mode == 'eb':
        vis.res_df = filter_eb_res_df(vis.res_df)

    plotter = Plotter2d(vis.res_df, para_btnks, n_flows, out_dir=out_dir,
                        do_show=out_dir==None, tag=tag, n_run=n_run, condition=condition)
    if 'mtc' in mode or 'all' in mode:
        assert type(metrics) == list
        # if metrics == ['t_classify']:
            # TODO: add DCW & use log scale for y-axis
        plotter.filter_methods(method_tags)
        ytick_list = None
        if label_mode == 'l':
            ytick_list = [[0.5, 0.75, 1.0], [0.25, 0.5, 0.75, 1.0], [0.25, 0.5, 0.75, 1.0]]
        elif label_mode == 'r':
            ytick_list = [[0.5, 0.75, 1.0], [0.5, 0.75, 1.0], [0.25, 0.5, 0.75, 1.0]]
        plotter.plot_metrics_vs_flow(metrics, ytick_list=ytick_list)
    if 'hmap' in mode or 'all' in mode:
        if method_tag is None:
            print('Warning: method_tag is None, meaning all methods are mixed!')
        for metric in metrics:
            plotter.plot_metric_heatmap(metric, method_tag)
    if mode[-2:] == '2d' or 'all' in mode:
        plotter.filter_methods(method_tags)
        for metric in metrics:
            plotter.plot_metric_vs_btnk_and_flow(metric)
    if 'btnk' in mode:
        plotter.filter_methods(method_tags)
        n_flow = btnk_flow['n_flow']
        plotter.plot_metrics_vs_btnk(metrics, n_flow=n_flow)
    print('  End visualization.')


def visualize_detect_time(folder, out_dir, detect_folder, tag,
                          n_run=None, mode=None, method_tag=None):
    vis = DataVisualizer(folder, out_dir=out_dir, tag=tag,
                         data_root=g_data_root, cache_root=g_cache_root)
    vis.calculate_detect_time(detect_folder=detect_folder, rewrite='rewrite' in mode)
    df = vis.t_detect_df
    ml_tag = method_tag or DEFAULT_ML_TAG
    assert 'ml' in ml_tag
    method_tags = ['dcw_0.8', 'rmcat_sbd', 'dc_sbd', ml_tag]
    df = df[df.tag.isin(method_tags)].copy()
    # intrinsic offset of detection, as the detection is done after the interval
    t_offset = {'dcw_0.8': 0.75, 'rmcat_sbd': 0.175, 'dc_sbd': 0.175, ml_tag: 0.75}
    df['t_detect'] = df.apply(lambda r: r.t_detect + t_offset[r.tag], axis=1)
    xlabel = 'Flow Count'
    if mode is None or mode == 't-detect':
        labels, xlabel = None, None
    elif 'r2d' in mode:
        labels = MODE_TO_BTNK_FLOW['r']['n_flows']
        df = df[df.run > df.run.max() - n_run * len(labels)]
    elif 'l2d' in mode:
        labels = MODE_TO_BTNK_FLOW['l']['n_flows']
        df = df[df.run > df.run.max() - n_run * len(labels)]
    elif 'eb2d' in mode:
        xlabel, labels = 'Bottleneck Count', MODE_TO_BTNK_FLOW['eb']['n_btnks']
    elif 'ef2d' in mode:
        labels = MODE_TO_BTNK_FLOW['ef']['n_flows']
    elif 'ebv2d' in mode:
        xlabel, labels = 'Bottleneck Count', MODE_TO_BTNK_FLOW['ebv']['n_btnks']
    else:
        raise ValueError(f'Unknown mode {mode}!')
    n_run = 15 if 'e' in mode[:mode.find('2d')] else n_run
    vis.plotter.plot_detect_time(df, labels=labels, n_run=n_run, xlabel=xlabel)


def visualize_overhead(folders, out_dir, tags, lmode, n_run=None,
                       n_flow=None, para_btnk=None, method_tag=None, fsize=None):
    vis = [None, None]
    assert len(folders) == 2
    ml_tag = method_tag or DEFAULT_ML_TAG
    assert 'ml' in ml_tag
    method_tags = ['dcw_0.8', 'rmcat_sbd', 'dc_sbd', ml_tag]
    plotter = []
    fsize = fsize or (4, 0.85)
    fig, ax = plt.subplots(1, 2, figsize=fsize, sharey=True)
    # TODO: lmode support are not complete yet, e.g. eb should be 4 groups, efv not added
    if not lmode:
        label_modes = ['l', 'r']
    elif 'v' not in lmode:
        label_modes = ['eb', 'ef']
    else:
        assert 'v' in lmode
        label_modes = ['ebv', 'efv']

    for i, label_mode in enumerate(label_modes):
        vis[i] = DataVisualizer(folders[i], out_dir=out_dir, tag=tags[i],
                            data_root=g_data_root, cache_root=g_cache_root)
        vis[i].read_detection()
        # if label_mode == 'eb':
        #     vis[i].res_df = filter_eb_res_df(vis[i].res_df)
        btnk_flow = MODE_TO_BTNK_FLOW[label_mode]
        para_btnks, n_flows = btnk_flow['n_btnks'], btnk_flow['n_flows']
        para_btnk, n_flow = btnk_flow['n_btnk'], btnk_flow['n_flow']
        if label_mode == 'eb' and vis[i].res_df.run.nunique() / n_run > len(para_btnks) \
            and len(para_btnks) > 1:
            assert vis[i].res_df.run.nunique() == 2 * len(para_btnks) * n_run
            vis[i].res_df = filter_eb_res_df(vis[i].res_df)
    
        # print(vis[i].res_df[['tag', 'run', 't_classify']].drop_duplicates().head(20))
        plotter.append(Plotter2d(vis[i].res_df, para_btnks, n_flows, out_dir=out_dir,
                            do_show=out_dir==None, tag=tags[i], n_run=n_run))
        plotter[i].filter_methods(method_tags)
        plotter[i].axplot_overhead_vs_x(ax[i], para_btnk, n_flow)
        ylabel = r'$\frac{\mathbf{\mathrm{Overhead}}}{\mathbf{\mathrm{Interval}}}$' if i == 0 else None
        ax[i].set_ylabel(ylabel, weight='bold', fontsize=12)

        # TODO: no ticks now...
        assert False, 'No ticks now!'

        plotter[i]._set_y_for_t_classify(ax[i], keep_ticks=i == 0)
        ax[i].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # plotter[i].add_legend(ax[i], tags[i], fontsize=7, loc='center right', bbox_to_anchor=(1.0, 0.65))
        plotter[i]._set_legend(ax[i], keep=i == 0, loc='lower center', bbox_to_anchor=(1.0, 1.01),
                            plt_type='point')
        # plotter[i]._set_miscellaneous(ax[i])
    
    for i in range(2):
        ax[i].tick_params(axis='y', labelsize=7, which='major', length=2)
        ax[i].tick_params(axis='y', labelsize=7, which='minor', length=0)
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=0, fontsize=7)
        # ax[i].set_yticklabels(ax[i].get_xticklabels(), rotation=0, fontsize=7)
        # ax[i].tick_params(axis='both', labelsize=7, which='minor', length=0)

    vstr = f'f{n_flow}' if n_flow else f'b{para_btnk}'
    plotter[0].show_or_save(f'overhead_{vstr}.pdf', tight_layout=False, merge_axis='x')


def get_latent_distances(folder, run, out_dir, tag, n_run, dist_tag,
                         ml_tag=DEFAULT_ML_TAG, mode='sc'):
    distance_df = None
    pdbg = PredictInfoDebugger(folder, out_dir, tag, run, folder)
    pdbg.decode(ml_tag)
    prefix = 'neighbor_dist_sc' if mode == 'sc' else 'neighbor_dist'
    fname = f'{prefix}_{dist_tag}_{run}-{run+1}.csv' if dist_tag else \
            f'{prefix}_{run}-{run+1}.csv'
    csv = os.path.join(folder, fname)
    if os.path.exists(csv):
        print(f' - Distance df loaded from {csv}')
        return pd.read_csv(csv, index_col=False)

    for t in pdbg.ts:
        df = pdbg.get_nearest_intra_inter_cluster_distances(t, mode)
        distance_df = df if distance_df is None else \
            pd.concat([distance_df, df], ignore_index=True)
    distance_df['grp'] = distance_df.run // n_run
    distance_df.to_csv(csv, index=False)
    print(f' - Distance df dumped to {csv}')
    return distance_df


# too slow:
#   1) pdbg: do calculation for each run, extend to interface, parallelize in cmd
#   2) dump the result for each run in csv
def visualize_latent_distances(folders, out_dir, tag, n_run, fsize=None,
                               dist_tag=None, ml_tag=DEFAULT_ML_TAG):
    """Visualize latent distances in 2x1 plots for two folders."""
    plotter = DataPlotter(out_dir, do_show=False, tag=tag)
    xlabel = ['Flow Count (iperf3)', 'Flow Count (Video)']
    xticklabels = [
        ['320', '400', '480', '560'],
        ['600', '750', '900', '1050', '1200']
    ]
    fsize = fsize or (4, plotter.height)
    fig, ax = plt.subplots(1, len(folders), figsize=fsize, sharey=True)
    for i, folder in enumerate(folders):
        distance_df = None
        dv = DataVisualizer(folder, out_dir=out_dir, tag=tag,
                            data_root=g_data_root, cache_root=g_cache_root)
        runs = dv._get_runs_to_read(cache_folder=folder)
        for run in runs:
            df = get_latent_distances(folder, run, out_dir, tag, n_run, dist_tag,
                                      ml_tag)
            distance_df = df if distance_df is None else \
                    pd.concat([distance_df, df], ignore_index=True)
        distance_df['grp'] = distance_df.run // n_run
        print(distance_df)
        plotter.axplot_latent_distances(distance_df, xtag='grp', ax=ax[i])
        plotter._set_legend(ax[i], keep=i == 0, bbox_to_anchor=(1.0, 1.01),
                            loc='lower center', ncol=2)
        if i == 1:
            ax[i].set(ylabel=None, yticks=[0, 0.5, 1.0])
        ax[i].set(xlabel=xlabel[i])
        ax[i].set(ylim=(0, 1.2))
        ax[i].set(xticklabels=xticklabels[i])
    plotter.show_or_save(f'latent_distance_{dist_tag}.pdf', tight_layout=False, whspace=(0,0))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--test', action='store_true',
                     help='Run the test function.')
    grp.add_argument('--folder', '-f', type=str, nargs='+',
                     help='The folders of the detection results.') 
    parser.add_argument('--data_root', '-dr', type=str, default=None,
                     help='The root directory of the data.')
    parser.add_argument('--cache_root', '-cr', type=str, default=None,
                     help='The root directory of the cache.')
    parser.add_argument('--cache_folder', '-cf', type=str, default=None,
                     help='The directory of the cache.')
    parser.add_argument('--detect_root', '-dtr', type=str, default=None,
                     help='The root directory of the detections')
    parser.add_argument('--out_dir', '-o', type=str, default='figures',
                        help='The output directory for the figures.')
    parser.add_argument('--tag', '-t', type=str, default='',
                        help='The tag for the figures.')
    parser.add_argument('--tags', '-tg', type=str, nargs='+', default=None,
                        help='The tags for the figures.')
    parser.add_argument('--mode', '-m', type=str, required=True,
                        choices=['detect', '3d-scan', '3d-scan-ml', '3d-scan-seq',
                                 'latent', 'latent-c', 'th-scan', 't-detect',
                                 'detect-mtc', 'path-lag', 'cross-load',
                                 'overall-load', 'detect-t', 'real-load',
                                 'raw',
                                 'l2d-mtc', 'l2d', 'l2d-hmap', 'l2d-all', 'l2d-btnk',
                                 'r2d-mtc', 'r2d', 'r2d-hmap', 'r2d-all', 'r2d-btnk',
                                 'r2d-td', 'l2d-td', 'r2d-td-rewrite', 'l2d-td-rewrite',
                                 'eb2d-mtc', 'eb2d', 'eb2d-btnk', 'eb2d-td',
                                 'ef2d-mtc', 'ef2d', 'ef2d-btnk', 'ef2d-td',
                                 'ebv2d-btnk',
                                 'ev2d-overhead', '2d-overhead', 'e2d-overhead',
                                 'adv-sim', 'adv-real', 'adv-sim-btnk', 'adv-sim-tdetect',
                                 'adv-nonbtnk', 'adv-large-overhead', 'adv-dash-final',
                                 'adv-dash-ms', 'adv-syn',
                                 'latent-dist', 'latent-dc'
                                ],
                        help='The mode of the visualization.'
                            'detect: detection results by run (mainly debug);\n'
                            '3d-scan: para-btnk, load, flow num scan for 3d-final dataset only;\n'
                            '3d-scan-ml: only used for 3d-scan w/ seq scan;\n'
                            '3d-scan-seq: specifically for seq scan;\n'
                            'th-scan: th scan for 3d-final dataset w/ th col in res_df;\n'
                            'path-lag, cross-load, overall-load: var scan w/ tick labels for paper\n'
                            'real-load: load scan of real traffic;\n'
                            'raw: raw ns-3 time series;\n'
                            'detect-t: detections vs time (mainly debug);\n'
                            't-detect: this for detecting time plot;\n'
                            'r2d-td/l2d-td: t-detect for 2d results (the most btnks);\n'
                            '2d-mtc, 2d, 2d-hmap: 2d metrics figures, l for left, r for right;\n'
                            '2d-btnk: 2d metrics vs btnk figures;\n'
                            'adv-sim: adv vis for sim btnk and flow;\n'
                            'adv-real: adv vis for real btnk and flow;\n'
                            'adv-dash-final: adv vis for dash-final with 16/32 btnsk;\n'
                            'adv-dash-ms: adv vis for ms comparison with video btns scan;\n'
                            'adv-sim-btnk: adv vis for sim btnk;\n'
                            'adv-sim-tdetect: adv vis for sim t-detect;\n'
                            'adv-nonbtnk: adv vis for scan in non-btnk scenario;\n'
                            'adv-large-overhead: adv vis for large overhead;\n'
                            'adv-syn: adv vis for synthesis experiments;\n'
                            'latent-dist: latent distance figures;\n'
                            'latent-dc: latent distance computation;\n')
    parser.add_argument('--metric', '-me', type=str, nargs='+',
                        # default=['f1', 'precision', 'recall',
                        # default=['pair_f1', 'pair_precision', 'pair_recall',
                        default=['rel_precision', 'rel_recall', 'rel_f1'],
                        choices=['f1', 'precision', 'recall', 'pair_f1',
                                 'pair_precision', 'pair_recall', 't_classify',
                                 'rel_precision', 'rel_recall', 'rel_f1'],
                        help='The metric for the detection.')
    parser.add_argument('--run_boundary', '-b', type=int, nargs='+', default=None,
                        help='The run boundary for the grouping, alternative of n_run.')
    parser.add_argument('--run', '-r', type=int, default=None,
                        help='Run No. for plots like visualize along time.')
    parser.add_argument('--runs', '-rs', type=int, nargs='+', default=None,
                        help='Run No. for debug plots.')
    parser.add_argument('--ts', '-ts', type=float, nargs='+', default=[],
                        help='Timestamps for plots like latents.')
    parser.add_argument('--n_run', '-n', type=int, default=4,
                        help='Number of runs for plots.')
    parser.add_argument('--n_flows', '-nf', type=int, nargs='+', default=None,
                        help='Number of flows for plots.')
    parser.add_argument('--method_tag', '-mt', type=str, default=DEFAULT_ML_TAG,
                        help='The method tag for the 2d figures.')
    parser.add_argument('--xdf_query', '-xq', type=str, default=None,
                        help='The query for the xdf, used in latent mode')
    parser.add_argument('--n_flow', '-nfl', type=int, default=None,
                        help='The number of flows for the t figures.')
    parser.add_argument('--n_btnk', '-nbt', type=int, default=None,
                        help='The number of bottleneck for the t figures.')
    parser.add_argument('--condition', '-co', type=str, default=None,
                        help='The filter condition for the res_df.')
    parser.add_argument('--dist_tag', '-dtag', type=str, default='dft',
                        help='The tag for the distance df.')
    args = parser.parse_args()
    args.folder = list(map(lambda x: os.path.basename(os.path.normpath(x)),
                           args.folder))
    g_data_root = args.data_root
    g_cache_root = args.cache_root
    g_detect_root = args.detect_root        # polish later
    detect_folders = args.folder
    print(args.folder)
    if len(args.folder) == 1:
        args.folder = args.folder[0]
    if args.tag:
        args.out_dir = os.path.join(args.out_dir, args.tag)
    if args.test:
        test_plot_queue_flow_detection()

    # adv vis modes first
    elif args.mode == 'adv-sim':
        adv_vis_sim_metrics(args.folder, args.out_dir, args.tag,
                            g_data_root, g_cache_root, args.metric)
    elif args.mode == 'adv-real':
        # for both iperf and video
        adv_vis_real_metrics(args.folder, args.out_dir, args.tag,
                             g_data_root, g_cache_root, args.metric)
    elif args.mode == 'adv-dash-ms':
        # final hacked version of dash figure
        adv_vis_dash_ms(args.folder, args.out_dir, args.tag,
                        g_data_root, g_cache_root, args.metric)
    elif args.mode == 'adv-dash-final':
        # final hacked version of dash figure
        adv_vis_dash_final(args.folder, args.out_dir, args.tag,
                           g_data_root, g_cache_root, args.metric)
    elif args.mode == 'adv-sim-btnk':
        adv_vis_sim_btnk(args.folder, args.out_dir, args.tag, g_data_root, g_cache_root)
    elif args.mode == 'adv-sim-tdetect':
        adv_vis_sim_tdetect(args.folder, args.out_dir, args.tag, g_data_root, g_cache_root)
    elif args.mode == 'adv-nonbtnk':
        adv_vis_nonbtnk(args.folder, args.out_dir, args.tag, g_data_root, g_cache_root)
    elif args.mode == 'adv-large-overhead':
        adv_vis_large_ovaerhead(args.folder[0], args.out_dir, args.tag, g_data_root, g_cache_root)
    elif args.mode == 'adv-syn':
        adv_vis_syn(args.folder, args.out_dir, args.tag, g_data_root, g_cache_root)

    elif args.mode == 'detect':
        visualize_detections(args.folder, args.out_dir,
                             args.tag, args.run_boundary)
    elif args.mode == 'detect-mtc':
        # TODO: broken as folder is not required for visualizer
        visualize_detections_by_metric(args.out_dir, detect_folders, args.tag)
    elif args.mode == '3d-scan-seq':
        visualize_seq_len_comparison(args.folder, args.out_dir, args.tag, args.metric,
                                     method='ml')
    elif '3d-scan' in args.mode:
        method = 'ml' if args.mode == '3d-scan-ml' else None
        for metric in args.metric:
            visualize_3d_scan(args.folder, args.out_dir, args.tag, metric,
                              method=method)
    elif 'raw' in args.mode:
        visualize_raw_flow(args.folder, args.out_dir, args.tag, args.runs, args.ts)
    elif 'latent' in args.mode:
        if args.mode != 'latent-dist':
            detect_folder = os.path.join(g_detect_root, args.folder)
        else:
            folders = [os.path.join(g_detect_root, folder) for folder in args.folder]
        if args.mode == 'latent':
            # visualize_flow_latent(args.folder, args.out_dir, args.cache_folder,
            #                       detect_folder, args.tag, args.run, args.ts,
            #                       ['dc_sbd', args.method_tag], args.xdf_query)
            visualize_flow_latent_official(args.folder, args.out_dir, args.cache_folder,
                                  detect_folder, args.tag, args.run, args.ts,
                                  ['dc_sbd', args.method_tag], args.xdf_query)
        elif args.mode == 'latent-c':
            for t in args.ts:
                # compare_latent_inference(args.folder, args.out_dir, detect_folder,
                #                         args.tag, args.run, t, args.method_tag)
                pvis = PredictInfoDebugger(args.folder, args.out_dir, args.tag,
                                           args.run, detect_folder)
                pvis.run_all(t, args.method_tag)
        elif args.mode == 'latent-dist':
            visualize_latent_distances(folders, args.out_dir, args.tag, args.n_run,
                                       dist_tag=args.dist_tag)
        elif args.mode == 'latent-dc':
            get_latent_distances(detect_folder, args.run, args.out_dir, args.tag,
                                 args.n_run, args.dist_tag)
    elif args.mode == 'th-scan':
        for metric in args.metric:
            visualize_th_scan(args.folder, args.out_dir, args.tag, metric)
    elif args.mode == 'grp-th-scan':
        for run in [0, 6, 12, 18]:
            visualize_grp_th_scan(args.folder, args.out_dir, args.tag, [run, run + 6])
    elif args.mode in ['path-lag', 'cross-load', 'overall-load', 'real-load']:
        visualize_vars(args.folder, args.out_dir, args.tag, args.mode)
    elif args.mode == 'detect-t':
        visualize_detections_vs_time(args.folder, args.out_dir,
                                     args.tag, args.run, args.metric)
    elif args.mode == 't-detect' or '-td' in args.mode:
        detect_folder = os.path.join(g_detect_root, args.folder)
        visualize_detect_time(args.folder, args.out_dir, detect_folder, args.tag,
                              args.n_run, args.mode, args.method_tag)
    elif 'overhead' in args.mode:
        label_mode = args.mode[:args.mode.find('2')]
        visualize_overhead(args.folder, args.out_dir, args.tags, label_mode, args.n_run,
                           args.n_flow, args.n_btnk, args.method_tag)
    elif '2d' in args.mode:
        label_mode = args.mode[:args.mode.find('2')]
        visualize_2d_metrics(args.folder, args.out_dir, args.tag, args.mode,
                            args.metric, args.method_tag, label_mode, args.n_run,
                            args.n_flows, args.condition)
   