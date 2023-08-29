import argparse
import glob
import os
import pandas as pd
import pickle
import numpy as np
import torch

from siamese.cluster import cluster
from data_visualizer import DataVisualizer
from siamese.pipeline import compare_segment
from siamese.pipeline import set_df_metadata, update_non_btnk_flows


class ClusterUpdater:
    """
    Requirements
    1. For all folders, read predict_info, get new res_df by rerunning
    clustering only, and output res_df to new folders.

    Vertical:
    1. Input: detect folder name
    2. Read predict info & detection (data visualizer)
    3. Redo the clustering based on latents
    4. Update the old res_df by new ML with new tag
    5. Output new res_df to new folder

    Horizontal:
    1. Scale up to more folders
    2. Support different clustering methods -> args: n_voter, th, r_ratio
    3. Output directories

    Note that last labels and flow labels should be kept for the next time.
    """
    def __init__(self, cluster_args=[(9, 0.4)], method_tag='ml_02_0.2', detect_root=None,
                 bin_classifier_path='/home/sapphire/neuralforecast/my_src/bin_cls/random_forest_93_92.pkl'):
        self.cluster_args = cluster_args    # [(n_voter, th), ...]
        self.method_tag = method_tag
        self.detect_root = detect_root or '/home/sapphire/hdd/detector_pipeline/detection'
        assert bin_classifier_path is not None
        with open(bin_classifier_path, 'rb') as f:
            self.classifier = pickle.load(f)
    
    def process_dir(self, src_folder, dst_folder):
        """Given src dir, read the predict_info, process, output to dst_dir.
        This alters self.vis."""
        print(f'Processing {src_folder} -> {dst_folder} under {self.detect_root}...')
        dst_dir = os.path.join(self.detect_root, dst_folder)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir)
        self.vis = DataVisualizer(src_folder, out_dir=dst_dir)
        detect_folder = os.path.join(self.detect_root, src_folder)
        runs = self.vis._get_runs_to_read(cache_folder=detect_folder) 
        for run in runs:
            self.process_run(detect_folder, run, dst_dir)
        print(f'Finished processing {src_folder} -> {dst_folder}, run {runs[0]} ~ {runs[-1]}.')

    def process_run(self, detect_folder, run, dst_dir):
        """Process each run & output data to dst_dir."""
        # read
        runs = [run, run + 1]
        self.vis.read_predict_infos(runs=runs, detect_folder=detect_folder)
        self.vis.read_detection(runs=runs, folders=[detect_folder])
        ys, y_hats, latent_map, flows = self.vis.decode_predict_infos(self.vis.all_predict_infos,
                                                                      self.method_tag)
        # cluster
        run_abs = self.vis.res_df.run.unique()[0]
        res_df = None
        for i, t in enumerate(latent_map.keys()):
            flow_labels = {f: -1 for f in flows[i]}
            latent = torch.tensor(latent_map[t])
            y = ys[i]
            for n_voter, th in self.cluster_args:
                y_hat1, flow_labels1, _ = cluster(latent, max_iter=20, n_voter=n_voter,
                                               th=th, flow_labels=flow_labels)
                accuracy_metrics, cols = compare_segment(y, y_hat1)
                tmp_df = pd.DataFrame([[t] + accuracy_metrics],
                                    columns=['time'] + cols)
                set_df_metadata(tmp_df, run_abs, 'ml', f'ml_v{n_voter}_th{th:.2f}')
                res_df = tmp_df if res_df is None else pd.concat([res_df, tmp_df],
                                                                 ignore_index=True)

        # update & output
        if 'n_flow' in self.vis.res_df.columns:
            res_df['n_flow'] = self.vis.res_df.n_flow.unique()[0]
        # print('new col', res_df.columns)
        # print('old col', self.vis.res_df.columns)
        res_df['t_classify'] = None
        res_df['pair_f1'] = None
        assert set(res_df.columns) == set(self.vis.res_df.columns)
        res_df = pd.concat([self.vis.res_df, res_df], ignore_index=True)
        dst_path = os.path.join(dst_dir, f'res_df_{run}-{run + 1}.csv')
        assert not os.path.exists(dst_path), f'{dst_path} already exists!'
        res_df.to_csv(dst_path, index=False)
        print(f'Output to {dst_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_folder', '-s', type=str, nargs='+', required=True,
                        help='Source folder names.')
    parser.add_argument('--dst_suffix', '-d', type=str, default='_v9',
                        help='Suffix of destination folder names.')
    parser.add_argument('--detect_root', '-r', type=str,
                        default='/home/sapphire/hdd/detector_pipeline/detection',
                        help='Root folder of detection results.')
    parser.add_argument('--method_tag', '-m', type=str, default='ml_02_0.2',
                        help='Method tag of detection results.')
    args = parser.parse_args()

    cu = ClusterUpdater(detect_root=args.detect_root, method_tag=args.method_tag)
    if '*' in args.src_folder[0]:
        args.src_folder = glob.glob(os.path.join(args.detect_root, args.src_folder[0]))
    for src_folder in args.src_folder:
        dst_folder = src_folder + args.dst_suffix
        cu.process_dir(src_folder, dst_folder)

