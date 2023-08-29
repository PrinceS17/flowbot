import glob
import os
import subprocess as sp
import matplotlib
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
import numpy as np
import pandas as pd
from collections import namedtuple
from copy import deepcopy
from typing import List, Tuple, Dict, Union, Optional

from siamese.preprocess import DataPreprocessor, DataModifier, DataGetter
from siamese.plotter import DataPlotter, Plotter2d, Plotter3d
from detector import get_detection_delay, set_df_metadata
from siamese.plotter import METHOD_NAME, MTC_NAME


def filter_eb_res_df(res_df):
    # only want 4, 8, 12, 16
    r_min = res_df.run.min()
    # print('r_min', r_min, 'runs', res_df.run.unique())
    chosen_runs = [r for r in res_df.run.unique() if int((r - r_min) // 100) % 2 == 1]
    print(f'[filter_eb_res_df] run_min: {r_min}, n_run now: {len(chosen_runs)} ')
    return res_df[res_df.run.isin(chosen_runs)]


def array_to_df(samples, i_max=100, t_unit=0.005):
    # convert samples in array to dataframe for easy visualization
    res_df = None
    flow_typ = ['anchor', 'pos', 'neg']
    for run in samples:
        for i, sample in enumerate(samples[run]):
            if i > i_max:
                break
            info = sample[-1]
            anchor, pos, neg = info[0], info[1], info[2]
            for j, flow in enumerate([anchor, pos, neg]):
                data = pd.DataFrame(sample[j], columns=['owd', 'rtt', 'slr', 'cwnd'])
                data['run'] = run
                data['i_triplet'] = i
                data['flow'] = flow
                data['type'] = flow_typ[j]
                data['t_in_sample'] = t_unit * data.index
                res_df = data if res_df is None else pd.concat([res_df, data])
    return res_df.reset_index(drop=True)


class DataVisualizer:
    """Specifications

    Data visualizer: self-containd data visualizer for dataset inspection,
    ground truth comparison, clustering and detection results. It includes
    several parts:

    1) read: read data from disk, including reading raw dataset, detection.
        Raw dataset: input folder to DataPreprocessor, then call prep.read(folder)
            It is convenient for data inspection, but it's not preprocessed.
        Triplet: read triplet from .pt files, and then convert to df.
        Detection: read from res_*.csv, and combine all of them.

    2) plot:
        flow & queue: like in final_clean.ipynb
        ground truth & label: like in data_preprocess_reader_test.ipynb
        detection: like in integrate_test_clean.ipynb
        *triplet: like in final_clean_pipeline.ipynb, but may not be that useful

    This class should provide a bunch of APIs for the usage in some other notebook
    as well as the automated official figure generation. Since the analysis
    is not mature, we will first use them extensively in notebook.

    The use cases include
    1) Raw data - queues for given run;
    2) Raw data - flow signals for given run;
    3) Raw data - queues vs labels for given run;  (limited num, able to draw them all)
    4) Detection - f1 figures for given testset;
    *5) Detection - processing time for given testset;
    *6) Triplet - flow signals for given samples, given run. (TODO: need more clarity)

    The features of the plot functions include:
    1) arbitrary flow precision, & plot them on n x 2 subplots if necessary;
    2) show or save;
    3) official quality, especially comparison subplots.

    The general development process is implementing it w/ plotter.py together.

    References:
    final_clean_pipeline.py: array_to_df & plot_triplet, for triplet visualization
    final_clean.ipynb: raw dataset inspection
    DataVisualizer & data_preprocess_reader_test.ipynb: truth & label plot
    integrate_test_clean.ipynb: global load of the res_df
    plotter.py: the collection of plot functions
    """

    def __init__(self, folder=None,
                out_dir=None,
                tag=None,
                data_root=None,
                cache_root=None,
                cache_folder=None,
                init_plotter=True):
        """Initialize the data visualizer.
        
        Args:
            folders: the dataset folders
            out_dir: the output directory for saving figures, None for show directly.
            data_root:  the root of the raw dataset
            cache_root: the root of the preprocessed cache where the triplet & detection
                        results are stored.
        """
        if '*' in folder:
            folders = glob.glob(os.path.join(data_root, folder))
            assert len(folders) == 1, 'Only one folder should be matched.'
            folder = os.path.basename(folders[0])
        data_root = data_root or \
            '/home/sapphire/NS_Simulation_Toolkit/BBR_test/ns-3.27/dataset_round2/round2_small'
        cache_root = cache_root or \
            '/home/sapphire/neuralforecast/my_src/pre_cache'
        if 'home' not in data_root:
            data_root = os.path.join(
                '/home/sapphire/NS_Simulation_Toolkit/BBR_test/ns-3.27/dataset_round2/',
                data_root)
        self.data_root = data_root
        self.cache_root = cache_root
        self.data_folder = os.path.join(data_root, folder) if folder else data_root
        cache_folder = cache_folder or folder
        self.cache_folder = os.path.join(cache_root, cache_folder) if cache_folder \
            else cache_root
        do_show = False
        if out_dir is None:
            do_show = True
        elif not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        if init_plotter:
            self.plotter = DataPlotter(out_dir=out_dir, do_show=do_show, tag=tag)

    def read_raw_from_cache(self, cache_folder, runs=None, config_run_base=0):
        config_runs = None if runs is None else \
            [r + config_run_base for r in runs]
        self.prep = DataPreprocessor(cache_folder, runs=runs,
                                     config_runs=config_runs)
        print('cache folder', cache_folder)
        assert os.path.isdir(cache_folder)
        self.prep.read_cache()
        self.xdf = self.prep.xdf
        self.truth_df = self.prep.truth_df

    def read_raw(self, runs=None, read_queue=False, config_run_base=0,
                 read_raw_only=False, no_cached_data=True, read_real=False):
        """Given runs (relative), read raw data through self.prep.
        config_run_base is important for choosing the correct relative
        run for btnk_df, especially for dummy runs."""
        config_runs = None if runs is None else \
            [r + config_run_base for r in runs]
        self.prep = DataPreprocessor(self.cache_folder, runs=runs,
                                     config_runs=config_runs)
        if os.path.isdir(self.cache_folder) and not read_raw_only and not no_cached_data:
            self.prep.read_cache()
        # print(f'Cache folder: {self.cache_folder}')
        # print(f'Data folder: {self.data_folder}')
        if (read_queue or not os.path.isdir(self.cache_folder) or no_cached_data) and \
            os.path.isdir(self.data_folder):
            if read_real:
                self.prep.read_real(self.data_folder)
            else:
                self.prep.read(self.data_folder)
            DataModifier.add_slr(self.prep.xdf)
            DataModifier.remove_zero_owds(self.prep.xdf)
            DataModifier.process_invalid_entry(self.prep.xdf)
        elif read_queue:
            assert False, f'{self.data_folder} doesn\'t exists for read_queue!'

        assert os.path.isdir(self.cache_folder) or os.path.isdir(self.data_folder), \
            f'Neither {self.cache_folder} nor {self.data_folder} exists.'

    def read_raw_across_runs(self, runs, read_queue=False,
                             config_run_base=0, read_raw_only=False,
                             no_cached_data=True, read_real=False):
        """Read raw data from separately chosen runs. The data is stored
        in self.xdf, self.qdf, etc.

        Args:
            runs (list): list of the separate runs.
            config_run_base (int): base of the relative config run
            read_raw_only (bool): if True, only read data instead of cache
            read_real (bool): if True, read real data instead of simulated data
        """
        self.xdf, self.truth_df = None, None
        self.qdf, self.qid_df = None, None
        print('   Read raw: runs', runs)
        for run in runs:
            self.read_raw([run, run + 1], read_queue=read_queue,
                          config_run_base=config_run_base,
                          read_raw_only=read_raw_only,
                          no_cached_data=no_cached_data,
                          read_real=read_real)
            var_names = []
            # if read raw ns3 data w/o preprocessing, there's no truth_df
            for name in ['xdf', 'truth_df']:
                if hasattr(self.prep, name):
                    var_names.append(name)
            if read_queue:
                var_names.extend(['qdf', 'qid_df'])
            for typ in var_names:
                exec(f'self.{typ} = self.prep.{typ} if self.{typ} is None '
                    f'else pd.concat([self.{typ}, self.prep.{typ}])')
        self.xdf.reset_index(drop=True, inplace=True)
        if hasattr(self.prep, 'truth_df'):
            self.truth_df.reset_index(drop=True, inplace=True)
        if read_queue:
            self.qdf.reset_index(drop=True, inplace=True)
            self.qid_df.reset_index(drop=True, inplace=True)

    def _get_runs_from_folder(self, folder, tag):
        """Read runs from folder using files with tag in name."""
        files = sp.getoutput(f'ls {folder}/*{tag}*').split('\n')
        runs = list(map(lambda f: int(f.split('/')[-1].split('_')[-1].split('-')[0]),
                        files))
        runs.sort()
        print(f' Runs found from {folder}: \n       {runs}')
        return runs

    def _get_runs_to_read(self, runs=None, prefix='res_df', cache_folder=None):
        cache_folder = cache_folder if cache_folder is not None else \
            self.cache_folder
        if runs is None:
            runs_to_read = self._get_runs_from_folder(cache_folder, prefix)
        else:
            runs_to_read = range(runs[0], runs[1])
        return runs_to_read
    
    def _print_loaded_runs(self, tag, runs, folder=None):
        folder = folder or self.cache_folder
        print(f'  {tag} of run [{runs[0]}, {runs[-1] + 1}) '
            f'loaded from {folder}')

    def read_detection(self, runs=None, folders=None, add_pair_f1=True):
        """Read detections from folders given runs. This is extended from
        the similar functions by adding multiple folders, and it is not
        applied to data and cache read as those are attached to the data
        preprocessor and hard to modify."""
        res_df = None
        if folders is not None:
            folders = [os.path.join(self.cache_root, f) for f in folders]
        else:
            folders = [self.cache_folder]
        self.res_df = None
        for folder in folders:
            load_any = False
            runs_to_read = self._get_runs_to_read(runs, cache_folder=folder)
            for run in runs_to_read:
                csv = f'res_df_{run}-{run+1}.csv'
                csv_path = os.path.join(folder, csv)
                if not os.path.exists(csv_path):
                    print(f'  Warning: {csv_path} does not exist.')
                    continue
                df = pd.read_csv(csv_path, index_col=False)
                res_df = df if res_df is None else pd.concat(
                    [res_df, df], ignore_index=True)
                load_any = True
            if add_pair_f1:
                res_df['pair_f1'] = 2 * res_df.pair_precision * res_df.pair_recall / \
                    (res_df.pair_precision + res_df.pair_recall)
            if load_any:
                self._print_loaded_runs('Detection', runs_to_read)
            if self.res_df is not None:
                assert set(res_df.columns) == set(self.res_df.columns)
            self.res_df = res_df if self.res_df is None else pd.concat(
                        [self.res_df, res_df], ignore_index=True)
    
    def read_predict_infos(self, runs=None, detect_folder=None):
        """Read predict infos into self.all_predict_infos.
           format:   {(run, n_flow): {run: {method: predict_info} } }
                      predict_info: [ys, y_hats, latent_map]
        """
        all_predict_infos = {}
        load_any = False
        runs_to_read = self._get_runs_to_read(runs, cache_folder=detect_folder)
        detect_folder = detect_folder or self.cache_folder
        for run in runs_to_read:
            pt = f'predict_infos_{run}-{run+1}.pt'
            pt_path = os.path.join(detect_folder, pt)
            if not os.path.exists(pt_path):
                print(f'pwd: {os.getcwd()}')
                print(f'  Warning: {pt_path} does not exist.')
                continue
            predict_infos = torch.load(pt_path)
            all_predict_infos.update(predict_infos)
            load_any = True
        self.all_predict_infos = all_predict_infos
        if load_any:
            self._print_loaded_runs('Predict infos', runs_to_read, detect_folder)

    def get_method_tags(self, all_predict_infos):
        """Get method tags from all_predict_infos."""
        assert len(all_predict_infos.keys()) == 1
        key = list(all_predict_infos.keys())[0]
        predict_infos = all_predict_infos[key]['infos']
        abs_run = list(predict_infos.keys())[0]
        method_tags = list(predict_infos[abs_run].keys())
        return method_tags

    def decode_predict_infos(self, all_predict_infos, method_tag):
        """Decode one run of predict infos w/ given method tag.
        Returns y, y_hat, latent_map, flows."""
        assert len(all_predict_infos.keys()) == 1
        key = list(all_predict_infos.keys())[0]
        predict_infos = all_predict_infos[key]['infos']
        # flows = all_predict_infos[key]['flows']
        abs_run = list(predict_infos.keys())[0]
        predict_info = predict_infos[abs_run][method_tag]
        ys, y_hats, latent_map, flows = predict_info[0], predict_info[1], predict_info[2], predict_info[3]
        return ys, y_hats, latent_map, flows

    def calculate_detect_time(self, runs=None, detect_folder=None, rewrite=False):
        """Given runs and method tags, calculate the detection time df for all
        runs."""
        res_df = None
        detect_folder = detect_folder or self.cache_folder
        # Note that this is different from t_detect_df_[RUN]-[RUN+1].csv, which is buggy
        t_detect_csv = os.path.join(detect_folder, 't_detect_df.csv')
        if os.path.exists(t_detect_csv) and not rewrite:
            print(f'  Loading from {t_detect_csv} ...')
            res_df = pd.read_csv(t_detect_csv, index_col=False)
            res_df.loc[res_df.t_detect < 0, 't_detect'] = np.nan
            self.t_detect_df = res_df
            if 'n_flow' not in res_df.columns:
                res_df['n_flow'] = res_df.groupby('run').flow.transform('nunique')
            return res_df

        print(f'  Calculating detection time ...')
        runs = self._get_runs_to_read(runs, cache_folder=detect_folder) 
        for run in runs:
            self.read_predict_infos(runs=[run, run + 1], detect_folder=detect_folder)
            for tag in self.get_method_tags(self.all_predict_infos):
                ys, y_hats, latent_map, flows = self.decode_predict_infos(
                    self.all_predict_infos, tag)
                interval = 0.35 if 'sbd' in tag else 1.5
                # ~ 2s consistent for correct detection
                if 'sbd' in tag:
                    n, n1 = 6, 4
                else:
                    n, n1 = 2, 2
                t_detect_df = get_detection_delay(ys, y_hats, flows, interval, n=n, n1=n1)
                # t_detect_df['n_flow'] = t_detect_df.flow.nunique()
                set_df_metadata(t_detect_df, run, None, tag)
                res_df = t_detect_df if res_df is None else pd.concat(
                        [res_df, t_detect_df], ignore_index=True)
        self.t_detect_df = res_df
        res_df.loc[res_df.t_detect < 0, 't_detect'] = np.nan
        if 'n_flow' not in res_df.columns:
            res_df['n_flow'] = res_df.groupby('run').flow.transform('nunique')
        res_df.to_csv(t_detect_csv, index=False)
        print(f'  Detection time calculated and saved to {t_detect_csv}.')
        return res_df

    def read_triplet(self, runs=None):
        triplet_df = None
        load_any = False
        runs_to_read = self._get_runs_to_read(runs, prefix='samples')
        for run in runs_to_read:
            pt = f'samples_{run}-{run+1}.pt'
            pt_path = os.path.join(self.cache_folder, pt)
            if not os.path.exists(pt_path):
                print(f'  Warning: {pt_path} does not exist.')
                continue
            sample = torch.load(pt_path)
            sample_df = array_to_df(sample)
            triplet_df = sample_df if triplet_df is None else pd.concat(
                [triplet_df, sample_df], ignore_index=True)
            load_any = True
        self.triplet_df = triplet_df
        if load_any:
            self._print_loaded_runs('Triplet', runs_to_read)

    def get_absolute_run(self, run):
        return sorted(self.prep.qid_df.run.unique())[run]

    # TODO: API below saves the need to find data so it's unified for analysis, but
    #       a little over-abstraction for official figures;
    #       the keywords like fsize, bbox_to_anchor for polish should be directly
    #       added in the plotter's top-level functions

    def plot_queues(self, run, t1, t2, qids=None):
        self.plotter.plot_queues(self.prep.qdf, run, t1, t2, qids=qids)

    def plot_flows_for_queue(self, run, qid, n_flow, t1, t2, n_col=2):
        qid_df = self.prep.qid_df
        flows = qid_df[(qid_df.run == run) &
                       ((qid_df.qid1 == qid) | (qid_df.qid2 == qid))].flow.unique()
        flows = np.random.choice(flows, n_flow, replace=False)
        self.plotter.plot_flows(self.prep.xdf, run, t1, t2, flows=flows, n_col=n_col)

    def plot_queue_vs_label(self, run):
        self.plotter.plot_queue_vs_label(self.prep.qid_df, self.prep.qdf, run)

    def dataplot_queue_for_label(self, run, t1, t2, labels_list, axes):
        """Plot queue for given labels in the given axes.

        This is a dataplot API, which is responsible for manipulating data, i.e.
        qdf here, and then call plotter's axplot_queues to plot.
        
        Args:
            run (int): run number
            t1 (int): start time
            t2 (int): end time
            labels_list (list): list of labels to plot, each for one axis
            axes (list): list of axes to plot
        """
        # TODO: flows list seems useful but slightly complicated
        #       as each flow may have multiple qids, currently fix it to labels
        qdf_run = self.qdf[(self.qdf.run == run)]
        for labels, ax in zip(labels_list, axes):
            g = self.plotter.axplot_queues(qdf_run, labels, 'packet_in_queue',
                                           t1, t2, ax)
            ax.set(title=f'run {run} queues')
        return axes

    def plot_group_detection_by_field(self, metric, xlabels=None, xfield=None,
        n_run=None, run_boundary=None, fsize=(8, 3), bbox_to_anchor=(1.25, 1)):
        self.plotter.plot_group_detection_by_field(self.res_df, metric, xlabels=xlabels,
            xfield=xfield, n_run=n_run, run_boundary=run_boundary, fsize=fsize,
            bbox_to_anchor=bbox_to_anchor)



"""
AdvPlotter Design

Requirements:
    - Support multiple data folders
    - Support multiple axis with the official format

Key idea:
    - Global config: font size, palette, fname, etc
    - Layout config: subplots (n_row, n_col, h/wspace), fsize
    - get_datacell(data, para_btnk, n_flow): prepare data as a unit
        - 1 data cell for 1 to n axis
        - data cell key features: (df_plot, x_field, y_field, hue)
        - in get_datacell(), we use filter to get df_plot from loaded data
            - Specifically, btnk_flow {para_btnks, n_flows, n_btnk, n_flow} to prepare
    - axplot(ax, data_cell, ax_config): config including labels, ticks, legend, no hardcode
        - xlabel, ylabel
        - *ytick: only used for overhead figure
        - set_legend(): call from plotter
        - !add_legend()
        - pointplot, boxplot, barplot, etc

Objects
    - DataCells: load one folder, prepare data cells
        - -> Plotter2d data preparation
    -> - AdvPlotter: plot one figure, use data_cells & ax_configs to plot
    - AxConfig: dict of ax config, each is complete to config one axis
        - -> label settings in leaf plot functions

Use cases:
    - Merge vs_btnk plots
    - Merge detection time plots
    - *Merge overhead plots
    - Merge metrics_vs_n_flow 3x1 plots

Sample run for btnk plots merge:
    - construct_ax_config(): manually construct & set it outside
    - get_datacell(): use data vis to load detection & prepare two data cells
    - config_global(): load global and layout config
"""

# data cell: the data unit for one axis
class DataCell:
    def __init__(self, df=None, x_field=None, y_field=None, hue=None) -> None:
        self.df = df
        self.x_field = x_field
        self.y_field = y_field
        self.hue = hue


class DataCells(DataVisualizer):
    """Load data from one folder, and prepare cells for later axplot."""
    def __init__(self, folder, out_dir, tag, data_root, cache_root):
        super().__init__(folder, out_dir=out_dir, tag=tag, data_root=data_root,
                         cache_root=cache_root, init_plotter=False)

    def init_detection(self, para_btnks, n_flows, n_run, ml_tag=None,
                       mode='detection'):
        """Add btnk and flow info to res_df, and add necessary fields.
        mode: detection, t_detect or nonbtnk
        """
        assert mode in ['detection', 't_detect', 'nonbtnk'], \
            f'Error: mode {mode} not supported.'
        if mode in ['detection', 'nonbtnk']:
            self.read_detection()
            df = self.res_df.copy()
        elif mode == 't_detect':
            self.calculate_detect_time()
            df = self.t_detect_df.copy()
        # TODO: hopefully this condition is correct, but we won't use it for now
        # if df.run.nunique() / n_run > len(para_btnks) * len(n_flows) and len(para_btnks) > 1:
        if df.run.nunique() == 2 * len(para_btnks) * n_run and len(para_btnks) > 1:
            df = filter_eb_res_df(df)
        self.df = self._add_metadata_to_df(df, para_btnks, n_flows, n_run, ml_tag=ml_tag,
                                           add_btnk_ratio=mode == 'nonbtnk')

    def _add_metadata_to_df(self, df, para_btnks, n_flows, n_run, ml_tag=None,
                            add_btnk_ratio=False):
        ml_tag = ml_tag or 'ml_v9_0.4'
        assert ml_tag in df.tag.unique()
        method_tags = ['dcw_0.8', 'rmcat_sbd', 'dc_sbd', ml_tag]
        expected_nrun = len(para_btnks) * len(n_flows) * n_run
        n_run_per_btnk = len(n_flows) * n_run
        self._check_equal_nrun(df, expected_nrun)
        self._add_rel_run(df)
        if 'para_btnk' not in df.columns:
            f_get_btnk = lambda r: para_btnks[int(r.rel_run // n_run_per_btnk)]
            df['para_btnk'] = df.apply(f_get_btnk, axis=1)
        # this function rounds df's n_flow to the closest in given n_flows
        f_get_nflow = lambda r: min(n_flows, key=lambda x: abs(x - r.n_flow))
        df['n_flow'] = df.apply(f_get_nflow, axis=1)
        if add_btnk_ratio:
            df['btnk_ratio'] = df.apply(lambda r: f'{r.para_btnk}/{16}', axis=1)
        print(f'[DataCells] init detection')
        print(f'        total # run: {expected_nrun}, n_btnks {para_btnks}, n_flows {n_flows}')
        return df[df.tag.isin(method_tags)]

    def _check_equal_nrun(self, res_df, expected_nrun):
        if res_df.run.nunique() == expected_nrun:
            return True
        res_runs = [r - res_df.run.min() for r in res_df.run.unique()]
        missing_runs = set(range(expected_nrun)) - set(res_runs)
        print(f'    Warning: res_df # runs: {res_df.run.nunique()} != total runs: {expected_nrun}')
        print(f'    Missing runs: {missing_runs}')
        return False

    def _add_rel_run(self, df: pd.DataFrame):
        """Given a df, update the run field to be relative to the first run,
        and consecutive."""
        assert 'run' in df.columns
        abs_to_rel = {r: i for i, r in enumerate(sorted(df.run.unique()))}
        df['rel_run'] = df.run.map(abs_to_rel)

    def get_cell(self, condition: Union[None, dict, str], x_field, y_field, hue='tag'):
        """Get cell by applying the condition to self.res_df.
        Typical usage includes filtering by para_btnk, n_flow, etc."""
        if type(condition) == dict:
            for k, v in condition.items():
                self.df = self.df[self.df[k] == v]
        elif type(condition) == str:
            self.df = self.df.query(condition)
        return DataCell(self.df.copy(), x_field, y_field, hue)

    def get_cells(self, conditions: np.ndarray, x_field, y_field, hue='tag'):
        """Get cells with a same layout as the conditions array."""
        if len(conditions.shape) == 1:
            return np.array([self.get_cell(c, x_field, y_field, hue) for c in conditions])
        if len(conditions.shape) == 2:
            return np.array([
                [self.get_cell(c, x_field, y_field, hue) for c in row]
                for row in conditions
            ])
        raise ValueError(f'conditions shape {conditions.shape} not supported')


# ax config: the config for one axis
#   legend:  dict {keep: , loc: , bbox_to_anchor: , fontsize:}
#       which is used in plotter._set_legend(),
#   top legend: for the top that is shared by all axes
#   sns_plot: the sns plot function, e.g. sns.pointplot, sns.boxplot
class AxConfig:
    def __init__(self, sns_plot=None, xlabel=None, ylabel=None,
                 yticks=None, yticklabels=None, yscale=None,
                 xticklabels=None,
                 legend=None, add_legend=None, yaxis_major_formatter=None,
                 xlabel_fontsize=None, ylabel_fontsize=None):
        self.sns_plot = sns_plot
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.yticks = yticks
        self.yticklabels = yticklabels
        self.xticklabels = xticklabels
        self.yscale = yscale
        self.legend = legend            # legend for methods tags
        self.add_legend = add_legend    # legend w/ only title for each folder's tag
        self.yaxis_major_formatter = yaxis_major_formatter
        self.xlabel_fontsize = xlabel_fontsize
        self.ylabel_fontsize = ylabel_fontsize

    # def update(self, ax):
    #     """Update the ax config to avoid wrongly remove some settings."""
    #     for k, v in self.__dict__.items():
    #         if v == 'UNSET':
    #             exec(f'self.{k} = ax.get_{k}()')


# sample ax config for metric, yticks should change for overhead
# ylabel: MTC_NAME[y_field]
# xlabel: 'Flow count' if j == len(metrics) -1 else None
METRIC_AX_CONFIG = AxConfig(
    sns_plot=sns.boxplot, xlabel=None, ylabel=None,
    yticks=None, yticklabels=None, yscale='linear',
    legend={'keep': True}, add_legend=None,
    # yaxis_float_formatter=matplotlib.ticker.ScalarFormatter(),    # used in vs_flow?
    yaxis_major_formatter=FormatStrFormatter('%.2f'),
)


# sample ax config for detect time, data should be prepared not like plot_detect_time
TDETECT_AX_CONFIG = AxConfig(
    sns_plot=sns.pointplot, xlabel='Flow Count', ylabel='Detection Delay (s)',
    yscale='linear',
    legend={'keep': True}, add_legend=None,
    # yaxis_float_formatter=matplotlib.ticker.ScalarFormatter(),    # used in vs_flow?
    yaxis_major_formatter=FormatStrFormatter('%.2f'),
)


# sample ax config for overhead, customize xlabel for each axis
OVERHEAD_AX_CONFIG = AxConfig(
    sns_plot=sns.pointplot, xlabel='TBD', ylabel='Overhead (s)',
    yticks=[1, 10, 100, 1000, 10000], yticklabels='UNSET', yscale='log',
    legend={'keep': False, 'labels': [], 'title': None, 'fontsize': 6,
            'loc': 'lower center', 'bbox_to_anchor': (0.5, 1.01), 'n_col': 4},
    add_legend={'title': 'TBD', 'loc': 'best'},
    yaxis_major_formatter=None,
    ylabel_fontsize=10,
)


class AdvPlotter(DataPlotter):
    """Advanced Plotter: plot a figure with multiple subplots and data sources."""
    def __init__(self, out_dir=None, do_show=True, tag=None) -> None:
        super().__init__(out_dir, do_show, tag)
  
    def plot(self, ftag: str, data_cells: np.ndarray, ax_configs: np.ndarray, fsize=None,
             tight_layout=True, whspace=None):
        """Plot the figure with given data cells and ax configs."""
        shape = data_cells.shape
        assert shape == ax_configs.shape
        print(f'[AdvPlotter] plot {ftag} with shape {shape}')
        fsize = fsize or (4, shape[0] * self.height)
        # sharey behavior: if used, then yticks of the right col should also be
        # set even if it's not shown, otherwise the left yticks will be overwritten
        # by the default right yticks
        fig, axs = plt.subplots(*shape, figsize=fsize,
                                sharey='row', sharex='col')
        if shape[0] == 1:
            axs = [axs]
        elif shape[1] == 1:
            axs = [[ax] for ax in axs]
        if len(shape) == 1:
            for ax, cell, ax_config in zip(axs, data_cells, ax_configs):
                self.axplot(ax, cell, ax_config)
        elif len(shape) == 2:
            for i in range(shape[0]):
                for ax, cell, ax_config in zip(axs[i], data_cells[i], ax_configs[i]):
                    self.axplot(ax, cell, ax_config)
        self.show_or_save(f'{ftag}.pdf', tight_layout=tight_layout, whspace=whspace)
    
    def axplot(self, ax, data_cell: DataCell, ax_cfg: AxConfig):
        """Plot each axis using given data cell and config.
        """
        df = data_cell.df
        kwargs = {}
        if ax_cfg.sns_plot == sns.boxplot:
            kwargs = {'flierprops': {'marker': '.', 'markersize': 1.5}}
        elif ax_cfg.sns_plot == sns.pointplot:
            kwargs = {
                    # 'errorbar': 'sd',
                    'ci': 'sd',
                      'scale': 0.55, 'style': data_cell.hue,
                      'dodge': 0.4,
                      'markers': ['s', 'v', '^', 'o'],
                      'linestyles': '--'
                      }
        # print(f'[AdvPlotter] axplot {data_cell.x_field} vs {data_cell.y_field} with hue {data_cell.hue}')
        df_plot = df[[data_cell.x_field, data_cell.y_field, data_cell.hue]]
        g = ax_cfg.sns_plot(x=data_cell.x_field, y=data_cell.y_field, hue=data_cell.hue,
                            data=df_plot, ax=ax, **kwargs)
        # print('ax_cfg:\n', ax_cfg.__dict__)
        g.set_ylabel(ax_cfg.ylabel, fontsize=ax_cfg.ylabel_fontsize)
        g.set_xlabel(ax_cfg.xlabel, fontsize=ax_cfg.xlabel_fontsize)
        g.set(yscale=ax_cfg.yscale)
        if ax_cfg.yticks is not None:
            print(f'[AdvPlotter] set yticks {ax_cfg.yticks}')
            g.set_yticks(ax_cfg.yticks)
        # if ax_cfg.yticklabels != 'UNSET':
        #     g.set_yticklabels(ax_cfg.yticklabels)
        if ax_cfg.xticklabels is not None:
            g.set_xticklabels(ax_cfg.xticklabels)
        if ax_cfg.yaxis_major_formatter:
            g.yaxis.set_major_formatter(ax_cfg.yaxis_major_formatter)
        if ax_cfg.sns_plot in [sns.barplot, sns.boxplot]:
            n_hatch, n_grp = df[data_cell.hue].nunique(), df[data_cell.x_field].nunique()
            plt_type = 'box' if ax_cfg.sns_plot == sns.boxplot else 'bar'
            self._set_hatches(g, n_hatch, n_grp, plt_type=plt_type)
        if ax_cfg.add_legend is not None:
            self.add_legend(g, **ax_cfg.add_legend)
        if ax_cfg.legend is not None:
            self._set_legend(g, **ax_cfg.legend)
        return g


def get_double_cells(folders, out_dir, tag, data_root, cache_root,
                     para_btnks, n_flows, n_run, metrics, conditions,
                     x_fields, mode='detection'):
    # convenient function to get data cells from double folders
    assert len(folders) == 2
    dcells = [[], []]
    for i, folder in enumerate(folders):
        tmp = DataCells(folder, out_dir, tag, data_root, cache_root)
        tmp.init_detection(para_btnks[i], n_flows[i], n_run, mode=mode)
        for metric in metrics:
            cell = tmp.get_cell(conditions[i], x_fields[i], metric)
            dcells[i].append(cell)
    dcells = np.array(dcells, dtype=object).T
    assert dcells.shape == (len(metrics), len(folders)), \
        f'Wrong data cells shape: {dcells.shape} != ({len(metrics)}, {len(folders)})'
    return dcells


def adv_vis_dash_ms(folders, out_dir, tag, data_root, cache_root, metrics=None):
    # 0. manual configs
    para_btnks = [[4, 8, 12, 16], [4, 8, 12, 16]]
    n_flows = [[80], [80]]
    n_run = 15
    metrics = metrics or ['rel_precision', 'rel_recall', 'rel_f1']
    x_fields = ['para_btnk', 'para_btnk']

    # 1. write the data conditions, prepare data cells
    # conditions = [None, 'n_flow in [600, 750, 900, 1050, 1200]']
    conditions = [None, None]
    dcells = get_double_cells(folders, out_dir, tag, data_root, cache_root,
                              para_btnks, n_flows, n_run, metrics, conditions,
                              x_fields)

    # 2. prepare ax configs from templates
    ax_configs = []
    suffix = ['(us OWD, original)', '(ms OWD, new)']
    for i in range(len(metrics)):
        ax_configs.append([])
        for j in range(len(folders)):
            cur_ax = deepcopy(METRIC_AX_CONFIG)
            cur_ax.xlabel = MTC_NAME[x_fields[j]] + '\n' + suffix[j]
            if j == 0:
                cur_ax.ylabel = MTC_NAME[metrics[i]]
            if i > 0:
                cur_ax.yticks = [0.25, 0.5, 0.75, 1.0]
            # else:
            #     cur_ax.yticks = []
            # #     cur_ax.yticklabels = []
            cur_ax.legend = {'keep': i == 0 and j == 0,
                            'loc': 'lower center',
                            'bbox_to_anchor': (1, 1.01)}
            ax_configs[i].append(cur_ax)
    ax_configs = np.array(ax_configs, dtype=object)

    # 3. call plot()
    adv_plotter = AdvPlotter(out_dir, do_show=False, tag=tag)
    adv_plotter.plot('mtc_ms', dcells, ax_configs, whspace=(0, 0))


def adv_vis_dash_final(folders, out_dir, tag, data_root, cache_root, metrics=None):
    # 0. manual configs
    para_btnks = [[4, 8, 12, 16], [16]]
    # n_flows = [[80], [320, 560, 800, 1120]]      # [80] wrong but doesn't matter if not used for filter
    n_flows = [[80], [600, 750, 900, 1050, 1200]]
    n_run = 15
    metrics = metrics or ['rel_precision', 'rel_recall', 'rel_f1']
    x_fields = ['para_btnk', 'n_flow']

    # 1. write the data conditions, prepare data cells
    # conditions = [None, 'n_flow in [600, 750, 900, 1050, 1200]']
    conditions = [None, None]
    dcells = get_double_cells(folders, out_dir, tag, data_root, cache_root,
                              para_btnks, n_flows, n_run, metrics, conditions,
                              x_fields)

    # 2. prepare ax configs from templates
    ax_configs = []
    for i in range(len(metrics)):
        ax_configs.append([])
        for j in range(len(folders)):
            cur_ax = deepcopy(METRIC_AX_CONFIG)
            if i == len(metrics) - 1:
                cur_ax.xlabel = MTC_NAME[x_fields[j]] if j == 0 \
                                    else 'Flow Count, 30 Bottlenecks'
                # if j == 1:
                    # cur_ax.xticklabels = ['320,16', '560,16',
                    #                       '800,32', '1120,32']
            if j == 0:
                cur_ax.ylabel = MTC_NAME[metrics[i]]
            if i > 0:
                cur_ax.yticks = [0.25, 0.5, 0.75, 1.0]
            # else:
            #     cur_ax.yticks = []
            # #     cur_ax.yticklabels = []
            cur_ax.legend = {'keep': i == 0 and j == 0,
                            'loc': 'lower center',
                            'bbox_to_anchor': (1, 1.01)}
            ax_configs[i].append(cur_ax)
    ax_configs = np.array(ax_configs, dtype=object)

    # 3. call plot()
    adv_plotter = AdvPlotter(out_dir, do_show=False, tag=tag)
    adv_plotter.plot('mtc_real', dcells, ax_configs, whspace=(0, 0))


def adv_vis_syn(folders, out_dir, tag, data_root, cache_root, metrics=None):
    """Synthesis results for real data."""
    # 0. manual configs
    para_btnks = [[16], [30]]
    n_flows = [[560, 1120, 2240, 4480], [1200, 2400, 4800, 9600]]
    n_run = 1
    metrics = metrics or ['rel_precision', 'rel_recall', 'rel_f1']
    x_fields = ['n_flow', 'n_flow']

    # 1. write the data conditions, prepare data cells
    conditions = [None] * len(folders)
    dcells = get_double_cells(folders, out_dir, tag, data_root, cache_root,
                              para_btnks, n_flows, n_run, metrics, conditions,
                              x_fields)

    # 2. prepare ax configs from templates
    suffix = [' (iperf3)', ' (Video)']
    ax_configs = []
    for i in range(len(metrics)):
        ax_configs.append([])
        for j in range(len(folders)):
            cur_ax = deepcopy(METRIC_AX_CONFIG)
            if i == len(metrics) - 1:
                cur_ax.xlabel = MTC_NAME[x_fields[j]] + suffix[j]
            if j == 0:
                cur_ax.ylabel = MTC_NAME[metrics[i]]
            if i > 0:
                cur_ax.yticks = [0.25, 0.5, 0.75, 1.0]
            # else:
            #     cur_ax.yticks = []
            # #     cur_ax.yticklabels = []
            cur_ax.legend = {'keep': i == 0 and j == 0,
                            'loc': 'lower center',
                            'bbox_to_anchor': (1, 1.01)}
            ax_configs[i].append(cur_ax)
    ax_configs = np.array(ax_configs, dtype=object)

    # 3. call plot()
    adv_plotter = AdvPlotter(out_dir, do_show=False, tag=tag)
    adv_plotter.plot('mtc_real', dcells, ax_configs, whspace=(0, 0))


def adv_vis_sim_metrics(folders, out_dir, tag, data_root, cache_root, metrics=None):
    # 0. manual configs
    para_btnks = [[1, 2, 3, 4], [4, 8, 12, 16]]
    n_flows = [[50, 100, 150, 200, 250],
               [100, 200, 300, 400, 500]]      # [80] wrong but doesn't matter if not used for filter
    n_run = 4
    metrics = metrics or ['rel_precision', 'rel_recall', 'rel_f1']
    x_fields = ['n_flow', 'n_flow']

    # 1. write the data conditions, prepare data cells
    conditions = ['para_btnk == 3', 'para_btnk == 12']
    dcells = get_double_cells(folders, out_dir, tag, data_root, cache_root,
                              para_btnks, n_flows, n_run, metrics, conditions,
                              x_fields)

    # 2. prepare ax configs from templates
    ax_configs = []
    suffix = ['\n(3 Left Bottlenecks)', '\n(12 Right Bottlenecks)']
    for i in range(len(metrics)):
        ax_configs.append([])
        for j in range(len(folders)):
            cur_ax = deepcopy(METRIC_AX_CONFIG)
            if i == len(metrics) - 1:
                cur_ax.xlabel = MTC_NAME[x_fields[j]] + suffix[j]
            if j == 0:
                cur_ax.ylabel = MTC_NAME[metrics[i]]
            if i > 0:
                cur_ax.yticks = [0.25, 0.5, 0.75, 1.0]
            cur_ax.legend = {'keep': i == 0 and j == 0,
                            'loc': 'lower center',
                            'bbox_to_anchor': (1, 1.01)}
            ax_configs[i].append(cur_ax)
    ax_configs = np.array(ax_configs, dtype=object)

    # 3. call plot()
    adv_plotter = AdvPlotter(out_dir, do_show=False, tag=tag)
    adv_plotter.plot('mtc_sim', dcells, ax_configs, whspace=(0, 0))


def adv_vis_real_metrics(folders, out_dir, tag, data_root, cache_root, metrics=None):
    # 0. manual configs
    para_btnks = [[4, 8, 12, 16], [16]]
    n_flows = [[80], [320, 400, 480, 560]]      # [80] wrong but doesn't matter if not used for filter
    n_run = 15
    metrics = metrics or ['rel_precision', 'rel_recall', 'rel_f1']
    x_fields = ['para_btnk', 'n_flow']

    # 1. write the data conditions, prepare data cells
    conditions = [None] * len(folders)
    dcells = get_double_cells(folders, out_dir, tag, data_root, cache_root,
                              para_btnks, n_flows, n_run, metrics, conditions,
                              x_fields)

    # 2. prepare ax configs from templates
    ax_configs = []
    for i in range(len(metrics)):
        ax_configs.append([])
        for j in range(len(folders)):
            cur_ax = deepcopy(METRIC_AX_CONFIG)
            if i == len(metrics) - 1:
                cur_ax.xlabel = MTC_NAME[x_fields[j]] if j == 0 \
                                    else 'Flow Count, 16 Bottlenecks'
            if j == 0:
                cur_ax.ylabel = MTC_NAME[metrics[i]]
            if i > 0:
                cur_ax.yticks = [0.25, 0.5, 0.75, 1.0]
            # else:
            #     cur_ax.yticks = []
            # #     cur_ax.yticklabels = []
            cur_ax.legend = {'keep': i == 0 and j == 0,
                            'loc': 'lower center',
                            'bbox_to_anchor': (1, 1.01)}
            ax_configs[i].append(cur_ax)
    ax_configs = np.array(ax_configs, dtype=object)

    # 3. call plot()
    adv_plotter = AdvPlotter(out_dir, do_show=False, tag=tag)
    adv_plotter.plot('mtc_real', dcells, ax_configs, whspace=(0, 0))


def adv_vis_sim_btnk(folders, out_dir, tag, data_root, cache_root, metrics=None):
    para_btnks = [[1, 2, 3, 4], [4, 8, 12, 16]]
    n_flows = [[50, 100, 150, 200, 250], [100, 200, 300, 400, 500]]
    n_run = 4
    metrics = metrics or ['rel_f1']
    x_fields = ['para_btnk', 'para_btnk']
    lg_titles = ['left', 'right']
    conditions = [{'n_flow': 150}, {'n_flow': 300}]
    assert len(folders) == 2
    dcells = get_double_cells(folders, out_dir, tag, data_root, cache_root,
                              para_btnks, n_flows, n_run, metrics, conditions,
                              x_fields)

    ax_configs = []
    for i in range(len(metrics)):
        ax_configs.append([])
        for j in range(len(folders)):
            cur_ax = deepcopy(METRIC_AX_CONFIG)
            if i == len(metrics) - 1:
                cur_ax.xlabel = MTC_NAME[x_fields[j]]
            if j == 0:
                cur_ax.ylabel = MTC_NAME[metrics[i]]
            cur_ax.legend = {'keep': i == 0 and j == 0,
                            'loc': 'lower center',
                            'bbox_to_anchor': (1, 1.01)}
            cur_ax.add_legend = {'title': lg_titles[j],
                                 'loc': 'lower right' if j == 0 else 'lower left',
                                 'bbox_to_anchor': (0.85, 0.02) if j == 0 else (0.15, 0.02),
                                 'fontsize': 7}
            ax_configs[i].append(cur_ax)
    ax_configs = np.array(ax_configs, dtype=object)
    adv_plotter = AdvPlotter(out_dir, do_show=False, tag=tag)
    adv_plotter.plot('f1_sim_btnk', dcells, ax_configs, whspace=(0, 0))


def adv_vis_sim_tdetect(folders, out_dir, tag, data_root, cache_root, metrics=None):
    para_btnks = [[1, 2, 3, 4], [4, 8, 12, 16]]
    n_flows = [[50, 100, 150, 200, 250], [100, 200, 300, 400, 500]]
    n_run = 4
    metrics = metrics or ['t_detect']
    x_fields = ['n_flow', 'n_flow']
    lg_titles = ['left', 'right']
    conditions = [{'para_btnk': 4}, {'para_btnk': 16}]
    assert len(folders) == 2
    dcells = get_double_cells(folders, out_dir, tag, data_root, cache_root,
                              para_btnks, n_flows, n_run, metrics, conditions,
                              x_fields, mode='t_detect')
    ax_configs = []
    for i in range(len(metrics)):
        ax_configs.append([])
        for j in range(len(folders)):
            cur_ax = deepcopy(TDETECT_AX_CONFIG)
            if j != 0:
                cur_ax.ylabel = None
            #     cur_ax.yticks = []
            cur_ax.legend = {'keep': i == 0 and j == 0,
                            'loc': 'lower center',
                            'bbox_to_anchor': (1, 1.01)}
            cur_ax.add_legend = {'title': lg_titles[j],
                                'loc': 'upper left',
                                'bbox_to_anchor': (0.1, 0.96),
                                'fontsize': 7}
            ax_configs[i].append(cur_ax)
    ax_configs = np.array(ax_configs, dtype=object)
    adv_plotter = AdvPlotter(out_dir, do_show=False, tag=tag)
    adv_plotter.plot('tdetect_sim', dcells, ax_configs, whspace=(0, 0))


def adv_vis_nonbtnk(folders, out_dir, tag, data_root, cache_root, metrics=None):
    para_btnks = [[2, 4, 6, 8], [2, 4, 6, 8]]
    n_flows = [[80], [160]]
    n_run = 15
    metrics = metrics or ['rel_precision', 'rel_recall', 'rel_f1']
    # x_fields = ['para_btnk', 'para_btnk']
    x_fields = ['btnk_ratio', 'btnk_ratio']
    lg_titles = ['80 flows', '160 flows']
    conditions = [None, None]
    assert len(folders) == 2
    dcells = get_double_cells(folders, out_dir, tag, data_root, cache_root,
                                para_btnks, n_flows, n_run, metrics, conditions,
                                x_fields, mode='nonbtnk')
    ax_configs = []
    for i in range(len(metrics)):
        ax_configs.append([])
        for j in range(len(folders)):
            cur_ax = deepcopy(METRIC_AX_CONFIG)
            if i == len(metrics) - 1:
                cur_ax.xlabel = MTC_NAME[x_fields[j]]
            if j == 0:
                cur_ax.ylabel = MTC_NAME[metrics[i]]
            cur_ax.legend = {'keep': i == 0 and j == 0,
                            'loc': 'lower center',
                            'bbox_to_anchor': (1, 1.01)}
            cur_ax.add_legend = {'title': lg_titles[j],
                                 'loc': 'lower center',
                                 'fontsize': 7}
            ax_configs[i].append(cur_ax)
    ax_configs = np.array(ax_configs, dtype=object)
    adv_plotter = AdvPlotter(out_dir, do_show=False, tag=tag)
    adv_plotter.plot('mtc_nonbtnk', dcells, ax_configs, whspace=(0, 0))


def adv_vis_large_ovaerhead(folder, out_dir, tag, data_root, cache_root, metric=None):
    para_btnks = [16]
    n_flows = [250, 500, 1000, 2000, 4000, 8000]
    n_run = 1
    metric = metric or 't_classify'
    x_field = 'n_flow'
    tmp = DataCells(folder, out_dir, tag, data_root, cache_root)
    tmp.init_detection(para_btnks=para_btnks, n_flows=n_flows, n_run=n_run)
    condition = 'n_flow < 8000'
    cell = tmp.get_cell(condition, x_field, metric)
    dcells = np.array([cell], dtype=object)
    ax_configs = []
    cur_ax = deepcopy(OVERHEAD_AX_CONFIG)
    cur_ax.xlabel = 'Flow Count'        # hack here, as n_flow internal is not correct
    cur_ax.ylabel = r'$\frac{\mathbf{\mathrm{Overhead}}}{\mathbf{\mathrm{Interval}}}$'
    cur_ax.legend = {'keep': True,
                    'loc': 'lower center',
                    'bbox_to_anchor': (0.5, 1.01)}
    cur_ax.add_legend = None
    ax_configs.append(cur_ax)
    ax_configs = np.array(ax_configs, dtype=object)
    adv_plotter = AdvPlotter(out_dir, do_show=False, tag=tag)
    g = adv_plotter.plot('large_overhead', dcells, ax_configs, whspace=(0, 0))


# TODO: extract the ax config into config files instead of hardcoding here