import argparse
import glob
import time
import os
import numpy as np
import pandas as pd
import sys
import subprocess as sp
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import seaborn as sns
import matplotlib.pyplot as plt

from enum import Enum
from multiprocessing import Process, active_children

sys.path.append('my_src')

from models import MyTransformerEncoder
from siamese.preprocess import DataPreprocessor
from siamese.dataset import SiameseDataset, CleanDataLoader
from siamese.model import MySiameseModel, ResSiameseModel
from siamese.cluster import cluster
from siamese.pipeline import predict_segment, compare_segment
from detector import get_detector_accuracy


def best_model(n_field, seq_len):
    model = {
        'c_in': n_field,
        'l_in': seq_len,
        'n_out': 16,
        # 'c_outs': [64] * 4,
        'c_outs': [64] * 3,
        'kernel_size': 13,
        # 'kernel_size': 9,
        # 'dilation': [1, 2, 4, 8],       # span = (k - 1) * d + 1
        'dilation': [1, 2, 4],
        'dropout': 0.0,
        'use_batch_norm': False,
        'use_local_pooling': False,
        'use_global_avg_pooling': True,
        'remove_mean': True,
        'loss_mode': 'abs_triplet',
    }
    return model


class PrepMode(Enum):
    """Preprocess mode for local_process()"""
    TRAIN = 'train'
    TEST = 'test'
    PREDICT = 'predict'
    REAL = 'real'


class FinalCleanPipeline:
    """This is the pipeline utilizing multi-process to preprocess
    different runs, and load them into unified dataframe for model
    train and test.

    The main objectives include:

        1. Take into folder and runs as args, and run the local preprocessor.
        2. Load all csv into one large dataframe.
        3. Train and test the model using loaded df.
        4. ? Interface: cmd + script first, (multiprocessing?)
    """

    def __init__(self, seq_len, fields, cache_root):
        # runs: two int, [run_start, run_end)
        self.seq_len = seq_len
        self.fields = fields
        self.cache_root = cache_root
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    def local_process(self, folder, runs, config_runs, local_cache_folder, prep_mode):
        # local_cache_folder: str, the folder to save the LOCAL preprocessed csv
        local_cache_folder = os.path.join(self.cache_root, local_cache_folder)
        prep = DataPreprocessor(local_cache_folder, runs=runs,
                                config_runs=config_runs)
        prep.read(folder)
        for_test = prep_mode == PrepMode.TEST
        prep.preprocess(self.seq_len, self.fields, for_test)       # samples saved in build_triplet

    def local_process_for_prediction(self, folder, runs, config_runs, local_cache_folder):
        local_cache_folder = os.path.join(self.cache_root, local_cache_folder)
        prep = DataPreprocessor(local_cache_folder, runs=runs, config_runs=config_runs)
        prep.read(folder)
        prep.preprocess_for_prediction(self.seq_len)
        prep.check()
        tag = f'{runs[0]}-{runs[1]}'
        prep.save_prediction(tag)
    
    def local_process_for_real(self, folder, runs, local_cache_folder):
        local_cache_folder = os.path.join(self.cache_root, local_cache_folder)
        prep = DataPreprocessor(local_cache_folder, runs=runs)
        prep.read_real(folder)
        prep.preprocess_for_real(self.seq_len)
        # prep.check_for_real(self.seq_len)
        tag = f'{runs[0]}-{runs[1]}'
        prep.save_prediction(tag)

    def global_load(self, tags, boundary1, boundary2, batch_size,
                    split_on='run', field_indices=None, subdirs=None, n_worker=1):
        """Load the samples from cache folder and construct dataloaders. It
        supports reading from multiple folders identified by different tags,
        and split the data w/ given boundaries. Note that the split can only
        happen for the first two tags, so only boundary1/2 are passed, and
        later boundary will be [0, 1] that are automatically added.

        For instance, tags = ['folder1', 'folder2'], boundary1 = [0, 1, 2],
        boundary2 = [0, 1], then the folder1 data will be split into 2 parts
        and serve as train/val, and the folder2 data will be used as test.
        More test folders are also allowed, and their boundary will be [0, 1],
        e.g. tags = ['f1', 'f2', 't1', t2'], boundary1 = boundary2 = [0, 1],
        then t1/t2 will be two testsets.

        Field indices are used to specify which fields to use, e.g. [0, 1]
        to use the first two fields. If None, then all fields will be used.
        """
        # actually CleanDataLoader doesn't depend on seq_len in load()
        self.loader = CleanDataLoader(self.seq_len, self.fields,
                                      field_indices=field_indices)
        self.field_indices = field_indices
        boundaries = [boundary1, boundary2]
        boundaries.extend([[0, 1]] * (len(tags) - 2))
        print('  Train / Val / Test / ... data from: ')
        i = 0
        for t, b in zip(tags, boundaries):
            for j in range(len(b) - 1):
                print(f'    No. {i}: {t} {b[j]}-{b[j + 1]}')
                i += 1
        print()

        # support input "prep_train" only for training
        subdirs = subdirs or [None] * len(tags)
        assert len(subdirs) == len(tags) or len(subdirs) == 1
        subdirs = subdirs * len(tags) if len(subdirs) == 1 else subdirs
        for i, (tag, boundary, subdir) in enumerate(zip(tags, boundaries, subdirs)):
            cur_index = self.loader.load(self.cache_root, tag=tag, subdir=subdir)
            self.loader.split(cur_index, boundary, batch_size, split_on,
                              shuffle_first=i == 0, n_worker=n_worker)

    def local_detect(self, run, model_path, ts, detect_folder,
                     out_dir=None,
                     max_flow=300,
                     methods=['dcw', 'rmcat_sbd', 'dc_sbd', 'ml'],
                     do_update=False,
                     model_th=0.2,
                     model_th_list=None,
                     n_flow_per_btnk=None,
                     n_flows_per_btnk_to_draw=[],
                     ths=None,
                     n_repeat=10):
        """Read prediction data for given run, give detections and write
        back the results into local csv.

        Assume per-run prediction data is saved in [type]_pred_[i]-[i+1].csv
        under given cache folder.
        ts: [t_start, t_end], used to filter the time.

        The res_df is saved in res_df_[i]-[i+1].csv, and the predict_infos
        is saved in predict_infos_[i]-[i+1].pt. Its format is:
            {(run, n_flow): {'flows': sampled_flows, 'infos': predict_infos} }
        Check get_detector_accuracy() for the details of data formats.

        There are 3 independent modes of detection:

        - Mode 1: one normal run, and max_flow applies to the maximum # of flows
                    to sample.
        - Mode 2: flow scan, where n_flow_per_btnk & n_flows_per_btnk_to_draw come
                  into play. The res_df is just generated once, and then flows are
                  sampled to construct the results of different total # of flows.
        - Mode 3: th scan, where only 'ml' is used & th are scanned.

        The entering order is: if n_flow_per_btnk, then mode 2; elif ths,
        then mode 3; else mode 1.

        If do_update is True, then the res_df, predict_infos from
        get_detector_accuracy() will be used to update the existing res_df,
        predict_infos instead of directly overwriting them. This can save time
        when the only thing changed is the ML method.

        Modifications:
        - res_df: set index to [run, method, time], first merge time_df, and
                then use df.update()
        - predict_infos: loop over run, then use dict.update()

        Get data:
        - new data: just from get_detector_accuracy()
        - old data: read from res_df_[i]-[i+1].csv, predict_infos_[i]-[i+1].pt

        Apr 5. update: use model_th_list to scan multiple models at once
        model_th_list: list of (model_tag, model_path, model_th), and overwrite
        the model_path & model_th if not None. In the res_df & time_df returned,
        tag, i.e. '{method}_{model_tag}_{model_th}' will be used as key to replace
        the method.
        """

        # TODO: this is temperory hack, shouldn't be needed if the labels are
        # converted in preprocessing
        converge_nonbtnk = True
        out_dir = out_dir or detect_folder
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        print(' Local detect: ', detect_folder, ' run: ', run)
        for typ in ['xdf', 'truth_df']:
            csv = f'{typ}_pred_{run}-{run+1}.csv'
            csv_path = os.path.join(detect_folder, csv)
            df = pd.read_csv(csv_path, index_col=False)
            exec(f'self.{typ} = df')

        # TODO: with model_th in model_th_list, the other way of scanning th
        # is simply feeding a list of ths of the same model, and it will
        # deprecate this function, so let it be for now.
        def _detect_for_different_th(xdf, truth_df, ths, corr_interval):
            print('  Detecting for different th: ', ths)
            final_res_df1 = None
            assert ths is not None
            methods = ['ml']
            print('  [WARN] Assigned ml to methods for ths != None')
            for th in ths:
                res_df, _, time_df, _ = get_detector_accuracy(
                    xdf, [run], self.fields, truth_df, model_path, methods,
                    corr_interval=corr_interval,
                    ts=ts, model_th=th, converge_nonbtnk=converge_nonbtnk)
                if res_df is None or time_df is None:
                    print('  [WARN] res_df or time_df is None, skip th: ', th)
                    continue
                time_df.set_index(['run', 'tag'], inplace=True)
                res_df['t_classify'] = res_df.apply(
                    lambda r: time_df.loc[(r.run, r.tag), 't_classify'], axis=1)
                res_df['th'] = th
                final_res_df1 = res_df if final_res_df1 is None else \
                    pd.concat([final_res_df1, res_df], ignore_index=True)
            return final_res_df1

        def _detect_for_all_flows(xdf, truth_df, corr_interval):
            print('  Detecting for all flows...')
            res_df, predict_infos, time_df, t_detect_df = get_detector_accuracy(
                xdf, [run], self.fields, truth_df, model_path, methods,
                corr_interval=corr_interval, ts=ts,
                model_th=model_th, converge_nonbtnk=converge_nonbtnk,
                model_th_list=model_th_list)
            if res_df is None or time_df is None:
                print('  [WARN] res_df or time_df is None')
                return None, None
            time_df.set_index(['run', 'tag'], inplace=True)
            print(time_df.index.unique())
            res_df['t_classify'] = res_df.apply(
                lambda r: time_df.loc[(r.run, r.tag), 't_classify'], axis=1)
            return res_df, predict_infos, t_detect_df

        def _detect_for_given_flows(predict_infos, sampled_flows):
            # provide detections for given flows by choosing from the full results
            # assume the flow No. in sampled_flows are the same as their indices
            assert type(sampled_flows) == set
            res_df = None
            cur_predict_infos = {}
            for run in predict_infos:
                cur_predict_infos[run] = {}
                for tag in predict_infos[run]:
                    method = tag.split('_')[0]
                    predict_info = predict_infos[run][tag]
                    ys, y_hats = predict_info[0], predict_info[1]
                    latent_map, flows = predict_info[2], predict_info[3]
                    times = sorted(latent_map.keys())
                    ys_cur, y_hats_cur, latent_map_cur = [], [], {}
                    for i, t in enumerate(times):
                        # print(i, t)
                        latent = latent_map[t]
                        common_flows = sorted(set(flows[i]) & sampled_flows)
                        i_sampled_flows = [flows[i].index(f) for f in common_flows]
                        y_cur = np.array(ys[i])[i_sampled_flows]
                        y_hat_cur = np.array(y_hats[i])[i_sampled_flows]
                        metrics, cols = compare_segment(y_cur.tolist(), y_hat_cur.tolist())
                        if latent is not None:
                            latent_cur = np.array(latent)[i_sampled_flows].tolist()
                        else:
                            latent_cur = None
                        # here t_classify is not measured, so set to -1
                        cols = ['run', 'tag', 'method', 'time'] + cols + ['t_classify', 'n_flow']
                        cur_df = pd.DataFrame([[run, tag, method, t] + metrics + [-1, len(sampled_flows)]],
                                              columns=cols)
                        res_df = cur_df if res_df is None else pd.concat([res_df, cur_df],
                                                                        ignore_index=True)
                        latent_map_cur[t] = latent_cur
                        ys_cur.append(y_cur)
                        y_hats_cur.append(y_hat_cur)
                    # flat_flows are set to None as we don't really detect here 
                    cur_predict_infos[run][tag] = [ys_cur, y_hats_cur,
                                                     latent_map_cur, None]
            return res_df, cur_predict_infos

        res_df = None
        all_predict_infos = None
        corr_interval = self.seq_len * 0.005
        if n_flow_per_btnk:
            # here we only run the detectors once for the full-flow case, and use
            # cached ys, y_hats and latent_map for the sampled-flow cases to speed up
            all_flows = sorted(self.xdf.flow.unique())
            full_res_df, full_predict_infos, full_t_detect_df = \
                _detect_for_all_flows(self.xdf, self.truth_df, corr_interval)
            if full_res_df is None:
                return

            # draw flows from each n_flows_per_btnk_to_draw, e.g. [2, 4, 6, 8, 10, 12]
            assert n_flows_per_btnk_to_draw
            all_predict_infos = {}
            for n_flow_per_btnk_to_draw in n_flows_per_btnk_to_draw:
                # repeat to make the sampling more broad
                assert n_flow_per_btnk_to_draw <= n_flow_per_btnk
                print(f'  Sampling {n_flow_per_btnk_to_draw} flows per btnk')
                for nr in range(n_repeat):
                    print(f'   Repeat {nr} / {n_repeat}')
                    sampled_flows = set()
                    for i in range(0, self.xdf.flow.nunique(), n_flow_per_btnk):
                        if self.xdf.flow.nunique() - i < n_flow_per_btnk_to_draw:
                            print(f'    Flows left ({self.xdf.flow.nunique() - i}) '
                                f'are fewer than n_flow_per_btnk_to_draw {n_flow_per_btnk_to_draw}')
                            continue
                        sampled_flows |= set(np.random.choice(all_flows[i: i + n_flow_per_btnk],
                            n_flow_per_btnk_to_draw, replace=False))
                    cur_df, predict_infos = _detect_for_given_flows(
                        full_predict_infos, sampled_flows)
                    # the duplicate rows are intended, as the visualization
                    # does not care the repeat group difference
                    res_df = cur_df if res_df is None else pd.concat([res_df, cur_df])
                    if n_flow_per_btnk_to_draw == n_flow_per_btnk:
                        # no other combinations
                        break
                info_dict = {'flows': sampled_flows, 'infos': predict_infos}
                all_predict_infos[(run, len(sampled_flows))] = info_dict
        elif ths is not None:
            res_df = _detect_for_different_th(self.xdf, self.truth_df, ths,
                                              corr_interval)
            if res_df is None:
                return
        else:
            all_flows = sorted(self.xdf.flow.unique())
            max_flow = min(max_flow, len(all_flows))
            sampled_flows = set(np.random.choice(all_flows, max_flow, replace=False))
            self.xdf = self.xdf[self.xdf.flow.isin(sampled_flows)]
            self.truth_df = self.truth_df[self.truth_df.flow.isin(sampled_flows)]
            res_df, predict_infos, t_detect_df = _detect_for_all_flows(self.xdf, self.truth_df,
                                                          corr_interval)
            res_df['n_flow'] = max_flow
            if res_df is None:
                return
            all_predict_infos = {(run, max_flow):
                {'flows': sampled_flows, 'infos': predict_infos}}

        res_csv_path = os.path.join(out_dir, f'res_df_{run}-{run+1}.csv')
        predict_info_path = os.path.join(out_dir, f'predict_infos_{run}-{run+1}.pt')
        if do_update:
            res_df, all_predict_infos = self._update_old_data(
                res_df, all_predict_infos, res_csv_path, predict_info_path,
                methods)

        res_df.to_csv(res_csv_path, index=False)
        # t_detect_df.to_csv(os.path.join(out_dir, f't_detect_df_{run}-{run+1}.csv'),
        #                    index=False)
        if all_predict_infos is not None:
            torch.save(all_predict_infos, predict_info_path)
            print(' Local detect: ', detect_folder, ' dumped to ', res_csv_path,
                ' and ', predict_info_path)
        else:
            print(' Local detect: ', detect_folder, ' dumped to ', res_csv_path)

    def _update_old_data(self, res_df, all_predict_infos, res_csv_path,
                         predict_info_path, methods):
        """Updating the old data instead of overwriting it. Note that
        all the detection above should be supported, the only difference
        is if updating the old or just overwriting.
        """
        old_res_df = pd.read_csv(res_csv_path, index_col=False)
        assert 'n_flow' in old_res_df.columns and 'th' not in old_res_df.columns, \
            print('   Columns mismatch, ensure you want to update the res_df!')

        idx_col = ['run', 'n_flow', 'tag']
        old_res_df.set_index(idx_col, inplace=True)            
        res_df.set_index(idx_col, inplace=True)
        for idx in res_df.index.unique():
            if idx in old_res_df.index.unique():
                old_res_df.drop(idx, inplace=True)
        res_df = pd.concat([old_res_df, res_df]).reset_index()

        if all_predict_infos is not None:
            old_all_predict_infos = torch.load(predict_info_path)
            for key in old_all_predict_infos:
                if key not in all_predict_infos:
                    print(f'Warning: {key} in old not found in new, skip!')
                    continue
                # in theory the different flows are not valid, but given
                # that mostly we only use predict infos for ml debugging,
                # here we want to overwrite it only when the new method
                # is ml
                if 'ml' in methods:
                    old_all_predict_infos[key]['flows'] = \
                        all_predict_infos[key]['flows']
                for run in old_all_predict_infos[key]['infos']:
                    old_all_predict_infos[key]['infos'][run].update(
                        all_predict_infos[key]['infos'][run])
            all_predict_infos = old_all_predict_infos
        return res_df, all_predict_infos

    def set_model(self, model_type='cnn', **params):
        if model_type == 'res-dc':
            default_params = {
                'c_in': len(self.fields),
                'l_in': self.seq_len,
                'n_out': 8,
            }
            default_params.update(params)
            self.model = ResSiameseModel(**default_params)
        elif model_type == 'cnn':
            default_params = {
                'c_in': len(self.fields),
                'l_in': self.seq_len,
                'n_out': 8,
                'c_outs': [64, 64],
                'dropout': 0.0,
            }
            default_params.update(params)
            self.model = MySiameseModel(**default_params)
        elif model_type == 'tf':
            default_params = {
                'n_in': len(self.fields),
                'n_out': 8,
                'l_in': self.seq_len,
                'd_model': 64,
                'nhead': 4,
                'd_hid': 64,
                'nlayers': 1,
            }
            default_params.update(params)
            self.model = MyTransformerEncoder(**default_params)

    def train(self, patience, max_epochs, check_val_every_n_epoch=1, dry_run=False):
        csv_logger = pl_loggers.CSVLogger(save_dir='logs')
        early_stopping = pl.callbacks.EarlyStopping(monitor='val_loss',
            min_delta=1e-4, patience=patience, verbose=False, mode='min')
        self.trainer = pl.Trainer(max_epochs=max_epochs,
                                  check_val_every_n_epoch=check_val_every_n_epoch,
                                  callbacks=[early_stopping],
                                #   log_every_n_steps=10000,
                                  accelerator=self.device,
                                #   accelerator='gpu',
                                  devices=1,
                                  logger=False)
        if not dry_run:
            assert self.model and self.loader
            self.trainer.fit(self.model, self.loader.get_train_loader(),
                             self.loader.get_val_loader())

    def test(self):
        assert self.trainer and self.loader
        for loader in self.loader.get_test_loaders():
            self.trainer.test(self.model, loader)
    
    def clear_dataloader(self, idx=[0, 1]):
        # clear to release memory, 0/1/2/... for train/val/test/...
        for i in idx:
            # del self.loader.loaders[i]
            self.loader.loaders[i] = None
            del self.loader.loaders[i]
        self.loader.i_test = 2 - len(idx)

    def save_model(self, path='final_model.pt', params=None):
        # currently support dilated cnn default only
        # note that state_dict can be used only after params are exactly
        # matched w/ state_dict
        default_params = {
            'c_in': len(self.fields),
            'l_in': self.seq_len,
            'n_out': 8,
            'c_outs': [64, 64],
            'dropout': 0.0,
        }
        params = params or default_params
        fields = self.fields
        if self.field_indices is not None:
            fields = [self.fields[i] for i in self.field_indices]
        model_info = {
            'params': params,
            'state_dict': self.model.state_dict(),
            'fields': fields,
        }
        torch.save(model_info, path)
        print(' Model saved to', path)
    
    def load_model(self, path='final_model.pt',
                   model_class=MySiameseModel):
        # support MySiameseModel only for now 
        model_info = torch.load(path)
        self.model = model_class(**model_info['params'])
        self.model.load_state_dict(model_info['state_dict'])
        print(' Model loaded from', path)

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

# multiprocessing for detections

def run_local_detect(seq_len, fields, cache_root, run, model_path, ts,
                     detect_folder, detect_out_dir, max_flow,
                     methods=['dcw', 'rmcat_sbd', 'dc_sbd', 'ml',
                              'naive_neg', 'naive_pos'],
                     do_update=False,
                     model_th=0.2,
                     model_th_list=None,
                     n_flow_per_btnk=None,
                     n_flows_per_btnk_to_draw=[],
                     ths=None):
    print('- Start local detect: ', detect_folder, ' run: ', run)
    cpipe = FinalCleanPipeline(seq_len, fields, cache_root)
    cpipe.local_detect(run, model_path, ts, detect_folder,
                       out_dir=detect_out_dir,
                       max_flow=max_flow,
                       methods=methods,
                       do_update=do_update,
                       model_th=model_th,
                       model_th_list=model_th_list,
                       n_repeat=5,
                       n_flow_per_btnk=n_flow_per_btnk,
                       n_flows_per_btnk_to_draw=n_flows_per_btnk_to_draw,
                       ths=ths)

def create_arg_list_for_detect(seq_len, fields, cache_root, model_path, ts,
                               detect_folders, detect_out_dirs, max_flow,
                               methods, do_update,
                               model_th,
                               model_th_list,
                               n_flow_per_btnk,
                               n_flows_per_btnk_to_draw,
                               ths):
    # build arg list for all runs in the given cache folders to run
    # local processes, used to decouple local worker and multiprocessing
    # scheduler
    arg_list = []
    cwd = os.getcwd()
    for detect_folder, detect_out_dir in zip(detect_folders, detect_out_dirs):
        os.chdir(detect_folder)
        for typ in ['xdf', 'truth_df']:
            assert len(glob.glob(f'{typ}_pred_*-*.csv'))
        ids = sp.getoutput(f'ls xdf_pred*csv').split('\n')
        # runs = list(range(len(ids)))        # assume all runs are saved
        runs = list(map(lambda x: int(x.split('_')[-1].split('-')[0]), ids))
        for run in runs:
            args = (seq_len, fields, cache_root, run, model_path, ts,
                    detect_folder, detect_out_dir, max_flow, methods,
                    do_update, model_th, model_th_list)
            if n_flow_per_btnk is not None:
                args += (n_flow_per_btnk, n_flows_per_btnk_to_draw)
            elif ths is not None:
                args += (None, -1, ths)
            arg_list.append(args)
        os.chdir(cwd)
    return arg_list

def run_multiprocess(f_worker, arg_list, n_proc=4, t_inter=1):
    ps = []
    for args in arg_list:
        p = Process(target=f_worker, args=args)
        ps.append(p)
    i = 0
    n_basic = len(active_children())
    while i < len(ps):
        if len(active_children()) - n_basic < n_proc:
            print(f'- Start process {i} ...')
            ps[i].start()
            time.sleep(t_inter)
            i += 1
            interval = 0
        else:
            interval = 5
        time.sleep(interval)
    for p in ps:
        p.join()

# multiprocessing for preprocessing

def run_local_process(seq_len, fields, cache_root, folder, runs, config_runs,
                      cache_folder, prep_mode):
    print('- Start local process: ', cache_folder, ' runs: ', runs)
    cpipe = FinalCleanPipeline(seq_len, fields, cache_root)
    if prep_mode == PrepMode.TRAIN or prep_mode == PrepMode.TEST:
        cpipe.local_process(folder, runs, config_runs, cache_folder, prep_mode)
    elif prep_mode == PrepMode.PREDICT:
        cpipe.local_process_for_prediction(folder, runs, config_runs, cache_folder)
    elif prep_mode == PrepMode.REAL:
        cpipe.local_process_for_real(folder, runs, cache_folder)
    else:
        raise ValueError('Invalid prep_mode')

def create_arg_list_for_preprocess(seq_len, fields, cache_root, folders,
                                   cache_folders,
                                   prep_mode=PrepMode.TRAIN, n_run=1,
                                   config_run_base=0, subdir='dats'):
    """Create arg list for all runs in the given cache folders to run.
    Here, we use the all-data to determine the runs to process from
    folders.
    
    Args:
        seq_len (int): sequence length
        fields (list): list of fields to use
        cache_root (str): root folder of cache
        folders (list): list of folders to process
        cache_folders (list): list of cache folders for output
        prep_mode (PrepMode): preprocessing mode
        n_run (int): number of runs to process for each file
        run_order (str): order of runs to process, 'asc' or 'desc'.
                        If 'asc', the first n_run runs will be processed.
                        If 'desc', the last n_run runs will be processed.
    """
    arg_list = []
    n_runs = {}
    cwd = os.getcwd()
    for folder in folders:
        # get all runs from folder
        subfolder = folder
        if subdir is not None:
            subfolder = os.path.join(folder, subdir)
        assert os.path.isdir(subfolder), \
            f'{subfolder} is not an existing directory'
        os.chdir(subfolder)
        ids = sp.getoutput(f'ls *all-data*').split('\n')
        ids = list(map(lambda x : int(x.split('_')[-1][:-4]), ids))
        n_runs[folder] = len(ids)
        os.chdir(cwd)

    if cache_folders is not None:
        assert len(folders) == len(cache_folders)
    for i, folder in enumerate(folders):
        n_all_run = n_runs[folder]
        cache_folder = cache_folders[i] if cache_folders is not None else \
            os.path.basename(os.path.normpath(folder))
        for i in range(0, n_all_run, n_run):
            runs = [i, min(i + n_run, n_all_run)]
            config_runs = [i + config_run_base,
                           min(i + n_run + config_run_base, n_all_run)]
            args = (seq_len, fields, cache_root, folder, runs, config_runs,
                    cache_folder, prep_mode)
            arg_list.append(args)
    
    return arg_list


def train_test_model(patience, max_epochs, model_params, model_type='cnn'):
    cpipe.set_model(model_type=model_type, **model_params)
    cpipe.train(patience, max_epochs)
    cpipe.test()


def scan_fcn_params(args, cpipe):
    patience, max_epochs = 3, 6
    cpipe.global_load(args.tags, args.boundary, args.boundary2, args.batch_size,
                      split_on=args.split_on, field_indices=args.field_indices,
                      subdirs=args.subdirs)
    model_params = {
        'c_in': len(args.fields),
        'l_in': args.seq_len,
        'n_out': 16,
        'c_outs': [64, 64],
        'dropout': 0.0,
        'use_batch_norm': True,
        'use_local_pooling': False, 
        'use_global_avg_pooling': True, 
        'remove_mean': True,
        'weight_decay': 0.1,
        'loss_mode': 'abs_triplet',         # not tested yet
    }
    
    c_outs_list = [[64, 64], [32, 64, 32], [16, 32, 32, 16]]

    # scan weight decay to control overfitting
    for wd in [0.2, 0.5, 1.0]:
       model_params['use_local_pooling'] = True
       model_params['use_global_avg_pooling'] = False
       model_params['weight_decay'] = wd
       print(f'- Start training w/ weight_decay = {wd}')
       train_test_model(patience, max_epochs, model_params)

    # scan kernel size
    for kernel_size in [5, 21]:
       model_params['kernel_size'] = kernel_size
       print(f'- Start training with kernel_size = {kernel_size}')
       train_test_model(patience, max_epochs, model_params)        
   
    # scan local pooling w/ arch
    for c_outs in c_outs_list:
       model_params['c_outs'] = c_outs
       model_params['use_local_pooling'] = True
       model_params['use_global_avg_pooling'] = False
       model_params['remove_mean'] = True
       print(f'- Start training with c_outs = {c_outs} & local pooling')
       train_test_model(patience, max_epochs, model_params)

    # scan no BN w/ GAP w/ arch
    for c_outs in c_outs_list:
        model_params['c_outs'] = c_outs
        model_params['use_batch_norm'] = False
        print(f'- Start training with c_outs = {c_outs} & no BN')
        train_test_model(patience, max_epochs, model_params)

    # scan no BN w/ local pooling
    for c_outs in c_outs_list:
        model_params['c_outs'] = c_outs
        model_params['use_batch_norm'] = False
        model_params['use_local_pooling'] = True
        model_params['use_global_avg_pooling'] = False
        print(f'- Start training with c_outs = {c_outs} & no BN w local pooling')
        train_test_model(patience, max_epochs, model_params)
    
    # scan remove_mean
    # for c_outs in c_outs_list:
    #    model_params['c_outs'] = c_outs
    #    model_params['remove_mean'] = False
    #    print(f'- Start training with c_outs = {c_outs} & local pooling')
    #    train_test_model(patience, max_epochs, model_params)

    cpipe.save_model()


def scan_dilated_cnn_params(args, cpipe):
    patience, max_epochs = 5, 4
    cpipe.global_load(args.tags, args.boundary, args.boundary2, args.batch_size,
                      split_on=args.split_on, field_indices=args.field_indices)
    n_field = len(args.fields) if args.field_indices is None else len(args.field_indices)
    model_params = best_model(n_field, args.seq_len)

    # scan weight, i.e. w * neg + (1 - w) * pos = loss
    t1 = time.time()
    print(f'- Start param scan at {t1:.3f} s')
    for w in [0.1, 0.3, 0.7, 0.9]:
        print(f'    - Scan w = {w}')
        model_params['w_loss'] = w
        train_test_model(patience, max_epochs, model_params)
        cpipe.save_model(f'models/dccnn_scan_w={w}.pt', params=model_params)
    t2 = time.time()
    print(f'- Finish param scan at {t2:.3f} s, total time = {t2 - t1:.3f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_len', '-s', type=int, default=300,
                        help='the length of the each sequence')
    parser.add_argument('--fields', '-fi', type=str, nargs='+',
                        default=['owd', 'rtt', 'slr', 'cwnd'],
                        help='fields of the signals, affects detection usage')
    parser.add_argument('--mode', '-m', type=str, required=True,
                        choices=['local', 'multi-local', 'multi-local-pred',
                        'multi-local-test', 'multi-local-real',
                        'global', 'global-tf', 'global-dc',
                        'scan', 'scan-dc', 'multi-detect', 'multi-detect-update'],
                        help='the mode of the pipeline: local for local process'
                            'global for load/train/test after preprocessing,'
                            'global-dc, global-tf: same mode for dc CNN and tf,'
                            'multi-local-xxx for multiprocess for folders w/ different mode,'
                            'scan/scan-dc for scan FCN/dcCNN params,'
                            'multi-detect for multiprocess for detections'
                            'multi-detect-update for update ml rather overwrite all')
    parser.add_argument('--cache_root', '-cr', type=str,
                        default='/home/sapphire/neuralforecast/my_src/pre_cache',
                        help='the root folder for all caches')
    parser.add_argument('--n_proc', '-np', type=int, default=4,
                        help='the number of processes for multi-local')
    local_grp = parser.add_argument_group('local')
    global_grp = parser.add_argument_group('global')
    detect_grp = parser.add_argument_group('detect')
    local_grp.add_argument('--runs', '-r', type=int, nargs=2,
                           help='[run_start, run_end) to be processed')
    local_grp.add_argument('--config_run_base', '-crb', type=int, default=0,
                           help='the base of config run number for btnk_df')
    local_grp.add_argument('--folder', '-f', type=str, nargs='+',
                           help='the folder to process')
    local_grp.add_argument('--cache_folder', '-c', type=str, nargs='+',
                           default=None,
                           help='the local cache folder name to process')
    # boundary arg name kept same for back compatibility
    global_grp.add_argument('--boundary', '-b', type=int, nargs='+',
                            help='the boundary1 of runs for train (val, test)')
    global_grp.add_argument('--boundary2', '-b2', type=int, nargs='+',
                            help='the boundary2 of runs for val/test')
    global_grp.add_argument('--batch_size', '-bs', type=int, default=64,
                            help='the batch size for train, val, test')
    global_grp.add_argument('--tags', '-t', type=str, nargs='+', default=None,
                            help='the tags for cache folder filtering')
    global_grp.add_argument('--subdirs', '-sd', type=str, nargs='+', default=None,
                            help='the subdirs after tag for cache folder')
    global_grp.add_argument('--n_worker', '-nw', type=int, default=1,
                            help='the number of workers for DataLoader')
    # global_grp.add_argument('--shuffle_run', '-sr', action='store_true', default=False,
    #                         help='whether to shuffle the runs')
    global_grp.add_argument('--split-on', type=str, default='sample',
                            choices=['run', 'sample'],
                            help='the way to split the training dataset')
    global_grp.add_argument('--field_indices', '-fidx', type=int, nargs='+', default=None,
                            help='the indices of fields to be used')
    detect_grp.add_argument('--detect_folder', '-df', type=str, nargs='+', default=None,
                            help='the cache folders to be read, support wildcard')
    detect_grp.add_argument('--detect_out_dir', '-do', type=str, nargs='+', default=None,
                            help='the output dir for detection results')
    detect_grp.add_argument('--model_path', '-mp', type=str,
                            default='dilated_cnn.pt',
                            help='the path to the model to be loaded/saved')
    detect_grp.add_argument('--ts', '-ts', type=float, nargs=2, default=[0, 50],
                            help='the time range to be detected')
    detect_grp.add_argument('--max_flow', '-mf', type=int, default=100000,
                            help='the max number of flows to be detected')
    detect_grp.add_argument('--model_th', type=float, default=0.2,
                            help='the threshold for general ml detection')
    detect_grp.add_argument('--model_th_csv', '-mt', type=str, default=None,
                            help='the csv file to store the model path & ths, used '
                                 'to provide multiple model configs for detection at once '
                                 'Rule 1: method should be the prefix of model tag; '
                                 'Rule 2: fields should be just skipped to get a None in code')
    detect_grp.add_argument('--methods', '-me', type=str, nargs='+',
                            default=['dcw', 'rmcat_sbd', 'dc_sbd', 'ml',
                                     'neg_naive', 'pos_naive'],
                            help='detection methods, if coexists with model_th, then first'
                            'the intersection is selected, then the rest of methods are used')
    detect_grp.add_argument('--n_flow_per_btnk', '-nf', type=int, default=None,
                            help='the number of flows per btnk in 3d scan')
    detect_grp.add_argument('--n_flows_per_btnk_to_draw', '-nfs', type=int, nargs='+',
                            default=[2, 4, 6, 8, 12, 14, 16],
                            help='the number of flows per btnk to draw')
    detect_grp.add_argument('--ths', type=float, nargs='+', default=None,
                            help='the thresholds to be scanned')

    args = parser.parse_args()

    cpipe = FinalCleanPipeline(args.seq_len, args.fields, args.cache_root)
    if args.field_indices is not None:
        assert max(args.field_indices) < len(args.fields)
    n_field = len(args.fields) if args.field_indices is None else len(args.field_indices)
    if args.mode == 'local':
        if args.cache_folder is None:
            assert len(args.folder) == 1
            args.folder = args.folder[0]
            args.cache_folder = os.path.basename(os.path.normpath(args.folder))
            print('cache folder name is set to be', args.cache_folder)    
        print('Deprecate for now: config_runs not added for new API.')
        # cpipe.local_process(args.folder, args.runs, args.cache_folder)
    elif args.mode == 'global':     # triplet loss: 0.4
        patience, max_epochs = 5, 20
        cpipe.global_load(args.tags, args.boundary, args.boundary2, args.batch_size,
                          split_on=args.split_on, field_indices=args.field_indices,
                          subdirs=args.subdirs)
        model_params = {
            'c_in': n_field,
            'l_in': args.seq_len,
            'n_out': 16,
            'c_outs': [64, 64],
            'dropout': 0.0,
            'use_batch_norm': False,
            'use_local_pooling': False, 
            'use_global_avg_pooling': True, 
            'remove_mean': True,
            'loss_mode': 'abs_triplet',
        }
        cpipe.set_model(**model_params)
        cpipe.train(patience, max_epochs)
        cpipe.clear_dataloader()
        # cpipe.test()
        cpipe.save_model(path=args.model_path, params=model_params)
    elif args.mode == 'global-dc':      # triplet loss: 0.12
        patience, max_epochs = 5, 10
        cpipe.global_load(args.tags, args.boundary, args.boundary2, args.batch_size,
                          split_on=args.split_on, field_indices=args.field_indices,
                          subdirs=args.subdirs, n_worker=args.n_worker)
        model_params = best_model(n_field, args.seq_len)
        cpipe.set_model(**model_params)
        cpipe.train(patience, max_epochs)
        # cpipe.clear_dataloader()
        cpipe.test()
        cpipe.save_model(path=args.model_path, params=model_params)
    elif args.mode == 'global-tf':
        patience, max_epochs = 3, 7
        cpipe.global_load(args.tags, args.boundary, args.boundary2, args.batch_size,
                          split_on=args.split_on, field_indices=args.field_indices,
                          subdirs=args.subdirs)
        cpipe.set_model(model_type='tf')
        cpipe.train(patience, max_epochs)
        cpipe.test()
        cpipe.save_model()
    elif 'multi-local' in args.mode:
        assert args.folder
        folders = args.folder
        if folders is not None and '*' in args.folder[0]:
            folders = glob.glob(args.folder[0])
            print('* for multi-local folder:', folders)
        cache_folders = args.cache_folder
        if cache_folders is not None and '*' in args.cache_folder[0]:
            cache_folders = glob.glob(args.cache_folder[0])
            print('* for multi-local cache folder:', cache_folders)
        for_prediction = 'pred' in args.mode
        prep_mode = PrepMode.PREDICT if 'pred' in args.mode else \
                    PrepMode.TEST if 'test' in args.mode else \
                    PrepMode.REAL if 'real' in args.mode else PrepMode.TRAIN
        subdir = None if 'real' in args.mode else 'dats'
        arg_list = create_arg_list_for_preprocess(args.seq_len, args.fields,
                    args.cache_root, folders, cache_folders, prep_mode=prep_mode,
                    config_run_base=args.config_run_base, subdir=subdir)
        run_multiprocess(run_local_process, arg_list, n_proc=args.n_proc)
    elif args.mode == 'scan':
        scan_fcn_params(args, cpipe)
    elif args.mode == 'scan-dc':
        scan_dilated_cnn_params(args, cpipe)
    elif 'multi-detect' in args.mode:
        do_update = 'update' in args.mode
        assert args.detect_folder
        detect_folders = args.detect_folder
        # TODO: possibly not needed as globbed automatically
        if '*' in args.detect_folder[0]:
            detect_folders = glob.glob(args.detect_folder[0])
            print('* for multi-detect folder:', detect_folders)
        detect_out_dirs = args.detect_out_dir
        if detect_out_dirs is None:
            detect_out_dirs = [None] * len(detect_folders)
        elif '*' in args.detect_out_dir[0]:
            detect_out_dirs = glob.glob(args.detect_out_dir[0])
            print('* for multi-detect out dir:', detect_out_dirs)
        model_th_list = None
        if args.model_th_csv is not None:
            model_th_df = pd.read_csv(args.model_th_csv, index_col=False,
                                      comment='#')
            model_th_list = model_th_df[['tag', 'path', 'th']].to_numpy()
            model_th_list = [tuple(row) for row in model_th_list]
        arg_list = create_arg_list_for_detect(args.seq_len, args.fields,
                    args.cache_root, args.model_path, args.ts,
                    detect_folders, detect_out_dirs,
                    args.max_flow, args.methods, do_update, args.model_th,
                    model_th_list,
                    args.n_flow_per_btnk,
                    args.n_flows_per_btnk_to_draw, args.ths)
        run_multiprocess(run_local_detect, arg_list, n_proc=args.n_proc)


# $ python3 final_clean_pipeline.py -m multi-local -f ~/NS_Simulation_Toolkit/BBR_test/ns-3.27/dataset_round1/results_cb-tleft-small-bkup_Dec-28-22\:40\:3560/
# $ python3 final_clean_pipeline.py -m global -b 0 5 8 10 
# $ python3 final_clean_pipeline.py -m global -b 0 5 8 10 -t run --split-on none
# $ python3 final_clean_pipeline.py -m multi-detect -np 1 -mp final_model_cnn.pt -ts 15 35 -df pre_cache/results_cbtest-large-flow_tiny
# 1.17.2023
# $ python3 final_clean_pipeline.py -m multi-local-pred -f /home/sapphire/NS_Simulation_Toolkit/BBR_test/ns-3.27/dataset_round2/round2_test/results_cbtest-*Dec-28* -cr /home/sapphire/neuralforecast/my_src/pre_cache/ -np 2
# $ python3 final_clean_pipeline.py -m multi-detect -np 3 -mp final_model_cnn.pt -ts 15 35 -df pre_cache/results_cbtest-large-flow_Dec-28-18\:05\:5486/
# $ python3 final_clean_pipeline.py -m global-dc --split-on sample -fidx 0 1 2 -b 0 1 -b2 0 1 2 -t train_data/eva* test_data/eva* -sd prep_train -cr ~/detector_pipeline/ -s 140 -mp ~/detector_pipeline/model/dccnn_s140_fi012_w55.pt
