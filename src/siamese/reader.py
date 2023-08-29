import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess as sp

class ConfigReader:
    """This tool checks the ground truth of the data by reading
    the config and data files. Runs [start, end) can be given to
    specify which runs to read.
    """
    def __init__(self, root, runs=None) -> None:
        self.root = root
        self.runs = None if runs is None else list(range(runs[0], runs[1]))
        self.cur_dir = os.getcwd()
        self._parse_config()
        print(f'ConfigReader: runs: {self.runs}')
    
    def _parse_config(self, typs=['link', 'flow', 'cross']):
        os.chdir(self.root)
        if "No such file" in sp.getoutput('ls *inflated*'):
            os.chdir('cfgs')
        self.config = {k: None for k in typs}
        for typ in typs:
            inflated = sp.getoutput(f'ls *{typ}_inflated*')
            self.config[typ] = pd.read_csv(inflated, index_col=False)
            if self.runs is None:
                continue
            n_run = self.config[typ].run.max() + 1
            self.runs = list(map(lambda x: x % n_run, self.runs))
            for run in self.runs:
                assert run in self.config[typ].run.unique()
            self.config[typ] = self.config[typ][self.config[typ].run.isin(
                self.runs)].copy()
        os.chdir(self.cur_dir)

    def get_btnk_info(self):
        """Process link & cross config to get btnk link info, i.e.
        [run, src, dst, ...] + [qid, bw_avail]
        """
        ldf = self.config['link']
        btnk_ldf = ldf[ldf.position.str.contains('_mid')].copy()
        res_df = None
        for run in btnk_ldf.run.unique():
            run_df = btnk_ldf[btnk_ldf.run == run].copy()
            run_df = run_df.sort_values(by=['run', 'src', 'dst']).reset_index(
                drop=True)
            indices = run_df[run_df.q_monitor == 'tx'].index.tolist()

            # 1) use index as the qid
            # qids = list(map(lambda x : indices.index(x), indices))
            # run_df.loc[indices, 'qid'] = qids

            # 2) use the 4 digit str (src, dst) as the qid
            run_df['qid'] = run_df.apply(lambda r: str(r.src).zfill(2) +
                                         str(r.dst).zfill(2), axis=1)

            res_df = run_df if res_df is None else pd.concat([res_df, run_df],
                                                             ignore_index=True)
        assert res_df.qid.isna().sum() == 0

        cdf = self.config['cross'].copy()
        df = res_df.merge(cdf, on=['run', 'src', 'dst'])
        df['bw_avail_mbps'] = df.bw_mbps * (1.0 - 
                              np.minimum(1.0, df.num * df.cross_bw_ratio))
        df = df.drop(columns=['mode', 'type', 'edge_rate_mbps', 'hurst',
                              'mean_duration'])
        return df if self.runs is None else df[df.run.isin(self.runs)]
        # return df
    
    def get_agg_flow_info(self):
        """Get aggregated flow rate over possible btnk links.
        """
        fdf = self.config['flow'].copy()
        fdf['total_rate_mbps'] = fdf.rate_mbps * fdf.num
        fdf['total_rate_mbps'] = fdf.groupby(['run', 'src_gw', 'dst_gw'])\
            ['total_rate_mbps'].transform('sum')
        fdf['num'] = fdf.groupby(['run', 'src_gw', 'dst_gw'])['num'].transform(
            'sum')
        fdf = fdf.drop_duplicates(subset=['run', 'src_gw', 'dst_gw'],
                                  ignore_index=True)
        fdf = fdf.drop(columns=['src', 'dst', 'delayed_ack', 'rate_mbps',
                                'start', 'end'])
        return fdf
        

    def check_by_config(self):
        """Compare if the total flow rate is over the available
        bandwidth of the btnk links.
        TODO: Seems need some optimization algorithm or iterative
        approach, first checked by eye.
        """
        btnk_info = self.get_btnk_info()
        flow_info = self.get_agg_flow_info()
        pass

class DataReader:
    """Reads the signals data, queue data from the data folder, and handle basic
    data processing.

    Dataframe are in the format below:

    xdf: [(run), flow, time, owd, rtt, ...]
    qid_df: [(run), flow, qid1, qid2]
    qdf: [(run), qid, time, packet_in_queue, max_q_size]

    xdf and qid_df can be joined on [run, flow],
    qdf and qid_df can be joined on [run, qid].
    """
    def __init__(self, folder, order_check=False, runs=None) -> None:
        self.folder = folder
        self.ids = None
        self.qids = None
        self.qruns = None
        self.order_check = order_check
        self.runs = None if runs is None else runs
        self.cur_dir = os.getcwd()

    def _get_ids_from_subdir(self, subdir, tag):
        """Get all the ids by parsing the file names."""
        if subdir is None:
            subfolder = self.folder
        else:
            subfolder = os.path.join(self.folder, subdir)
        assert os.path.isdir(subfolder)
        os.chdir(subfolder)
        ids = sp.getoutput(f'ls *{tag}*').split('\n')
        if 'No such file or directory' in ids[0]:
            print(f'Warning: no {tag} files in {subfolder}')
            return []
        ids = list(map(lambda x : x.split('_')[-1][:-4], ids))
        if self.order_check:
            print('ids from subdir: ', ids)
            c = input('Is the order from ls correct? (y/n)')
            if c == 'n':
                exit(0)
        os.chdir(self.cur_dir)
        return ids
    
    def get_missing_runs(self):
        """Get the missing runs, i.e. the No. of the runs in the original
        config csvs.
        
        Here we decide the missing runs as those exist in logs but not in
        dats.

        TODO: empty all-data csv?"""
        log_ids = self._get_ids_from_subdir('logs', 'log_debug')
        dat_ids = self._get_ids_from_subdir('dats', 'all-data')
        missing_runs = list(set(log_ids) - set(dat_ids))
        missing_ids = list(map(lambda x : log_ids.index(x), missing_runs))
        return missing_runs, missing_ids

    def get_qids(self, runs):
        """Get the qids and qruns from the queue csvs."""
        ids = self.get_ids(runs)
        if self.qids is None or self.qruns is None:
            self.qids, self.qruns = [], []
            for id in ids:
                qids = self._get_ids_from_subdir('dats', f'queue_{id}')
                if not qids:
                    continue
                self.qids += qids
                self.qruns += [int(id)] * len(qids)
        return self.qids, self.qruns

        # [To deprecate]
        # if self.qids is None or self.qruns is None:    
        #     self.qids = self._get_ids_from_subdir('dats', 'queue')
        #     self.qruns = [int(str(x)[:5]) for x in self.qids]
        #     # use absolute run now
        #     # qruns = sorted(list(set(self.qruns)))
        #     # self.qruns = list(map(lambda x: qruns.index(x), self.qruns))
        # if runs is None:
        #     return self.qids, self.qruns
        
        # unique_runs = sorted(list(set(self.qruns)))
        # runs = set(unique_runs[runs[0] : runs[1]])
        # qids, qruns = [], []
        # for qid, qrun in zip(self.qids, self.qruns):
        #     if qrun in runs:
        #         qids.append(qid)
        #         qruns.append(qrun)
        # return qids, qruns

    def get_ids(self, runs, subdir='logs'):
        if self.ids is None:
            if subdir == 'dats':
                self.ids = self._get_ids_from_subdir(subdir, 'all-data')
            elif subdir == 'logs':
                self.ids = self._get_ids_from_subdir(subdir, 'log_debug')
        if runs is None:
            return self.ids
        return self.ids[runs[0] : runs[1]]

    def get_data_df(self, subdir='dats'):
        """Get data df / xdf from all-data csv files."""
        xdf = None
        ids = self.get_ids(self.runs)
        print('data reader ids:', ids)
        for i, id in enumerate(ids):
            if not subdir:
                data_csv = os.path.join(self.folder, f'all-data_{id}.csv')
            else:
                data_csv = os.path.join(self.folder, subdir, f'all-data_{id}.csv')
            if os.path.exists(data_csv):
                df = pd.read_csv(data_csv, index_col=False)
                df['run'] = int(id)
                xdf = df if xdf is None else pd.concat([xdf, df], ignore_index=True)
                xdf = xdf.sort_values(by=['run', 'flow', 'time']).reset_index(drop=True)
            else:
                print(f' Warning: missing {data_csv}')
        return xdf

    def get_qid_df(self):
        """Get qid_df [run, flow, qid1, qid2] from toc files."""
        ids = self.get_ids(self.runs)
        qid_df = None
        for i, id in enumerate(ids):
            toc_csv = os.path.join(self.folder, 'dats', f'toc_{id}.csv')
            if os.path.exists(toc_csv):
                toc_df = pd.read_csv(toc_csv, index_col=False)
                toc_df = self._parse_toc_df(toc_df)
                toc_df['run'] = int(id)
                qid_df = toc_df if qid_df is None else pd.concat([qid_df, toc_df],
                                                                ignore_index=True)
            else:
                print(f' Warning: missing {toc_csv}')
        return qid_df.drop(columns=['qid'])

    def get_truth_df(self):
        ids = self.get_ids(self.runs)
        truth_df = None
        for i, id in enumerate(ids):
            truth_csv = os.path.join(self.folder, f'truth_{id}.csv')
            df = pd.read_csv(truth_csv, index_col=False)
            df['run'] = int(id)
            truth_df = df if truth_df is None else pd.concat([truth_df, df],
                                                             ignore_index=True)
        truth_df = truth_df.sort_values(by=['run', 'flow']).reset_index(drop=True)
        return truth_df

    def _parse_toc_df(self, toc_df):
        """Expand each row of toc_df to qids."""
        def parse_qid(row):
            qids = eval(row.qid.replace(' ', ','))
            row['qid1'], row['qid2'] = str(qids[0])[-4:], str(qids[1])[-4:]
            return row

        toc_df = toc_df.apply(parse_qid, axis=1)
        return toc_df

    def get_queue_df(self):
        """Get queue df / qdf [run, qid, time, ...] from queue csv files."""
        qdf = None
        qids, runs = self.get_qids(self.runs)
        for qid, run in zip(qids, runs):
            queue_csv = os.path.join(self.folder, 'dats', f'queue_{qid}.csv')
            if os.path.exists(queue_csv):
                df = pd.read_csv(queue_csv, index_col=False)
                df['qid'] = str(qid)[-4:]
                df['run'] = run
                qdf = df if qdf is None else pd.concat([qdf, df], ignore_index=True)
            else:
                print(f' Warning: missing {queue_csv}')
        qdf = qdf.sort_values(by=['run', 'qid', 'time']).reset_index(drop=True)
        return qdf


class MergedDataReader(DataReader):
    """Reads the config and data from merged dataset folder.
    
    Delegation for ConfigReader for each cfg subfolder, and inheritation
    as current folder architecture can be seen as same as data reader.
    """
    def __init__(self, root):
        self.original_dir = os.getcwd()
        super().__init__(root, order_check=False)
        
    
    """Steps
    1) get all valid ids using self.get_ids(), which is from dats
    2) parse merged_run.json to get run list per sub-folder
    3) for each sub-folder, apply run -> run_id using previous run list values()[i]
       and then concat all config dfs
    4) filter all df using valid ids: df[df['run'].isin(valid_ids)]
    5) replace with the valid index, assert it's the same # as valid ids

    Note: deprecate the missing runs part in preprocess.
    """




def test_config_reader():
    root = '/home/sapphire/NS_Simulation_Toolkit/BBR_test/ns-3.27'
    # folder = 'adam_dbg_1208'
    folder = 'adam_dbg_cluster_1212'
    folder = os.path.join(root, folder)
    tc = ConfigReader(folder)
    df = tc.get_btnk_info()
    fdf = tc.get_agg_flow_info()
    print('Bottleneck info:\n', df)
    print('Aggregated flow info:\n', fdf)

def test_data_reader():
    root = '/home/sapphire/NS_Simulation_Toolkit/BBR_test/ns-3.27'
    folder = 'adam_newdbg1_1212'
    folder = os.path.join(root, folder)
    dr = DataReader(folder)
    xdf = dr.get_data_df()
    qid_df = dr.get_qid_df()
    qdf = dr.get_queue_df()
    print('xdf:\n', xdf)
    print('qid_df:\n', qid_df)
    print('qdf:\n', qdf)

if __name__ == '__main__':
    test_config_reader()