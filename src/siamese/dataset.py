import os
import numpy as np
import pandas as pd
import subprocess as sp
# import modin.pandas as pd
import random

import torch
from torch.utils.data import dataset, DataLoader, Dataset, IterableDataset

import unittest

from .preprocess import DataChecker, DataModifier, DataGetter, DataPreprocessor
from .indexed_sample_io import IndexedSampleIO, convert_segment_to_array


class SiameseDataset(Dataset):
    """Dataset for Siamese model training. The dataframe here is splitted by
    run and interval, and each sample consists of all flows' signals within
    an interval, i.e. training sample [n_flow, seq_len, n_feature_in], ground
    truth [n_flow].

    TODO: outdated due to truth changes, not tested
    """
    
    def __init__(self, df, truth_df, seq_len, fields):
        super().__init__()
        self.seq_len = seq_len
        basic_fields = ['run', 'time', 'flow']
        for field in  basic_fields + fields:
            assert field in df.columns, f"Field {field} not in dataframe."
        self.df = df[basic_fields + fields].copy()
        self.df = self.df.sort_values(['run', 'flow', 'time']).reset_index(drop=True)
        self.fields = fields
        # offset = 0
        # # TODO: need test with multiple run and flow
        # for i in self.df.run.unique():
        #     tmp = self.df[self.df.run == i]
        #     for j in self.df.flow.unique():
        #         cnt = len(tmp[tmp.flow == j])
        #         idx = offset + cnt + np.linspace(-(cnt % seq_len), -1, cnt % seq_len)
        #         self.df = self.df.drop(idx).reset_index(drop=True)
        #         offset += cnt - cnt % seq_len
        self.df = self.df.sort_values(['run', 'time', 'flow']).reset_index(drop=True)
        self.sample_idx = self.df[['run', 'time']].drop_duplicates(['run', 'time'],
                                                                                                                                ignore_index=True)
        self.n_sample = int(len(self.sample_idx) / seq_len)
        self.truth_df = truth_df

    def get_df(self):
        return self.df
    
    def __len__(self):
        return self.n_sample
    
    def __getitem__(self, i):
        """Returns i-th sample in [n_flow, seq_len, n_feature_in] and its ground truth.
        TODO: not tested after using DataModifier
        """
        sample_start = self.sample_idx.iloc[i * self.seq_len]
        sample_end = self.sample_idx.iloc[(i + 1) * self.seq_len - 1]
        run, t_start, t_end = sample_start.run, sample_start.time, sample_end.time
        sample = self.df[(self.df.run == run) & (self.df.time >= t_start) & (self.df.time <= t_end)]
        sample = sample.sort_values(['flow', 'time']).reset_index(drop=True)
        # SiameseDataset is solely used for prediction now,
        # currently we use flow padding for prediction, so no cropping here
        # sample = DataModifier.crop_sample_for_flow_change(sample)
        # consistent_flows = sample.flow.unique()

        x = sample[self.fields].to_numpy(dtype=np.float32)
        x = x.reshape(sample.flow.nunique(), self.seq_len, len(self.fields))

        # TODO: below not tested!
        y = self.truth_df[self.truth_df.run == run]
        if 'time' in self.truth_df.columns:
            y = DataGetter.fetch_truth_for_segment(y, t_start, t_end)
        # y = y[y.flow.isin(consistent_flows)]

        y = y['label'].to_numpy(dtype=np.float32).reshape(
            (y.flow.nunique(),))
        return x, y


class IndexedTripletDataset(Dataset):
    """Indexed triplet dataset based on indexed sample I/O. Decouple
    the complexity of triplet indicies w/ the sample data s.t. more
    samples can be generated w/o increasing the space.

    We support selecting a subset of fields to use for training, e.g.
    only OWD and RTT by using field_indices = [0, 1].
    
    Data structures:
        - sample_data: dict, sample_data[run][ti][flow] = sample_array
        - index_list: list of triplet indices, i.e. [run, ti, anchor, pos, neg]
    """
    def __init__(self, sample_data, index_list, field_indices=None):
        """field_indices_to_use: list of indices of fields to use, e.g. original
        data has 4 fields [owd, rtt, slr, cwnd], then field_indices_to_use = [0, 1]
        will fetch only [owd, rtt] for train/test."""
        self.sample_data = sample_data
        self.index_list = index_list
        self.field_indices = field_indices

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, i):
        idx = self.index_list[i]
        run, ti, anchor, pos, neg = tuple(idx)
        arrays = [self.sample_data[run][ti][flow] for flow in [anchor, pos, neg]]
        if self.field_indices is None:
            return tuple(arrays)
        else:
            return tuple([array[:, self.field_indices] for array in arrays])


class SiameseTripletDataset(SiameseDataset):
    """Dataset for triplet loss, i.e. return (anchor, positive, negative) for
    each sample. Here positive/negative samples are randomly selected from
    the same run.
    """
    def __init__(self, samples):
        # samples: list of samples, make it extreme simple
        #    offload the processing outside
        self.samples = samples
        self.n_sample = len(samples)

    def __len__(self):
        return self.n_sample

    # def get_label_type_df(self):
        # return self.label_type_df

    def __getitem__(self, i):
        return self.samples[i][0], self.samples[i][1], self.samples[i][2]


class SiameseFlowTripletDataset(SiameseTripletDataset):
    """Dataset for sequential model like RNN which requires one sample
    to be the whole time series of a flow. Thus, compared to triplet
    dataset, here we output three flows in __getitem__ each time, and
    the later processing is offloaded to the trainer for the sequential
    training customization.
    """
    def __init__(self, df, truth_df, seq_len, fields):
        super().__init__(df, truth_df, seq_len, fields)
        # we need to iterate over all flows to get the index to get
        self.samples = []        # store (run, anchor, pos, neg)
        for run in self.df.run.unique():
            tdf = self.truth_df[self.truth_df.run == run].copy()
            label_to_flow = {label: list(tdf.loc[tdf.label == label, 'flow'].unique()) 
                            for label in tdf.label.unique()}
            df_run = self.df[self.df.run == run]
            for label, flows in label_to_flow.items():
                for i in range(len(flows) - 1):
                    for j in range(i, len(flows)):
                        # iterate over all (anchor, positive) pairs, but sample
                        # the negative entry 
                        anchor, pos = flows[i], flows[j]
                        neg_candidates = tdf[(tdf.label != label)].flow
                        neg = np.random.choice(neg_candidates)

                        # iterate over time to get segments for the flow pair                                                
                        for k in range(int(self.df.time.nunique() // self.seq_len)):
                            t0 = df_run.time.unique()[k * self.seq_len]
                            t1 = df_run.time.unique()[(k + 1) * self.seq_len - 1]
                            df_seg = df_run[(df_run.time >= t0) & (df_run.time <= t1)]
                            is_flow_start = k == 0
                            res = []
                            for flow in [anchor, pos, neg]:
                                segment = df_seg[df_seg.flow == flow][self.fields].to_numpy(
                                        dtype=np.float32)
                                assert segment.size == self.seq_len * len(self.fields), \
                                    f"Wrong segment size {len(segment)} for flow {flow}"
                                segment = segment.reshape(self.seq_len, len(self.fields))
                                res.append(segment)
                            self.samples.append((res, is_flow_start,
                                [run, anchor, pos, neg]))

        # TODO: reimplement the per sample by append all samples
        #       Note to add the indicator for the first sample of a flow to
        #       help _step() to refersh the last_y 

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, i):
        res, is_flow_start, metadata = self.samples[i]
        return res[0], res[1], res[2], is_flow_start, metadata

class SiameseClusterPairDataset(IterableDataset):
    """Iterable dataset support for cluster triplet loss, next() returns 
    next unvisited cluster pair.
    
    Note that batch_size > 1 is not supported by current implementation,
    as n_flow of different clusters may be different and thus make adjacent
    samples unable to be stacked.
    """
    def __init__(self, df, truth_df, seq_len, fields):
        super().__init__()
        self.seq_len = seq_len
        basic_fields = ['run', 'time', 'flow']
        for field in  basic_fields + fields:
            assert field in df.columns, f"Field {field} not in dataframe."
        self.df = df[basic_fields + fields].copy()
        self.df = self.df.sort_values(['run', 'flow', 'time']).reset_index(drop=True)
        self.fields = fields
        offset = 0
        for i in self.df.run.unique():
                tmp = self.df[self.df.run == i]
                for j in self.df.flow.unique():
                        cnt = len(tmp[tmp.flow == j])
                        idx = offset + cnt + np.linspace(-(cnt % seq_len), -1, cnt % seq_len)
                        self.df = self.df.drop(idx).reset_index(drop=True)
                        offset += cnt - cnt % seq_len
        self.df = self.df.sort_values(['run', 'time', 'flow']).reset_index(drop=True)
        self.sample_idx = self.df[['run', 'time']].drop_duplicates(['run', 'time'],
                                                                                                                                ignore_index=True)
        self.n_sample = int(len(self.sample_idx) / seq_len)
        self.truth_df = truth_df
    
    def __iter__(self):
        self.i_run, self.i_seq, self.j, self.k = 0, 0, 0, 1
        return self

    def __next__(self):
        """Generates the next cluster pair (run, i_seq, i, j) by plusing one.
        Returns clusters of run with labels of labels[i] and labels[j].
        """
        if self.i_run == self.df.run.nunique():
            raise StopIteration

        cur_run = self.df.run.unique()[self.i_run]
        run_df = self.df[self.df.run == cur_run]
        run_df = run_df.sort_values(['time', 'flow']).reset_index(drop=True)
        n_flow = run_df.flow.nunique()
        cur_df = run_df[self.i_seq * self.seq_len * n_flow :
                                        (self.i_seq + 1) * self.seq_len * n_flow].copy()
        tdf_run = self.truth_df[self.truth_df.run == cur_run]
        labels = tdf_run.label.unique()
        res = []
        for ci in [self.j, self.k]:
            flows = tdf_run[tdf_run.label == labels[ci]].flow.unique()
            cluster = cur_df[cur_df.flow.isin(flows)].sort_values(['flow', 'time'])
            cluster = cluster.reset_index(drop=True)
            val = cluster[self.fields].to_numpy(dtype=np.float32)
            val = val.reshape(len(flows), self.seq_len, len(self.fields))
            res.append(val)

        info = [cur_run, self.i_seq, self.j, self.k]

        self.k += 1
        if self.k == len(labels):
            self.j += 1
            if self.j == len(labels) - 1:
                self.j = 0
                self.i_seq += 1
                if self.i_seq == run_df.time.nunique() // self.seq_len:
                    self.i_seq = 0
                    self.i_run += 1
            self.k = self.j + 1

        return res[0], res[1], info


class CleanDataLoader:
    """Latest data loader with the support of reader and preprocess.

    Now it only takes in the clean df from preprocess and then construct
    the dataset, split it, and output train/test/val dataloaders. This
    is main version handling the large simulated dataset from cluster
    for Atc paper.
    """
    def __init__(self, seq_len, fields, xdf=None, truth_df=None,
                 field_indices=None) -> None:
        """Initialize the data loader.

        Use field_indices to specify the indices of fields to be used,
        e.g. [0, 1] to use the first two fields (OWD & RTT).
        """
        self.seq_len = seq_len
        self.fields = fields
        if xdf is not None:
            self.xdf = xdf
        if truth_df is not None:
            self.truth_df = truth_df
        self.loaders = []
        self.datasets = []
        self.i_test = 2   # order of testset in loaders, will change if train/val are deleted
        self.isio = IndexedSampleIO(fields, seq_len)
        self.field_indices = field_indices
        if field_indices is not None:
            print(f'[CleanDataLoader]: field_indices = {field_indices}')

    def load(self, cache_folder, tag=None, subdir=None):
        """Load samples from cache_folder/*tag*/subdir using IndexedSampleIO.
        Returns the triplet indices loaded for later split.
        E.g. train1_*/prep_train, test1_*/prep_train, can support the
        automation as well."""
        cwd = os.getcwd()
        if not os.path.exists(cache_folder):
            os.makedirs(cache_folder)
        os.chdir(cache_folder)
        tag = '' if tag is None else tag
        dirs = sp.getoutput(f'ls -d *{tag}*/').split('\n')
        dirs = [d[:-1] if subdir is None else os.path.join(d[:-1], subdir)
                for d in dirs]
        cur_index = self.isio.load(dirs)
        os.chdir(cwd)
        return cur_index

    def split(self, triplet_index, boundary, batch_size, split_on, n_worker=1,
        shuffle_first=False):
        """Split the triplet indices by boundary, and construct data loaders and
        append to the current loaders list.
        
        Args:
            triplet_index (dict): dict of all triplets' indices {run: indices}
            boundary (list): boundaries of the dataset split, its length
                    should be n_split + 1, e.g. 4 for train/val/test split.
            batch_size (int): size of the batch
            split_on (str): the field to split on
            n_worker (int, optional): number of workers for dataloader. Default 1.
            shuffle_first (bool, optional): if shuffle the first loader. Default False.
        """
        do_shuffle = True      # if shuffle the triplets
        index_lists = self.isio.split(triplet_index, boundary, do_shuffle, split_on)
        for i, index_list in enumerate(index_lists):
            # TODO: init the dataset
            dataset1 = IndexedTripletDataset(self.isio.get_sample_data(), index_list,
                                             field_indices=self.field_indices)
            self.datasets.append(dataset1)
            loader = DataLoader(dataset1, batch_size=batch_size,
                                shuffle=(i == 1) and shuffle_first, num_workers=n_worker,
                                pin_memory=False,
                                persistent_workers=True)
            self.loaders.append(loader)
            # must ensure they are using the same sample_data instance for memory!
            assert id(loader.dataset.sample_data) == id(self.isio.get_sample_data())

    # TODO: deprecate
    # def load_all(self, cache_folder, tag=None, shuffle_run=False):
    #     """Load all samples under cache_folder, and combine into unified
    #     dfs (xdf, truth_df, info_df, etc).
        
    #     The folder architecture is cache_folder/[folder]/xxx_df_[run0-run1].csv.

    #     Args:
    #         cache_folder (str): cache folder to scan.
    #     """
    #     cwd = os.getcwd()
    #     if not os.path.exists(cache_folder):
    #         os.makedirs(cache_folder)
    #     os.chdir(cache_folder)
    #     tag = '' if tag is None else tag
    #     dirs = sp.getoutput(f'ls -d *{tag}*/').split('\n')
    #     dirs = [d[:-1] for d in dirs]

    #     # just load the samples
    #     all_samples = {}
    #     for d in dirs:
    #         print(f'Loaded samples from {d}')
    #         os.chdir(d)
    #         pts = sp.getoutput('ls sample*.pt').split('\n')
    #         for pt in pts:
    #             print(' ', pt)
    #             samples = torch.load(pt)
    #             all_samples.update(samples)
    #         os.chdir('..')
    #     new_keys = sorted(all_samples.keys())
    #     if shuffle_run:
    #         # sorted runs, and make it a list
    #         random.shuffle(new_keys)
    #         print('  Shuffled keys: ', new_keys)
    #     all_samples = [all_samples[k] for k in new_keys]
    #     os.chdir(cwd)
    #     return all_samples

    def _translate_runs(self, df):
        runs = sorted(df.run.unique())
        run_map = {runs[i]: i for i in range(len(runs))}
        df['run'] = df.run.map(run_map)

    # def split(self, all_samples, boundary, batch_size, split_on, n_worker=1,
    #     shuffle_first=False, multi_test_loader=False):
    #     """Split the given samples into datasets based on boundary, and then
    #     append the data loaders to self.loaders. Note that specific
    #     order of the dataset is not determined here but by the final order
    #     of self.loaders.

    #     Currently, we keep the sample shuffling in 'none' mode, as it does not
    #     hurt when the all_samples of only one folder is passed to.

    #     TODO: currently only support SiameseTripletDataset

    #     Args:
    #         all_samples (dict): dict of all samples {run: [samples]}
    #         boundary (list): boundaries of the dataset split, its length
    #                 should be n_split + 1, e.g. 4 for train/val/test split.
    #         batch_size (int): size of the batch
    #         split_on (str): the field to split on
    #         n_worker (int): number of workers for dataloader
    #     """
    #     # DataChecker.check_time_alignment(self.xdf, self.truth_df)
    #     # mid = ['train', 'val', 'test']
    #     # assert len(boundary) == len(mid) + 1
    #     n_total = len(all_samples)
    #     print('- CleanDataLoader: given boundary: ', boundary)
    #     print(f'        total runs: {n_total}, batch size: {batch_size}, '
    #           f'split on: {split_on}')

    #     if split_on == 'none':
    #         shuffled_samples = sum(all_samples, [])
    #         random.shuffle(shuffled_samples)
    #         boundary = list(map(lambda x: int(x / max(boundary) * len(shuffled_samples)),
    #             boundary))
    #         print(f'        Shuffled {len(shuffled_samples)} samples ...')
    #     else:
    #         boundary = list(map(lambda x: int(x / max(boundary) * n_total),
    #             boundary))
    #     print(f'        derived boundary: {boundary}')

    #     # TODO: design here is: split the indices, and then use (index, sample_data)
    #     #       to construct the datasets.

    #     for i, _ in enumerate(boundary):
    #         if i == 0:
    #             continue
    #         # run_samples: [ [run 0: [sample 0, sample 1, ...]], run 1: ... ]
    #         if split_on == 'none':
    #             samples = shuffled_samples[boundary[i-1]:boundary[i]]
    #         else:
    #             run_samples = all_samples[boundary[i-1]:boundary[i]]
    #             samples = sum(run_samples, [])
    #         self.dataset = SiameseTripletDataset(samples)
    #         # enable multiple workers only when test, as we can release memory
    #         # in train/val then
    #         n_test_worker = 12 if multi_test_loader else 1
    #         n_worker = n_worker if n_worker > 1 else n_test_worker if len(self.loaders) > 1 else 1
    #         loader = DataLoader(self.dataset, batch_size=batch_size,
    #                             shuffle=(i == 1) and shuffle_first,
    #                             num_workers=n_worker)
    #         self.loaders.append(loader)

    def smooth(self, f_smooth):
        """Smooth the time series w/ given function."""
        if not f_smooth:
            return
        for run in self.df.run.unique():
            for flow in self.df.flow.unique():
                idx = (self.df.run == run) & (self.df.flow == flow)
                self.df.loc[idx, 'owd'] = f_smooth(self.df.loc[idx, 'owd'])

    def get_train_loader(self):
        return self.loaders[0]
    
    def get_val_loader(self):
        return self.loaders[1]
    
    def get_test_loaders(self):
        return self.loaders[self.i_test:]


class SiameseNetDataLoader:
    """Net data loader for Siamese model training.
    """
    def __init__(self, seq_len, fields) -> None:
        self.seq_len = seq_len
        self.fields = fields

    def load(self, root, folder_to_ids, labels=None, no_dats=False,
             caches=[]):
        """Loads data from given runs of each folder. If caches are specified,
        then do cacheing, i.e. write cache if there's none, or read cache.

        Args:
            root (str): Root folder of data.
            folder_to_ids (dict): {folder: ids} of selected runs' data.
            labels (dataframe, optional): labels of selected runs w/ columns [run, flow, label].
            no_dats (bool, optional): Whether to add dats/ subfolder. Defaults to False.
            cache (list, optional): List of cached file. Defaults to [].
        """
        if caches:
            assert len(caches) >= 2
            overwrite = caches[2]
            if os.path.exists(caches[0]) and os.path.exists(caches[1]) and not overwrite:
                self.df = pd.read_csv(caches[0], index_col=False)
                self.truth_df = pd.read_csv(caches[1], index_col=False)
                assert not self.df.empty and not self.truth_df.empty
                print(f" - df & truth_df loaded from cache {caches[0]}, {caches[1]}.")
                return

        df, truth_df = None, None
        for folder, ids in folder_to_ids.items():
            folder_path = os.path.join(root, folder, 'dats')
            if no_dats:
                folder_path = os.path.join(root, folder)
            xdf = self._load(folder_path, ids)
            if labels is not None:
                for run in labels.run.unique():
                    run_flows = labels[labels.run == run].flow.unique()
                    xdf = xdf.drop(xdf[(xdf.run == run) & (~xdf.flow.isin(run_flows))].index)
            elif 'label' not in xdf.columns:
                xdf['label'] = xdf.apply(lambda r: 0 if r.run == ids[0] else r.flow, axis=1)
                ydf = xdf[['run', 'flow', 'label']].drop_duplicates(['run', 'flow'],
                                                                ignore_index=True)
                truth_df = ydf if truth_df is None else pd.concat([truth_df, ydf],
                                                                  ignore_index=True)
            else:
                # label inferred from the queue
                ydf = xdf[['run', 'flow', 'time', 'label']].copy()
                truth_df = ydf if truth_df is None else pd.concat([truth_df, ydf],
                                                                  ignore_index=True)
            df = xdf if df is None else pd.concat([df, xdf], ignore_index=True)
        if labels is not None:
            truth_df = labels

        self.df, self.truth_df = df, truth_df
        if caches:
            self.df.to_csv(caches[0], index=False)
            self.truth_df.to_csv(caches[1], index=False)
            print(f" - df & truth_df dumped to {caches[0]}, {caches[1]}.")

    def _load(self, folder, ids):
        """Loads data from given folder and ids. Deprecate NetDataLoader
        as queue data is not needed now.
        TODO: test the result after merging qdf
        
        """
        res_df = None
        for id in ids:
            data_csv = os.path.join(folder, f'all-data_{id}.csv')
            xdf = pd.read_csv(data_csv, index_col=False)
            xdf['run'] = id
            toc_csv = os.path.join(folder, f'toc_{id}.csv')
            # if os.path.exists(toc_csv):
            try:
                toc_df = pd.read_csv(toc_csv, index_col=False)
                n_normal = len(toc_df)            
                xdf = xdf[xdf.flow < n_normal]

                qdf = self._parse_queue(toc_df, folder)
                qdf = self._get_truth_from_queue(toc_df, qdf)
                qdf['run'] = id

                # if trace delay is not added, qdf may have a different timestamp
                # than xdf, but we only need to match the total numbers here
                assert xdf.time.nunique() <= qdf.time.nunique()
                if len(xdf) < len(qdf):
                    qdf = qdf[qdf.time.isin(xdf.time.unique())].reset_index(drop=True)
                    assert len(xdf) == len(qdf)
                if not (xdf.time.unique() == qdf.time.unique()).all():
                    qdf['time'] = xdf['time']
                xdf = xdf.merge(qdf, on=['run', 'flow', 'time'])
            except Exception as e:
                if type(e) == FileNotFoundError:
                    print('Dataset: no toc file found')
                else:
                    print(e)
                    exit(1)

            res_df = pd.concat([res_df, xdf]) if res_df is not None else xdf
        res_df = res_df.reset_index(drop=True).sort_values(by=['run', 'flow', 'time'])

        return res_df

    def _parse_queue(self, toc_df, folder):
        """Parse queue data from toc_df.
        """
        qdf = None
        for _, row in toc_df.iterrows():
            if type(row['qid']) == str:
                assert ']' in row['qid']
                qids = row['qid'].replace(' ', ',')
                qids = eval(qids)
            else:
                qids = [int(row['qid'])]
            row_df = None
            for i, qid in enumerate(qids):
                queue_csv = os.path.join(folder, f'queue_{qid}.csv')
                tmp_qdf = pd.read_csv(queue_csv, index_col=False)
                tmp_qdf = tmp_qdf[['time', 'packet_in_queue']]
                tmp_qdf.rename(columns={'packet_in_queue': f'queue_{i}'}, inplace=True)
                if row_df is None:
                    row_df = tmp_qdf
                else:
                    assert len(row_df) == len(tmp_qdf)
                    row_df[f'queue_{i}'] = tmp_qdf[f'queue_{i}']
            row_df['flow'] = row['flow']
            if qdf is not None:
                assert (qdf.columns == row_df.columns).all()
            qdf = pd.concat([qdf, row_df]) if qdf is not None else row_df
        qdf = qdf.reset_index(drop=True).sort_values(by=['flow', 'time'])
        return qdf 

    def _get_truth_from_queue(self, toc_df, qdf):
        flow_qid = {}
        for _, row in toc_df.iterrows():
            if type(row['qid']) == str:
                assert ']' in row['qid']
                qids = row['qid'].replace(' ', ',')
                qids = eval(qids)
            else:
                qids = [int(row['qid'])]
            flow_qid[row.flow] = qids

        # argmax queue size -> index -> qid
        f_qargmax = lambda r: np.argmax([r[f'queue_{i}'] for i in range(len(qids))])
        qdf['label'] = qdf.apply(f_qargmax, axis=1)
        qdf['label'] = qdf.apply(lambda r: flow_qid[int(r.flow)][int(r.label)], axis=1)
        # filter some outliers, optional, as will be filtered for each interval
        # later as well
        qdf['label'] = qdf['label'].rolling(window=4, min_periods=1).apply(lambda x: x.mode()[0])
        return qdf

    def split(self, ts, dataset_type, batch_size, split_on='time'):
        """Split the dataset into train, validation and test dataset given
        the boundaries in ts.

        Args:
            ts (list): boundaries for the split
            seq_len (int): length of each sequence
            fields (list): fields to be used
            dataset_type (str): type of dataset, 'triplet', 'cluster',
                                'flow_triplet', or 'cluster_triplet'.
            batch_size (int): batch size
            split_on (str, optional): field to split on. Defaults to 'time'.
        """
        self.loaders = []
        for i, _ in enumerate(ts):
            if i == 0:
                continue
            data_df = self.df[(self.df[split_on] >= ts[i - 1]) & (self.df[split_on] < ts[i])]
            n_worker = 6
            if dataset_type == 'cluster_triplet':
                dataset = SiameseClusterPairDataset(data_df, self.truth_df, self.seq_len,
                                                    self.fields)
                loader = DataLoader(dataset, batch_size=1, num_workers=1)
            else:
                if dataset_type == 'triplet':
                    dataset = SiameseTripletDataset(data_df, self.truth_df, self.seq_len,
                                                    self.fields)
                elif dataset_type == 'flow_triplet':
                    dataset = SiameseFlowTripletDataset(data_df, self.truth_df,
                                                    self.seq_len, self.fields)
                elif dataset_type == 'cluster':
                    dataset = SiameseDataset(data_df, self.truth_df, self.seq_len,
                                            self.fields)
                else:
                    raise NotImplementedError    
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=(i == 1),
                                    num_workers=n_worker)
            self.loaders.append(loader)
    
    def smooth(self, f_smooth):
        """Smooth the time series w/ given function."""
        if not f_smooth:
            return
        for run in self.df.run.unique():
            for flow in self.df.flow.unique():
                idx = (self.df.run == run) & (self.df.flow == flow)
                self.df.loc[idx, 'owd'] = f_smooth(self.df.loc[idx, 'owd'])

    def get_train_loader(self):
        return self.loaders[0]
    
    def get_val_loader(self):
        return self.loaders[1]
    
    def get_test_loader(self):
        return self.loaders[2]
    
    def get_dataframes(self):
        return self.df, self.truth_df

# TODO: deprecated by CleanDataLoader
class SiameseNetDataLoaderTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        seq_len = 20        # segment duration: 20 x 5ms = 0.1s
        self.loader = SiameseNetDataLoader(seq_len, ['owd', 'drop'])
        folder_to_ids = {'db_pie_f5_0905': [49917 + i for i in range(3)]}
        root = '~/dumbbell_datasets_0908'
        labels = pd.DataFrame({
            'run': [49917] * 5 + [49918] * 5 + [49919] * 5, 
            'flow': list(range(5)) * 3,
            'label': ([0] * 3 + [1] * 2) * 3
        })
        self.loader.load(root, folder_to_ids, labels)

    # def test_load(self):
    #     df, truth_df = self.loader.get_dataframes()
    #     self.assertEqual(len(df), 200 * 30 * 5 * 3)
    #     self.assertEqual(truth_df.shape, (15, 3))

    # def test_split(self):
    #     assert self.loader.df is not None
    #     ts = [1, 2, 3, 4]
    #     batch_size = 5
    #     # TODO: test cluster_triplet / cluster later
    #     self.loader.split(ts, 'triplet', batch_size, split_on='time')
    #     train_loader = self.loader.get_train_loader()
    #     val_loader = self.loader.get_val_loader()
    #     test_loader = self.loader.get_test_loader()
    #     for the_loader in [train_loader, val_loader, test_loader]:
    #         self.assertEqual(len(the_loader), 10 * 5 * 3 / batch_size)
    #         x, y, z, _ = the_loader.dataset[0]
    #         for t in [x, y, z]:
    #             self.assertEqual(t.shape, (20, 2))

    # def test_load_qid(self):
    #     seq_len = 20
    #     loader = SiameseNetDataLoader(seq_len, ['owd', 'drop'])
    #     folder_to_ids = {'MboxStatistics': [86224]}
    #     root = '/home/sapphire/NS_Simulation_Toolkit/BBR_test/ns-3.27'
    #     labels = pd.DataFrame({
    #         'run': [86224] * 4,
    #         'flow': list(range(4)),
    #         'label': [0, 0, 1, 1]
    #     })
    #     loader.load(root, folder_to_ids, labels, no_dats=True)
    #     df, _ = loader.get_dataframes()
    #     for name in ['queue_0', 'queue_1']:
    #         self.assertIn(name, df.columns)
    
    def test_truth_from_queue(self):
        seq_len = 20        # segment duration: 20 x 5ms = 0.1s
        self.loader = SiameseNetDataLoader(seq_len, ['owd', 'drop'])
        folder_to_ids = {'edb_new_cross_1127': [78197 + i for i in range(12)]}
        root = '~/NS_Simulation_Toolkit/BBR_test/ns-3.27'
        self.loader.load(root, folder_to_ids)
        _, truth_df = self.loader.get_dataframes()
        # 3 runs, 5 flows, 6000 time
        self.assertEqual(truth_df.shape, (12 * 4 * 6001, 4))
        for folder, ids in folder_to_ids.items():
            for run in ids:
                df = truth_df[truth_df.run == run].copy()
                n_violation = 0
                for time in df.time.unique():
                    row_label = df[df.time == time].sort_values(['flow']).label
                    if not (row_label.iloc[0] == row_label.iloc[1] and \
                            row_label.iloc[2] == row_label.iloc[3] and \
                            row_label.iloc[0] != row_label.iloc[2]):
                        n_violation += 1
                self.assertLess(n_violation, 10)
                print(f'run {run} has {n_violation} violations')

        ts = [1, 3, 5, 7]
        batch_size = 5
        self.loader.split(ts, 'triplet', batch_size, split_on='time')
        train_loader = self.loader.get_train_loader()
        val_loader = self.loader.get_val_loader()
        test_loader = self.loader.get_test_loader()
        for the_loader in [train_loader, val_loader, test_loader]:
            self.assertEqual(len(the_loader), 12 * 4 * 20 / batch_size)
            x, y, z, _ = the_loader.dataset[0]
            for t in [x, y, z]:
                self.assertEqual(t.shape, (20, 2))


    # TODO: broken, current need label arg in add_sample_data
    def test_isio(self):
        print('TODO: current broken, add label test here')
        cache_root = '/home/sapphire/neuralforecast/my_src/pre_cache'
        folder = 'results_cbtest-one-to-n_tiny'
        cache_folder = os.path.join(cache_root, folder)
        data_folder = os.path.join('/home/sapphire/NS_Simulation_Toolkit/BBR_test/ns-3.27/dataset_round2/round2_test',
                                folder)

        runs = [0, 1]
        seq_len = 300
        fields = ['owd', 'rtt', 'slr', 'cwnd']
        prep = DataPreprocessor(cache_folder, runs=runs)
        prep.read(data_folder)

        fields = ['owd', 'rtt', 'drop', 'cwnd']
        seq_len = 300
        isio = IndexedSampleIO(fields, seq_len)

        # test save side
        xdf = prep.xdf
        run = 60758
        segment = xdf[(xdf.run == run) & (xdf.flow == 0) & (xdf.time >= 10.0) & (xdf.time < 11.5)]
        for ti in range(3):
            for ri in range(3):
                isio.add_sample_data(run + ri, ti, 0, segment)
                isio.add_sample_data(run + ri, ti, 1, segment)
                isio.add_sample_data(run + ri, ti, 2, segment)
        isio.add_triplet(run, 0, 0, 1, 2)
        assert isio.triplet_index == {60758: [[60758, 0, 0, 1, 2]]}
        for i in range(3):
            assert (isio.sample_data[run][0][i] == convert_segment_to_array(segment, fields, seq_len)).all()

        folder = 'isio_test'
        if not os.path.isdir(folder):
            os.mkdir(folder)
        isio.save(folder, '0-1')
        assert os.path.exists(os.path.join(folder, f'indexed_samples_0-1.pt'))
        isio.triplet_index = {}
        isio.add_triplet(run + 1, 0, 0, 1, 2)
        isio.add_triplet(run + 1, 1, 0, 1, 2)
        isio.save(folder, '1-2')
        isio.triplet_index = {}
        isio.add_triplet(run + 2, 1, 0, 1, 2)
        isio.add_triplet(run + 2, 2, 0, 1, 2)
        isio.save(folder, '2-3')

        # test load side
        isio.clear()
        boundary = [0, 1, 2]
        shuffle = True
        split_on = 'run'
        folders = [folder]
        cur_index = isio.load(folders)
        index_list = isio.split(cur_index, boundary, shuffle, split_on=split_on)
        print(index_list)
        x = isio.get_sample_data()
        assert (x[run][0][0] == convert_segment_to_array(segment, fields, seq_len)).all()


if __name__ == '__main__':
    unittest.main()
