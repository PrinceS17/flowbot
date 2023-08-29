import numpy as np
import os
import pandas as pd
import subprocess as sp
import torch
import glob


def convert_segment_to_array(segment, fields, seq_len):
    segment = segment[fields].to_numpy(dtype=np.float32)
    assert segment.size == seq_len * len(fields), \
        f"Wrong segment size {len(segment)}"
    segment = segment.reshape(seq_len, len(fields))
    return segment


def print_nonbtnk_flow_seg_number(tag, 
                                  root= '/home/sapphire/neuralforecast/my_src/pre_cache'):
    """Print the non-btnk flow seg number to check of the folders globbed by tag
    under given root."""
    folders = glob.glob(root + f'/*{tag}*')
    for folder in folders:
        fbase = os.path.basename(folder)
        print(f' Checking non-btnk number in {fbase}...')
        os.chdir(folder)
        prefix = 'indexed_samples_'
        files = sorted(glob.glob(f'{prefix}*'),
                       key=lambda s: int(s[len(prefix):-3].split('-')[0]))
        total_cnt, total_n = 0, 0
        for file in files:
            cnt, n_total = 0, 0
            path = os.path.join(folder, file)
            samples, indices, labels = torch.load(path)
            for run in labels:
                for ti in labels[run]:
                    for flow, label in labels[run][ti].items():
                        n_total += 1
                        if flow == float(label) or float(label) == -1:
                            cnt += 1
            print(f'    {file}: {cnt} / {n_total}')
            total_cnt += cnt
            total_n += n_total
        print(f'   Total: {total_cnt} / {total_n} = {total_cnt / total_n:.3f}')


class IndexedSampleIO:
    """
    Indexed Sample IO: an I/O class for the efficient sample save/load for
    preprocessing and load/split. The basic idea is: instead of saving all
    triplet samples, we only save the flow samples data, and the triplets
    are saved using indices of the flow samples. Dataset can then construct
    these triplets as quick as fetching them directly, and we can save spaces
    with at least a factor of 3.

    Data structure:
        - Sample Data: a dict w/ O(1) access. Keys: run, ti, flow, i.e.
            sample_data[run][ti][flow] = sample_array
        - Tripet Index: a dict of all triplet indices, one row for a triplet.
            triplet_index[run] = [ [ti1, i1, j1, k1], [ti2, i2, j2, k2], ... ]
        - Flow Labels: dict of flow labels, flow_label[run][ti][flow] = label
        - File: save [sample_data, triplet_index] to a pt file.

    Specifications:
        - Save: replace build_triplet(). Caller provides all the necessary data
          (run, ti, flow, and segment data), and IO records the triplets and
          saves the data to disk in proper format.
        - Load: global load specified runs from the given dir. Returns sample_data
          and triplet_index later for dataset construction.
            - Dataset takes input of all sample data and partial triplet list,
                thus we support the conversion after load().
            - Support the split w/ boundary and shuffle to split triplet index into
                several triplet lists.

    """
    def __init__(self, fields, seq_len) -> None:
        self.fields = fields
        self.seq_len = seq_len
        self.sample_data = {}
        self.flow_label = {}
        self.triplet_index = {}

    def add_sample_data(self, run, ti, label, flow, segment):
        """Add sample data with memorization."""
        if run not in self.sample_data:
            self.sample_data[run] = {}
            self.flow_label[run] = {}
        if ti not in self.sample_data[run]:
            self.sample_data[run][ti] = {}
            self.flow_label[run][ti] = {}
        if flow not in self.sample_data[run][ti]:
            sample = convert_segment_to_array(segment, self.fields, self.seq_len)
            self.sample_data[run][ti][flow] = sample
            self.flow_label[run][ti][flow] = label

    def add_triplet(self, run, ti, anchor, pos, neg):
        for flow in [anchor, pos, neg]:
            assert flow in self.sample_data[run][ti]
        if run not in self.triplet_index:
            self.triplet_index[run] = []
        self.triplet_index[run].append([run, ti, anchor, pos, neg])

    def save(self, folder, tag):
        fname = f'indexed_samples_{tag}.pt'
        path = os.path.join(folder, fname)
        torch.save((self.sample_data, self.triplet_index, self.flow_label), path)
        print(f'[Indexed Sample IO] Data, index & label saved to {fname}')

    def safe_load(self, path):
        """Load from ISIO path, back compatible to (data, index) format."""
        objects = torch.load(path)
        if len(objects) == 3:
            return objects
        return objects[0], objects[1], None

    def resample(self, folders, overwrite=False, bkup_dir='indexed_samples_bkup',
                 pos_factor=2, neg_factor=1, neg_to_pos_ratio=2):
        """Resample the indexed samples in the given folders by sampling
        more triplets from the data. If overwrite if True, overwrite the
        current index sample files by appending to previous data; otherwise,
        keep the original files in back up directory. 

        Current hazard is: the sample data may not be complete as the flows
        are sampled from the xdf flows, so raw samples that don't exist in
        the previous dataset cannot appear here as well.

        Args:
            folders (list): folders to resample from
            overwrite (bool, optional): whether to overwrite the existing data
            bkup_dir (str, optional): back up dir if not overwrite
            pos_factor (int, optional): factor of # of pos pair draws / # of pos
                                        per cluster. Defaults to 2.
            neg_to_pos_ratio (int, optional): # neg draws per pos pair. Defaults
                                        to 2.
        """
        cwd = os.getcwd()
        for folder in folders:
            print(f'[Indexed Sample IO] Resample from {folder} ...')
            os.chdir(folder)
            if not overwrite:
                os.mkdir(bkup_dir)
            fnames = sp.getoutput('ls')
            for fname in fnames:
                if fname.startswith('indexed_samples_'):
                    data, index, labels = self.safe_load(fname)
                    assert labels is not None, 'Cannot resample without labels.'
                    for run in data:
                        for ti in data[run]:
                            all_flows = set(list(data[run][ti].keys()), [])
                            flow_labels = labels[run][ti]
                            for label in set(labels[run][ti].values()):
                                pos_candidates = set([f for f in all_flows if flow_labels[f] == label])
                                neg_candidates = all_flows - pos_candidates
                                n_per_cluster = max(pos_factor * len(pos_candidates),
                                                    neg_factor * len(neg_candidates)) 
                                for _ in range(n_per_cluster):
                                    res = np.random.choice(pos_candidates, 2, replace=False)
                                    anchor, pos = res[0], res[1]
                                    for _ in range(neg_to_pos_ratio):
                                        neg = np.random.choice(neg_candidates)
                                    index[run].append([run, ti, anchor, pos, neg]) 

                    if not overwrite:
                        os.system(f'mv {fname} {bkup_dir}')
                    torch.save((data, index, labels), fname)
                    print(f'         {fname} resampled and dumped.')
            os.chdir(cwd) 

    def load(self, folders):
        """Load all runs from the given folders. The split still happens
        in the unit of folders, i.e. load(), split() come together.

        Returns the index of the loaded runs to use for dataset split.
        """
        cur_index = {}
        for folder in folders:
            print(f'[Indexed Sample IO] Loading from {folder} ...')
            for fname in os.listdir(folder):
                if fname.startswith('indexed_samples_'):
                    path = os.path.join(folder, fname)
                    data, index, labels = self.safe_load(path)
                    self.sample_data.update(data)
                    if labels is not None:
                        self.flow_label.update(labels)
                    for run in list(index.keys()):
                        assert run not in self.triplet_index
                    cur_index.update(index)
                    self.triplet_index.update(index)
        return cur_index

    def split(self, triplet_index, boundary, shuffle, split_on):
        """Split the loaded data into several splits. The split is done
        on the triplet index w/o touching the underlying sample data.
        split_on can be 'run' or 'sample'.

        Returns a list of triplet indices for each split.
        """
        index_by_run = [triplet_index[run] for run in sorted(triplet_index.keys())]
        assert split_on in ['sample', 'run'], \
            f'split_on should be in [sample, run], current {split_on}'
        if split_on == 'sample':
            all_index = sum(index_by_run, [])
            if shuffle:
                np.random.shuffle(all_index)
            boundary = list(map(lambda x: int(x / max(boundary) * len(all_index)),
                        boundary))
            print('                   Split: derived boundary:', boundary)
            return [all_index[boundary[i-1] : boundary[i]]
                for i in range(1, len(boundary))]
        elif split_on == 'run':
            if shuffle:
                np.random.shuffle(index_by_run)
            boundary = list(map(lambda x: int(x / max(boundary) * len(index_by_run)),
                        boundary))
            print('                   Split: derived boundary:', boundary)
            return [sum(index_by_run[boundary[i-1] : boundary[i]], [])
                for i in range(1, len(boundary))]

    def get_sample_data(self):
        return self.sample_data

    def clear(self):
        self.sample_data = {}
        self.triplet_index = {}
        self.flow_label = {}

    def get_label_count_df(self):
        """Get the label count of the current indexed samples.
        """
        label_counts = []
        for run in self.flow_label:
            for ti in self.flow_label[run]:
                for flow in self.flow_label[run][ti]:
                    label = self.flow_label[run][ti][flow]
                    label_counts.append([run, ti, label, flow])
        return pd.DataFrame(label_counts, columns=['run', 'ti', 'label', 'flow'])

    def print_label_count(self):
        """Print the label count of the current indexed samples.
        """
        df = self.get_label_count_df()
        print('=== Number of labels per cluster ===')
        print(df.groupby(['run', 'ti']).label.count())
        print('=== Number of flows per label ===')
        print(df.groupby(['run', 'ti', 'label']).count())