import argparse
import os
import re
import numpy as np
import pandas as pd


def replicate(csv1, csv2, ns, out_dir):
    """Replicate the csv file n times w/ flow number set to unique
    to emulate a run w/ larger flow numbers.
    """
    for csv in [csv1, csv2]:
        df = pd.read_csv(csv, index_col=False)
        print(f'- Read {csv}')
        for n in ns:
            df = pd.concat([df] * n, ignore_index=True)
            df['flow'] = df.groupby(['run', 'time']).cumcount()
            df['run'] = df.run + n
            print(df)
            new_csv = os.path.basename(csv)[:-4] + f'_n{n}.csv'
            new_csv = os.path.join(out_dir, new_csv)
            df.to_csv(new_csv, index=False)
            print(f'  Replicated {n} times, dumped to {new_csv}')


def combine(csvs, ns, out_dir, name_mode='rep'):
    """Combine the csv files by using different of them and dump to out_dir.
    """
    # first read the csvs needed by the max n
    n_max = max(ns)
    assert n_max <= len(csvs)
    dfs = []
    for i in range(n_max):
        df = pd.read_csv(csvs[i], index_col=False)
        print(f'- Read {csvs[i]}')
        dfs.append(df)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # then combine the dfs
    ns.sort()
    run_base = dfs[0].run.min()
    for i, n in enumerate(ns):
        if i == 0:
            cur_df = pd.concat(dfs[:n], ignore_index=True, copy=False)
        else:
            cur_df = pd.concat([cur_df] + dfs[ns[i-1]:n], ignore_index=True,
                               copy=False)
        # don't ruin the original run and flow in cur_df, as needed later too
        cur_df['flow_new'] = (cur_df.run * 10000 + cur_df.flow).astype(int)
        if 'label' in cur_df.columns:
            cur_df['label_new'] = cur_df.apply(lambda r: r.run * 10000 + r.label if r.label >= 0 else r.label,
                                               axis=1)
        cur_df['run_new'] = (run_base + n).astype(int)
        print(f'  new run: {cur_df.run_new.unique()}, '
              f'new n_flow: {cur_df.flow_new.nunique()}')

        out_df = cur_df.copy()
        out_df.drop(columns=['flow', 'run'], inplace=True)
        if 'label' in out_df.columns:
            out_df.drop(columns=['label'], inplace=True)
        out_df = out_df.rename(columns={'flow_new': 'flow',
                                        'run_new': 'run',
                                        'label_new': 'label'})

        if name_mode == 'app':
            new_csv = os.path.basename(csvs[0])[:-4] + f'_m{n}.csv'
        elif name_mode == 'rep':
            prefix = re.search(r'xdf_pred|truth_df_pred',
                               os.path.basename(csvs[0])).group()
            new_csv = f'{prefix}_{n}-{n+1}.csv'
        new_csv = os.path.join(out_dir, new_csv)
        out_df.to_csv(new_csv, index=False)
        print(f'    Combined {n} runs from {os.path.basename(csvs[0])}\n'
              f'    Dumped to {new_csv}\n')


def get_csvs(dir_in, runs, prefix):
    """Get the csv files to be combined from input/output directories & runs,
    prefix can be 'xdf_pred' or 'truth_df_pred'.
    """
    assert len(runs) == 2
    csvs = []
    for run in range(runs[0], runs[1]):
        csv = os.path.join(dir_in, f'{prefix}_{run}-{run+1}.csv')
        csvs.append(csv)
    return csvs


def test_combine(in_dir, runs, n, out_dir):
    """Given in / out directories, runs, and n, check the out csv passes
    the tests below.
        test 1: aggregate flow count should match
        test 2: [run, flow] should have the same flow No. in new csv
        test 3: run, time, etc should keep the same
    """
    assert runs[1] - runs[0] == n
    for prefix in ['xdf_pred', 'truth_df_pred']:
        cur_df = None
        if out_dir == in_dir:
            fname = f'{prefix}_{runs[0]}-{runs[0]+1}_m{n}.csv'
        else:
            fname = f'{prefix}_{n}-{n+1}.csv'
        out_csv = os.path.join(out_dir, fname)
        out_df = pd.read_csv(out_csv, index_col=False)
        csvs = get_csvs(in_dir, runs, prefix)
        for csv in csvs[:n]:
            df = pd.read_csv(csv, index_col=False)
            cur_df = df if cur_df is None else \
                pd.concat([cur_df, df], ignore_index=True) 
        print(f'- Read out csv: {out_csv} \n   in csv{csvs[:n]}')

        flow_no = {}        # {(run, flow): new flow No.}
        for t in out_df.time.unique():
            n_flow_in = len(cur_df.loc[cur_df.time == t, ['run', 'flow']].drop_duplicates())
            n_flow_out = out_df[out_df.time == t].flow.nunique()
            print(f'  time {t}: n_flow in: {n_flow_in} == n_flow out: {n_flow_out}')
            assert n_flow_in == n_flow_out

            tmp_df = cur_df[cur_df.time == t]
            flow_nos = []
            for run, flow in tmp_df.groupby(['run', 'flow']).groups.keys():
                flow_nos.append(run * 10000 + flow)
                if (run, flow) not in flow_no:
                    flow_no[(run, flow)] = run * 10000 + flow
                else:
                    # flow No. should match between in and out df
                    assert flow_no[(run, flow)] == run * 10000 + flow
                    out_seg = out_df[(out_df.time == t) & (out_df.flow == flow_no[(run, flow)])].copy()
                    out_seg.reset_index(drop=True, inplace=True)
                    # print('out_seg:\n', out_seg)
                    in_seg = tmp_df[(tmp_df.run == run) & (tmp_df.flow == flow)].copy()
                    in_seg.reset_index(drop=True, inplace=True)
                    # print('in_seg:\n', in_seg)
                    other_col = list(set(in_seg.columns) - set(['run', 'flow']))
                    assert in_seg[other_col].equals(out_seg[other_col]), \
                        print(f'  {in_seg} != {out_seg}')
            assert flow_nos == list(out_df[out_df.time == t].flow.unique())
            print(f'  time {t}: flow No. and each row match')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', '-o', type=str, default='.',
                        help='output directory')
    parser.add_argument('--n', '-n', type=int, nargs='+', required=True,
                        help='number of times to replicate')
    parser.add_argument('--mode', '-m', type=str,
                        choices=['replicate', 'combine', 'test-combine'],
                        default='combine', help='mode to run')
    grp1 = parser.add_argument_group('replicate')
    grp1.add_argument('--csv1', '-c1', type=str, help='csv file to replicate')
    grp1.add_argument('--csv2', '-c2', type=str, help='csv file to replicate')
    grp2 = parser.add_argument_group('combine')
    grp2.add_argument('--in_dir', '-i', type=str, help='input directory')
    grp2.add_argument('--runs', '-r', type=int, nargs=2, metavar=('start', 'end'),
                      help='runs to combine [start, end)')
    args = parser.parse_args()

    if args.mode == 'replicate':
        replicate(args.csv1, args.csv2, args.n, args.out_dir)
    elif args.mode == 'combine':
        for prefix in ['xdf_pred', 'truth_df_pred']:
            csvs = get_csvs(args.in_dir, args.runs, prefix)
            combine(csvs, args.n, args.out_dir)
    elif args.mode == 'test-combine':
        for n in args.n:
            test_combine(args.in_dir, args.runs, n, args.out_dir)

