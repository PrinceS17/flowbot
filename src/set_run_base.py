import argparse
import os
import re
import pandas as pd

def set_offset(dir_path, offset, out_dir=None, runs=None,
               prefix=['res_df', 'predict_infos'],
               dry_run=True):
    # for all the files in the directory, add offset to the files
    # out_dir: if not None, then copy the files to out_dir, otherwise modify in place
    print(f'- Begin: offset: {offset}, dir_path: {dir_path}, out_dir: {out_dir}')
    assert os.path.isdir(dir_path)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    runs = range(runs[0], runs[1] + 1) if runs is not None else None


    for f in os.listdir(dir_path):
        # print('file', f)
        prefix = 'predict_infos' if f.startswith('predict_infos') else 'res_df' \
            if f.startswith('res_df') else None
        ext = 'csv' if prefix == 'res_df' else 'pt'
        if prefix is None:
            continue
        m = re.search(r'(\d+)-(\d+)', f)
        # print('m groups', m.groups())
        run1, run2 = int(m.group(1)), int(m.group(2))
        if runs is not None and run1 not in runs:
            continue
        run1, run2 = run1 + offset, run2 + offset
        f_new = f'{prefix}_{run1}-{run2}.{ext}'
        path1 = os.path.join(dir_path, f)
        if out_dir is None:
            path2 = os.path.join(dir_path, f_new)
            cmd = f'mv {path1} {path2}'
        else:
            path2 = os.path.join(out_dir, f_new)
            cmd = f'cp {path1} {path2}'
        print(cmd)
        if not dry_run:
            os.system(cmd)
    if dry_run:
        print('dry run')
    print('- End')


# Not quite good, should just support a python script to read df, add para btnk & dump back
# file ops should leave to the shell!
# then use current script to add offset, and make sure rel_run is not used


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, required=True,
                        help='input directory of detections')
    parser.add_argument('--offset', '-of', type=int, required=True,
                        help='offset to add to the run numbers')
    parser.add_argument('--out_dir', '-o', type=str, default=None)
    parser.add_argument('--runs', '-r', type=int, nargs='+', default=None,
                        help='[run1, run2], None for all runs in the dir')
    dry_run = False
    args = parser.parse_args()
    set_offset(args.dir, args.offset, args.out_dir, args.runs, dry_run=dry_run)
