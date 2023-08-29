import argparse
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import textwrap

from data_visualizer import DataVisualizer
from siamese.preprocess import DataModifier, measure_time
from sklearn import naive_bayes, svm, tree, ensemble, neighbors, linear_model


class RawTextArgumentDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter
):
    pass


class DataBinClassifier(DataVisualizer):
    """This class reads dataset including xdf and truth_df, calculates
    the stats of the features (OWD, RTT, SLR, etc) for each segment, and
    then fits and tests binary classifiers to determine if a segment is
    traversing btnk or non-btnk.

    Here we also provide visualizer to provide the boxplot of the stats
    to have an intuitive understanding of the data, and probably come up
    with a direct threshold to use in update_nonbtnk() in detection quickly.

    The core data structures of this class are the stat_df, which contains
    columns [run, flow, time, stat, value, isbtnk], and the train/test set
    for the binary classification.
    """
    def __init__(self, folder, out_dir=None, tag=None, data_root=None,
                 cache_root=None, save_dir=None):
        if data_root is not None and cache_root is not None:
            super().__init__(folder, out_dir=out_dir, tag=tag,
                data_root=data_root, cache_root=cache_root)
        elif data_root is not None:
            super().__init__(folder, out_dir=out_dir, tag=tag,
                data_root=data_root)
        elif cache_root is not None:
            super().__init__(folder, out_dir=out_dir, tag=tag,
                cache_root=cache_root)
        else:
            super().__init__(folder, out_dir=out_dir, tag=tag)
        self.stat_df = None         # [run, flow, time, owd_std, ..., isbtnk]
        self.x_train, self.x_test = None, None      # [n_seg, n_features]
        self.y_train, self.y_test = None, None      # [n_seg, 1]
        self.tag = tag
        self.save_dir = save_dir if save_dir is not None else \
            os.path.join('bin_cls', tag)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.stat_field_func = [
            ('owd_std', 'owd', lambda x: x.std()),
            ('rtt_std', 'rtt', lambda x: x.std()),
            ('slr_avg', 'slr', lambda x: x.mean()),
            ('slr_std', 'slr', lambda x: x.std()),
            # ('cwnd_avg', 'cwnd', lambda x: x.mean()),
            # ('cwnd_std', 'cwnd', lambda x: x.std()),
        ]
        # def _groupby_per_run(run_df, field, func):
        #     return run_df.groupby(['flow', 'time'])[field].transform(func)
        self.stats = [stat for stat, _, _ in self.stat_field_func]

    def read(self, runs=None, config_run_base=0):
        """Read cached dataset, i.e. xdf_pred and truth_df_pred, and stored
        in stat_df and self.truth_df.

        Args:
            runs (list, optional): Runs to read in [run0, run1).
                                Defaults to None for all runs.
            config_run_base (int, optional): Config run base to start from.
                                             Defaults to 0.
        """
        print('-read(): runs =', runs, ', config_run_base =', config_run_base)
        super().read_raw(runs, config_run_base=config_run_base,
                     read_queue=False, read_raw_only=False, no_cached_data=False)
        assert hasattr(self.prep, 'xdf') and hasattr(self.prep, 'truth_df')
        self.xdf, self.truth_df = self.prep.xdf, self.prep.truth_df
        assert self.xdf.run.nunique() == self.truth_df.run.nunique()
    
    @measure_time()
    def calculate_stats(self, interval=1.5):
        """Calculate stats for all segments, and store the results in
        self.stat_df. 

        stat_df: [run, flow, time, owd_std, ..., isbtnk], where mid field can
            be owd_std, rtt_std, slr_avg, slr_std, cwnd_avg, cwnd_std. This
            is organized for better visualization later.
        """
        print('-calculate_stats(): ')
        keys = ['run', 'flow', 'time']
        tick_truth_df = DataModifier.fetch_truth_for_segments(self.truth_df,
            interval=interval)
        stat_df = self.xdf.copy()
        stat_df['time'] = (stat_df.time - stat_df.time.min()) // interval \
                            * interval + stat_df.time.min()
        assert (stat_df.time.unique() == tick_truth_df.time.unique()).all()
        stat_df_keys = stat_df[keys].drop_duplicates(ignore_index=True)
        for run in stat_df.run.unique():
            if not stat_df_keys[stat_df_keys.run == run].reset_index(drop=True).equals(
                tick_truth_df.loc[tick_truth_df.run == run, keys].reset_index(drop=True)):
                print(f'  run {run}: bad')

        for stat, field, func in self.stat_field_func:
            # stat_df[stat] = stat_df.groupby(keys)[field].transform(func,
            #     engine='numba',
            #     engine_kwargs={'nopython': True, 'nogil': True, 'parallel': True})
            stat_df[stat] = stat_df.groupby(keys)[field].transform(func)
        stat_df.drop_duplicates(subset=keys, inplace=True, ignore_index=True)

        tick_truth_df.set_index(keys, inplace=True)
        def _isbtnk(row):
            label = tick_truth_df.loc[(row.run, row.flow, row.time)].label
            return -1 if label == -1 or label == row.flow else 1
        stat_df['isbtnk'] = stat_df.apply(_isbtnk, axis=1)
        self.tick_truth_df = tick_truth_df.reset_index(drop=True)
        self.stat_df = stat_df

    def save_stats(self, prefix='stat_df'):
        csv = os.path.join(self.save_dir, f'{prefix}_{self.tag}.csv')
        self.stat_df.to_csv(csv, index=False)
        print(f'-save_stats(): stat_df saved to {csv}')
    
    def load_stats(self, prefix='stat_df'):
        csv = os.path.join(self.save_dir, f'{prefix}_{self.tag}.csv')
        self.stat_df = pd.read_csv(csv, index_col=False)
        print(f'-load_stats(): stat_df loaded from {csv}')

    def polish_stats_data(self, split=0.8):
        """Polish the stats data to make it easier to plot and train.
        The results are stored in self.stat_df_for_plot, self.x_train,
        self.x_test, self.y_train, self.y_test.
 
            stat_df_for_plot: [run, flow, time, stat, value, isbtnk], where we can
                use stat as x and isbtnk as hue.

            x_train / x_test: [n_segments, n_features]
            y_train / y_test: [n_segments, 1]
        """
        print('-polish_stats_data(): ')
        stat_df_for_plot = pd.melt(self.stat_df,
            id_vars=['run', 'flow', 'time', 'isbtnk'],
            value_vars=self.stats)
        stat_df_for_plot.rename(columns={'variable': 'stat'}, inplace=True)
        self.stat_df_for_plot = stat_df_for_plot

        self.stat_df.sort_values(by=['run', 'flow', 'time'], inplace=True)
        xy = self.stat_df[self.stats + ['isbtnk']].to_numpy()
        np.random.shuffle(xy)
        n_split = int(len(xy) * split)
        self.x_train, self.y_train = xy[:n_split, :len(self.stats)], xy[:n_split, -1]
        self.x_test, self.y_test = xy[n_split:, :len(self.stats)], xy[n_split:, -1]

    def visualize_stats(self):
        """Visualize stat df using boxplot to observe the threshold."""
        print('-visualize_stats(): ')
        assert hasattr(self, 'stat_df_for_plot')
        self.plotter.plot_stats(self.stat_df_for_plot)

    def get_nonbtnk_ratio(self):
        """Get the ratio of -1 per run in stat df for checking."""
        df = self.stat_df.copy()
        df['nonbtnk_ratio'] = df.groupby(['run'])['isbtnk'].transform(
            lambda x: (x == -1).sum() / len(x))
        df.drop_duplicates(subset=['run', 'nonbtnk_ratio'],
                           inplace=True, ignore_index=True)
        return df[['run', 'nonbtnk_ratio']]

    def visualize_flows_w_stat(self, run, t1, flow, interval=1.5):
        """Show the flows w/ given stat to validate the calculation."""
        print(f'-visualize_flows_w_stat(): run {run}, t1 {t1}, flow {flow}')
        t2 = t1 + interval
        run_abs = self.xdf.run.min() + run
        assert run_abs in self.xdf.run.unique()
        flow_seg_df = self.xdf[(self.xdf.run == run_abs) & (self.xdf.flow == flow) &
                            (self.xdf.time >= t1) & (self.xdf.time < t2)]
        stats_df = self.stat_df[(self.stat_df.run == run_abs) & (self.stat_df.flow == flow) &
                            (self.stat_df.time >= t1) & (self.stat_df.time < t2)]
        assert stats_df.time.nunique() == 1
        field_stats = {
            'owd': ['owd_std'],
            'rtt': ['rtt_std'],
            'slr': ['slr_avg', 'slr_std'],
            'cwnd': ['cwnd_avg', 'cwnd_std'],
        }
        plt.figure()
        fig, ax = plt.subplots(4, 1, figsize=(12, 5))
        for i, (field, stats) in enumerate(field_stats.items()):
            self.plotter.axplot_flows(flow_seg_df, [flow], t1, t2, field, ax[i])
            info_str = ', '.join([f'{stat} {stats_df[stat].values[0]:.3f}' for stat in stats])
            ax[i].set(title=f'run {run} run_abs {run_abs} flow {flow} {field}: ' + info_str)
        self.plotter.show_or_save('flows_w_stat.pdf')

    def fit_test(self, algorithms='all', th1=1e-3, th2=2e-3):
        """Given algorithms in [naive bayes, logistic regression, KNN, SVM, ...],
        fit the model and test it on our dataset.
        """
        print('-fit_test(): ')
        assert hasattr(self, 'x_train') and hasattr(self, 'x_test') and \
            hasattr(self, 'y_train') and hasattr(self, 'y_test')
        def func_th(x: np.array, th1: float, th2: float) -> np.array:
            # [owd_std, rtt_std, slr_avg, slr_std, cwnd_avg, cwnd_std]
            # return np.where((x[:, 0] > th1) & (x[:, 2] > th2), 1, -1)
            return np.where(x[:, 2] > th2, 1, -1)

        classifier_dict = {
            'th': None,
            'naive_bayes': sklearn.naive_bayes.MultinomialNB(), # or GaussianNB?
            'logistic_regression': sklearn.linear_model.LogisticRegression(max_iter=1000),
            'knn': sklearn.neighbors.KNeighborsClassifier(algorithm='brute', n_jobs=-1),
            'svm': sklearn.svm.LinearSVC(C=1),
            'decision_tree': sklearn.tree.DecisionTreeClassifier(),
            'random_forest': sklearn.ensemble.RandomForestClassifier(n_estimators=100,
                                                                     max_depth=9),
        }
        algorithms = algorithms if algorithms != 'all' else classifier_dict.keys()
        assert all([algorithm in classifier_dict for algorithm in algorithms])
        res_df = None
        columns = ['algorithm', 'train_score', 'test_score',
                   'btnk_precision', 'nonbtnk_precision',
                   'btnk_recall', 'nonbtnk_recall',
                   'btnk_f1', 'nonbtnk_f1']
        for alg in algorithms:
            print(f' - Fitting {alg} ')
            if alg == 'th':
                y_train_hat = func_th(np.array(self.x_train), th1, th2)
                y_test_hat = func_th(np.array(self.x_test), th1, th2)
                train_score = sklearn.metrics.accuracy_score(self.y_train, y_train_hat)
                test_score = sklearn.metrics.accuracy_score(self.y_test, y_test_hat)
            else:
                classifier = classifier_dict[alg]
                classifier.fit(self.x_train, self.y_train)
                train_score = classifier.score(self.x_train, self.y_train)
                test_score = classifier.score(self.x_test, self.y_test)
                y_test_hat = classifier.predict(self.x_test)
                fname = f'{alg}.pkl'
                with open(os.path.join(self.save_dir, fname), 'wb') as f:
                    pickle.dump(classifier, f)
            precision, recall, f1, support = sklearn.metrics.precision_recall_fscore_support(
                self.y_test, y_test_hat, average=None, labels=[1, -1])
            row_df = pd.DataFrame([[alg, train_score, test_score, *precision, *recall, *f1]],
                                  columns=columns)
            res_df = row_df if res_df is None else pd.concat([res_df, row_df],
                                                             ignore_index=True)
            print(f'        train score: {train_score:.2f}, test score: {test_score:.2f}')
            print(f'        precision: {precision}')
            print(f'        recall: {recall}')
            print(f'        f1: {f1}')
            print(f'        support: {support}')
        print(res_df)
        res_df.to_csv(os.path.join(self.save_dir, f'fit_test_{self.tag}.csv'), index=False)


def main(folder, tag='bin_cls', runs=None, split=0.8, out_dir='figures/bin_cls'):
    dbc = DataBinClassifier(folder, out_dir=out_dir, tag=tag)
    prefix = 'stat_df'
    csv = f'{prefix}_{tag}.csv'
    if os.path.exists(os.path.join(dbc.save_dir, csv)):
        dbc.load_stats(prefix=prefix)
    else:
        dbc.read(runs=runs)
        dbc.calculate_stats()
        dbc.save_stats(prefix=prefix)

    dbc.polish_stats_data(split=split)
    dbc.visualize_stats()

    # validate purpose
    ratio_df = dbc.get_nonbtnk_ratio()
    print('  ratio_df', ratio_df)
    print('  mean', ratio_df.mean())
  
    if not os.path.exists(os.path.join(dbc.save_dir, csv)):
        sample_keys = [
            (0, 3, 1), (0, 6, 2), (0, 9, 3),
            # (10, 4, 4), (10, 7, 5), (10, 10, 6),
        ]
        for run, t1, flow in sample_keys:
            dbc.visualize_flows_w_stat(run, t1, flow)
 
    # fit test all algorithms, th to be changed
    dbc.fit_test(algorithms='all')


if __name__ == '__main__':
    epilog = textwrap.dedent('''
        This tool is used to build a binary classifier for non-btnk detection.

        It reads cached time series, calculates the statistics including
        owd_std, rtt_std, slr_avg, slr_std, cwnd_avg, cwnd_std, fits and tests
        the data with different algorithms, and save the model for detector's
        usage. 
        
        The output incluedes:
            1) figures of the statistics of the flows;
            2) print the accuracy of models;
            3) save the models.
        ''')
    parser = argparse.ArgumentParser(description='Data Binary Classifier',
                                     formatter_class=RawTextArgumentDefaultsHelpFormatter,
                                     epilog=epilog)
    parser.add_argument('--folder', '-f', type=str, required=True,
                        help='result folder name of the cached data')
    parser.add_argument('--runs', '-r', type=int, nargs=2, default=None,
                        help='[run0, run1) to read, None to read all')
    parser.add_argument('--split', '-s', type=float, default=0.8,
                        help='split ratio of train/test')
    parser.add_argument('--tag', '-t', type=str, required=True,
                        help='tag of the output figures')
    args = parser.parse_args()
    main(args.folder, tag=args.tag, runs=args.runs, split=args.split)
