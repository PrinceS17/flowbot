import pandas as pd
import numpy as np
from models import Pipeline
from collections import OrderedDict
from sklearn.feature_selection import mutual_info_regression

# TODO: to do this clustering, the predicted relative q delay needs to be recovered
#       to the original delay value

class SimpleClustering:
    """This is used to cluster the flows based on the pairwise correlation,
    i.e. the ordinary O(N^2) approach.
    
    TODO
        1. the input structure: df ['time', 'flow', 'queue']?
        2. window length is needed for correlation, and shift it or not?
        3. speed: apart from N, depends on averaging window L1 & correlation window L2

    Design
        input: df ['time', 'flow', 'queue'], 
        average: average queue value over time      # kind of wierd to use
        ? sync offset: assuming solved by the averaging interval?
        N to N: correlation coefficient between any two flows
        clustering algorithm: loss function to group the flows
        
    TODO:
        this method makes people feel the info is still projected into 1-d, and probably 
        lose some information.
    """
    def __init__(self, th) -> None:
        self.th = th        # clustering threshold for corr coefficient
        self.score_func = OrderedDict([
            ('linear', lambda s1, s2: s1.corr(s2, method='pearson')),
            ('rank', lambda s1, s2: s1.corr(s2, method='spearman')),
            ('continuous_minfo', lambda s1, s2: mutual_info_regression(s1, s2))
        ])

    def _average(self, df, interval):
        # average the df using interval to reduce the sequence length
        df['tick'] = df.time // interval
        grouped = df.groupby('tick').transform(np.mean)
        df = pd.concat(df.tick, grouped, axis=1)
        df = df.drop(columns='time').drop_duplicates()
        return df

    def group(self, df, interval, typ='linear'):
        df = self._average(df, interval)
        flows = df.flow.unique()
        n_flow = df.flow.nunique()
        corr_func = self.score_func[typ]
        r_mat = np.zeros((n_flow, n_flow))
        for i in range(n_flow - 1):
            for j in range(i + 1, n_flow):
                f1 = df[df.flow == flows[i]]
                f2 = df[df.flow == flows[j]]
                r_mat[i, j] = corr_func(f1, f2)

        # TODO: currently fully depends on the threshold
        #       maybe use some more self-adaptive method later
        groups = [1]        # group start from 1
        n_avail = 2
        for i in range(1, n_flow):
            for j in range(i):
                if r_mat[j, i] > self.th:   # use the 1st marked neighbor j
                    groups.append(groups[j])
                    break
            if len(groups) == i:            # no neighbors
                groups.append(n_avail)
                n_avail += 1
        res_df = pd.DataFrame({'flow': flows, 'group': groups})
        return res_df

