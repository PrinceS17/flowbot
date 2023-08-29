import os
import unittest
import numpy as np
import pandas as pd


class NetDataLoader:
    """This class loads the observation of flows and queue data from ns3 simulation
    as well as the real experiments. 
    """
    def __init__(self, ids, folder, btnk_rate, qnorm_mode, pkt_size=1500):
        self.ids = ids        # run ids of network ids
        self.folder = folder
        self.btnk_rate = btnk_rate
        self.pkt_size = pkt_size
        self.qnorm_mode = qnorm_mode

    def load(self, store=False):
        """
        The expected files in the folder are three csvs of one run with ID:

            all_data_ID.csv: contains all flows' observations, e.g. loss, rtt, owd, etc;
            queue_QID.csv: contains the occupancy data of queue QID;
            toc_ID.csv: table of contents, giving the mapping between each flow and its
                bottleneck queue's QID.

        TODO: normalize the xdf features?


        

        Returns:
            dataframe: the feature dataframe of all flows of requested runs;
            dataframe: the queue dataframe of all flows of requested runs.
        """
        res_df, tmp_qdf = None, None
        qbase = 0.03           # const value
        inter_pkt_time = self.pkt_size * 8 / self.btnk_rate

        for id in self.ids:
            data_csv = os.path.join(self.folder, f'all-data_{id}.csv')
            xdf = pd.read_csv(data_csv, index_col=False)
            toc_csv = os.path.join(self.folder, f'toc_{id}.csv')
            toc_df = pd.read_csv(toc_csv, index_col=False)
            n_normal = len(toc_df)
            
            xdf = xdf[xdf.flow < n_normal]
            xdf['run'] = id

            for _, row in toc_df.iterrows():
                queue_csv = os.path.join(self.folder, f"queue_{row['qid']}.csv")
                qdf = pd.read_csv(queue_csv, index_col=False)
                qdf['flow'] = row['flow']
                qdf['run'] = id

                xdf_flow = xdf[xdf.flow == row.flow].copy().reset_index(drop=True)
                if self.qnorm_mode in ['owd', 'rtt']:
                    qbase = xdf_flow[self.qnorm_mode]
                elif self.qnorm_mode == 'dowd':
                    qbase = max(xdf_flow['owd']) - min(xdf_flow['owd'])
                qdf['relative_q_delay'] = qdf['packet_in_queue'] * inter_pkt_time / qbase
                qdf['relative_q_delay'].fillna(0, inplace=True)
                tmp_qdf = pd.concat([tmp_qdf, qdf]) if tmp_qdf is not None else qdf

            assert (xdf.time.unique() == tmp_qdf.time.unique()).all()
            tmp_df = xdf.merge(tmp_qdf, on=['run', 'flow', 'time'])
            res_df = pd.concat([res_df, tmp_df]) if res_df is not None else tmp_df

        res_df = res_df.reset_index(drop=True).sort_values(by=['run', 'flow', 'time'])
        if store:
            self.df = res_df

        return res_df


class NetDataLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def test_load(self):
        folder = '/home/sapphire/NS_Simulation_Toolkit/BBR_test/ns-3.27/MboxStatistics'
        ids = [6253, 6254]
        btnk_rate = 200e6
        qmode = 'owd'
        loader = NetDataLoader(ids, folder, btnk_rate, qmode)
        df = loader.load()
        res = []
        for flow in range(5):
            tmp = df.query(f'run == 6253 and flow == {flow}')
            if flow > 0:
                self.assertEqual (len(tmp), len(res[-1]))
                self.assertTrue ((tmp.time.unique() == res[-1].time.unique()).all())
            res.append(tmp)
            # qtmp = qdf.query(f'run == 6254 and flow == {flow}')
            # self.assertEqual(len(xtmp), len(qtmp))
        # self.assertTrue((xdf.flow.unique() == qdf.flow.unique()).all())

        print(f'# of entries: df: {len(df)}')
        print(df.head())
        print(df[['owd', 'relative_q_delay']].describe())



if __name__ == '__main__':
    unittest.main()
