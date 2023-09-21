import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        import numpy as np
        import matplotlib.pyplot as plt

        cnt = 0.2  # g Aceq/L
        # ylistplus =  [0.4, 0.6, 0.8, 1.0, 1.2]  # y-coordinates
        # xlist =      [0.2, 0.4, 0.6, 0.8, 1.0]  # x-coordinates
        # ylistminus = [0.0, 0.2, 0.4, 0.6, 0.8]  # y-coordinates
        xlist = np.linspace(0.1, 1, 100)

        ylistplusrelative = []
        ylistminusrelative = []
        for i in xlist:
            abs_value = (float(cnt / i)) * 100
            ylistplusrelative.append(100 + abs_value)
            ylistminusrelative.append(100 - abs_value)
        #
        # z = np.polyfit(xlist, ylistplusrelative, 2)
        # f = np.poly1d(z)
        #


        accept_lim = 20  # chosen by the client or by law
        acceptability_interval_up = 100 + accept_lim
        acceptability_interval_down = 100 - accept_lim

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.axline((0, acceptability_interval_up), slope=0, ls='--', label='acceptability_interval_up', color='red')
        ax.axline((0, acceptability_interval_down), slope=0, ls='--', label='acceptability_interval_down', color='red')

        #xx = np.linspace(np.array(xlist).min(), np.array(xlist).max(), 100)
        ax.plot(xlist, ylistplusrelative,ls='--', label='acceptability_interval_up', color='red')
        ax.plot(xlist, ylistminusrelative,ls='--', label='acceptability_interval_up', color='red')
        #plt.plot(xx, f(xx))
        plt.show  # add assertion here

    def test_insert_drop_multidex_dataframe(self):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.DataFrame(np.random.randn(3, 2), index=["A", "B", "C"], columns=['io','tu'])
        df = pd.DataFrame(np.random.randn(3, 4), index=["A", "B", "C"],
                          columns=[('io', 'tu'), ('io', 'lei'), ('tu', 'egli'), ('tu', 'lei')])

        d = dict()
        d['first_level'] = pd.DataFrame(columns=['idx', 'a', 'b', 'c'],
                                        data=[[10, 0.89, 0.98, 0.31],
                                              [20, 0.34, 0.78, 0.34]])
        d['first_level'].set_index(['idx', 'a'], inplace=True, drop=True)
        d['SECOND_level'] = pd.DataFrame(columns=['idx', 'a', 'b', 'c'],
                                         data=[[10, 0.55, 0.98, 0.31],
                                               [20, 0.34, 0.78, 0.34]])
        d['SECOND_level'].set_index(['idx', 'a'], inplace=True, drop=True)

        # to erase
        d.pop('SECOND_level')
        # add column
        d['SECOND_level']['f'] = 1

        # df of df modifications
        # add column
        for j in dict_param_relation[param]:
            for i in profil_param:
                data['SNAC_res_' + i, j] = np.nan
         # erase column
        del data['SNAC_res_std_kr', 'zzz']


if __name__ == '__main__':
    unittest.main()

