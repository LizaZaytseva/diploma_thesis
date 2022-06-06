import filecmp
import os
import unittest
import pandas as pd
import analysis


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        self.analysis = analysis.Analysis(pd.read_excel('Files/test.xlsx', header=0))

    def test_prep_df(self):
        f = open("Files/res.txt", "w", encoding='utf8')
        f.write(str(self.analysis.df.to_dict()))
        f.close()
        self.assertTrue(filecmp.cmp('Files/prep_df.txt', 'Files/res.txt', shallow=False))
        os.remove('Files/res.txt')

    def test_filter_type(self):
        f = open("Files/res_pers.txt", "w", encoding='utf8')
        f.write(str(self.analysis.filter_type('I.Persulcatus').to_dict()))
        f.close()
        f = open("Files/res_ric.txt", "w", encoding='utf8')
        f.write(str(self.analysis.filter_type('I.Ricinus').to_dict()))
        f.close()
        self.assertTrue(filecmp.cmp('Files/res_pers.txt', 'Files/pers.txt', shallow=False))
        self.assertTrue(filecmp.cmp('Files/res_ric.txt', 'Files/ric.txt', shallow=False))
        os.remove('Files/res_pers.txt')
        os.remove('Files/res_ric.txt')

    def test_corr_estimation(self):
        res = {}
        f = open("Files/res_corr.txt", "w", encoding='utf8')
        f.write(str(self.analysis.corr_estimation(self.analysis.df, 'Количество', 'Среднесуточная температура', res)))
        f.close()
        self.assertTrue(filecmp.cmp('Files/res_corr.txt', 'Files/corr.txt', shallow=False))
        os.remove('Files/res_corr.txt')
        with self.assertRaises(AssertionError):
            self.analysis.corr_estimation(self.analysis.df, 'Качество', 'Среднесуточная температура', res)

    def test_descr_statistics(self):
        self.analysis.descr_statistics('location', 'Files/res_location.xlsx')
        self.analysis.descr_statistics('years', 'Files/res_years.xlsx')
        self.assertTrue(pd.read_excel('Files/res_location.xlsx').equals(pd.read_excel('Files/location.xlsx')))
        self.assertTrue(pd.read_excel('Files/res_years.xlsx').equals(pd.read_excel('Files/years.xlsx')))
        with self.assertRaises(AssertionError):
            self.analysis.descr_statistics('abc', 'Files/res_location.xlsx')
        os.remove('Files/res_location.xlsx')
        os.remove('Files/res_years.xlsx')

    def test_month_descr_statistics(self):
        self.analysis.descr_statistics('location', 'Files/res_month.xlsx')
        self.assertTrue(pd.read_excel('Files/res_month.xlsx').equals(pd.read_excel('Files/month.xlsx')))
        os.remove('Files/res_month.xlsx')

    def test_types_statistics(self):
        self.analysis.types_statistics('Files/res_types1.xlsx', 'Files/res_types2.xlsx')
        self.assertTrue(pd.read_excel('Files/res_types1.xlsx').equals(pd.read_excel('Files/types1.xlsx')))
        self.assertTrue(pd.read_excel('Files/res_types2.xlsx').equals(pd.read_excel('Files/types2.xlsx')))
        os.remove('Files/res_types1.xlsx')
        os.remove('Files/res_types2.xlsx')

    def test_forest_type(self):
        self.analysis.forest_type('Files/res_forest.xlsx')
        self.assertTrue(pd.read_excel('Files/res_forest.xlsx').equals(pd.read_excel('Files/forest.xlsx')))
        os.remove('Files/res_forest.xlsx')

    def test_reg_analysis(self):
        self.analysis.df = pd.read_excel('Files/test2.xlsx')
        self.analysis.loc_list = sorted(set(self.analysis.df['Название локации'].tolist()))
        self.analysis.descr_statistics('location', 'Files/res_location2.xlsx')
        self.analysis.reg_analysis(False, 'Files/res_location2.xlsx', 'Files/res_regression.xlsx')
        self.assertTrue(pd.read_excel('Files/res_regression.xlsx').equals(pd.read_excel('Files/regression.xlsx')))
        os.remove('Files/res_regression.xlsx')
        os.remove('Files/res_location2.xlsx')


if __name__ == '__main__':
    unittest.main()
