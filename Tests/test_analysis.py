import pytest
import filecmp
import os
import pandas as pd
import analysis
import graphics


class TestClass:
    @pytest.fixture
    def data(self):
        data_analysis = analysis.Analysis(pd.read_excel('Files/test.xlsx', header=0))
        return data_analysis

    @pytest.fixture
    def data_regression(self):
        data_analysis = analysis.Analysis(pd.read_excel('Files/test2.xlsx', header=0))
        return data_analysis

    def test_prep_df(self, data):
        f = open("Files/res_prep.txt", "w", encoding='utf8')
        f.write(str(data.df.to_dict()))
        f.close()
        assert filecmp.cmp('Files/prep_df.txt', 'Files/res_prep.txt', shallow=False)
        os.remove('Files/res_prep.txt')

    def test_filter_type(self, data):
        f = open("Files/res_pers.txt", "w", encoding='utf8')
        f.write(str(data.filter_type('I.Persulcatus').to_dict()))
        f.close()
        f = open("Files/res_ric.txt", "w", encoding='utf8')
        f.write(str(data.filter_type('I.Ricinus').to_dict()))
        f.close()
        assert filecmp.cmp('Files/res_pers.txt', 'Files/pers.txt', shallow=False)
        assert filecmp.cmp('Files/res_ric.txt', 'Files/ric.txt', shallow=False)
        os.remove('Files/res_pers.txt')
        os.remove('Files/res_ric.txt')

    def test_corr_estimation(self, data):
        res = {}
        f = open("Files/res_corr.txt", "w", encoding='utf8')
        f.write(str(data.corr_estimation(data.df, 'Количество', 'Среднесуточная температура', res)))
        f.close()
        assert filecmp.cmp('Files/res_corr.txt', 'Files/corr.txt', shallow=False)
        os.remove('Files/res_corr.txt')
        with pytest.raises(AssertionError):
            data.corr_estimation(data.df, 'Качество', 'Среднесуточная температура', res)

    def test_descr_statistics(self, data):
        data.descr_statistics('location', 'Files/res_location.xlsx')
        data.descr_statistics('years', 'Files/res_years.xlsx')
        assert pd.read_excel('Files/res_location.xlsx').equals(pd.read_excel('Files/location.xlsx'))
        assert pd.read_excel('Files/res_years.xlsx').equals(pd.read_excel('Files/years.xlsx'))
        with pytest.raises(AssertionError):
            data.descr_statistics('abc', 'Files/res_location.xlsx')
        os.remove('Files/res_location.xlsx')
        os.remove('Files/res_years.xlsx')

    def test_month_descr_statistics(self, data):
        data.month_descr_statistics('Files/res_month.xlsx')
        assert pd.read_excel('Files/res_month.xlsx').equals(pd.read_excel('Files/month.xlsx'))
        os.remove('Files/res_month.xlsx')

    def test_types_statistics(self, data):
        data.types_statistics('Files/res_types1.xlsx', 'Files/res_types2.xlsx')
        assert pd.read_excel('Files/res_types1.xlsx').equals(pd.read_excel('Files/types1.xlsx'))
        assert pd.read_excel('Files/res_types2.xlsx').equals(pd.read_excel('Files/types2.xlsx'))
        os.remove('Files/res_types1.xlsx')
        os.remove('Files/res_types2.xlsx')

    def test_forest_type(self, data):
        data.forest_type('Files/res_forest.xlsx')
        assert pd.read_excel('Files/res_forest.xlsx').equals(pd.read_excel('Files/forest.xlsx'))
        os.remove('Files/res_forest.xlsx')

    def test_reg_analysis(self, data_regression):
        data_regression.descr_statistics('location', 'Files/res_location2.xlsx')
        plot = graphics.Graphics()
        data_regression.reg_analysis(False, 'Files/res_location2.xlsx', 'Files/res_regression.xlsx', plot)
        assert pd.read_excel('Files/res_regression.xlsx').equals(pd.read_excel('Files/regression.xlsx'))
        os.remove('Files/res_regression.xlsx')
        os.remove('Files/res_location2.xlsx')

