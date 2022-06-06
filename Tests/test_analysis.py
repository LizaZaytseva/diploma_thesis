import filecmp
import os
import pathlib
import unittest

import pandas as pd

import analysis


class TestAnalysis(unittest.TestCase):
    def setUp(self):
        data = {'Название локации': ['loc 1', 'loc 2'], 'Год': [2000, 2010], 'Месяц': ['май', 'май'],
                'Факторы и характеристики': ['близость к заливу, лиственный лес', 'хвойный лес'],
                'Широта': [123456, 654321], 'Долгота': [123456, 321654], 'I.Persulcatus': ['+', '-'],
                'I.Ricinus': ['-', '+'], 'Имаго I.Persulcatus': [5.0, 0.0], 'Нимфы I.Persulcatus': [5.0, 0.0],
                'Имаго I.Ricinus': [0.0, 5.5], 'Нимфы I.Ricinus': [0.0, 1.5],
                'Среднесуточная температура': [12.0, 21.1],
                'Количество осадков, мм': [25.0, 22.2]}
        self.analysis = analysis.Analysis(pd.DataFrame(data))

    def test_prep_df(self):
        data_after = {'Название локации': ['loc 1', 'loc 2'], 'Год': [2000, 2010], 'Месяц': ['май', 'май'],
                      'Факторы и характеристики': ['близость к заливу, лиственный лес', 'хвойный лес'],
                      'Широта': [123456, 654321], 'Долгота': [123456, 321654], 'I.Persulcatus': ['+', '-'],
                      'I.Ricinus': ['-', '+'], 'Имаго I.Persulcatus': [5.0, 0.0], 'Нимфы I.Persulcatus': [5.0, 0.0],
                      'Имаго I.Ricinus': [0.0, 5.5], 'Нимфы I.Ricinus': [0.0, 1.5],
                      'Среднесуточная температура': [12.0, 21.1],
                      'Количество осадков, мм': [25.0, 22.2], 'Номер месяца': [5, 5], 'Количество': [10.0, 7.0],
                      'Тип леса': ['Лиственный лес', 'Хвойный лес']}
        df_after = pd.DataFrame(data_after)
        self.assertTrue(self.analysis.df.equals(df_after))

    def test_filter_type(self):
        df_pers = {'Название локации': ['loc 1'], 'Год': [2010], 'Месяц': ['май'], 'Имаго I.Persulcatus': [5.0],
                   'Нимфы I.Persulcatus': [5.0], 'Среднесуточная температура': [12.0],
                   'Количество осадков, мм': [25.0]}
        df_ric = {'Название локации': ['loc 2'], 'Год': [2000], 'Месяц': ['май'], 'Имаго I.Ricinus': [5.5],
                  'Нимфы I.Ricinus': [1.5], 'Среднесуточная температура': [21.1],
                  'Количество осадков, мм': [22.2]}
        self.assertTrue(self.analysis.filter_type('I.Persulcatus').equals(df_pers))
        self.assertTrue(self.analysis.filter_type('I.Ricinus').equals(df_ric))

    def test_corr_estimation(self):
        res = {}
        add_data = {'Название локации': 'loc 3', 'Год': 2005, 'Месяц': 'апрель',
                    'Факторы и характеристики': 'смешанный лес',
                    'Широта': 111111, 'Долгота': 333333, 'I.Persulcatus': '+',
                    'I.Ricinus': '-', 'Имаго I.Persulcatus': 7.0, 'Нимфы I.Persulcatus': 0.0,
                    'Имаго I.Ricinus': 0.0, 'Нимфы I.Ricinus': 0.0,
                    'Среднесуточная температура': 11.2,
                    'Количество осадков, мм': 55.0, 'Номер месяца': 4, 'Количество': 7.0,
                    'Тип леса': 'Смешанный лес'}
        self.analysis.df = self.analysis.df.append(pd.Series(add_data), ignore_index=True)
        self.assertEquals(self.analysis.corr_estimation(self.analysis.df, 'Количество', 'Среднесуточная температура', res),
                    {0: {'coef. t': -0.484, 'corr. coef': -0.436, 'crit. t': 3.078, 'estimation': 0.9, 'factor 1': 'Количество',
                    'factor 2': 'Среднесуточная температура'}})
        with self.assertRaises(AssertionError):
            #Задание фактора, у которого нет соответствующего столбца в таблице
            self.analysis.corr_estimation(self.analysis.df, 'Качество', 'Среднесуточная температура', res)
            #Условие, при котором слишком мало данных для проведения расчетов
            self.analysis.corr_estimation(self.analysis.df, 'Имаго I.Ricinus', 'Среднесуточная температура', res)

    def test_descr_statistics(self):
        add_data = {'Название локации': 'loc 1', 'Год': 2000, 'Месяц': 'апрель',
                    'Факторы и характеристики': 'смешанный лес',
                    'Широта': 111111, 'Долгота': 333333, 'I.Persulcatus': '+',
                    'I.Ricinus': '-', 'Имаго I.Persulcatus': 7.0, 'Нимфы I.Persulcatus': 1.0,
                    'Имаго I.Ricinus': 0.0, 'Нимфы I.Ricinus': 0.0,
                    'Среднесуточная температура': 11.2,
                    'Количество осадков, мм': 55.0, 'Номер месяца': 4, 'Количество': 7.0,
                    'Тип леса': 'Смешанный лес'}
        self.analysis.df = self.analysis.df.append(pd.Series(add_data), ignore_index=True)
        self.analysis.descr_statistics('location', 'res_location.xlsx')
        self.analysis.descr_statistics('years', 'res_years.xlsx')
        res = pd.read_excel('res_years.xlsx', header=0)
        df_standart_loc = pd.DataFrame({'Unnamed: 0': {0: None, 1: 'Название локации', 2: 'loc 1', 3: 'loc 2'}, 'Unnamed: 1':
            {0: None, 1: 'Год', 2: 2000, 3: 2010}, 'Количество': {0: 'Медиана', 1: None, 2: 8.5, 3: 7}, 'Unnamed: 3': {0:
            'Среднее знач.', 1: None, 2: 8.5, 3: 7}, 'Unnamed: 4': {0: 'Мин. знач.', 1: None, 2: 7, 3: 7}, 'Unnamed: 5':
            {0: 'Макс. знач.', 1: None, 2: 10, 3: 7}, 'Unnamed: 6': {0: 'Дисперсия', 1: None, 2: 4.5, 3: None}, 'Unnamed: 7':
            {0: 'Станд. отклонение', 1: None, 2: 2.121, 3: None}, 'Unnamed: 8': {0: 'Среднее абс. отклонение', 1: None, 2: 1.5, 3: 0},
            'Подвиды': {0: 'Подвиды', 1: None, 2: '+-', 3: '-+'}})
        df_standart_years = pd.DataFrame({'Unnamed: 0': {0: None, 1: 'Год', 2: 2000, 3: None, 4: 2010}, 'Unnamed: 1':
            {0: None, 1: 'Месяц', 2: 'апрель', 3: 'май', 4: 'май'}, 'Unnamed: 2': {0: None, 1: 'Номер месяца', 2: 4, 3: 5, 4: 5},
            'Количество': {0: 'Медиана', 1: None, 2: 7, 3: 10, 4: 7}, 'Unnamed: 4': {0: 'Среднее знач.', 1: None, 2: 7, 3: 10, 4: 7},
            'Unnamed: 5': {0: 'Мин. знач.', 1: None, 2: 7, 3: 10, 4: 7}, 'Unnamed: 6': {0: 'Макс. знач.', 1: None, 2: 7, 3: 10, 4: 7},
            'Unnamed: 7': {0: 'Дисперсия', 1: None, 2: None, 3: None, 4: None}, 'Unnamed: 8': {0: 'Станд. отклонение', 1: None, 2: None, 3: None, 4: None},
            'Unnamed: 9': {0: 'Среднее абс. отклонение', 1: None, 2: 0, 3: 0, 4: 0}})
        self.assertTrue(df_standart_loc.equals(pd.read_excel('res_location.xlsx', header=0)))
        self.assertTrue(df_standart_years.equals(pd.read_excel('res_years.xlsx', header=0)))
        with self.assertRaises(AssertionError):
            #Использование неверного фактора анализа
            self.analysis.descr_statistics('abc', 'res_location.xlsx')
        os.remove('res_location.xlsx')
        os.remove('res_years.xlsx')

if __name__ == '__main__':
    unittest.main()
