import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy import stats
import graphics


def prep_df(df):
    month_dict = {'март': 3, 'апрель': 4, 'май': 5, 'июнь': 6, 'июль': 7, 'август': 8, 'сентябрь': 9, 'октябрь': 10,
                  'ноябрь': 11}
    df['Номер месяца'] = [month_dict.get(x) for x in df['Месяц']]
    df['Количество'] = df['Имаго I.Ricinus'] + df['Имаго I.Persulcatus'] + df['Нимфы I.Ricinus'] + \
                       df['Нимфы I.Persulcatus']
    df['Тип леса'] = ['Cмешанный лес' if 'смешанный лес' in x else 'Хвойный лес' if 'хвойный лес' in x else
    'Лиственный лес' if 'лиственный лес' in x else 'Не определен' for x in df['Факторы и характеристики']]
    return df


class Analysis(object):

    def __init__(self, df):
        self.param = ['median', 'mean', 'min', 'max', 'var', 'std', 'mad']
        self.df = prep_df(df)
        self.loc_list = set(df['Название локации'].tolist())

    # Фильтрация по подтипу
    def filter_type(self, type_k):
        assert type_k in ['I.Ricinus', 'I.Persulcatus'], 'Некорректно указан тип клещей'
        df_type = self.df.loc[self.df[type_k] == '+']
        df_t = df_type[
            ['Название локации', 'Год', 'Месяц', f'Имаго {type_k}', f'Нимфы {type_k}', 'Среднесуточная температура',
             'Количество осадков, мм']]
        return df_t

    # Вычисление коэффициента корреляции Пирсона, построение графиков
    # factor1 - кол-во клещей factor 2 - фактор сравнения
    def corr_estimation(self, df, factor1, factor2, res):
        factors = df.columns.values
        assert factor1 in factors and factor2 in factors, 'Некорректные факторы анализа'
        r = df[factor1].corr(df[factor2])
        n = df.shape[0]
        assert n > 2, 'Невозможно применить формулы к данным'
        if n > 100:
            err = np.sqrt((1 - r ** 2) / n)
        else:
            err = np.sqrt((1 - r ** 2) / (n - 2))
        coef_t = r / err
        alpha = 0.1
        # Табличное (критическое) значение t-критерия
        cv = stats.t.ppf(1 - alpha, n - 2)
        new_row = {'factor 1': factor1, 'factor 2': factor2, 'corr. coef': round(r, 3), 'estimation': round(err, 3),
                   'coef. t': round(coef_t, 3), 'crit. t': round(cv, 3)}
        res[len(res)] = new_row
        return res

    # Описательная статистика
    # Факторы: location - анализ по локациям, years - анализ по годам
    def descr_statistics(self, factor, file):
        assert factor in ['location', 'years'], 'Некорректно указан фактор анализа'
        agg_func_math = {'Количество': self.param, 'Подвиды': [pd.Series.mode]}
        self.df['Подвиды'] = self.df['I.Persulcatus'] + self.df['I.Ricinus']
        if factor == 'location':
            res = self.df.groupby(['Название локации', 'Год']).agg(agg_func_math).round(3)
        else:
            res = self.df.groupby(['Год', 'Месяц', 'Номер месяца']).agg(agg_func_math).round(3)
        res.rename(columns={'mean': 'Среднее знач.', 'median': 'Медиана', 'min': 'Мин. знач.', 'max': 'Макс. знач.',
                            'std': 'Станд. отклонение', 'var': 'Дисперсия', 'mad': 'Среднее абс. отклонение',
                            'mode': 'Подвиды'},
                   inplace=True)
        if factor == 'years':
            res = res.drop(columns='Подвиды')
        res.to_excel(file)

    # Вычисление параметров описательной статистики для каждого месяца, когда велось исследование
    def month_descr_statistics(self, file):
        agg_func_math = {'Количество': self.param}
        res = self.df.groupby(['Номер месяца', 'Месяц']).agg(agg_func_math).round(3)
        res.rename(columns={'mean': 'Среднее знач.', 'median': 'Медиана', 'min': 'Мин. знач.', 'max': 'Макс. знач.',
                            'std': 'Станд. отклонение', 'var': 'Дисперсия', 'mad': 'Среднее абс. отклонение'},
                   inplace=True)
        res.to_excel(file)

    # Вычисление параметров описательной статистики для каждого вида клещей
    def types_statistics(self, file1, file2):
        types = ['I.Ricinus', 'I.Persulcatus']
        files = [file1, file2]
        for type in types:
            df_type = self.df[['Название локации', 'Год', 'Месяц', f'{type}', f'Имаго {type}', f'Нимфы {type}']]
            df_type = df_type.loc[self.df[type] == '+']
            df_type['Количество'] = df_type[f'Имаго {type}'] + df_type[f'Нимфы {type}']
            df_type = df_type.groupby(['Год'])['Количество'].agg(self.param).round(3)
            df_type.rename(
                columns={'mean': 'Среднее знач.', 'median': 'Медиана', 'min': 'Мин. знач.', 'max': 'Макс. знач.',
                         'std': 'Станд. отклонение', 'var': 'Дисперсия', 'mad': 'Среднее абс. отклонение'},
                inplace=True)
            df_type.to_excel(files[types.index(type)])

    # Вычисление параметров описательной статистики для каждого типа леса
    def forest_type(self, file):
        res = self.df.groupby(['Тип леса', 'Год', 'Номер месяца'])['Количество'].agg(self.param).round(3)
        res.rename(columns={'mean': 'Среднее знач.', 'median': 'Медиана', 'min': 'Мин. знач.', 'max': 'Макс. знач.',
                            'std': 'Станд. отклонение', 'var': 'Дисперсия', 'mad': 'Среднее абс. отклонение'},
                   inplace=True)
        res.to_excel(file)

    # Множественная линейная регрессия
    def multi_reg_analysis(self, location):
        df_reg = self.df[
            ['Название локации', 'Год', 'Месяц', 'Имаго I.Persulcatus', 'Нимфы I.Persulcatus', 'Имаго I.Ricinus',
             'Нимфы I.Ricinus', 'Среднесуточная температура', 'Количество осадков, мм']]
        df_w = pd.read_excel('weather.xlsx', header=0).drop(columns=['Среднесуточная температура', 'Количество осадков, мм'])
        df_reg = pd.merge(df_reg, df_w, how='left', on=['Год', 'Месяц'])
        df_reg = df_reg.loc[self.df['Название локации'] == location]
        df_reg = df_reg.reset_index(drop=True)
        X = df_reg[
            ['Год', 'Номер месяца', 'Среднесуточная температура', 'Количество осадков, мм',
             'Температура на 1 м. раньше']]
        y = df_reg['Количество']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=0)
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
        y_pred = regressor.predict(X_test)
        df_res = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        coeff_df.to_excel(f'Results/coeff_df_{location}.xlsx')
        df_res.to_excel(f'Results/df_res_{location}.xlsx')
        print('Среднее абсолютное отклонение:', metrics.mean_absolute_error(y_test, y_pred))
        print('Среднее квадратичное отклонение:', metrics.mean_squared_error(y_test, y_pred))
        print('Значение среднеквадратичной ошибки:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # Линейная регрессия
    def reg_analysis(self, with_plot, file_in, file_out):
        df_r = pd.read_excel(file_in, header=1)
        df_r.rename(columns={'Unnamed: 0': 'Название локации', 'Unnamed: 1': 'Год'}, inplace=True)
        df_r = df_r[['Название локации', 'Год', 'Медиана', 'Среднее знач.']]
        df_r['Название локации'].fillna(method='pad', inplace=True)
        res = {}
        for loc in self.loc_list:
            index = list(self.loc_list).index(loc)
            df_loc = df_r.loc[df_r['Название локации'] == loc]
            X = df_loc.iloc[:, 1].values
            X = X.reshape(-1, 1)
            y = df_loc.iloc[:, 3].values
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, train_size=0.5, random_state=0)
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            arr = np.arange(2008, 2025)
            if with_plot:
                plot_pred = regressor.predict(arr.reshape(-1, 1))
                plot = graphics.Graphics()
                plot.plot_regression(X, y, arr, plot_pred, loc)
            if regressor.coef_[0] > 0:
                type_trend = 'возр.'
            else:
                type_trend = 'убыв.'
            new_row = {'location': loc, 'a': round(regressor.intercept_, 3), 'b': round(regressor.coef_[0], 3),
                       'trend': type_trend,
                       'std': round(metrics.mean_squared_error(y_test, y_pred), 3),
                       'mad': round(metrics.mean_absolute_error(y_test, y_pred), 3), 'R^2':
                           round(r2_score(y_test, y_pred), 3)}
            res[index] = new_row
        df_res = (pd.DataFrame(res)).transpose()
        df_res.to_excel(file_out)
