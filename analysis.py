import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import graphics


# Фильтрация по подтипу
def filter_type(df, type_k):
    df_type = df.loc[df[type_k] == '+']
    df_t = df_type[
        ['Название локации', 'Год', 'Месяц', f'Имаго {type_k}', f'Нимфы {type_k}', 'Среднесуточная температура',
         'Количество осадков, мм']]
    # df_t.to_excel(f'Results/{type_k}.xlsx')
    return df_t


# Создание списка локаций
def location_list(df):
    loc_list = df['Название локации'].tolist()
    return set(loc_list)


def prep_df(df):
    month_dict = {'март': 3, 'апрель': 4, 'май': 5, 'июнь': 6, 'июль': 7, 'август': 8, 'сентябрь': 9, 'октябрь': 10,
                  'ноябрь': 11}
    df['Номер месяца'] = [month_dict.get(x) for x in df['Месяц']]
    df['Количество'] = df['Имаго I.Ricinus'] + df['Имаго I.Persulcatus'] + df['Нимфы I.Ricinus'] + \
                       df['Нимфы I.Persulcatus']
    df['Тип леса'] = ['Cмешанный лес' if 'смешанный лес' in x else 'Хвойный лес' if 'хвойный лес' in x else
                    'Лиственный лес' if 'лиственный лес' in x else 'Не определен' for x in df['Факторы и характеристики']]
    return df


# Вычисление коэффициента корреляции Пирсона, построение графиков
# factor1 - кол-во клещей factor 2 - фактор сравнения
def corr_estimation(df, factor1, factor2):
    r = df[factor1].corr(df[factor2])
    n = df.shape[0] - 2
    t = r * np.sqrt(n / (1 - r ** 2))
    graphics.plot_corr(df, factor1, factor2, r, t)


# Описательная статистика
# Типы: 1 - Имаго и нимфы обоих подвидов, 2 - имаго и нимфы I.Ricinus, 3 - имаго и нимфы  I.Persulcatus,
# 4 - имаго обоих подвидов, 5 - нимфы обоих подвидов
# Факторы: location - анализ по локациям, years - анализ по годам
def descr_statistics(df, factor, data_type):
    agg_func_math = {'Количество': ['median', 'mean', 'min', 'max', 'var', 'std', 'mad'], 'Подвиды': [pd.Series.mode]}
    df['Подвиды'] = df['I.Persulcatus'] + df['I.Ricinus']
    if data_type == 1:
        df['Количество'] = df['Имаго I.Ricinus'] + df['Имаго I.Persulcatus'] + df['Нимфы I.Ricinus'] + df[
            'Нимфы I.Persulcatus']
    elif data_type == 2:
        df['Количество'] = df['Имаго I.Ricinus'] + df['Нимфы I.Ricinus']
    elif data_type == 3:
        df['Количество'] = df['Имаго I.Persulcatus'] + df['Нимфы I.Persulcatus']
    elif data_type == 4:
        df['Количество'] = df['Имаго I.Ricinus'] + df['Имаго I.Persulcatus']
    else:
        df['Количество'] = + df['Нимфы I.Ricinus'] + df['Нимфы I.Persulcatus']
    if factor == 'location':
        res = df.groupby(['Название локации', 'Год']).agg(agg_func_math).round(3)
    else:
        res = df.groupby(['Год', 'Месяц', 'Номер месяца']).agg(agg_func_math).round(3)
        # res = df.groupby(['Год']).agg(agg_func_math).round(3)
    res.rename(columns={'mean': 'Среднее знач.', 'median': 'Медиана', 'min': 'Мин. знач.', 'max': 'Макс. знач.',
                        'std': 'Станд. отклонение', 'var': 'Дисперсия', 'mad': 'Среднее абс. отклонение', 'mode': 'Подвиды'},
               inplace=True)
    if factor == 'years':
        res = res.drop(columns='Подвиды')
    res.to_excel(f'Results/Descr_statistics/res_{factor}_{data_type}.xlsx')


# Вычисление параметров описательной статистики для каждого месяца, когда велось исследование
def month_descr_statistics(df):
    agg_func_math = {'Количество': ['median', 'mean', 'min', 'max', 'var', 'std', 'mad']}
    res = df.groupby(['Номер месяца', 'Месяц']).agg(agg_func_math).round(3)
    res.rename(columns={'mean': 'Среднее знач.', 'median': 'Медиана', 'min': 'Мин. знач.', 'max': 'Макс. знач.',
                        'std': 'Станд. отклонение', 'var': 'Дисперсия', 'mad': 'Среднее абс. отклонение'}, inplace=True)
    res.to_excel(f'Results/Descr_statistics/month_stat.xlsx')


# Вычисление параметров описательной статистики для каждого вида клещей
def types_statistics(df):
    types = ['I.Ricinus', 'I.Persulcatus']
    for type in types:
        df_type = df[['Название локации', 'Год', 'Месяц', f'{type}', f'Имаго {type}', f'Нимфы {type}']]
        df_type = df_type.loc[df[type] == '+']
        df_type['Количество'] = df_type[f'Имаго {type}'] + df_type[f'Нимфы {type}']
        df_type = df_type.groupby(['Год'])['Количество'].agg(['median', 'mean', 'min', 'max', 'var', 'std', 'mad']).round(3)
        df_type.rename(columns={'mean': 'Среднее знач.', 'median': 'Медиана', 'min': 'Мин. знач.', 'max': 'Макс. знач.',
                        'std': 'Станд. отклонение', 'var': 'Дисперсия', 'mad': 'Среднее абс. отклонение'},
                       inplace=True)
        df_type.to_excel(f'Results/Descr_statistics/{type}.xlsx')


# Вычисление параметров описательной статистики для каждого типа леса
def forest_type(df):
    res = df.groupby(['Тип леса', 'Год', 'Номер месяца'])['Количество'].agg(['median', 'mean', 'min', 'max',
                                                                             'var', 'std', 'mad']).round(3)
    res.rename(columns={'mean': 'Среднее знач.', 'median': 'Медиана', 'min': 'Мин. знач.', 'max': 'Макс. знач.',
                        'std': 'Станд. отклонение', 'var': 'Дисперсия', 'mad': 'Среднее абс. отклонение'},
               inplace=True)
    res.to_excel('Results/Descr_statistics/forest_types.xlsx')


# Множественная линейная регрессия
def multi_reg_analysis(df, location):
    df_reg = df[['Название локации', 'Год', 'Месяц', 'Имаго I.Persulcatus', 'Нимфы I.Persulcatus', 'Имаго I.Ricinus',
                 'Нимфы I.Ricinus', 'Среднесуточная температура', 'Количество осадков, мм']]
    df_w = pd.read_excel('weather.xlsx', header=0).drop(
        columns=['Среднесуточная температура', 'Количество осадков, мм'])
    df_reg = pd.merge(df_reg, df_w, how='left', on=['Год', 'Месяц'])
    df_reg = df_reg.loc[df['Название локации'] == location]
    df_reg = df_reg.reset_index(drop=True)
    X = df_reg[
        ['Год', 'Номер месяца', 'Среднесуточная температура', 'Количество осадков, мм', 'Температура на 1 м. раньше']]
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


def reg_analysis(location):
    df_r = pd.read_excel('Results/Descr_statistics/res_location_1.xlsx', header=1)
    df_r.rename(columns={'Unnamed: 0': 'Название локации', 'Unnamed: 1': 'Год'}, inplace=True)
    df_r = df_r[['Название локации', 'Год', 'Медиана', 'Среднее знач.']]
    df_r['Название локации'].fillna(method='pad', inplace=True)
    df_r = df_r.loc[df_r['Название локации'] == location]
    X = df_r.iloc[:, 1].values
    X = X.reshape(-1, 1)
    y = df_r.iloc[:, 3].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    plt.scatter(X, y)
    arr = np.arange(2008, 2025)
    plt.plot(arr, regressor.predict(arr.reshape(-1, 1)), color='red', linewidth=2)
    plt.title(f'{location}')
    plt.ylabel('Кол-во клещей на 1 флагочас')
    pyplot.ylim([0, 35])
    plt.savefig(f'Results/reg_analysis_{location}.png')
    print('a =', regressor.intercept_)
    print('b =', regressor.coef_)
    print('Среднее абсолютное отклонение =', metrics.mean_absolute_error(y_test, y_pred))
    print('Среднее квадратичное отклонение =', metrics.mean_squared_error(y_test, y_pred))
    print('Значение среднеквадратичной ошибки =', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
