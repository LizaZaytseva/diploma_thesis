from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from matplotlib.ticker import LinearLocator, MultipleLocator

param = ['Среднее знач.', 'Медиана', 'Мин. знач.', 'Макс. знач.', 'Станд. отклонение',
         'Дисперсия', 'Среднее абс. отклонение']
color = ['g', 'b', 'y', 'r', 'k', 'm', 'c']
marker = ['.', 'o', '^', '*', 's', '1', 'd', 'h']
line = ['-', '-.', ':']


# Графики по корреляции Пирсона
def plot_corr(df, factor1, factor2, r, t):
    xs = df[factor2]
    ys = df[factor1]
    pd.DataFrame(np.array([xs, ys]).T).plot.scatter(0, 1, s=12, grid=True, color='g', marker='x')
    plt.xlabel(factor2)
    plt.ylabel(f'Кол-во {factor1} на 1 флагочас')
    plt.title(f'Коэффициент корреляции Пирсона = {r},\n Оценка достоверности = {t}', fontsize=10)
    plt.savefig(f'Results/Correlation/cor_{factor1} - {factor2}.png')


# Графики по локациям для описательной статистики
def plot_for_locations(loc_list):
    df = pd.read_excel('Results/Descr_statistics/res_location_1.xlsx', header=1)
    df.rename(columns={'Unnamed: 0': 'Название локации', 'Unnamed: 1': 'Год'}, inplace=True)
    df.drop(labels=[0], axis=0, inplace=True)
    df['Название локации'].fillna(method='pad', inplace=True)
    df.fillna(-1, inplace=True)
    for i in range(0, 7):
        for loc in loc_list:
            loc_index = list(loc_list).index(loc)
            df_location = df.loc[df['Название локации'] == loc]
            if loc_index == 0:
                 ax = df_location.plot(x='Год', y=param[i], color=color[loc_index % 7], marker=marker[7 - loc_index % 8],
                                        linestyle=line[loc_index % 3], title=param[i], label=loc, figsize=(30, 10))
            else:
                df_location.plot(ax=ax, x='Год', y=param[i], color=color[loc_index % 7], marker=marker[7 - loc_index % 8],
                                linestyle=line[loc_index % 3], title=param[i], label=loc)
        plt.hlines(y=0, xmin=2008, xmax=2023, linewidth=4)
        leg = ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'Results/Descr_statistics/all_loc_{param[i]}_stat.png')


# Графики по годам для описательной статистики
def plot_for_years():
    df = pd.read_excel('Results/Descr_statistics/res_years_1.xlsx', header=1)
    df.rename(columns={'Unnamed: 0': 'Год', 'Unnamed: 1': 'Месяц', 'Unnamed: 2': 'Номер месяца'}, inplace=True)
    df.drop(labels=[0], axis=0, inplace=True)
    df['Год'].fillna(method='pad', inplace=True)
    df['Год и месяц'] = df["Номер месяца"].astype(str) + '/' + df['Год'].astype(int).astype(str)
    df['Год и месяц'] = df['Год и месяц'].apply(lambda _: datetime.strptime(_, "%m/%Y"))
    for i in range(0, 7):
        ax = df.plot(x='Год и месяц', y=param[i], color=color[i], title=f'{param[i]}', marker='o', figsize=(20, 4))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        plt.tight_layout()
        # ax.xaxis.set_major_locator(MultipleLocator(base=1))
        plt.savefig(f'Results/Descr_statistics/all_years_stat_{param[i]}.png')


# Графики для ср. значения и медианы по типу леса
def plot_forest_types():
    labels = ['Cмешанный лес', 'Хвойный лес', 'Лиственный лес']
    df = pd.read_excel('Results/Descr_statistics/forest_types.xlsx', header=0)
    df.rename(columns={'Unnamed: 0': 'Год', 'Unnamed: 1': 'Месяц'}, inplace=True)
    df['Тип леса'].fillna(method='pad', inplace=True)
    df['Год'].fillna(method='pad', inplace=True)
    df['Год и месяц'] = df["Номер месяца"].astype(str) + '/' + df['Год'].astype(int).astype(str)
    df['Год и месяц'] = df['Год и месяц'].apply(lambda _: datetime.strptime(_, "%m/%Y"))
    for p in range(0, 7):
        for i in range(0, 3):
            df_t = df.loc[df['Тип леса'] == labels[i]]
            if i == 0:
                ax = df_t.plot(x='Год и месяц', y=param[p], color=color[i], label=labels[i], title=param[p], marker='x',
                               figsize=(28, 4))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
                ax.set_ylabel('Кол-во клещей на 1 флагочас')
            else:
                df_t.plot(ax=ax, x='Год и месяц', y=param[p], color=color[i], label=labels[i], marker='x')
        plt.tight_layout()
        plt.savefig(f'Results/Descr_statistics/forest_{param[p]}.png')


# Матрица корреляции
def plot_corr_matrix(df):
    df_corr = pd.read_excel('weather.xlsx', header=0).drop(
        columns=['Среднесуточная температура', 'Количество осадков, мм'])
    df_corr = pd.merge(df, df_corr, how='left', on=['Год', 'Месяц'])
    df_corr = df_corr[['Год', 'Имаго I.Persulcatus', 'Нимфы I.Persulcatus', 'Имаго I.Ricinus',
                       'Нимфы I.Ricinus', 'Среднесуточная температура', 'Температура на 1 м. раньше',
                       'Количество осадков, мм', 'Тип леса']]
    df_corr['Тип леса'] = [0 if x == 'Cмешанный лес' else 1 if x == 'Хвойный лес' else -1 if x == 'Лиственный лес'
                                else 100 for x in df_corr['Тип леса']]
    df_corr = df_corr.loc[df_corr['Тип леса'] != 100]
    c_matrix = df_corr.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(0.25, 0.25, 0.93, 0.93)
    sns.heatmap(c_matrix, ax=ax, cmap="YlGnBu", linewidths=0.1)
    fig.savefig('Results/Correlation/corr_matrix.png')


# Графики по локациям для описательной статистики
def plot_for_location(file, location):
    df = pd.read_excel(file, header=1)
    df.rename(columns={'Unnamed: 0': 'Название локации', 'Unnamed: 1': 'Год'}, inplace=True)
    df.drop(labels=[0], axis=0, inplace=True)
    df.fillna(method='pad', inplace=True)
    df_location = df.loc[df['Название локации'] == location]
    fig, axes = plt.subplots(1, 7, figsize=(40, 6))
    fig.suptitle(location, fontsize=15)
    fig.tight_layout()
    for i in range(0, 7):
        df_location.plot(ax=axes[i], x='Год', y=param[i], color='b', title=f'{param[i]}', legend=False, xlabel='',
                         marker='x')
    axes[0].set_ylabel('Кол-во клещей на 1 флагочас')
    fig.savefig(f'Results/Descr_statistics/{location}_stat.png')


# Графики по годам для описательной статистики
def plot_for_year(year):
    df = pd.read_excel('Results/Descr_statistics/res_years_1.xlsx', header=1)
    df.rename(columns={'Unnamed: 0': 'Год', 'Unnamed: 1': 'Месяц', 'Unnamed: 2': 'Номер месяца'}, inplace=True)
    df.drop(labels=[0], axis=0, inplace=True)
    df['Год'].fillna(method='pad', inplace=True)
    df = df.loc[df['Год'] == year]
    fig, axes = plt.subplots(1, 7, figsize=(40, 5))
    for i in range(0, 7):
        df.plot(ax=axes[i], x='Месяц', y=param[i], color='b', title=f'{param[i]}', marker='o', legend=False)
    axes[0].set_ylabel('Кол-во клещей на 1 флагочас')
    fig.tight_layout()
    fig.savefig(f'Results/Descr_statistics/stat_{year}.png')


# Графики для описательной статистики по месяцам
def plot_for_months():
    df = pd.read_excel('Results/Descr_statistics/month_stat.xlsx', header=1)
    df.rename(columns={'Unnamed: 0': 'Номер месяца', 'Unnamed: 1': 'Месяц'}, inplace=True)
    df.drop(labels=[0], axis=0, inplace=True)
    for i in range(0, 7):
        df.plot(x='Месяц', y=param[i], color=color[i], title=f'{param[i]}', marker='o', legend=False,
                ylabel='Кол-во клещей на 1 флагочас')
        plt.tight_layout()
        plt.savefig(f'Results/Descr_statistics/stat_months_{param[i]}.png')


# Графики для описательной статистики по месяцам
def plot_for_types():
    df_r = pd.read_excel('Results/Descr_statistics/I.Ricinus.xlsx', header=0)
    df_p = pd.read_excel('Results/Descr_statistics/I.Persulcatus.xlsx', header=0)
    for i in range(0, 7):
        ax = df_r.plot(x='Год', y=param[i], color=color[1], label='I.Ricinus', title=param[i], marker='x',
                       ylabel='Кол-во клещей на 1 флагочас', figsize=(10, 5))
        df_p.plot(ax=ax, x='Год', y=param[i], color=color[2], label='I.Persulcatus', title=param[i], marker='o')
        plt.tight_layout()
        plt.savefig(f'Results/Descr_statistics/stat_types_{param[i]}.png')


#Линейная регрессия
def plot_regression(X, y, arr, res_reg, location):
    plt.clf()
    plt.scatter(X, y)
    plt.plot(arr, res_reg, color='red', linewidth=2)
    plt.title(f'{location}')
    plt.ylabel('Кол-во клещей на 1 флагочас')
    pyplot.ylim([-1, max(y) + 2])
    plt.savefig(f'Results/Regression/reg_analysis_{location}.png')