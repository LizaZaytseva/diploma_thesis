import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import analysis
import gen_geojson
import graphics


def load_data():
    try:
        data = pd.read_excel('data.xlsx', header=0)
    except FileNotFoundError:
        print('Проверьте название файла')
    return data


def correlation(df_ric, df_pers):
    res = {}
    factors = ['Количество осадков, мм', 'Среднесуточная температура', 'Температура на 1 м. раньше']
    types = ['Имаго I.Ricinus', 'Нимфы I.Ricinus', 'Имаго I.Persulcatus', 'Нимфы I.Persulcatus']
    df_corr = pd.read_excel('weather.xlsx', header=0).drop(
        columns=['Среднесуточная температура', 'Количество осадков, мм', ])
    df_ric = pd.merge(df_ric, df_corr, how='left', on=['Год', 'Месяц'])
    df_pers = pd.merge(df_pers, df_corr, how='left', on=['Год', 'Месяц'])
    for t in range(0, 4):
        for f in range(0, len(factors)):
            if t < 2:
                res = analysis.corr_estimation(df_ric, types[t], factors[f], res)
            else:
                res = analysis.corr_estimation(df_pers, types[t], factors[f], res)
    res_df = pd.DataFrame(res).transpose()
    res_df.to_excel(f'Results/Correlation_results.xlsx')

def main():
    s = load_data()
    assert s is not None, 'Пустой файл'
    df = analysis.prep_df(s)
    loc_list = analysis.location_list(s)
    ric = analysis.filter_type(df, 'I.Ricinus')
    pers = analysis.filter_type(df, 'I.Persulcatus')
    # Корреляционный анализ
    correlation(ric, pers)
    graphics.plot_corr_matrix(df)
    # Описательная статистика
    analysis.descr_statistics(df, 'location')
    analysis.descr_statistics(df, 'years')
    analysis.month_descr_statistics(df)
    analysis.types_statistics(df)
    analysis.forest_type(df)
    graphics.plot_for_years()
    graphics.plot_for_locations(loc_list)
    graphics.plot_for_months()
    graphics.plot_for_types()
    graphics.plot_for_year(2021)
    graphics.plot_forest_types()
    # Регрессионный анализ
    analysis.reg_analysis(loc_list)
    # plt.show()
    # Генерация файлов в GeoJSON
    for i in range(2008, 2022):
        gen_geojson.df_to_geojson(df, i)

main()