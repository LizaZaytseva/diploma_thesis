import pandas as pd
from matplotlib import pyplot as plt
import analysis
import gen_geojson
import graphics
import loading


def correlation(plots, data_analysis, df_ric, df_pers):
    res = {}
    factors = ['Количество осадков, мм', 'Среднесуточная температура', 'Температура на 1 м. раньше']
    types = ['Имаго I.Ricinus', 'Нимфы I.Ricinus', 'Имаго I.Persulcatus', 'Нимфы I.Persulcatus']
    df_corr = pd.read_excel('Files/weather.xlsx', header=0).drop(
        columns=['Среднесуточная температура', 'Количество осадков, мм', ])
    df_ric = pd.merge(df_ric, df_corr, how='left', on=['Год', 'Месяц'])
    df_pers = pd.merge(df_pers, df_corr, how='left', on=['Год', 'Месяц'])
    for t in range(0, 4):
        for f in range(0, len(factors)):
            if t < 2:
                res = data_analysis.corr_estimation(df_ric, types[t], factors[f], res)
                plots.plot_corr(df_ric, res[len(res) - 1]['factor 1'], res[len(res) - 1]['factor 2'],
                                res[len(res) - 1]['corr. coef'], res[len(res) - 1]['coef. t'])
            else:
                res = data_analysis.corr_estimation(df_pers, types[t], factors[f], res)
                plots.plot_corr(df_pers, res[len(res) - 1]['factor 1'], res[len(res) - 1]['factor 2'],
                                res[len(res) - 1]['corr. coef'], res[len(res) - 1]['coef. t'])
    res_df = pd.DataFrame(res).transpose()
    res_df.to_excel(f'Results/Correlation_results.xlsx')


def main():
    data = loading.LoadData('Files/data.xlsx')
    df = data.load_data()
    data_analysis = analysis.Analysis(df)
    plots = graphics.Graphics()
    ric = data_analysis.filter_type('I.Ricinus')
    pers = data_analysis.filter_type('I.Persulcatus')
    # Корреляционный анализ
    correlation(plots, data_analysis, ric, pers)
    plots.plot_corr_matrix(df)
    # Описательная статистика
    data_analysis.descr_statistics('location', 'Results/Descr_statistics/res_location.xlsx')
    data_analysis.descr_statistics('years', f'Results/Descr_statistics/res_years.xlsx')
    data_analysis.month_descr_statistics('Results/Descr_statistics/month_stat.xlsx')
    data_analysis.types_statistics('Results/Descr_statistics/I.Ricinus.xlsx',
                                   'Results/Descr_statistics/I.Persulcatus.xlsx')
    data_analysis.forest_type('Results/Descr_statistics/forest_types.xlsx')
    # Графики
    plots.plot_for_years()
    plots.plot_for_locations(data_analysis.loc_list)
    plots.plot_for_months()
    plots.plot_for_types()
    plots.plot_for_year(2021)
    plots.plot_forest_types()
    # Регрессионный анализ
    data_analysis.reg_analysis(True, 'Results/Descr_statistics/res_location.xlsx', 'Results/Regression_results.xlsx')
    # Генерация файлов в GeoJSON
    for i in range(2008, 2022):
        gen_geojson.df_to_geojson(df, i)
    # Отображение графиков
    plt.show()


main()
