import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import analysis
import gen_geojson
import graphics


def load_data():
    data = pd.read_excel('data.xlsx', header=0)
    return data


# Чтение данных из файла
s = load_data()
df = analysis.prep_df(s)

# Создание списка локаций
loc_list = analysis.location_list(s)
print(loc_list)
# Фильтрация по подтипу
ric = analysis.filter_type(s, 'I.Ricinus')
pers = analysis.filter_type(s, 'I.Persulcatus')

# Вычисление коэффициента коррелляции
analysis.corr_estimation(ric, 'Имаго I.Ricinus', 'Количество осадков, мм')
analysis.corr_estimation(ric, 'Имаго I.Ricinus', 'Среднесуточная температура')
analysis.corr_estimation(ric, 'Нимфы I.Ricinus', 'Количество осадков, мм')
analysis.corr_estimation(ric, 'Нимфы I.Ricinus', 'Среднесуточная температура')
analysis.corr_estimation(pers, 'Имаго I.Persulcatus', 'Количество осадков, мм')
analysis.corr_estimation(pers, 'Имаго I.Persulcatus', 'Среднесуточная температура')
analysis.corr_estimation(pers, 'Нимфы I.Persulcatus', 'Количество осадков, мм')
analysis.corr_estimation(pers, 'Нимфы I.Persulcatus', 'Среднесуточная температура')

# Описательная статистика
analysis.descr_statistics(df, 'location')
analysis.descr_statistics(df, 'years')
analysis.month_descr_statistics(df)
analysis.types_statistics(s)

#Описательная статистика: графики
graphics.plot_for_years()
graphics.plot_for_locations(loc_list)
graphics.plot_for_months()
graphics.plot_for_types()
graphics.plot_for_year(2021)

#Вычисление среднего и медианы для каждого типа леса + отрисовка графиков
analysis.forest_type(df)
graphics.plot_forest_types()

# Матрица корреляции
graphics.plot_corr_matrix(df)

#Регрессионный анализ
analysis.reg_analysis('ДОЛ Луч')
# analysis.m_reg_analysis('ДОЛ Луч')
plt.show()

# Генерация файлов в GeoJSON
for i in range(2008, 2022):
    gen_geojson.df_to_geojson(df, i)
