import json
import pandas as pd


def df_to_geojson(df, year, lat='Широта', lon='Долгота'):
    properties = ['location', 'median', 'mean', 'min', 'max', 'var', 'std', 'mad', 'types of ticks']
    df_geo = pd.read_excel('Results/Descr_statistics/res_location_1.xlsx', header=1)
    df_geo.rename(columns={'Unnamed: 0': 'Название локации', 'Unnamed: 1': 'Год'}, inplace=True)
    df_geo.drop(labels=[0], axis=0, inplace=True)
    df_geo['Название локации'].fillna(method='pad', inplace=True)
    df_geo = df_geo.merge(
        df[['Название локации', 'Широта', 'Долгота']].drop_duplicates(['Название локации'], keep='first'),
        how='left', on=['Название локации'])
    df_geo = df_geo.loc[df_geo['Год'] == year]
    df_geo['Подвиды'] = [x if len(x) == 2 else x[2:4] for x in df_geo['Подвиды']]
    df_geo['Подвиды'] = ['I.Persulcatus' if x == '+-' else 'I.Ricinus' if x == '-+' else 'I.Persulcatus и I.Ricinus'
                         for x in df_geo['Подвиды']]
    df_geo.rename(columns={'Название локации': 'location', 'Среднее знач.': 'mean', 'Медиана': 'median', 'Мин. знач.':
        'min', 'Макс. знач.': 'max', 'Станд. отклонение': 'std', 'Дисперсия': 'var', 'Среднее абс. отклонение': 'mad',
                           'Подвиды': 'types of ticks'}, inplace=True)
    df_geo.fillna('Нет данных', inplace=True)
    geojson = {'type': 'FeatureCollection', 'features': []}
    for _, row in df_geo.iterrows():
        feature = {'type': 'Feature',
                   'properties': {},
                   'geometry': {'type': 'Point',
                                'coordinates': []}}
        feature['geometry']['coordinates'] = [row[lon], row[lat]]
        for prop in properties:
            feature['properties'][prop] = row[prop]
        geojson['features'].append(feature)
    output_filename = f'Results/Integration/{year}.geojson'
    with open(output_filename, 'w') as output_file:
        json.dump(geojson, output_file, indent=2)
