import pandas as pd


class LoadData(object):
    def __init__(self, file):
        self.file = file

    def load_data(self):
        try:
            data = pd.read_excel(self.file, header=0)
        except FileNotFoundError:
            print('Проверьте название файла')
        assert data is not None, 'Пустой файл'
        return data
