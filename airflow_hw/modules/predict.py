import json
import dill
import pandas as pd
import os
import sys

path = os.path.expanduser('~/airflow_hw')
# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)

with open(f'{path}/data/models/cars_pipe.pkl', 'rb') as file:
    model = dill.load(file)


def predict():
    h1 = []

    for filename in os.listdir(f'{path}/data/test'):
        with open(f'{path}/data/test/' + filename, 'r') as f:
            data = json.loads(f.read())
            h1.append(data)

    df = pd.DataFrame(h1)
    df = df[['id', 'url', 'region', 'region_url', 'price', 'year', 'manufacturer',
       'model', 'fuel', 'odometer', 'title_status', 'transmission',
       'image_url', 'description', 'state', 'lat', 'long', 'posting_date'
             ]]
    y = model.predict(df)

    dfr = pd.DataFrame(zip(df.id, y))
    dfr.columns = [['id', 'predict']]

    dfr.to_csv(f'{path}/data/predictions/cars_predict.csv', index=False)


if __name__ == '__main__':
    predict()
