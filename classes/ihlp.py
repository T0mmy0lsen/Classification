import datetime
import os
import warnings

import numpy as np
import pandas as pd

from classes.objs.Item import Item
from classes.objs.Request import Request
from classes.objs.Relation import Relation
from classes.text import Text

warnings.filterwarnings("ignore", category=UserWarning, module='bs4')


# Lemmatizer
# Tokenizer
# Remove stopwords
# MongoDB for text search
# https://github.com/kazimirzak/Bachelor/blob/b3c5441ccb46d100b9eb8632a47c69b08761df90/main.py#L96
# https://jovian.ai/diardanoraihan/ensemble-cr/v/2?utm_source=embed#C39
# https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/03.%20Feature%20Engineering/03.%20Feature%20Engineering.ipynb

class IHLP:

    def ready_data(
            self,
            file_path_input='{}/input/data.xlsx'.format(os.path.dirname(__file__)),
            use_cache=True,
            use_all=False
    ):
        str_from = str(datetime.datetime(2016, 1, 1))
        str_to = str(datetime.datetime(2016, 1, 31))

        if os.path.isfile(file_path_input) and use_cache:
            return pd.read_excel(file_path_input)

        def get_date(x, index):
            tmp = x[index]
            if isinstance(x[index], str):
                if tmp[0] != '2':
                    return None
                tmp = datetime.datetime.strptime(tmp, "%Y-%m-%d %H:%M:%S")
            return tmp

        def get_process_time(x):
            x = int(x.dateStart.timestamp()) - int(x.dateEnd.timestamp())
            if x < 1:
                return 0
            return np.log(x) / np.log(10)

        if use_all:
            sql_requests = Request.get_sql()
        else:
            sql_requests = Request.get_between_sql(str_from, str_to)

        sql_relations = Relation.get_relations_by_type('ItemRole')
        sql_items = Item.get_items()

        print('Data is loaded.')

        df_items = pd.DataFrame(sql_items, columns=Item().fillables)
        df_requests = pd.DataFrame(sql_requests, columns=Request().fillables)
        df_relations = pd.DataFrame(sql_relations, columns=Relation().fillables)

        df_items = df_items.rename(columns={'id': 'item.id', 'description': 'item.description', 'subject': 'item.subject'})
        df_requests = df_requests.rename(columns={'id': 'request.id', 'description': 'request.description', 'subject': 'request.subject'})

        data = pd.merge(df_requests, df_relations, left_on='request.id', right_on='leftId')
        data = pd.merge(data, df_items, left_on='rightId', right_on='item.id')

        print('Merge is done.')

        data['dateStart'] = data.apply(lambda x: get_date(x, index='solutionDate'), axis=1)
        data = data[~data.dateStart.isnull()]

        data['dateEnd'] = data.apply(lambda x: get_date(x, index='receivedDate'), axis=1)
        data = data[~data.dateEnd.isnull()]

        data['processSolver'] = data['username']

        data['processTime'] = data.apply(lambda x: get_process_time(x), axis=1)
        data = data[data.processTime > 0]

        text = Text()

        data['processText'] = data.apply(lambda x: text.get_process_text(
            "{} {}".format(x['request.subject'], x['request.description'])
        ), axis=1)

        data = data[['processTime', 'processText', 'processSolver']]
        data.to_excel(file_path_input, index=False)

        return self.ready_data()

    def get(self, categories=2, use_cache=True, use_all=False):

        data = self.ready_data(use_cache=use_cache, use_all=use_all)
        data = data.fillna('')
        data = data[data['processText'].str.split().str.len().lt(300)]
        data = data[data['processTime'] >= 0]

        _min = data.processTime.min()
        _max = data.processTime.max()

        data['processTimeDense'] = data.apply(lambda x: (x['processTime'] - _min) / (_max - _min), axis=1)
        data['processCategory'] = pd.qcut(data['processTime'], q=[i/categories for i in range(categories + 1)], labels=[int(i) for i in range(categories)])

        return data[['processText', 'processTimeDense', 'processCategory']]
