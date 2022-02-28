import datetime
import os
import warnings
from ast import literal_eval

import numpy as np
import pandas as pd

from assort.add_solvers import setSolver
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
            data = pd.read_excel(file_path_input)
            data['solvers'] = data.apply(lambda x: literal_eval(x['solvers']), axis=1)
            return data

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

        print('[IHLP] Data is loaded.')

        data = pd.DataFrame(sql_requests, columns=Request().fillables)

        data['dateStart'] = data.apply(lambda x: get_date(x, index='solutionDate'), axis=1)
        data = data[~data.dateStart.isnull()]

        data['dateEnd'] = data.apply(lambda x: get_date(x, index='receivedDate'), axis=1)
        data = data[~data.dateEnd.isnull()]

        data['time'] = data.apply(lambda x: get_process_time(x), axis=1)
        data = data[data.time > 0]

        text = Text()

        data['text'] = data.apply(lambda x: text.get_process_text(
            "{}: {}".format(x['subject'], x['description'])
        ), axis=1)

        data = data.fillna('')
        data = data[data['text'].str.split().str.len().lt(300)]

        print('[IHLP] Text processed.')

        data = data[data['time'] >= 0]

        _min = data.time.min()
        _max = data.time.max()

        categories = 6

        data['timeDense'] = data.apply(lambda x: (x['time'] - _min) / (_max - _min), axis=1)
        data['timeCategory'] = pd.qcut(data['time'], q=[i / categories for i in range(categories + 1)], labels=[int(i) for i in range(categories)])

        print('[IHLP] Time processed.')

        data = setSolver(data)

        print('[IHLP] Solvers processed.')

        data = data[['id', 'text', 'solvers', 'time', 'timeDense', 'timeCategory']]
        data.to_excel(file_path_input, index=False)

        return self.ready_data()

    def get(self, use_cache=True, use_all=False):
        data = self.ready_data(use_cache=use_cache, use_all=use_all)
        return data[['id', 'text', 'solvers', 'timeDense', 'timeCategory']]
