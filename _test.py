import numpy as np
import pickle
import json

import tensorflow
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope

from classes.objs.Relation import Relation
from classes.objs.Request import Request

from tcn import TCN

from classes.text import Text


def findSolvers():

    df_requests = Request.get_sql(10)
    df_relations = Relation.get_relations_by_type('ItemRole')
    df_requests['solvers'] = df_requests.apply(lambda x: determineSolvers(x, df_relations), axis=1)
    print(df_requests)


def determineSolvers(x, relations):

    tmp = relations[relations['leftId'] == x['id']]
    tmp = tmp[tmp['rightType'] == 'ItemRole']
    if len(tmp) > 2:
        tmp.sort_values(by=['id'], inplace=True)
        rm = tmp.iloc[0:2]
        rm_list = rm['rightId'].tolist()
        tmp = tmp[~tmp['rightId'].isin(rm_list)]
        return tmp['rightId'].tolist()
    else:
        return []


def main():

    with CustomObjectScope({'TCN': TCN}):
        model = tensorflow.keras.models.load_model('saved_model/model_ihlp_time/1.h5')
        model.summary()

    text = Text()

    with open('saved_model/model_ihlp_time/1.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        data = json.load(open('saved_model/model_ihlp_time/1.json'))
        sample_text = text.get_process_text(data['text'])
        test_sequences = tokenizer.texts_to_sequences([sample_text])
        test = pad_sequences(test_sequences, maxlen=data['max_len'], padding=data['padding_type'], truncating=data['trunc_type'])
        predictions = model.predict(test)
        print(predictions)

    pass


if __name__ == '__main__':
    main()