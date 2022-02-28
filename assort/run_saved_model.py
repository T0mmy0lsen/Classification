import pickle
import json

import tensorflow
from tcn import TCN
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope

from classes.text import Text


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