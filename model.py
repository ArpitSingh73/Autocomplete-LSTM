import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts([faqs])
len(tokenizer.word_index)

input_sequences = []
for sentence in faqs.split("\n"):
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[: i + 1])

max_len = max([len(x) for x in input_sequences])

from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding="pre")

X = padded_input_sequences[:, :-1]

y = padded_input_sequences[:, -1]

X.shape

y.shape

from tensorflow.keras.utils import to_categorical

y = to_categorical(y, num_classes=283)
