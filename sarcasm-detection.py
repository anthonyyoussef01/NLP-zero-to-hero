import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('Data/sarcasm.json', 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

tonizer = Tokenizer(oov_token="<OOV>")
tonizer.fit_on_texts(sentences)
word_index = tonizer.word_index

sequences = tonizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)

# training_size = 20000
# training_sentences = sentences[0:training_size]
# training_labels = labels[0:training_size]
# testing_sentences = sentences[training_size:]
# testing_labels = labels[training_size:]
#
# tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
# tokenizer.fit_on_texts(training_sentences)
#
# word_index = tokenizer.word_index
#
# training_sequences = tokenizer.texts_to_sequences(training_sentences)
# training_padded = pad_sequences(training_sequences, padding='post', maxlen=max_length, truncating='post')
#
# testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
# testing_padded = pad_sequences(testing_sequences, padding='post', maxlen=max_length, truncating='post')


