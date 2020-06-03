import csv
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer # generate token_index for word
from tensorflow.keras.preprocessing.sequence import pad_sequences #uniform the size of text
#setting the basic infromation
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_portion = .8

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be",
        "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does",
        "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he",
        "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
        "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me",
        "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours",
        "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
        "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
        "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's",
        "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd",
        "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

# save text and use ' ' to replace of stopwords
sentences = []
labels = []
with open("/Users/annawang/Documents/GitHub/Coursera_ML/bbc-text.csv",'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            labels.append(row[0])
            sentence = row[1]
            for word in stopwords:
                token = " " + word + " "
                sentence = sentence.replace(token, " ")
                sentence = sentence.replace("  ", " ")
            sentences.append(sentence)

print(len(sentences))

# tokenize the words and sentences
#def Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?...', lower=True, split=' ', char_level=False, oov_token=None, document_count=0, **kwargs)
# train dataset
train_size = int(len(sentences) * training_portion)
train_sentences = sentences[:train_size]
train_labels = labels[:train_size]

validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]

print(train_size)
print(len(train_sentences))
print(len(train_labels))
print(len(validation_sentences))
print(len(validation_labels))

tokenizer = Tokenizer(num_words = vocab_size,oov_token = oov_tok) # give unseen word oov as special token value
tokenizer.fit_on_texts(train_sentences) # try to give code to any word in the sentence
word_index = tokenizer.word_index # build a dictionary with key-value, key is the code and value is the word
print(len(word_index))

train_sequences = tokenizer.texts_to_sequences(train_sentences) # give code to the word in sentences, encode the sentence by using code
train_padded = pad_sequences(train_sequences, padding= padding_type, maxlen=max_length) # 0 will at the end of sentence and uniform to the longest sentence
print(len(train_padded[0]))
print(len(train_sequences[0]))

print(len(train_sequences[1]))
print(len(train_padded[1]))

print(len(train_sequences[10]))
print(len(train_padded[10]))


label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)

print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)

label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(labels)
print(label_seq) #labe sequence (the token for the sentence)
print(label_word_index) #label dic

#build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 30
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "acc")
plot_graphs(history, "loss")


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

# Expected output
# (1000, 16)

import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')