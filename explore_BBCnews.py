#!wget https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
import csv
from tensorflow.keras.preprocessing.text import Tokenizer # generate token_index for word
from tensorflow.keras.preprocessing.sequence import pad_sequences #uniform the size of text

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
with open("/Users/annawang/Documents/RAproject/coursera_ml/bbc-text.csv",'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    next(data)
    for row in data:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
        sentences.append(sentence)
print(len(sentences))
print(sentences[0])

# tokenize the words and sentences
#def Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?...', lower=True, split=' ', char_level=False, oov_token=None, document_count=0, **kwargs)
# train dataset
tokenizer = Tokenizer(oov_token = "<oov>") # give unseen word oov as special token value
tokenizer.fit_on_texts(sentences) # try to give code to any word in the sentence
word_index = tokenizer.word_index # build a dictionary with key-value, key is the code and value is the word
print(len(word_index))

sequences = tokenizer.texts_to_sequences(sentences) # give code to the word in sentences, encode the sentence by using code
padded = pad_sequences(sequences, padding='post') # 0 will at the end of sentence and uniform to the longest sentence
print(padded[0])
print(padded.shape)


label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(labels)
print(label_seq) #labe sequence (the token for the sentence)
print(label_word_index) #label dic