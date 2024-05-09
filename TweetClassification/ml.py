import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import nltk
from nltk.corpus import stopwords

df = pd.read_csv('../data/tweets_one_hot.csv')
def remove_short_word(tweet):
    tweet_ = []
    for word in tweet.split():
        if len(word) < 4:
            continue
            
        tweet_.append(word)

    return " ".join(tweet_)

#Character job
df.tweet = (df.tweet.str.lower()
    ).replace("l’", "", regex=True
    ).replace("-", " ", regex=True
    ).replace("[éèêë]", "e", regex=True
    ).replace("[ç]", "c", regex=True
    ).replace("[àâ]", "a", regex=True
    ).replace("[îï]", "i", regex=True
    ).replace("[ûùü]", "u", regex=True
    ).replace('http?\:\/\/\S*', "", regex=True
    ).replace('^@\S*', "", regex=True
    ).replace('[^A-Za-z\s]', "", regex=True
    ).replace('stminfo', "", regex=True
    ).replace('ligne verte', "ligne", regex=True
    ).replace('ligne orange', "ligne", regex=True
    ).replace('ligne jaune', "ligne", regex=True
    ).replace('ligne bleue', "ligne", regex=True
    )

#Remove short word
df.tweet = df.tweet.apply(remove_short_word)

#Remove english
sw = stopwords.words('english')
swe = set(stopwords.words('english'))
swf = set(stopwords.words('french'))
swe = swe.difference(swf)

def sw_in(tweet):
    for word in tweet.split():
        #exeception
        if word in ['show']:
            continue
            
        if word in swe:
            return True
        
    return False

max_words = 1000
max_len = 50
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df.tweet.values)
sequences = tokenizer.texts_to_sequences(df.tweet.values)
X_processed = pad_sequences(sequences, maxlen=max_len)
X_processed

model = Sequential()
model.add(Embedding(max_words, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X_processed, df.stop.values, test_size=0.2, random_state=42)
X_train_, X_test_, y_train_, y_test_ = train_test_split(df.tweet.values, df.stop.values, test_size=0.2, random_state=42)
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

for i in range(10):
    print(X_test_[i])
    print(model.predict(X_test[i].reshape(1, -1)), y_test_[i])