# Text Learning

In order to extract features from text for text learning models, one technique we use is
called the [bag-of-words](https://en.wikipedia.org/wiki/Bag-of-words_model)
model which counts the frequency of words in a sentence but totally ignores
grammar or word order. It's commonly used in in document classification, where
the occurrence of each word is a feature for training a classifier.


### Q&A

Q. What are stopwords? (representation)

A. They are low informational value words that we can remove from our texts
because of their low value. We can usually get stopword lists online pretty easily.

We can grab stopword lists in python from nltk.

---

Q. What is word stemming? (representation)

A. Not all unique words are really unique. There are some words that have permutations,
but they generally mean the same thing (`response`, `respond`, `responsive`).

So we can take stem these words so that we can reduce input space while still
maintaining the same information (reducing all of the above words to `respon`).

---

Q. What is TfIdf Representation?

A.

  Tf = term frequency (term frequency in a document).

  Idf = inverse document frequency (weighting considering how much the term
  occurs relative to the entire corpus of documents).

We give a low weighting to terms that occur frequently documents that encompass
our corpus and a high weighting to rare words that occur infrequently.

### Example

bag-of-words (CountVectorizer) in sklearn

```
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
vectorizer = CountVectorizer()

string1 = "Hey buddy what color is the sky today buddy?"
string2 = "I can't believe that machine learning is this easy"

string1 = stemmer.stem(string1)
string2 = stemmer.stem(string2)

string_list = [string1, string2]

bag_of_words = vectorizer.fit(string_list)

print vectorizer.transform(string_list)
```
