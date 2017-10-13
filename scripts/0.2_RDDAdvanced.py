# coding: utf-8

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


import pyspark

sc = pyspark.SparkContext('local[*]')


def tokenize(s):
    import re
    stopwords = set(
        ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
         'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
         'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
         'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
         'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
         'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
         'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
         'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
         'should', 'now'])
    word_regex = '^[a-z][a-z\'-]+[a-z]$'
    s = s.lower()
    arr = s.split()
    terms = []
    for term in arr:
        if re.match(word_regex, term) != None and len(term) > 3 and term not in stopwords:
            terms.append(term)
    return terms





test_strings = ['the quick brown fox jumps over the brown fence.',
                'the boy paints a tall fence brown!',
                'basketball players are tall.',
                'quick basketball players jump high']




tokens = sc.parallelize(test_strings).map(tokenize)



logging.info('Tokens')
logging.info(tokens.collect())




vocab = tokens.flatMap(lambda words: words).distinct()
logging.info('Vocab.collect')
logging.info(vocab.collect())




from collections import Counter
import numpy as np

# sc.broadcast shares an immutable object throughout the cluster
broadcastVocab = sc.broadcast(vocab.collect())


def bow_vectorize(tokens):
    word_counts = Counter(tokens)
    logging.info('word_counts')
    logging.info(word_counts)
    vector = [word_counts[v] if v in word_counts else 0 for v in broadcastVocab.value]
    return np.array(vector)



collected_tokens = tokens.map(bow_vectorize).collect()
logging.info('collected_tokens')
logging.info(collected_tokens)
print(collected_tokens)

term_freq = tokens.map(lambda terms: Counter(terms))
logging.info('term_freq')
logging.info(term_freq)




doc_freq = term_freq.flatMap(lambda counts: counts.keys()).map(lambda keys: (keys, 1)).reduceByKey(lambda a, b: a + b)
logging.info('doc_freq.collect()')
logging.info(doc_freq.collect())





total_docs = term_freq.count()
logging.info('total_docs')
logging.info(total_docs)





import math

idf = doc_freq.map(lambda tup: (tup[0], math.log(float(total_docs) / (1 + tup[1])))).collect()
logging.info('idf')
logging.info(idf)




broadcast_idf = sc.broadcast(idf)





def tfidf_vectorize(tokens):
    word_counts = Counter(tokens)
    doc_length = sum(word_counts.values())

    vector = [(word_counts.get(word[0], 0) / float(doc_length) ) * word[1]  for word in broadcast_idf.value]
    return np.array(vector)





tfidf = tokens.map(tfidf_vectorize)
logging.info('tfidf.collect()')
logging.info(tfidf.collect())




bow = tokens.map(bow_vectorize).cache()
logging.info('bow.collect()')
logging.info(bow.collect())

from pyspark.mllib.clustering import KMeans, KMeansModel
from math import sqrt




clusters = KMeans.train(tfidf, 2, maxIterations=10, initializationMode="random")





def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x ** 2 for x in (point - center)]))





WSSSE = tfidf.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))


logging.info('clusters.centers')
logging.info(clusters.centers)






top_n = 3
print([idf[idx][0] for idx in [np.argsort(x)[::-1][:top_n] for x in clusters.centers][0]])
print([idf[idx][0] for idx in [np.argsort(x)[::-1][:top_n] for x in clusters.centers][1]])

# In[ ]:


sc.stop()


# In[ ]:



