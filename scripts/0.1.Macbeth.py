
# coding: utf-8




import pyspark
sc = pyspark.SparkContext('local[*]')





# load text of macbeth from s3
shakespeare = sc.textFile("macbeth.txt")





lines = shakespeare.map(lambda line: line.lower())





lines.cache()





macbeth = lines.filter(lambda l: l.find("macbeth") >= 0)
import sys
print(macbeth)





macbeth.count()





macbeth.take(5)





macbeth_and_macduff = macbeth.filter(lambda l: l.find("macduff") >= 0)





print(macbeth_and_macduff)
macbeth_and_macduff.collect()





print(macbeth_and_macduff.toDebugString())





import re
words = lines.flatMap(lambda l: re.split(r'\W+', l))





groupedWords = words.map(lambda w: (w, 1))
unsorted = groupedWords.reduceByKey(lambda a, b: a + b)
unsorted.collect()





print(unsorted.map(lambda t: (t[1], t[0])).sortByKey(False).collect())





sc.stop()





