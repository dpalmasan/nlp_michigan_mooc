from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.metrics.distance import edit_distance as lev
from nltk.stem.porter import *

##############
# Question 1 #
##############

# brown information content
brown_ic = wordnet_ic.ic('ic-brown.dat')

# Words
dog = wn.synset('dog.n.01')
cat = wn.synset('cat.n.01')
elephant = wn.synset('elephant.n.01')

# print similarity
print ("dog - cat: %.3f") % dog.lin_similarity(cat, brown_ic)
print ("dog - elephant: %.3f") % dog.lin_similarity(elephant, brown_ic)

##############
# Question 2 #
##############

# Print Levenshtein edit distance assuming no transpositions and cost = 1
print("%d") % lev("apples", "pears", transpositions=False)

##############
# Question 3 #
##############

# Initialize and use the stemmer
stemmer = PorterStemmer()
print stemmer.stem("computational")
