import sys
import nltk
import math
import time
import re

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []
    
    for s in brown_train:
        words = [START_SYMBOL, START_SYMBOL]
        tags = [START_SYMBOL, START_SYMBOL]
        for m in re.findall(r'(\S+)/([\w.]+)', s):
            words.append(m[0])
            tags.append(m[1])
        words.append(STOP_SYMBOL)
        tags.append(STOP_SYMBOL)
        brown_words.append(words)
        brown_tags.append(tags)

    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}
    bigram_p = {}
    num_sentences = len(brown_tags)
    for tokens in brown_tags:
        bigram_tuples = list(nltk.bigrams(tokens))
        for bigram in set(bigram_tuples):
            if bigram in bigram_p:
                bigram_p[bigram] += bigram_tuples.count(bigram)
            else:
                bigram_p[bigram] = bigram_tuples.count(bigram)

        trigram_tuples = list(nltk.trigrams(tokens))
        for trigram in set(trigram_tuples):
            if trigram in q_values:
                q_values[trigram] += trigram_tuples.count(trigram)
            else:
                q_values[trigram] = trigram_tuples.count(trigram)

    for key in q_values:
        if key[0] == START_SYMBOL and key[1] == START_SYMBOL:
            q_values[key] = math.log(q_values[key] / float(num_sentences), 2)
        else:
            q_values[key] = math.log(q_values[key] / float(bigram_p[(key[0], key[1])]), 2)
    return q_values

# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()
    trigrams.sort()  
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    known_words = set([])
    freq_counts = {}
    for sent in brown_words:
        for word in set(sent):
            if word in freq_counts:
                freq_counts[word] += sent.count(word)
            else:
                freq_counts[word] = sent.count(word)

    for key in freq_counts:
        if freq_counts[key] > RARE_WORD_MAX_FREQ:
            known_words.add(key)

    return known_words

# TODO: IMPLEMENT THIS FUNCTION
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = []
    
    for sent in brown_words:
        new_sent = []
        for word in sent:
            if word in known_words:
                new_sent.append(word)
            else:
                new_sent.append(RARE_SYMBOL)
        brown_words_rare.append(new_sent)

    return brown_words_rare

# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}
    taglist = set([])

    N = len(brown_tags)
    tag_count = {}

    for idx in range(N):
        n_tok = len(brown_tags[idx])
        for jdx in range(n_tok):
            if (brown_words_rare[idx][jdx], brown_tags[idx][jdx]) in e_values:
                e_values[(brown_words_rare[idx][jdx], brown_tags[idx][jdx])] += 1
            else:
                e_values[(brown_words_rare[idx][jdx], brown_tags[idx][jdx])] = 1
            if brown_tags[idx][jdx] in tag_count:
                tag_count[brown_tags[idx][jdx]] += 1
            else:
                tag_count[brown_tags[idx][jdx]] = 1
                taglist.add(brown_tags[idx][jdx])
    
    for key in e_values:
        e_values[key] = math.log(e_values[key] / float(tag_count[key[1]]), 2)    
    return e_values, taglist

# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    emissions.sort()  
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []
    tags = list(taglist)
    T = len(tags)
    for oo in brown_dev_words:
        vb = {}
        bp = {}
        
        # Initialization
        n = len(oo)
        
        for i in xrange(0, T):
            for j in xrange(0, T):
                vb[(0, tags[i], tags[j])] = LOG_PROB_OF_ZERO
        vb[(0, START_SYMBOL, START_SYMBOL)] = 0

        # Recursion step
        for t in range(0, n):
            xk = oo[t]
            if xk not in known_words:
                xk = RARE_SYMBOL
            
            for i in xrange(T):
                for j in xrange(T):
                    u = tags[i]
                    v = tags[j]
                    w = tags[0]
                    best = LOG_PROB_OF_ZERO

                    if (w, u, v) in q_values and (xk, v) in e_values:
                        best = vb[(t, w, u)] + q_values[(w, u, v)] + e_values[(xk, v)]
                    for k in xrange(1, T):
                        cur_w = tags[k]
                        if (cur_w, u, v) in q_values and (xk, v) in e_values:                                
                            cur = vb[(t, cur_w, u)] + q_values[(cur_w, u, v)] + e_values[(xk, v)]
                            if cur > best:
                                best = cur
                                w = cur_w
                    vb[(t + 1, u, v)] = best
                    bp[(t + 1, u, v)] = w

        # Last step for backpointers
        best = LOG_PROB_OF_ZERO
        y1 = tags[0]
        y0 = tags[0]        
        for i in xrange(T):
            for j in xrange(T):
                u = tags[i]
                v = tags[j]
                if (u, v, STOP_SYMBOL) in q_values:
                    if (n, u, v) in vb:
                        cur = vb[(n, u, v)] + q_values[(u, v, STOP_SYMBOL)]
                        if cur > best:
                            y0 = u
                            y1 = v
        
        # Initializing tag vector        
        y = range(n - 1, -1, -1)
        y[n - 1] = y1
        y[n - 2] = y0

        for k in xrange(n - 3, -1, -1):
            y[k] = bp[(k + 3, y[k + 1], y[k + 2])] 
        
        # Transforming to tagged sentence
        s = ''
        for i in xrange(n - 1):
            s += oo[i] + '/' + y[i] + ' '
        s += oo[n - 1] + '/' + y[n - 1]
        tagged.append(s)

    tagged = "\r\n".join(tagged)

    return tagged

# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    '''
    Some commented lines, were my first attempt to the problem. Accuracy
    could be increased using a unigram based on my previous results.
    To get the same as the specs, the tag for the default tagger should be
    NOUN.
    '''
    # Hint: use the following line to format data to what NLTK expects for training
    training = [ zip(brown_words[i],brown_tags[i]) for i in xrange(len(brown_words)) ]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []
    default_tagger = nltk.DefaultTagger('NOUN')
    #unigram_tagger = nltk.UnigramTagger(training, backoff=default_tagger)
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    
    # Tagging sentences
    for sent in brown_dev_words:
        y = trigram_tagger.tag(sent)
        
        # Transforming to tagged sentence
        s = ''
        for word, tag in y:
            s += word + '/' + tag + ' '
        tagged.append(s)
    tagged = "\r\n".join(tagged)
        
    return tagged

# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        outfile.write(sentence)
    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

def main():
    # start timer
    time.clock()

    # open Brown training data
    infile = open(DATA_PATH + "Brown_tagged_train.txt", "r")
    brown_train = infile.readlines()
    infile.close()

    # split words and tags, and add start and stop symbols (question 1)
    brown_words, brown_tags = split_wordtags(brown_train)

    # calculate tag trigram probabilities (question 2)
    q_values = calc_trigrams(brown_tags)

    # question 2 output
    q2_output(q_values, OUTPUT_PATH + 'B2.txt')

    # calculate list of words with count > 5 (question 3)
    known_words = calc_known(brown_words)

    # get a version of brown_words with rare words replace with '_RARE_' (question 3)
    brown_words_rare = replace_rare(brown_words, known_words)

    # question 3 output
    q3_output(brown_words_rare, OUTPUT_PATH + "B3.txt")

    # calculate emission probabilities (question 4)
    e_values, taglist = calc_emission(brown_words_rare, brown_tags)

    # question 4 output
    q4_output(e_values, OUTPUT_PATH + "B4.txt")

    # delete unneceessary data
    del brown_train
    del brown_words_rare

    # open Brown development data (question 5)
    infile = open(DATA_PATH + "Brown_dev.txt", "r")
    brown_dev = infile.readlines()
    infile.close()

    # format Brown development data here
    brown_dev_words = []
    for sentence in brown_dev:
        brown_dev_words.append(sentence.split(" ")[:-1])

    # do viterbi on brown_dev_words (question 5)
    viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

    # question 5 output
    q5_output(viterbi_tagged, OUTPUT_PATH + 'B5.txt')

    # do nltk tagging here
    nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)

    # question 6 output
    q6_output(nltk_tagged, OUTPUT_PATH + 'B6.txt')

    # print total time to run Part B
    print "Part B time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
