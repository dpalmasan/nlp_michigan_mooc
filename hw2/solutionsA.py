import math
import nltk
import time

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000

# TODO: IMPLEMENT THIS FUNCTION
# Calculates unigram, bigram, and trigram probabilities given a training corpus
# training_corpus: is a list of the sentences. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function outputs three python dictionaries, where the keys are tuples expressing the ngram and the value is the log probability of that ngram
def calc_probabilities(training_corpus):
    unigram_p = {}
    bigram_p = {}
    trigram_p = {}

    #bigram_tuples = list(nltk.bigrams(tokens))
    #trigram_tuples = list(nltk.trigrams(tokens))

    # Para contar tokens
    total_tokens = 0
    num_sentences = len(training_corpus)
    # Se itera sobre todas las oraciones del corpus
    for sentence in training_corpus:
        # Para unigramas no se agrega el *
        new_sentence = sentence + ' ' + STOP_SYMBOL

        # Se tokeniza la oracion
        tokens = new_sentence.split()

        # Se agrega al conteo de tokens para obtener el total
        total_tokens += len(tokens)

        # Para cada token del conjunto de token, se cuenta su frec. de aparicion
        for item in set(tokens):
            if (item,) in unigram_p:
                unigram_p[(item,)] += tokens.count(item)
            else:
                unigram_p[(item,)] = tokens.count(item)

        ############
        # Bigramas #
        ############
        # Se agrega el start symbol a la oracion        
        new_sentence = START_SYMBOL + ' ' + new_sentence
        tokens = new_sentence.split()
        
        bigram_tuples = list(nltk.bigrams(tokens))
        for bigram in set(bigram_tuples):
            if bigram in bigram_p:
                bigram_p[bigram] += bigram_tuples.count(bigram)
            else:
                bigram_p[bigram] = bigram_tuples.count(bigram)

        #############
        # trigramas #
        ############
        # Se agrega un segundo start symbol a la oracion        
        new_sentence = START_SYMBOL + ' ' + new_sentence
        tokens = new_sentence.split()
        
        trigram_tuples = list(nltk.trigrams(tokens))
        for trigram in set(trigram_tuples):
            if trigram in trigram_p:
                trigram_p[trigram] += trigram_tuples.count(trigram)
            else:
                trigram_p[trigram] = trigram_tuples.count(trigram)


    # Logaritmo de la probabilidad para trigramas
    for key in trigram_p:
        if key[0] == START_SYMBOL and key[1] == START_SYMBOL:
            trigram_p[key] = math.log(trigram_p[key] / float(num_sentences), 2)
        else:
            trigram_p[key] = math.log(trigram_p[key] / float(bigram_p[(key[0], key[1])]), 2)
    
    # Logaritmo de la probabilidad para bigramas
    for key in bigram_p:
        if key[0] == START_SYMBOL:
            bigram_p[key] = math.log(bigram_p[key] / float(num_sentences), 2)
        else:
            bigram_p[key] = math.log(bigram_p[key] / float(unigram_p[(key[0],)]), 2)

    # Logaritmo de la probabilidad para unigramas
    for key in unigram_p:
        unigram_p[key] = math.log(unigram_p[key]/float(total_tokens), 2)

    return unigram_p, bigram_p, trigram_p

# Prints the output for q1
# Each input is a python dictionary where keys are a tuple expressing the ngram, and the value is the log probability of that ngram
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence
# ngram_p: python dictionary of probabilities of uni-, bi- and trigrams.
# n: size of the ngram you want to use to compute probabilities
# corpus: list of sentences to score. Each sentence is a string with tokens separated by spaces, ending in a newline character.
# This function must return a python list of scores, where the first element is the score of the first sentence, etc. 
def score(ngram_p, n, corpus):
    scores = []

    # Agregar start symbol dependiendo del tipo de ngrama
    sent_start = (n - 1) * (START_SYMBOL + ' ')

    # Para cada oracion del corpus calcular probabilidad
    for sentence in corpus:
        new_sent = sent_start + sentence + ' ' + STOP_SYMBOL
        tokens = new_sent.split()
        if n == 2:
            tokens = list(nltk.bigrams(tokens))   
        elif n == 3:
            tokens = list(nltk.trigrams(tokens))
        else:
            tokens = [(token,) for token in tokens]
        
        prob = 0
        for ngram in tokens:
            if ngram not in ngram_p:
                prob = MINUS_INFINITY_SENTENCE_LOG_PROB
                break           
            prob += ngram_p[ngram]        
        scores.append(prob)

    return scores

# Outputs a score to a file
# scores: list of scores
# filename: is the output file name
def score_output(scores, filename):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + '\n')
    outfile.close()

# TODO: IMPLEMENT THIS FUNCTION
# Calculates scores (log probabilities) for every sentence with a linearly interpolated model
# Each ngram argument is a python dictionary where the keys are tuples that express an ngram and the value is the log probability of that ngram
# Like score(), this function returns a python list of scores
def linearscore(unigrams, bigrams, trigrams, corpus):
    scores = []
    lambda1 = float(1)/3
    lambda2 = lambda1
    lambda3 = lambda1

    for sentence in corpus:
        new_sent = sentence + ' ' + STOP_SYMBOL
        tokens = new_sent.split()

        uni = [(token, ) for token in tokens]
        tokens.insert(0, START_SYMBOL)
        bi = list(nltk.bigrams(tokens))
        tokens.insert(0, START_SYMBOL)
        tri = list(nltk.trigrams(tokens))
        prob = 0
        for idx in range(len(uni)):
            if uni[idx] not in unigrams:
                prob = MINUS_INFINITY_SENTENCE_LOG_PROB
                break        
            unigram_p = lambda1*2**unigrams[uni[idx]]

            if bi[idx] not in bigrams:
                prob = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
            bigram_p = lambda2*2**bigrams[bi[idx]]

            if tri[idx] not in trigrams:
                prob = MINUS_INFINITY_SENTENCE_LOG_PROB
                break
            
            trigram_p = lambda2*2**trigrams[tri[idx]]
            prob += math.log(unigram_p + bigram_p + trigram_p, 2)
        scores.append(prob) 
        
    return scores

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'

# DO NOT MODIFY THE MAIN FUNCTION
def main():
    # start timer
    time.clock()

    # get data
    infile = open(DATA_PATH + 'Brown_train.txt', 'r')
    corpus = infile.readlines()
    infile.close()

    # calculate ngram probabilities (question 1)
    unigrams, bigrams, trigrams = calc_probabilities(corpus)

    # question 1 output
    q1_output(unigrams, bigrams, trigrams, OUTPUT_PATH + 'A1.txt')

    # score sentences (question 2)
    uniscores = score(unigrams, 1, corpus)
    biscores = score(bigrams, 2, corpus)
    triscores = score(trigrams, 3, corpus)

    # question 2 output
    score_output(uniscores, OUTPUT_PATH + 'A2.uni.txt')
    score_output(biscores, OUTPUT_PATH + 'A2.bi.txt')
    score_output(triscores, OUTPUT_PATH + 'A2.tri.txt')

    # linear interpolation (question 3)
    linearscores = linearscore(unigrams, bigrams, trigrams, corpus)

    # question 3 output
    score_output(linearscores, OUTPUT_PATH + 'A3.txt')

    # open Sample1 and Sample2 (question 5)
    infile = open(DATA_PATH + 'Sample1.txt', 'r')
    sample1 = infile.readlines()
    infile.close()
    infile = open(DATA_PATH + 'Sample2.txt', 'r')
    sample2 = infile.readlines()
    infile.close() 

    # score the samples
    sample1scores = linearscore(unigrams, bigrams, trigrams, sample1)
    sample2scores = linearscore(unigrams, bigrams, trigrams, sample2)

    # question 5 output
    score_output(sample1scores, OUTPUT_PATH + 'Sample1_scored.txt')
    score_output(sample2scores, OUTPUT_PATH + 'Sample2_scored.txt')

    # print total time to run Part A
    print "Part A time: " + str(time.clock()) + ' sec'

if __name__ == "__main__": main()
