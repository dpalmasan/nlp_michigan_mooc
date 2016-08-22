from main import replace_accented
from sklearn import svm
from sklearn import neighbors
import nltk
import codecs

# don't change the window size
window_size = 10

# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}

    # implement your code here
    for lexelt in data:
        Si = []
        for instance in data[lexelt]:
            LC = nltk.word_tokenize(instance[1])
            RC = nltk.word_tokenize(instance[3])
            Si = Si + LC[-window_size:] + RC[0:window_size]

        s[lexelt] = set(Si)   

    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}

    # implement your code here
    for instance in data:
        LC = nltk.word_tokenize(instance[1])
        RC = nltk.word_tokenize(instance[3])
        Si = LC + RC
        vectors[instance[0]] = [Si.count(i) for i in s]
        labels[instance[0]] = instance[4]

    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''

    svm_results = []
    knn_results = []

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()

    training_x = []
    training_y = []
    for key in X_train:
        training_x.append(X_train[key])
        training_y.append(y_train[key])
 
    # implement your code here
    svm_clf.fit(training_x, training_y)
    knn_clf.fit(training_x, training_y)

    for instance in X_test:
        svm_results.append((instance, svm_clf.predict(X_test[instance])))
        knn_results.append((instance, knn_clf.predict(X_test[instance])))

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results, output_file):
    '''
    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output
    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing

    outfile = codecs.open(output_file, encoding='utf-8', mode='w')
    for lexelt, instances in sorted(results.iteritems(), key=lambda d: replace_accented(d[0].split('.')[0])):
        for instance in sorted(instances, key=lambda d: int(d[0].split('.')[-1])):
            instance_id = instance[0]
            sid = instance[1]
            outfile.write(replace_accented(lexelt + ' ' + instance_id + ' ' + sid[0] + '\n'))
    outfile.close()

# run part A
def run(train, test, language, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)

    print_results(svm_results, svm_file)
    print_results(knn_results, knn_file)



