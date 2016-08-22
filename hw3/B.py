import A
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn import ensemble
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from time import time
from sklearn.feature_selection import SelectKBest, chi2

# You might change the window size
window_size = 15
colloc_size = 2

# B.1.a,b,c,d
def extract_features(data, language):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :return: features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }
    '''
    features = {}
    labels = {}

    # Asumimos que los experimentos seran en lenguajes que soporta el stemmer (No mejora mucho en realidad)
    # stemmer = nltk.stem.snowball.SnowballStemmer(language)

    # implement your code here
    S = set([])
    sur_words = set([])

    # Testeando con otro tokenizer que remueve signos de puntuacion (Empeora la precision)
    # tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
    for instance_id, left_context, head, right_context, sense_id in data:
        LC = nltk.word_tokenize(left_context)
        RC = nltk.word_tokenize(right_context)
        Si = LC[-window_size:] + RC[0:window_size]
    
        # Agregar word collocations
        sur_words.add(tuple(LC[-colloc_size:]))
        sur_words.add(tuple(RC[0:colloc_size]))

        # Remover stop words
        try:
            stop = stopwords.words(language)
        except IOError:
            stop = []
        Si = [word for word in Si if word.lower() not in stop]
        
        for s in Si:
            S.add(s)


    for instance_id, left_context, head, right_context, sense_id in data:
        feat = {}    
        
        # Etiqueta para la instancia
        labels[instance_id] = sense_id

        # Extraer palabras para modelo de espacio vectorial
        LC = nltk.word_tokenize(left_context)
        RC = nltk.word_tokenize(right_context)
        Si = LC[-window_size:] + RC[0:window_size]
        

        for word in S:
            feat[word] = 0
        for word in Si:
            if word in S:
                feat[word] = Si.count(word)

        # Extraer collocational features (palabras)
        sur = [tuple(LC[-colloc_size:]), tuple(RC[0:colloc_size])]
        for colloc in sur_words:
            feat[u'WL-{}'.format(colloc)] = 0
        for colloc in sur:
            if colloc in sur_words:
                feat[u'WL-{}'.format(colloc)] = 1

        features[instance_id] = feat


    return features, labels

# implemented for you
def vectorize(train_features,test_features):
    '''
    convert set of features to vector representation
    :param train_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :param test_features: A dictionary with the following structure
             { instance_id: {f1:count, f2:count,...}
            ...
            }
    :return: X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
            X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    '''
    X_train = {}
    X_test = {}

    vec = DictVectorizer()
    vec.fit(train_features.values())
    for instance_id in train_features:
        X_train[instance_id] = vec.transform(train_features[instance_id]).toarray()[0]

    for instance_id in test_features:
        X_test[instance_id] = vec.transform(test_features[instance_id]).toarray()[0]

    return X_train, X_test

#B.1.e
def feature_selection(X_train,X_test,y_train):
    '''
    Try to select best features using good feature selection methods (chi-square or PMI)
    or simply you can return train, test if you want to select all features
    :param X_train: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param X_test: A dictionary with the following structure
             { instance_id: [f1_count,f2_count, ...]}
            ...
            }
    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }
    :return:
    '''

    return X_train, X_test
    # implement your code here
    X_train_new = {}
    X_test_new = {}

    x_train = []
    x_test = []
    y = []
    for instance_id in X_train:
        x_train.append(X_train[instance_id])
        y.append(y_train[instance_id])
    for instance_id in X_test:
        x_test.append(X_test[instance_id])
        
    # Como criterio de prueba nos quedamos con el 60% de las features
    N = len(x_train[0])
    ch2 = SelectKBest(chi2, k=int(round(0.9*N)))
    x_train_new = ch2.fit_transform(x_train, y)

    k = 0
    for instance_id in X_train:
        X_train_new[instance_id] = x_train_new[k]
        k += 1

    x_test = ch2.transform(x_test)
    k = 0
    for instance_id in X_test:
        X_test_new[instance_id] = x_test[k]
        k += 1
    
    return X_train_new, X_test_new
    # or return all feature (no feature selection):
    # return X_train, X_test

# B.2
def classify(X_train, X_test, y_train):
    '''
    Train the best classifier on (X_train, and y_train) then predict X_test labels

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

    :return: results: a list of tuples (instance_id, label) where labels are predicted by the best classifier
    '''

    results = []


    # implement your code here
    clf = svm.LinearSVC()
    # clf = ensemble.RandomForestClassifier()
    # clf = svm.SVC(kernel="linear", C=0.025)
    # clf = svm.SVC(gamma=2, C=1)
    # clf = svm.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    training_x = []
    training_y = []
    for key in X_train:
        training_x.append(X_train[key])
        training_y.append(y_train[key])
 
    # implement your code here
    clf.fit(training_x, training_y)

    for instance in X_test:
        results.append((instance, clf.predict(X_test[instance])))

    return results

# run part B
def run(train, test, language, answer):
    results = {}

    for lexelt in train:

        train_features, y_train = extract_features(train[lexelt], language)
        test_features, _ = extract_features(test[lexelt], language)

        X_train, X_test = vectorize(train_features,test_features)
        X_train_new, X_test_new = feature_selection(X_train, X_test,y_train)
        results[lexelt] = classify(X_train_new, X_test_new,y_train)

    A.print_results(results, answer)
