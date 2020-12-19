# Random Linear Oracle by: las3

import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import Perceptron
from sklearn.cluster import KMeans
from collections import Counter
import pandas as pd

class dumb_base:
    classe: int   
        
    def __init__(self, value):
        self.classe = value
    
    def fit(self, X_train, y_train):
        pass
    
    def predict(self,X_test):
        saida = []
        for index in range(len(X_test)):
            saida.append(self.classe)
        
        return saida


class clf_base_r:
    clf_1: Perceptron
    clf_2: Perceptron
    h: KMeans   
        
    def __init__(self):
        self.clf_1 = Perceptron()
        self.clf_2 = Perceptron()
        self.h = KMeans
    
    def predict(self,X_test):
        saida = []
        for index, row in X_test.iterrows():
            dim = self.h.predict([row])[0]
            #print('kmeasn dim',dim)
            if(dim == 0):
                saida.append(self.clf_1.predict([row])[0])
            else:
                saida.append(self.clf_2.predict([row])[0])
        
        return saida
        
    def fit(self,X_train, y_train):
        
        self.h = KMeans(n_clusters=2, n_init=1, max_iter=1).fit(X_train)
         
        k_selection = self.h.predict(X_train)
        
        base1 =  []
        y1 = []
        
        base2 =  []
        y2 = []
                
        for index, d1 in enumerate(k_selection):
            if(d1 == 0):
                base1.append(X_train.iloc[index])
                y1.append(y_train[index])
            else:
                base2.append(X_train.iloc[index])
                y2.append(y_train[index])                
        
        base1 = pd.DataFrame(base1,columns=X_train.columns)
        base2 = pd.DataFrame(base2,columns=X_train.columns)
        if len(Counter(y1).keys()) >1:
            self.clf_1.fit(base1,y1)
        else:
             self.clf_1 = dumb_base(y1[0])
        if len(Counter(y2).keys()) >1:
            self.clf_2.fit(base2,y2)
        else:
             self.clf_2 = dumb_base(y2[0])
        
        return self

    
class random_linear_oracle:
    n_: int
    pool_classifiers: []
        
        
    def __init__(self, n_value=10):
        self.n_ = n_value
        self.pool_classifiers = []
        
        for x in range(self.n_):            
            self.pool_classifiers.append(clf_base_r())
    
    def fit(self,X_train, y_train):
        for x in range(self.n_): 
            self.pool_classifiers[x] = self.pool_classifiers[x].fit(X_train, y_train)
        return self
    
    def predict(self,X_test):
        saidas = []
        for x in range(self.n_): 
            saidas.append(self.pool_classifiers[x].predict(X_test))                
        
        saidas_clfs = [list(x) for x in zip(*saidas)]
        saida = []
        for v in saidas_clfs:
            occurence_count = Counter(v) 
            saida.append(occurence_count.most_common(1)[0][0])
        

        return saida
    
    def oracle(self,X_test, y_test):
        saidas = []
        for x in range(self.n_): 
            saidas.append(self.pool_classifiers[x].predict(X_test))                
        
        saidas_clfs = [list(x) for x in zip(*saidas)]
        hits = 0
        for index, v in enumerate(saidas_clfs):
             if(y_test[index] in v):
                hits = hits + 1
        

        return hits/len(y_test)
