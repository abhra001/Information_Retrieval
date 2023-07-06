#from util import *

# Add your import statements here
import math
import pandas as pd 
import numpy as np
import gensim
import gensim.downloader
from collections import Counter

class InformationRetrieval():

    def __init__(self):
        self.index = None
        #self.tf = None
        #self.docs = None
        self.docIDs = None
        self.documents = None

    def tf_idf_query(self,Query,IDF_docs,wiki_words,docs_wiki): 
    
    #Number of articles: 
        numart = wiki_words.shape[1]
        
        #Getting Term Frequency
        termf = dict()
        for sent in Query:
            for word in sent:
                if word in termf.keys():
                    termf[word] = termf[word]+1
                else:
                    termf[word] = 1
        
        default = np.log10((1400+1)/(1)) + 1
        
        total = np.zeros(4543)    
        for key in termf.keys():
            try:
                idf = IDF_docs[key]
            except: 
                idf = default
                
            tf_idf = termf[key]*idf
            
            try:
                vec = list(wiki_words.loc[key])
            except:
                vec = [0]*numart
            vec_np = np.array(vec)
            weighted_vec = tf_idf*vec_np
            
            total = total + weighted_vec
            
        norm_q = np.linalg.norm(total)
        
        cosine = list(docs_wiki.apply(lambda y: np.dot(y,total)/(np.linalg.norm(y)*norm_q) if (np.linalg.norm(y)!=0) else 0, axis = 1))
        ids = list(range(1,1401))
        cosine_list = list(zip(cosine,ids))
        cosine_list.sort(key=lambda y: y[0], reverse=True)
        temp = [y[1] for y in cosine_list]
        
        return temp

    def ir_esa(self,Queries, IDF_docs,wiki_words,docs_wiki):
        
        overall = []
        count = 0
        for query in Queries: 
            ordered = self.tf_idf_query(query, IDF_docs,wiki_words,docs_wiki)
            overall.append(ordered)
            print(count)
            count = count + 1 
            
        return overall
