#from util import *

# Add your import statements here
import math
import pandas as pd 
import numpy as np
import gensim
from collections import Counter

class InformationRetrieval():

    def __init__(self):
        self.index = None
        #self.tf = None
        #self.docs = None
        self.docIDs = None
        self.dictionary = None
        self.corpus = None

    def buildIndex(self, docs, docIDs):
        """
        Builds the document index in terms of the document
        IDs and stores it in the 'index' class variable

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is
            a document and each sub-sub-list is a sentence of the document
        arg2 : list
            A list of integers denoting IDs of the documents
        Returns
        -------
        None
        """
        #Computing the inverted index: 
        index = dict()
        #Fill in code here
        proc_docs = []
        for doc, docID in zip(docs, docIDs):
            d=[]
            for sent in doc:
                d=d+sent
                for word in sent:
                    if word in index.keys():
                        index[word].append(docID)
                    else:
                        index[word] = [docID]
            proc_docs+=[d]
        
        self.dictionary = gensim.corpora.Dictionary(proc_docs)
        self.corpus = [self.dictionary.doc2bow(text) for text in proc_docs]

        self.index = index
        self.docIDs = docIDs
        

    def rank(self, queries):
        """
        Rank the documents according to relevance for each query

        Parameters
        ----------
        arg1 : list
            A list of lists of lists where each sub-list is a query and
            each sub-sub-list is a sentence of the query
        

        Returns
        -------
        list
            A list of lists of integers where the ith sub-list is a list of IDs
            of documents in their predicted order of relevance to the ith query
        """
        '''
        index = self.index
        tf = self.tf
        docIDs = self.docIDs
        docs = self.docs
        '''
       # Definitions: 
           # idf = log((N+1)/(n+1)) 
           # tf = term frequency = count of each word in that document (Not normalised)
            
        tfidf = gensim.models.TfidfModel(self.corpus, smartirs='npu')
        corpus_tfidf = tfidf[self.corpus]
        
        lsi = gensim.models.LsiModel(corpus_tfidf, num_topics=700)
        ind = gensim.similarities.MatrixSimilarity(lsi[corpus_tfidf])        

        doc_IDs_ordered = []
        
        for query in queries:
            q=[]
            for sent in query:
                q+=sent
            vec = self.dictionary.doc2bow(q)
            vec_bow_tfidf = tfidf[vec]
            vec_lsi = lsi[vec_bow_tfidf]
            sims = ind[vec_lsi]
            sort_sim=sorted(enumerate(sims), key=lambda item: -item[1])
            sort_ids=[]
            for s in sort_sim:
                sort_ids+=[s[0]+1]
            doc_IDs_ordered+=[sort_ids]

        return doc_IDs_ordered