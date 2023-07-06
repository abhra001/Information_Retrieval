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
            for sent in doc:
                proc_docs+=[sent]
                for word in sent:
                    if word in index.keys():
                        index[word].append(docID)
                    else:
                        index[word] = [docID]

        self.index = index
        self.docIDs = docIDs
        # self.glove_vectors = gensim.downloader.load('glove-wiki-gigaword-200')
        self.word2vec = gensim.models.Word2Vec(proc_docs, min_count=1, window=5, sg=1)

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
        
        index = self.index
        docIDs = self.docIDs
        
        doc_IDs_ordered = []
        N = len(docIDs)
        
        #Fill in code here
       
        data_df = [Counter(index[key]) for key in index.keys()]
        data = pd.DataFrame(data_df, index=index.keys(), columns=docIDs)
        data.fillna(0, inplace=True)
        dfSeries = data.sum(axis=1)
        data['IDF'] = np.log10((N+1)/(dfSeries + 1)) + 1
        
        docVec = {}
        normDict = {}

        for docid in docIDs:
            doc = None
            for word in data.index:
                if data[docid][word] != 0:
                    if word in self.word2vec.wv.key_to_index.keys():
                        if doc is not None:
                            doc += (data['IDF'][word] * data[docid][word]) * self.word2vec.wv[word]
                        else:
                            doc = (data['IDF'][word] * data[docid][word]) * self.word2vec.wv[word]
			
            docVec[docid] = doc
            if doc is not None:
                normDict[docid] = np.linalg.norm(doc)
            else:
                normDict[docid] = 0


        for q in queries:
            q_tf = dict()

            #Finding the TF of the Query 
            for s in q:
                for w in s: 
                    if w not in list(q_tf.keys()):
                        q_tf[w] = 0
                    q_tf[w]+=1

            query_data = pd.DataFrame.from_dict(q_tf,orient='index',columns =['Query'])
            data_full = data.join(query_data, how = 'outer')
            data_full.fillna(0, inplace=True)
            # data_full['tf_idfQuery'] = data_full['Query']*data_full['IDF']
            data_full = data_full.loc[self.index.keys()]

            qvec = None
            for word in data_full.index:
                if data_full['Query'][word] != 0:
                    if word in self.word2vec.wv.key_to_index.keys():
                        if qvec is not None:
                            qvec += (data_full['IDF'][word] * data_full['Query'][word]) * self.word2vec.wv[word]
                        else:
                            qvec = (data_full['IDF'][word] * data_full['Query'][word]) * self.word2vec.wv[word]
            qnorm = np.linalg.norm(qvec)

            #Finding cosine similarity 
            cosine_list = []
            for i in docIDs:
                if normDict[i] == 0:
                    continue

                cosine = np.dot(docVec[i],qvec)/(normDict[i]*qnorm)
                cosine_list.append((cosine,i))

            cosine_list.sort(key=lambda y: y[0], reverse=True)
            temp = [y[1] for y in cosine_list]
            doc_IDs_ordered.append(temp)


        return doc_IDs_ordered