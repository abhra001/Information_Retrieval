#from util import *

# Add your import statements here
import math
import pandas as pd 
import numpy as np
from collections import Counter

# #Put in Main (right before calling the function) 
# data = pd.read_csv('doc_words_final.csv')
# data = data.set_index('Unnamed: 0')
# #print("Docs Data Imported") 
# idf_values = data['IDF'].to_dict()
# norm = {}
# data_tf_idf = pd.DataFrame()
# for i in range(1,1401):
#     data_tf_idf[i] = data[str(i)]*data['IDF']
#     norm[i] = np.linalg.norm(data_tf_idf[i])
# data_tf_idf['IDF'] = data['IDF']

# #Call the function like this: 
# informationRetriever.rank(Queries,data_tf_idf,norm)

# informationRetriever.rank([query,data_tf_idf,norm)

class InformationRetrieval():

    def __init__(self):
        self.index = None
        #self.tf = None
        #self.docs = None
        self.docIDs = None

    def rank(self, queries, data, norm):
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
        
        
        doc_IDs_ordered = []
        N = 1400
        
        count = 0 
        for q in queries:
            q_tf = dict()
            #Finding the TF of the Query 
            for s in q:
                for w in s: 
                    if w not in list(q_tf.keys()):
                        q_tf[w] = 1
                    q_tf[w]+=1

            query_data = pd.DataFrame.from_dict(q_tf,orient='index',columns =['Query'])
            data_full = data.join(query_data, how = 'outer')
            data_full.fillna(0, inplace=True)
            data_full['tf_idfQuery'] = data_full['Query']*data_full['IDF']
            qnorm = np.linalg.norm(data_full['tf_idfQuery'])

            #Finding cosine similarity 
            cosine_list = []
            for i in range(1,1401):
                if norm[i] == 0:
                    continue

                cosine = np.dot(data_full[i],data_full['tf_idfQuery'])/(norm[i]*qnorm)
                cosine_list.append((cosine,i))

            cosine_list.sort(key=lambda y: y[0], reverse=True)
            temp = [y[1] for y in cosine_list]
            doc_IDs_ordered.append(temp)
            print(count)
            count = count + 1

        return doc_IDs_ordered