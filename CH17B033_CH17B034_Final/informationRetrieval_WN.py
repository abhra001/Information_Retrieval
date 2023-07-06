## Main Part 
import numpy as np
import pandas as pd
from nltk.corpus import wordnet

class InformationRetrieval():
    
    def WeightedQuery(self, query,vocab): 
        
        print(query)

        query_df = pd.DataFrame(columns = vocab)

        query_vocab = dict()
        for sent in query:
            for word in sent:
                if word not in query_vocab.keys():
                    query_vocab[word] = 1
                else:
                    query_vocab[word] = query_vocab[word]+1
    
        synonyms = query_vocab.copy()
        alpha = 0.9
        
        for w in query_vocab.keys():
            
            word_weight = query_vocab[w]

            try:
                syn1 = wordnet.synsets(w)
            except:
                continue
                
            for s in syn1:
                syn_name = s.lemmas()[0].name().split('_')
                for synonym in syn_name:
                    if (synonym != w): 
                        if synonym not in synonyms.keys():
                            synonyms[synonym] = alpha*query_vocab[word]
                        else:
                            synonyms[synonym] = synonyms[synonym] + alpha*query_vocab[word]
                        
        query_df.loc['Query'] = synonyms
        
        x = query_df.sum(axis = 0).to_frame(name="Query")
        print(list(x[x['Query']>0].index))
        return x

        # return query_df.sum(axis = 0).to_frame(name="Query")

    def relatednessRetrieval(self, query, data ,norm, vocab):
    
        Q = self.WeightedQuery(query, vocab)
    
        data_full = data.join(Q, how = 'outer')
        data_full.fillna(0, inplace=True)
        data_full['tf_idfQuery'] = data_full['Query']*data_full['IDF']
        
        qnorm = np.linalg.norm(data_full['tf_idfQuery'])
        
        cosine_list = []
        
        if (qnorm!=0):
            for i in range(1,1401):

                if norm[i] == 0:
                    continue

                cosine = np.dot(data_full[i],data_full['tf_idfQuery'])/(norm[i]*qnorm)
                cosine_list.append((cosine,i))

            cosine_list.sort(key=lambda y: y[0], reverse=True)
            temp = [y[1] for y in cosine_list]
            doc_IDs_ordered = temp
            
            return doc_IDs_ordered
        else:
            return list(range(1,1401))
        
    def ir_wordnet(self, Queries, data,norm,vocab):
        overall = []
        count = 1
        for query in Queries: 
            ordered = self.relatednessRetrieval(query, data, norm, vocab)
            overall.append(ordered)
            print(count)
            count = count + 1
        return overall