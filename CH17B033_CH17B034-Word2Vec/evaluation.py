from util import *

# Add your import statements here

import math

class Evaluation():
    
    def QRelDocs(self, qrels):
        rel_docs = {}
        for dic in qrels:
            if dic["query_num"] in rel_docs.keys():
                rel_docs[dic["query_num"]].append((int(dic["id"]), dic["position"]))
            else:
                rel_docs[dic["query_num"]] = [(int(dic["id"]), dic["position"])]

        return rel_docs
    
    def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""
        
        precision = -1

		#Fill in code here
        
        count = 0
        i = 0
        while(i<k):
            if (query_doc_IDs_ordered[i] in true_doc_IDs):
                count=count+1
            i=i+1
        precision = count/k
        
        return precision


    def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""
        
        meanPrecision = -1
        total_precision = 0

		#Fill in code here

        count = 0
        for query in query_ids:
            true_doc_IDs=[]
            for q in qrels:
                if(int(q["query_num"]) == query):
                    true_doc_IDs.append(int(q["id"]))
            total_precision = total_precision + self.queryPrecision(doc_IDs_ordered[count], query, true_doc_IDs, k)
            count = count+1
        meanPrecision = total_precision/count
        
        return meanPrecision
    
    def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""
        
        recall = -1

		#Fill in code here
        
        count = 0
        i = 0
        while (i<k):
            if (query_doc_IDs_ordered[i] in true_doc_IDs):
                count=count+1
            i=i+1
        if true_doc_IDs:
            recall = count/len(true_doc_IDs)
        else:
            recall = 1
        
        return recall
    
    def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""
        
        meanRecall = -1
        total_recall = 0

		#Fill in code here
        
        count = 0
        for query in query_ids:
            true_doc_IDs=[]
            for q in qrels:
                if(int(q["query_num"]) == query):
                    true_doc_IDs.append(int(q["id"]))
            total_recall = total_recall + self.queryRecall(doc_IDs_ordered[count], query, true_doc_IDs, k)
            count = count+1
        meanRecall = total_recall/count
        
        return meanRecall
    
    def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""
        
        fscore = -1
        P=-1
        R=-1

		#Fill in code here
        
        count = 0
        i = 0
        while (i<k):
            if (query_doc_IDs_ordered[i] in true_doc_IDs):
                count=count+1
            i=i+1
        P = count/k
        if true_doc_IDs:
            R = count/len(true_doc_IDs)
        else:
            R=1
        if(P + R > 0):
            fscore = 2 * P * R / (P + R)
        else:
            fscore = 0

        return fscore


    def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

        meanFscore = -1
        total_Fscore = 0

		#Fill in code here
        
        count = 0
        for query in query_ids:
            true_doc_IDs=[]
            for q in qrels:
                if(int(q["query_num"]) == query):
                    true_doc_IDs.append(int(q["id"]))
            total_Fscore = total_Fscore + self.queryFscore(doc_IDs_ordered[count], query, true_doc_IDs, k)
            count = count+1
        meanFscore = total_Fscore/count

        return meanFscore
	

    def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

        nDCG = -1

		#Fill in code here
        DCG = 0
        for i in range(k):
            for j in true_doc_IDs:
                if query_doc_IDs_ordered[i] == j[0]:
                    DCG += (5-j[1])/math.log(i+2,2)
                    break

        ideal_order = sorted(true_doc_IDs, key=lambda x: x[1])
        IDCG = 0
        for i in range(min(k, len(true_doc_IDs))):
            IDCG += (5-ideal_order[i][1])/math.log(i+2,2)

        nDCG = DCG/IDCG

        return nDCG


    def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
        """
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

        meanNDCG = -1
        totalNDCG = 0

		#Fill in code here
        
        rel_docs = self.QRelDocs(qrels)

        for qid in query_ids:
            totalNDCG += self.queryNDCG(doc_IDs_ordered[qid-1], qid, rel_docs[str(qid)], k)

        meanNDCG = totalNDCG/len(query_ids)
        return meanNDCG


    def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
        """
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

        avgPrecision = -1

		#Fill in code here
        
        count = 0
        total_precision=0
        i = 0
        while (i < k):
            if(query_doc_IDs_ordered[i] in true_doc_IDs):
                total_precision = total_precision + (count+1)/(i+1)
                count = count+1
            i=i+1
        if(count > 0):
            avgPrecision = total_precision/count
        else:
            avgPrecision = 0

        return avgPrecision


    def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
        """
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

        meanAveragePrecision = -1
        total_precision = 0
		#Fill in code here
        
        count = 0
        for query in query_ids:
            true_doc_IDs=[]
            for q in q_rels:
                if(int(q["query_num"]) == query):
                    true_doc_IDs.append(int(q["id"]))
            total_precision = total_precision + self.queryAveragePrecision(doc_IDs_ordered[count], query, true_doc_IDs, k)
            count = count+1
        meanAveragePrecision = total_precision/count

        return meanAveragePrecision
