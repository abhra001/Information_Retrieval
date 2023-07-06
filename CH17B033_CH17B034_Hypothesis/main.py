from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from informationRetrieval_LSA import InformationRetrieval as InformationRetrieval_LSA
from informationRetrieval_ESA import InformationRetrieval as InformationRetrieval_ESA
from informationRetrieval_W2V import InformationRetrieval as InformationRetrieval_W2V
from informationRetrieval_WN import InformationRetrieval as InformationRetrieval_WN
from scipy.stats import ttest_rel
import random

from evaluation import Evaluation

from sys import version_info
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")

class SearchEngine:
    
    def __init__(self, args):
        self.args = args
        
        self.tokenizer = Tokenization()
        self.sentenceSegmenter = SentenceSegmentation()
        self.inflectionReducer = InflectionReduction()
        self.stopwordRemover = StopwordRemoval()
        
        self.informationRetriever = InformationRetrieval()
        self.informationRetrieverLSA = InformationRetrieval_LSA()
        self.informationRetrieverESA = InformationRetrieval_ESA()
        self.informationRetrieverW2V = InformationRetrieval_W2V()
        self.informationRetrieverWN = InformationRetrieval_WN()

        self.evaluator = Evaluation()


    def segmentSentences(self, text):
        """
		Call the required sentence segmenter
		"""
        if self.args.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.args.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)

    def tokenize(self, text):
        """
		Call the required tokenizer
		"""
        if self.args.tokenizer == "naive":
            return self.tokenizer.naive(text)
        elif self.args.tokenizer == "ptb":
            return self.tokenizer.pennTreeBank(text)

    def reduceInflection(self, text):
        """
		Call the required stemmer/lemmatizer
		"""
        return self.inflectionReducer.reduce(text)

    def removeStopwords(self, text):
        """
		Call the required stopword remover
		"""
        return self.stopwordRemover.fromList(text)


    def preprocessQueries(self, queries):
        """
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
        segmentedQueries = []
        for query in queries:
            segmentedQuery = self.segmentSentences(query)
            segmentedQueries.append(segmentedQuery)
        json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
        tokenizedQueries = []
        for query in segmentedQueries:
            tokenizedQuery = self.tokenize(query)
            tokenizedQueries.append(tokenizedQuery)
        json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		# Stem/Lemmatize queries
        reducedQueries = []
        for query in tokenizedQueries:
            reducedQuery = self.reduceInflection(query)
            reducedQueries.append(reducedQuery)
        json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
        stopwordRemovedQueries = []
        for query in reducedQueries:
            stopwordRemovedQuery = self.removeStopwords(query)
            stopwordRemovedQueries.append(stopwordRemovedQuery)
        json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))
            
        return stopwordRemovedQueries

    def preprocessDocs(self, docs):
        """
		Preprocess the documents
		"""
	
		# Segment docs
        segmentedDocs = []
        for doc in docs:
            segmentedDoc = self.segmentSentences(doc)
            segmentedDocs.append(segmentedDoc)
        json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
        tokenizedDocs = []
        for doc in segmentedDocs:
            tokenizedDoc = self.tokenize(doc)
            tokenizedDocs.append(tokenizedDoc)
        json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		# Stem/Lemmatize docs
        reducedDocs = []
        for doc in tokenizedDocs:
            reducedDoc = self.reduceInflection(doc)
            reducedDocs.append(reducedDoc)
        json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		# Remove stopwords from docs
        stopwordRemovedDocs = []
        for doc in reducedDocs:
            stopwordRemovedDoc = self.removeStopwords(doc)
            stopwordRemovedDocs.append(stopwordRemovedDoc)
        json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))
        
        return stopwordRemovedDocs

    def t_test(self,base,new):
        
        alpha = 0.05
        t, pval = ttest_rel(base,new,alternative = 'less')
        if pval < alpha:
            print("p value = ", pval, "\nNull Hypothesis Rejected. New Method is better")
        else: 
            print("p value = ", pval, "\nNull Hypothesis cannot be rejected. New method is not conclusively better")
    
        # _, double_p = ttest_rel(base,new,equal_var = False)

        # if alternative == 'both-sided':
        #     pval = double_p

        # elif alternative == 'greater':
        #     if np.mean(base) > np.mean(new):
        #         pval = double_p/2.
        #     else:
        #         pval = 1.0 - double_p/2.

        # elif alternative == 'less':
        #     if np.mean(base) < np.mean(new):
        #         pval = double_p/2.
        #     else:
        #         pval = 1.0 - double_p/2.

        # if pval<alpha:
        #     print("p value = ", pval, "\nNull Hypothesis Rejected. New Method is better")
        # else:
        #     print("p value = ", pval, "\nNull Hypothesis cannot be rejected. New method is not conclusively better")

    def evaluateDataset(self):
        """
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		# Read queries
        queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
        query_ids, queries = [item["query number"] for item in queries_json], \
								[item["query"] for item in queries_json]
		# Process queries 
        processedQueries = self.preprocessQueries(queries)

		# Read documents
        docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
        doc_ids, docs = [item["id"] for item in docs_json], \
								[item["body"] for item in docs_json]
		# Process documents
        processedDocs = self.preprocessDocs(docs)
        
        data = pd.read_csv('doc_words_final.csv')
        data = data.set_index('Unnamed: 0')
        #print("Docs Data Imported") 
        idf_values = data['IDF'].to_dict()
        norm = {}
        data_tf_idf = pd.DataFrame()
        for i in range(1,1401):
            data_tf_idf[i] = data[str(i)]*data['IDF']
            norm[i] = np.linalg.norm(data_tf_idf[i])
        data_tf_idf['IDF'] = data['IDF']
        
        #Hypothesis Testing 
        qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]
        
        #Getting Queries: 
        n = 25
        list_of_list_of_qids = []
        list_of_list_of_queries = []
       
        for i in range(0,50):
            list_of_qids = random.sample(query_ids, n)
            list_of_queries = []
            for qid in list_of_qids:
                list_of_queries.append(processedQueries[qid-1])
            list_of_list_of_queries.append(list_of_queries)
            list_of_list_of_qids.append(list_of_qids)
        
        precisions_vs, recalls_vs, fscores_vs, MAPs_vs, nDCGs_vs = [], [], [], [], []
        
        count_query = -1
        for loq in list_of_list_of_queries:
            print(count_query)
            count_query+=1
            doc_IDs_ordered = self.informationRetriever.rank(loq,data_tf_idf,norm)
            # Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
            
            k = 5
            precision = self.evaluator.meanPrecision(
				doc_IDs_ordered, list_of_list_of_qids[count_query], qrels, k)
            precisions_vs.append(precision)
            
            recall = self.evaluator.meanRecall(
				doc_IDs_ordered, list_of_list_of_qids[count_query], qrels, k)
            recalls_vs.append(recall)
            
            fscore = self.evaluator.meanFscore(
				doc_IDs_ordered, list_of_list_of_qids[count_query], qrels, k)
            fscores_vs.append(fscore)
            
            print("Precision, Recall and F-score @ " +  
				str(k) + " : " + str(precision) + ", " + str(recall) + 
				", " + str(fscore))
            
            MAP = self.evaluator.meanAveragePrecision(
				doc_IDs_ordered, list_of_list_of_qids[count_query], qrels, k)
            MAPs_vs.append(MAP)
            
            nDCG = self.evaluator.meanNDCG(
				doc_IDs_ordered, list_of_list_of_qids[count_query], qrels, k)
            nDCGs_vs.append(nDCG)
            
            print("MAP, nDCG @ " +  
				str(k) + " : " + str(MAP) + ", " + str(nDCG))
            
        if self.args.method == "ESA":
        
            wiki = pd.read_csv('wiki_words_intersect.csv')
            print("Wiki Imported") 
            wiki_x = wiki.set_index('Unnamed: 0')
            
            docs_wiki = pd.read_csv('docs_wiki_final.csv')
            print("Docs - Wiki Imported")
            docs_wiki = docs_wiki.set_index('Unnamed: 0')
            
            idf_values = data['IDF'].to_dict()
            
            precisions_esa, recalls_esa, fscores_esa, MAPs_esa, nDCGs_esa = [], [], [], [], []
            
            cnt=-1
            for loq in list_of_list_of_queries:
                print(cnt)
                cnt+=1
                # Build document index
                # self.informationRetrieverLSA.buildIndex(processedDocs, doc_ids)
                # Rank the documents for each query
                doc_IDs_ordered = self.informationRetrieverESA.ir_esa(loq,idf_values,wiki_x,docs_wiki)
                
                k = 5
                precision = self.evaluator.meanPrecision(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                precisions_esa.append(precision)

                recall = self.evaluator.meanRecall(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                recalls_esa.append(recall)

                fscore = self.evaluator.meanFscore(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                fscores_esa.append(fscore)

                print("Precision, Recall and F-score @ " +  
                    str(k) + " : " + str(precision) + ", " + str(recall) + 
                    ", " + str(fscore))

                MAP = self.evaluator.meanAveragePrecision(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                MAPs_esa.append(MAP)

                nDCG = self.evaluator.meanNDCG(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                nDCGs_esa.append(nDCG)
                
            fig,(ax1,ax2) = plt.subplots(1,2)
            ax1.scatter(MAPs_vs,MAPs_esa)
            lims=[np.min([ax1.get_xlim(),ax1.get_ylim()]),np.max([ax1.get_xlim(),ax1.get_ylim()])]
            ax1.plot(lims,lims,color='red')
            ax1.set_xlim(lims)
            ax1.set_ylim(lims)
            ax1.set(xlabel="mean MAPs - ESA",ylabel="mean MAPs - Vector Space")
            ax2.hist(MAPs_vs)
            ax2.hist(MAPs_esa)
            ax2.set(xlabel="mean MAPs",ylabel="Frequency")
            fig.savefig(args.out_folder + "hypothesis.png")
            
            self.t_test(MAPs_vs,MAPs_esa)
            
    		# Build document index
            # self.informationRetriever.buildIndex(processedDocs, doc_ids)
    		# Rank the documents for each query
            # doc_IDs_ordered = self.informationRetrieverESA.ir_esa(processedQueries,idf_values,wiki_x,docs_wiki)
            
        elif self.args.method == "LSA":
            
            precisions_lsa, recalls_lsa, fscores_lsa, MAPs_lsa, nDCGs_lsa = [], [], [], [], []
            
            cnt=-1
            for loq in list_of_list_of_queries:
                print(cnt)
                cnt+=1
                # Build document index
                self.informationRetrieverLSA.buildIndex(processedDocs, doc_ids)
                # Rank the documents for each query
                doc_IDs_ordered = self.informationRetrieverLSA.rank(loq)
                
                k = 5
                precision = self.evaluator.meanPrecision(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                precisions_lsa.append(precision)

                recall = self.evaluator.meanRecall(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                recalls_lsa.append(recall)

                fscore = self.evaluator.meanFscore(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                fscores_lsa.append(fscore)

                print("Precision, Recall and F-score @ " +  
                    str(k) + " : " + str(precision) + ", " + str(recall) + 
                    ", " + str(fscore))

                MAP = self.evaluator.meanAveragePrecision(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                MAPs_lsa.append(MAP)

                nDCG = self.evaluator.meanNDCG(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                nDCGs_lsa.append(nDCG)
                
            fig,(ax1,ax2) = plt.subplots(1,2)
            ax1.scatter(nDCGs_vs,nDCGs_lsa)
            lims=[np.min([ax1.get_xlim(),ax1.get_ylim()]),np.max([ax1.get_xlim(),ax1.get_ylim()])]
            ax1.plot(lims,lims,color='red')
            ax1.set_xlim(lims)
            ax1.set_ylim(lims)
            ax1.set(xlabel="mean nDCGs - LSA",ylabel="mean nDCGs - Vector Space")
            ax2.hist(nDCGs_vs)
            ax2.hist(nDCGs_lsa)
            ax2.set(xlabel="mean nDCGs",ylabel="Frequency")
            fig.savefig(args.out_folder + "hypothesis.png")
            
            self.t_test(nDCGs_vs,nDCGs_lsa)

      #   elif self.args.method == "VS":
            
            
    		# # Build document index
      #       # self.informationRetriever.buildIndex(processedDocs, doc_ids)
    		# # Rank the documents for each query
      #       doc_IDs_ordered = self.informationRetriever.rank(processedQueries,data_tf_idf,norm)
            
        # elif self.args.method == "W2V":
        #     # Build document index
        #     self.informationRetrieverW2V.buildIndex(processedDocs, doc_ids)
        #     # Rank the documents for each query
        #     doc_IDs_ordered = self.informationRetrieverW2V.rank(processedQueries)  
        
        elif self.args.method == "WN":
            vocab = []
            
            # data = pd.read_csv('doc_words_final.csv')
            # data = data.set_index('Unnamed: 0')
            # print("Docs Data Imported") 
            # idf_values = data['IDF'].to_dict()
            # norm = {}
            # data_tf_idf = pd.DataFrame()
            
            # for i in range(1,1401):
            #     data_tf_idf[i] = data[str(i)]*data['IDF']
            #     norm[i] = np.linalg.norm(data_tf_idf[i])
            # data_tf_idf['IDF'] = data['IDF']
            
            for doc in processedDocs:
                for sent in doc:
                    for word in sent: 
                        if word not in vocab:
                            vocab.append(word)
                            
            for query in processedQueries:
                for sent in query:
                    for word in sent:
                        if word not in vocab:
                            vocab.append(word)
            for i in range(1,1401):
                data_tf_idf[i] = data[str(i)]*data['IDF']
                norm[i] = np.linalg.norm(data_tf_idf[i])
            
            precisions_wn, recalls_wn, fscores_wn, MAPs_wn, nDCGs_wn = [], [], [], [], []
            
            cnt=-1
            for loq in list_of_list_of_queries:
                print(cnt)
                cnt+=1
                # Build document index
                # self.informationRetrieverLSA.buildIndex(processedDocs, doc_ids)
                # Rank the documents for each query
                doc_IDs_ordered = self.informationRetrieverWN.ir_wordnet(loq,data_tf_idf,norm,vocab)
                
                k = 5
                precision = self.evaluator.meanPrecision(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                precisions_wn.append(precision)

                recall = self.evaluator.meanRecall(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                recalls_wn.append(recall)

                fscore = self.evaluator.meanFscore(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                fscores_wn.append(fscore)

                print("Precision, Recall and F-score @ " +  
                    str(k) + " : " + str(precision) + ", " + str(recall) + 
                    ", " + str(fscore))

                MAP = self.evaluator.meanAveragePrecision(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                MAPs_wn.append(MAP)

                nDCG = self.evaluator.meanNDCG(
                    doc_IDs_ordered, list_of_list_of_qids[cnt], qrels, k)
                nDCGs_wn.append(nDCG)
                
            fig,(ax1,ax2) = plt.subplots(1,2)
            ax1.scatter(nDCGs_vs,nDCGs_wn)
            lims=[np.min([ax1.get_xlim(),ax1.get_ylim()]),np.max([ax1.get_xlim(),ax1.get_ylim()])]
            ax1.plot(lims,lims,color='red')
            ax1.set_xlim(lims)
            ax1.set_ylim(lims)
            ax1.set(xlabel="mean nDCGs - WordNet",ylabel="mean nDCGs - Vector Space")
            ax2.hist(nDCGs_vs)
            ax2.hist(nDCGs_wn)
            ax2.set(xlabel="mean nDCGs",ylabel="Frequency")
            fig.savefig(args.out_folder + "hypothesis.png")
            
            self.t_test(nDCGs_vs,nDCGs_wn)
    		# Build document index
            # self.informationRetriever.buildIndex(processedDocs, doc_ids)
    		# Rank the documents for each query
            # doc_IDs_ordered = self.informationRetrieverWN.ir_wordnet(processedQueries,data_tf_idf,norm,vocab)

		# Read relevance judements
#         qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]

# 		# Calculate precision, recall, f-score, MAP and nDCG for k = 1 to 10
#         precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
#         for k in range(1, 11):
#             precision = self.evaluator.meanPrecision(
# 				doc_IDs_ordered, query_ids, qrels, k)
#             precisions.append(precision)
#             recall = self.evaluator.meanRecall(
# 				doc_IDs_ordered, query_ids, qrels, k)
#             recalls.append(recall)
#             fscore = self.evaluator.meanFscore(
# 				doc_IDs_ordered, query_ids, qrels, k)
#             fscores.append(fscore)
#             print("Precision, Recall and F-score @ " +  
# 				str(k) + " : " + str(precision) + ", " + str(recall) + 
# 				", " + str(fscore))
#             MAP = self.evaluator.meanAveragePrecision(
# 				doc_IDs_ordered, query_ids, qrels, k)
#             MAPs.append(MAP)
#             nDCG = self.evaluator.meanNDCG(
# 				doc_IDs_ordered, query_ids, qrels, k)
#             nDCGs.append(nDCG)
#             print("MAP, nDCG @ " +  
# 				str(k) + " : " + str(MAP) + ", " + str(nDCG))

# 		# Plot the metrics and save plot 
#         plt.plot(range(1, 11), precisions, label="Precision")
#         plt.plot(range(1, 11), recalls, label="Recall")
#         plt.plot(range(1, 11), fscores, label="F-Score")
#         plt.plot(range(1, 11), MAPs, label="MAP")
#         plt.plot(range(1, 11), nDCGs, label="nDCG")
#         plt.legend()
#         plt.title("Evaluation Metrics - Cranfield Dataset")
#         plt.xlabel("k")
#         plt.savefig(args.out_folder + "eval_plot.png")

		
#     def handleCustomQuery(self):
#         """
# 		Take a custom query as input and return top five relevant documents
# 		"""

# 		#Get query
#         print("Enter query below")
#         query = input()
# 		# Process documents
#         processedQuery = self.preprocessQueries([query])[0]

# 		# Read documents
#         docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
#         doc_ids, docs = [item["id"] for item in docs_json], \
# 							[item["body"] for item in docs_json]
                            
# 		# Process documents
#         processedDocs = self.preprocessDocs(docs)
        
#         wiki = pd.read_csv('wiki_words_intersect.csv')
#         print("Wiki Imported") 
#         wiki_x = wiki.set_index('Unnamed: 0')
        
#         docs_wiki = pd.read_csv('docs_wiki_final.csv')
#         print("Docs - Wiki Imported")
#         docs_wiki = docs_wiki.set_index('Unnamed: 0')
        
#         data = pd.read_csv('doc_words_final.csv')
#         data = data.set_index('Unnamed: 0')
#         print("Docs Data Imported") 
#         idf_values = data['IDF'].to_dict()

# 		# Build document index
#         # self.informationRetriever.buildIndex(processedDocs, doc_ids)
# 		# Rank the documents for the query
#         doc_IDs_ordered = self.informationRetriever.ir_esa([processedQuery],idf_values,wiki_x,docs_wiki)[0]

# 		# Print the IDs of first five documents
#         print("\nTop five document IDs : ")
#         for id_ in doc_IDs_ordered[:5]:
#             print(id_)



if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-method',  default = "LSA",
	                    help = "Retrieval Method [VS|LSA|ESA|W2V|WN]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
