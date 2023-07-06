from util import *

# Add your import statements here
from nltk.corpus import stopwords



class StopwordRemoval():

    def fromList(self, text):
        """
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

        stopwordRemovedText = None
        stopwordRemovedText = []
        
        stop_words = set(stopwords.words('english'))

        for sentence in text:
            stopword_removed_sent = []
        
            for word in sentence:
                if word not in stop_words:
                    stopword_removed_sent = stopword_removed_sent + [word]
            stopwordRemovedText=stopwordRemovedText+[stopword_removed_sent]      
        return stopwordRemovedText
