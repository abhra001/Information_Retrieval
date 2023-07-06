from util import *

# Add your import statements here


class SentenceSegmentation():

    def naive(self, text):
        """
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

        segmentedText = None
        
        lowercaseDocs = text.lower()
        
        segmentedText=[]
        sentence=''
        
        unambiguous = ['?','!']
        
        # '.' marked as ambiguous because abbreviations like 'Dr.' and 'Mr.' do not indicate the end of a sentence
        ambiguous = ['.']
        
        for char in enumerate(lowercaseDocs):

            if char[1] in (unambiguous+ambiguous):
            # Add if we want to remove space at the end and beginning of sentences
            # sentence = sentence.strip()
        
                sentence = sentence + char[1]
                segmentedText.append(sentence)
                sentence=''
            else:
                sentence = sentence + char[1]
        
        segmentedText.append(sentence)

        return segmentedText



    def punkt(self, text):
        """
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

        segmentedText = None
        
        lowercaseDocs = text.lower()

        segmentedText = nltk.tokenize.punkt.PunktSentenceTokenizer().tokenize(lowercaseDocs)
		
        return segmentedText