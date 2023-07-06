from util import *

# Add your import statements here
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

class InflectionReduction:
    
    def nltk2wordnetTag(self, nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None

    def reduce(self, text):
        """
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
        """

        reducedText = None
        
        #Fill in code here
        reducedText = []
        wordnet_lemmatizer_object = WordNetLemmatizer()
        
        for sentence in text:

            nltk_tag = nltk.pos_tag(sentence)  

            wordnet_tag = map(lambda x: (x[0], self.nltk2wordnetTag(x[1])), nltk_tag)
			
            reduced_sentence = []
            for word, tag in wordnet_tag:
                if tag is None:
                    reduced_sentence=reduced_sentence+[word]
                else:        
                    reduced_sentence=reduced_sentence+[wordnet_lemmatizer_object.lemmatize(word, tag)]
        
            reducedText=reducedText+[reduced_sentence]
    
        return reducedText


