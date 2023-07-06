from util import *
from nltk.tokenize import TreebankWordTokenizer
from spellchecker import SpellChecker


# Add your import statements here

class Tokenization():
    
    def naive(self, text):
        """
  		Tokenization using a Naive Approach
  
  		Parameters
  		----------
  		arg1 : list
  			A list of strings where each string is a single sentence
  
  		Returns
  		-------
  		list
  			A list of lists where each sub-list is a sequence of tokens
  		"""

        tokenizedText = None
        tokenizedText = []
        
        for sentence in (text):
            #Splitting the sentence by the occurence of a blankspace
            words=sentence.split(' ')
            clean_words=[]
            
            #Removing any punctuation attached to the word and removing tokens which are punctuations
            for word in words:
                if word and word[-1] in [',','.','!','?',';'] and len(word[:-1])>0:
                    clean_words=clean_words+[word[:-1]]
                elif word not in ['.','?','!','']:
                    clean_words=clean_words+[word]
            tokenizedText=tokenizedText+[clean_words]
            
        # spell = SpellChecker()
        
        # spellcorrected = []
        
        # for sentence in tokenizedText:
        #     misspelt=spell.unknown(sentence)
        #     corrected=[]
        #     for word in sentence:
        #         if word in misspelt:
        #             corrected=corrected+[spell.correction(word)]
        #         else:
        #             corrected=corrected+[word]
        #     spellcorrected.append(corrected)

        return tokenizedText



    def pennTreeBank(self, text):
        """
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

        tokenizedText = None
        tokenizedText = []
    
        for sentence in text:
            tokenizedText.append(TreebankWordTokenizer().tokenize(sentence))
            
        # spell = SpellChecker()
        
        # spellcorrected = []
        
        # for sentence in tokenizedText:
        #     misspelt=spell.unknown(sentence)
        #     corrected=[]
        #     for word in sentence:
        #         if word in misspelt:
        #             corrected=corrected+[spell.correction(word)]
        #         else:
        #             corrected=corrected+[word]
        #     spellcorrected.append(corrected)

        return tokenizedText