
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import  stopwords 
from  nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import  BigramAssocMeasures
import nltk
import itertools

import string
STEMMER=PorterStemmer()
STOPWORDS=set(stopwords.words("english"))
#### Features Extraction

def tokenize(text):
    text=re.sub("<.*?>"," ",text)# remove html tags like : <br />
    text=re.sub("[0-9]+"," ",text) # remove numbers
    punctuation = '['+re.escape(re.sub('[?!\\.]','',string.punctuation))+']'
        # we want to keep the sentence structure
        #..................................don't remove ?!. replace ?/! with .
                                            #this part exludes ?!.from the list of punctuations
        
    text=re.sub(punctuation,' ',text)
        
    text=re.sub("\s+"," ",text)
    tokens=word_tokenize(text)
    lowercased=[t.lower() for t in tokens]
    
    #### TODO Stopword Removal or Elimination
    without_stopwords=[w for w in lowercased if not w in STOPWORDS]
    
    #### TODO Stemming
    stemmed=[STEMMER.stem(w) for w in without_stopwords]
    #print [w for w in stemmed if w]
    return [w for w in stemmed if w]
   
#### Using N-Grams
### TODO Bi-grams good for SVM apparently

   
def bigram_return(text):
   tokenized_text= tokenize(text)
   bigramFeatureVector=[]
   for item in nltk.bigrams(tokenized_text):
       
       bigramFeatureVector.append(' '.join(item))
   return bigramFeatureVector    
   
def terms(doc):

    sentences = sent_tokenize(doc.decode('utf-8'))
    Terms = []
    for sentence in sentences:
        #Change ---------------------------------
        Terms.extend(tokenize(sentence))  # when u want to work with bigram  call bigram_return funtion here instead of tokenize
    return Terms  
#This code could be useful for chi sq bigram feature selection mechanism
   
   #finder=BigramCollocationFinder.from_words(tokenized_text)
   #print finder
   #bigrammed_words=sorted(finder.nbest(BigramAssocMeasures.chi_sq,300))
   #print dict([(ngram)  for ngram in itertools.chain(tokenized_text,bigrammed_words)])
   #return bigrammed_words

#print answer