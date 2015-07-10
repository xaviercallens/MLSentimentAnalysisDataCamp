import re
import nltk.data
from nltk.stem.porter import *
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

def terms(doc) :
  Terms=[]
  sentences=sent_tokenize (doc.decode("utf-8")) #break the doc into sentences
  for sentence in sentences :
    Terms.extend(sentence_terms(sentence.lower())) #add the terms found per sentence
  return Terms

  def sentence_terms(sentence) :
    stop_words=stopwords.words('english') #liste par défaut de stop words
    sentence=re.sub('[?!\\.]+','',sentence).strip() #remove punctuation
    sentence=re.sub('\s+,' ',sentence) #retire les espaces en trop
    stemmer = ForterStemmer()
    #split les phrases en mots
    #si le mot n'est pas dans la liste des stop words > application du stemming
    #ensuite ajouter à la liste. si un mot apparait plusieurs fois, il est ajouté plusieurs fois
    terms=[stemmer.stem(w) for w in sentence.split(" ") if w not in stop_words]
    return terms
