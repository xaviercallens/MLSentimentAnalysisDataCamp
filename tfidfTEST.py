import loadFiles as lf

import extract_terms as et
import preProcess as pp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as nps

#### TODO change to the cluster folders
data,Y=lf.loadLabeled("D://Data//DSSP//Data Camp 2//xcode//train")

#/home/xavier.callens/DataCamp
data,Y=lf.loadLabeled("/home/xavier.callens/DataCamp/train")

print data
#analyzer and preprocessor dont work together...one or the other
pp.proc(data) # we apply the preprocessing by ourselves
m = TfidfVectorizer(analyzer=et.terms)

tt = m.fit_transform(data)
print tt
rows,cols=tt.shape
print "number of features :" +str(len(m.get_feature_names())) #this is the same as the number of columns
#the function get_feature_names() returns a list of all the terms found 

print "non compressed matrix expected size:" + str(rows*cols*8/(1024*1024*1024))+"GB"

