import loadFiles as lf
import preProcess as pp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from  pyspark.mllib.regression import LabeledPoint
from random import randint
from  pyspark.mllib.classification import NaiveBayes
from pyspark import SparkContext
from pyspark import SparkFiles
from functools import partial

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.regression import LabeledPoint
from numpy import array

#Locally
#trainF="D://Data//DSSP//Data Camp 2//xcode//train" #the path to where the train data is
#testF="D://Data//DSSP//Data Camp 2//xcode//test" # the path to the unlabelled data 
#saveF="./predictions.txt" #where to save the predictions

#### TODO change to the cluster directory
#Server /home/xavier.callens/DataCamp
trainF="/home/xavier.callens/DataCamp/train" #the path to where the train data is
testF="/home/xavier.callens/DataCamp/test" # the path to the unlabelled data 
saveF="./predictions.txt" #where to save the predictions


sc = SparkContext(appName="Simple App")  #initialize the spark context
#since we are not in the command line interface we need to add to the spark context
#some of our classes so that they are available to the workers
sc.addFile("helpers.py") 
sc.addFile("exctract_terms.py")
#now if we import these files they will also be available to the workers
from helpers import *
import extract_terms as et



# load data : data is a list with the text per doc in each cell. Y is the respective class value
#1 :positive , 0 negative
print "loading local data"
data,Y=lf.loadLabeled(trainF) 

print "preprocessing"
pp.proc(data) #clean up the data from  number, html tags, punctuations (except for "?!." ...."?!" are replaced by "."
m = TfidfVectorizer(analyzer=et.terms) # m is a compressed matrix with the tfidf matrix the terms are extracted with our own custom function 

'''
we need an array to distribute to the workers ...
the array should be the same size as the number of workers
we need one element per worker only
'''
ex=np.zeros(8) 
rp=randint(0,7)
ex[rp]=1 #one random worker will be selected so we set one random element to non-zero

md=sc.broadcast(m) #broadcast the vectorizer so that he will be available to all workers
datad=sc.broadcast(data) # broadcast teh data

#execute vectorizer in one random  remote machine 
#partial is a python function that calls a function and can assign partially some of the parameters 
#numSlices determins how mnay partitions should the data have
#numslices is also helpfull if we want to reduce the size of each task for each worker
tmpRDD=sc.parallelize(ex,numSlices=8).filter(lambda x: x!=0).map(partial(compute, model=md, data=datad))
print "transforming the data in a remote machine"
data=tmpRDD.collect() # get back the coordinate matrix and the fitted vectorizer
#data = [[[matrix][vectorizer]]] (double nested)
tfidf_coo=data[0][0] 
m=data[0][1] #the fitted vectorizer re-assign it just in case

datad.unpersist() # we don't need this broadcasted variable anymore




#distribute the coordinate data 
# data =[ [value,row_index,column_index],[value,row_index,column_index]..]
ttodp=sc.parallelize(tfidf_coo,512) 

comb = (lambda x,y : np.vstack([np.array(x),np.array(y)])) # a function to combine tuples into a "vertical" array
#organize the coordinate matrix into the row index and a tuple containing the value and column index
#group by the row index 
tmp=ttodp.map(lambda x: (x[1],(x[0],x[2]))).aggregateByKey([0,0],comb,comb)

bY=sc.broadcast(Y) #broadcast the class variable (in order to create labeled points)
# the number of features is the columns of the matrix
#we need this information to convert to vectors and label point the coordinate data
cols=sc.broadcast(len(m.get_feature_names())) 
print "number of features"+str(cols.value)

#convert to labeled point in parallel
tmpLB=tmp.map(partial(toLB,cols=cols,class_v=bY)) 
print "training the machine learning algorithm"


#model = NaiveBayes.train(tmpLB) # train a naive bayes model

### Change XCA
### TODO Test different MLs

# 1) LogisticsRegression

#clf = LogisticRegression()

model =LogisticRegressionwWithSGD.train(tmpLB)  #This is used for Logistic regression classification
#model =LogisticRegression.train(tmpLB)

# 2) SVM Classification 
#model=SVMWithSGD.train(tmpLB)  This used for SVM classiffication

# 3) RandomForest 
#************Random forest model in pyspark is experimental so not sure whether works perfectly or not
#model=RandomForest.trainClassifier(tmpLB,2,{},300,seed=2)  here 300 is best solution as per literature for this dataset

# 4) kNN perhaps bu more complex

##### TODO 
# This could be done in parallel to test the different one 


##### TODO
# Perhaps combined such AdaBoost & GradientBoost


print "loading unlabeled data"
test,names=lf.loadUknown(testF) #load the unlabelled data . test : text per document. names : the respective file names
namesb=sc.broadcast(names) # broadcast the file names as we will need them for predictions

md=sc.broadcast(m) # broadcast the fitted model of the vecotrizer 
datadt=sc.broadcast(test) # broadcast the unlabeled data so that we may call the vectorizer in same manner

#apply the vestorization in a random worker
print "transforming unlabelled data"
test_data=sc.parallelize(ex,numSlices=16).filter(lambda x: x!=0).map(partial(computeTest, model=md,data=datadt)).collect()


datadt.unpersist()
print "convert data to non-compressed vector and predict the class"
#Steps:
#distribute the coordinate tf-idf of the transformed unlabelled data
#organize the coordinate by row
##aggregate the data by row
#convert the coordinate data to a vector
#apply for each vector the prediction and return the prediction along with the name of the file for that prediction
test_data_d=sc.parallelize(test_data[0],numSlices=512).map(lambda x: (x[1],(x[0],x[2]))).aggregateByKey([0,0],comb,comb).map(partial(toVector,cols=cols)).map(lambda x: (namesb.value[int(x[0])],model.predict(x[1])))

predictions=test_data_d.collect() # get all the (filename, prediction) tuples 
print "writing prediction"
#write the predictions to a file
f=open(saveF,'w')
for x in predictions:
    f.write(x[0])
    f.write(',')
    f.write(str(x[1]))
    f.write('\n')
f.close()