import loadFiles as lf
import preProcess as pp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from  pyspark.mllib.regression import LabeledPoint
from random import randint
from  pyspark.mllib.classification import NaiveBayes
from pyspark import SparkContext
from pyspark import SparkFiles
from sklearn.naive_bayes import GaussianNB
from functools import partial
from random import shuffle

trainF="./train" #the path to where the train data is

sc = SparkContext(appName="Simple App")  #initialize the spark context
#since we are not in the command line interface we need to add to the spark context
#some of our classes so that they are available to the workers
sc.addFile("/home/christos.giatsidis/data_camp_2015_june/helpers.py") 
sc.addFile("/home/christos.giatsidis/data_camp_2015_june/helpers2.py") 
sc.addFile("/home/christos.giatsidis/data_camp_2015_june/exctract_terms.py")
#now if we import these files they will also be available to the workers
from helpers import *
from helpers2 import *
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

bY=sc.broadcast(Y)
rows=sc.broadcast(len(Y))
cols=sc.broadcast(len(m.get_feature_names())) 

#assign random partition number to each row
partitions=tmp.map(partial(toVector,cols=cols)).map(lambda x : (np.random.randint(0,50),x))

#split the data into train and test based on the partition number
#append the class values in the train data and the row index in the test
#group the data into lists of arrays with the groupByKey
train_partition=partitions.filter(lambda x: x[0]<40).map(lambda x: (x[0],np.append(x[1][1],bY.value[x[1][0]]))).groupByKey()
test_partition=partitions.filter(lambda x: x[0]>=40).map(lambda x: (x[0],np.append(x[1][1],x[1][0]))).groupByKey()

#train NEW classifiers on the partitioned data
print "training 1"
part_classifiers=train_partition.map(partial(trainClassifier, cols=cols)).collect()
#we want to shuffle the trained classifiers but we need a list ...
classifiers=[]
for x in part_classifiers:
		classifiers.append(x)


shuffle(classifiers)

#broadcast the models for retraining
models=sc.broadcast(classifiers)
print "training 2"
#retrain and re-broadcast the classifiers (this time for evaluation)
classifiers=train_partition.map(partial(updateClassifier, cols=cols,models=models)).collect()
models=sc.broadcast(classifiers)
print "evaluation by matrix"
# evaluate the broadcasted models on the test data
voted_results=test_partition.map(partial(evaluateByMatrixPartitioned, cols=cols,models=models)).collect()	

#calculate accuracy
correct=0
all=0
for x in voted_results:
 row_i=x[0]
 result=x[1]
 for i,v in enumerate(row_i):
	all=all+1
	if bY.value[v]==result[i]:
		correct=correct+1


accurac=float(correct)/float(all)
print "accuracy:"+str(accurac)

''' commend the next lines out if you undestand them
they only serve as an example of the usage of evaluateByRow
'''

#assume tmp has the unlabelled data for this example...you can fill in the code to load the unlabelled data from the other examples
data=tmp.map(partial(toVector,cols=cols))
voted_results_by_row=data.map(partial(evaluateByRow, models=models)).collect()

correct=0
all=0
for x in voted_results_by_row:
 row_i=x[0]
 result=x[1]
 all=all+1
 if bY.value[row_i]==result:
		correct=correct+1


accurac=float(correct)/float(all)
print "accuracy:"+str(accurac)
	