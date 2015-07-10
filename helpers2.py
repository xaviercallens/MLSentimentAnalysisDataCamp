import numpy as np
from sklearn.naive_bayes import GaussianNB

#create new classifier train it on the data and return it
#partition_data : (partition_number,matrix_data)
#matrix data : a list-like structure of row data 
#row data: all the column features except from the last column which contains the class
#length of row data == cols.value+1
def trainClassifier(partition_data,cols):
      #### Naive Bayes 
	gnb = GaussianNB()
      #### TODO change and put in parallel
 
	data=np.array(list(partition_data[1]))
	#the first call to partial_fit must include a list of all the classes 
	gnb=gnb.partial_fit(data[:,0:cols.value],data[:,cols.value],[0,1]) 
	return gnb
#this is essentially the same as trainClassifier 
#the only difference : we DO NOT inititalize the classifier; we get it from the broadcasted models
def updateClassifier(partition_data,cols,models):
	gnb=models.value[partition_data[0]]
	data=np.array(list(partition_data[1]))
	gnb=gnb.partial_fit(data[:,0:cols.value],data[:,cols.value])
	return gnb
 
 
#this function is used to get the predictions of data that are organized as
#(partition_key,[row_data,row_data,...]) where row_data : [features,row_index]
#the length of the row data is #cols+1
#it returns a matrix where each row has the prediction and the row index of that prediction
def evaluateByMatrixPartitioned(dataGroupedByKey, models,cols):
	#the data here should contain the row number as
	data=np.array(list(dataGroupedByKey[1]))#the matrix with features and row_indices
	rows=data[:,cols.value] #an array with row indices; the data contain the row_index in the last column
	
      #### TODO Check the voting mecanism
      votes=np.zeros((data.shape[0],2)) #lets assume for simplicity that class 0 is at column 0 etc
	for x in models.value:
		#this will return an array of all the predictions by row
		predictions=x.predict_proba(data[:,0:cols.value])# the data contain the row_index in the last column
		votes=votes+predictions
	final_pred=np.zeros(data.shape[0])
	for i,x in enumerate(votes):
		final_pred[i]= 0 if votes[i,0]>=votes[i,1] else 1
	return (rows,final_pred) #we need the row information for the evaluation

#sklearn can predict predict by row so we don't have to organize our data in matrices
#This might be useful for the final classification of the unlabelled data
#The function assumes that data is one row with (row_index,data)
def evaluateByRow(data,models):
	preds=np.zeros(2)
	for x in models.value:
		prediction=x.predict(data[1])
		preds[int(prediction)]=preds[int(prediction)]+1
	final_pred=0 if preds[0]>=preds[1] else 1
	return (data[0],final_pred)