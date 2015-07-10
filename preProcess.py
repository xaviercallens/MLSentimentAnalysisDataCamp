import re
import string
#applies a few simple regex rules to clean up the data 
def proc(data):

	for i,d in enumerate(data):
		data[i]=re.sub("<.*?>"," ",data[i]) # remove html tags like : <br />
		data[i]=re.sub("[0-9]+"," ",data[i]) # remove numbers
		# we want to keep the sentence structure
		#..................................don't remove ?!. replace ?/! with .
											#this part exludes ?!.from the list of punctuations
		punctuation = '['+re.escape(re.sub('[?!\\.]','',string.punctuation))+']'
		data[i]=re.sub(punctuation,' ',data[i])
		
		data[i]=re.sub("\s+"," ",data[i])# remove multiple spaces
		