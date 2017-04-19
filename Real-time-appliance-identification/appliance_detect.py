import random
import json
import time
import os
import socket
import sys
import subprocess
import numpy as np
from datetime import datetime
import sklearn
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from itertools import chain, combinations
import warnings

#The below path is Path to the folder containing code, training #and test data. Please edit this accordingly
Data_path = '/Path-to-the-folder-containing-code-training-and-test-data/'
test_csv_path = Data_path + 'test/';
train_csv_path = Data_path + 'CSV/';

start_time=time.time()

type_Ids = {}
start = datetime.now()
num_clf =1


def myclassifiers(X_train,y_train,X_test):
    
    knn = KNeighborsClassifier(n_neighbors=1)
        
    names = ["Nearest Neighbors(k=1)"]
    classifiers = [knn]
    y_predict = []
    acc = []

    for (i,clf) in enumerate(classifiers):
        clf.fit(X_train,y_train)
        y_predict.append(clf.predict(X_test))
        
    return (y_predict,names)




# functions to read data and meta data
def read_data_given_id(path,ids,progress=True,last_offset=0):
    '''read data given a list of ids and CSV paths'''
    start = datetime.now()
    n = len(ids)
    if n == 0:
        return {}
    else:
        data = {}
        for (i,ist_id) in enumerate(ids, start=1):
            if last_offset==0:
                data[ist_id] = np.genfromtxt(path+str(ist_id)+'.csv',delimiter=',',\
                                         names='current,voltage',dtype=(float,float))
            else:
                p=subprocess.Popen(['tail','-'+str(int(last_offset)),path+str(ist_id)+'.csv'],\
                                   stdout=subprocess.PIPE)
                data[ist_id] = np.genfromtxt(p.stdout,delimiter=',',names='current,voltage',dtype=(float,float))
        return data

def clean_meta(ist):
    type_Ids = {}
    #remove None elements in Meta Data ''' 
    clean_ist = ist.copy()
    for k,v in ist.items():
        if len(v) == 0:
            del clean_ist[k]
    return clean_ist

#function to read the meta data                
def parse_meta(meta):
    #parse meta data for easy access'''
    M = {}
    for m in meta:
        for app in m:
            M[int(app['id'])] = clean_meta(app['meta'])
    return M

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def subsets(s):
    return map(set, powerset(s))



with open(Data_path + 'meta.json') as data_file:    
    meta_file = json.load(data_file)
Meta = parse_meta([meta_file])

# read data

# appliance types of all instances
Types = [x['type'] for x in Meta.values()]
# unique appliance types
Unq_type = list(set(Types)) 
Unq_type.sort()
IDs_for_read_data = list(Meta.keys())

npts = 1344
fs = 3200
f0 = 50
NS = fs/f0 # number of samples per period
NP = npts/NS # number of periods for npts

Data = read_data_given_id(train_csv_path,IDs_for_read_data,progress=True, last_offset=npts)
n=len(Data)

type_label = np.zeros(n,dtype='int')
for (ii,t) in enumerate(Unq_type):
    type_Ids[t] = [i-1 for i,j in enumerate(Types,start=1) if j == t]
    type_label[type_Ids[t]] = ii+1

rep_I = np.empty([n,NS])
rep_V = np.empty([n,NS])

for i in range(n):
    tempI = np.sum(np.reshape(Data[i+1]['current'],[NP,NS]),0)/NP
    tempV = np.sum(np.reshape(Data[i+1]['voltage'],[NP,NS]),0)/NP
    # align current to make all samples start from 0 and goes up
    ix = np.argsort(np.abs(tempV))
    j = 0
    while True:
        if ix[j]<(NS-1) and tempV[ix[j]+1]>tempV[ix[j]]:
            real_ix = ix[j]
            break
        else:
            j += 1

    rep_I[i,] = np.hstack([tempI[real_ix:],tempI[:real_ix]])
    rep_V[i,] = np.hstack([tempV[real_ix:],tempV[:real_ix]])

RawCF=rep_I

##########################################################################################################################################################

n=len(Data)
devices=np.empty([n])
 
for i in range(len(Unq_type)):
    devices[i]=type_label[i]

subset=subsets(devices[:])
sum0=np.zeros([len(subset),NS])
sum1=np.zeros([len(subset),NS])


for i in range(len(subset)-1):
    for j in range(len(subset[i+1])):
	sum0[i]=sum0[i]+RawCF[(list(subset[i+1])[j])-1]
	sum1[i]=sum1[i]+rep_V[(list(subset[i+1])[j])-1]
    sum1[i]=sum1[i]/(j+1)

concur=np.empty([len(subset),(npts-64)])
convol=np.empty([len(subset),(npts-64)])

for i in range(len(sum0)):
    concur[i]=np.concatenate([sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i],sum0[i]])
    convol[i]=np.concatenate([sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i],sum1[i]])


########################################################################################################################################################
n=len(concur)

rep_I1 = np.empty([n,NS])
rep_V1 = np.empty([n,NS])

rep_I1=sum0
rep_V1=sum1

RawCF_new=rep_I1

		###############################################################################################################################################################

PQ = np.empty([n,NP,2])
for i in range(n-1):
    for j in range(NP-2):
			# extract current and voltage in two cycle
			temp_I = concur[i][j*NS:(j+2)*NS]
			temp_V = convol[i][j*NS:(j+2)*NS]
			# normalize voltage, since it's not well calibrated, calibrate current
			temp_I = temp_I*0.00246
			temp_V = temp_V*1.063
			# extract abs part(apparent component), divided by the number of points
			apparI = np.abs(2*np.fft.fft(temp_I))/NS
			apparV = np.abs(2*np.fft.fft(temp_V))/NS
			# phase difference
			theta = np.angle(np.fft.fft(temp_V)) - np.angle(np.fft.fft(temp_I))
			# calculate real/reactive power
			power=apparI*apparV
			ang1=np.cos(theta)
			ang2=np.sin(theta)
			intm_tempp=power*ang1
			intm_tempq=power*ang2
			tempP=intm_tempp/1.4
			tempQ=intm_tempq
			#PQ[i,j,0] = (tempP[2])
			PQ[i,j,1] = np.abs(tempQ[2])
			#PQ[i,j,1] = np.sum(tempQ)
			PQ[i,j,0] = np.abs(np.sum(tempP))
PQ = np.delete(PQ,np.where(np.isnan(PQ))[1],1)
PQ = np.median(PQ,1)

y = np.empty([n,100])


for i in range(n-1):
    tempII=concur[i]*0.00246
    tempII=tempII[0:100]
    y[i]=np.abs(np.fft.fft(tempII))
        
allF= np.concatenate([PQ,y,rep_I1],axis=1)
X_train = np.empty([n-1,166])
y_train = np.zeros([n-1])
for i in range(n-1):
		    X_train[i,]=allF[i]
		    y_train[i,]=i

elapsed_time = time.time() - start_time


while True:
	try:

		start_time = time.time()
		#os.system("rm -r /root/open_day/test/test.csv")
		print "reading for test"
		os.system("echo 1 > /root/flag_read.txt")
		time.sleep(11)
		os.system("sed -i 1,100d /root/open_day/test/test.csv")

		
	
		test_data= np.genfromtxt(test_csv_path+'test.csv',delimiter=',',names='current,voltage',dtype=(float,float),skip_header=556)
		rep_I2=np.empty(NS)
		rep_V2=np.empty(NS)
		tempI = np.sum(np.reshape(test_data['current'],[NP,NS]),0)/(NP)
		tempV = np.sum(np.reshape(test_data['voltage'],[NP,NS]),0)/(NP)
# align current to make all samples start from 0 and goes up
		ix = np.argsort(np.abs(tempV))
		j = 0
		while True:
			if ix[j]<63 and tempV[ix[j]+1]>tempV[ix[j]]:
	    			real_ix = ix[j]
	    			break
			else:
	    			j += 1
		rep_I2 = np.hstack([tempI[real_ix:],tempI[:real_ix]])
		rep_V2 = np.hstack([tempV[real_ix:],tempV[:real_ix]])
		rep_V2=  rep_V[1]
		
		

		concur[len(concur)-1]=np.concatenate([rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2,rep_I2])
		convol[len(convol)-1]=np.concatenate([rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2,rep_V2])
		

		i=len(concur)-1
		rep_I1[i,]=rep_I2
		rep_V1[i,]=rep_V2
		for j in range(NP-2):
					# extract current and voltage in two cycle
					temp_I = concur[i][j*NS:(j+2)*NS]
					temp_V = convol[i][j*NS:(j+2)*NS]
					# normalize voltage, since it's not well calibrated, calibrate current
					temp_I = temp_I*0.00246
					temp_V = temp_V*1.063
					# extract abs part(apparent component), divided by the number of points
					apparI = np.abs(2*np.fft.fft(temp_I))/NS
					apparV = np.abs(2*np.fft.fft(temp_V))/NS
					# phase difference
					theta = np.angle(np.fft.fft(temp_V)) - np.angle(np.fft.fft(temp_I))
					# calculate real/reactive power
					power=apparI*apparV
					ang1=np.cos(theta)
					ang2=np.sin(theta)
					intm_tempp=power*ang1
					intm_tempq=power*ang2
					tempP=intm_tempp/1.4
					tempQ=intm_tempq
					#PQ[i,j,0] = (tempP[2])
					PQ[i,1] = np.abs(tempQ[2])
					#PQ[i,j,1] = np.sum(tempQ)
					PQ[i,0] = np.abs(np.sum(tempP))
	        tempII = concur[i]*0.00246
    		y[len(concur)-1] = np.abs(np.fft.fft(tempII))[0:100]
		
		
		print "test ready"
		time.sleep(5)

		allF[len(concur)-1]= np.concatenate([PQ[len(concur)-1],y[len(concur)-1],rep_I1[len(concur)-1]],axis=0)

				
		X_test=allF[len(concur)-1]
		(y_p,names) = myclassifiers(X_train,y_train,X_test)
		
		if int(PQ[len(concur)-1][0])<10:
			strs="No devices available"
			wattage='0'
		else:
			strs = [str(Unq_type[int(list(subset[int(y_p[0])+1])[j])-1]) for j in range(len(subset[int(y_p[0])+1]))]
			wattage=str(int(PQ[len(concur)-1][0]))



		strin=str(strs).translate(None, "[']!#")
		
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)		
		try:
				  
				  HOST = "192.168.43.1"
				  PORT = 5000
				  s.connect((HOST, PORT))
				  
				  a={'req':'Active Devices are:, '+strin+', ,--------------------------,Total power in watts:    '+wattage+' W,--------------------------'}
				  
				  
				  b=json.dumps(a)
		  		  s.send(b)
				  print "sent"
				  try:
				      data = s.recv(19)
		    		      if data:
				        print "data"
				  except:
				       	s.close()
					print "time not received"
				  
		except KeyboardInterrupt:
				  print('Interruption from Keyboard')
				  s.close()


		
                elapsed_time = time.time() - start_time
		os.system("rm -r /root/open_day/test/test.csv")
		print "deleted test"
		
		print elapsed_time

	except KeyboardInterrupt:
				  print('Interruption from Keyboard')
				  break

	#except:
	#	print "General Exception"

