#	A simple test of Scikit's Gausian Naive-Bayes classifier on weather prediction
# 	by: Valentina Narvaez

import csv
import os
import sys

import numpy as np

from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn import metrics
from sklearn.model_selection import cross_val_score


def main1():
	#	Get data from csv file
	dataSetName = 'data/seattle-weather.csv'
	reader = csv.reader(open(dataSetName, 'r'), delimiter=',')

	date = list()
	pres = list()
	tMax = list()
	tMin = list()
	wind =  list()
	weatherClass = list()

	#	Get columns into lists 
	i = 0
	for r in enumerate(reader):
		if i > 0:
			date.append(r[1][0])
			pres.append( float( r[1][1]) ) 
			tMax.append( float( r[1][2]) )
			tMin.append( float( r[1][3]) )
			wind.append( float( r[1][4]) )

			weatherClass.append(r[1][5])
		i = 1

	

	#	Map class labels from strings to integers
	lEncoder = preprocessing.LabelEncoder()
	cLabel = lEncoder.fit_transform(weatherClass)
	dateEnc = lEncoder.fit_transform(date)

	#	Make columns into a list of tuples
	data = list( zip(dateEnc, pres, tMax, tMin, wind) )
	

	#	Holdout data partitioning into train and test sets (30%), shuffled by default
	dataTrain, dataTest, cLabelTrain, cLabelTest = train_test_split( data, cLabel, test_size = 0.30, random_state = 0 )

	#	Train and test a gaussian NB model (single test)
	nbGauss = GaussianNB()
	nbGauss.fit( dataTrain, cLabelTrain )
	cLabelPred = nbGauss.predict( dataTest )
	met = metrics.accuracy_score(cLabelTest, cLabelPred)

	#	Test with cross validation, hold out, k = 5, 30% to test
	cvSpliter = ShuffleSplit( n_splits=50, test_size=0.30, random_state=0 )
	scores = cross_val_score(nbGauss, data, cLabel, cv = cvSpliter)

	print( 'Accuracy of single test: ', met )
	print( 'Accuracy of cross val test: ', scores)
	print( 'Mean acc cross val: ', np.mean(scores) )

if __name__ == '__main__':
	main1()