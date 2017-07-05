#import tensorflow
#import tflearn
#import pandas
import matplotlib.pyplot as plt
import numpy as np
from exchanges import *
from sklearn import linear_model

key = None
secret = None

def main(key, secret):
	plt.ion()
	while (True):
		ark = getHistory("BTC-ARK", key, secret)
		dgb = getHistory("BTC-DGB", key, secret)

		#plt.title('ARK')
		#plt.plot(ark[0], ark[1], label="Actual Price")

		ark_train_x = np.array(ark[0][:-100])
		ark_test_x = np.array(ark[0][-100:])
		ark_train_y = np.array(ark[1][:-100])
		ark_test_y = np.array(ark[1][-100:])
		
		ark_x = np.array(ark[0])
		ark_y = np.array(ark[1])
		regr = linear_model.LinearRegression()
		
		regr.fit(ark_x.reshape(1,-1), ark_y.reshape(1,-1))
		
		plt.plot(ark_x, ark_y, label = "Actual Price")
		#plt.scatter(ark_x[:-1]+100, regr.predict((ark_x[:-1])+100)[0], label = "prediction")
		print(regr.predict((1400+100)))
		plt.legend()

		plt.pause(1)
		plt.clf()
		plt.show()


	
	

def getHistory(token, key, secret):
	bt = bittrex(key, secret)
	ark = bt.getmarkethistory(token ,count = 2000)
	results = ark['result']
	
	prices, time, quant, type, total = [],[],[],[],[]
	for i in results:
		time.append(i['Id'])
		prices.append(i['Price'])
		quant.append(i['Quantity'])
	return([time, prices, quant])
	
	
if __name__ == '__main__':
	main(key, secret)