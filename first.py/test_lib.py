import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_mean(X):
	sumofX=0
	for x in X:
		sumofX += x
	return sumofX/len(X)

class LinearRegressionModel:
	def __init__(self):
		self.coefficients=[]
		self.intercept=0
		self.Y=[]
		self.X1=[]
		self.X2=[]
		self.pY=[]

	def find_coeffs(self,independent_vars,dependent_var):
		X1 = independent_vars[0]
		X2 = independent_vars[1]
		Y  = dependent_var

		self.X1=X1
		self.X2=X2
		self.Y=Y

		#temp variables
		sumofX1=0
		sumofX2=0
		sumofY =0

		sumofX1_squared=0
		sumofX2_squared=0

		sumofX1Y = 0
		sumofX2Y = 0
		sumofX1X2= 0

		for x1,x2,y in zip(X1,X2,Y):

			sumofX1 += x1
			sumofX2 += x2
			
			sumofY  += y
			
			sumofX1_squared += x1 ** 2
			sumofX2_squared += x2 ** 2

			sumofX1Y += x1*y
			sumofX2Y += x2*y
			sumofX1X2 += x1*x2


		sumof_X1_squared = (sumofX1_squared) - (sumofX1) ** 2 / len(Y)
		sumof_X2_squared = (sumofX2_squared) - (sumofX2) ** 2 / len(Y)
		
		sumof_X1Y = (sumofX1Y) - (sumofX1) * (sumofY) / len(Y)
		sumof_X2Y = (sumofX2Y) - (sumofX2) * (sumofY) / len(Y)

		sumof_X1X2 = (sumofX1X2) - (sumofX1) * (sumofX2) / len(Y)


		print()
		#print("sigma X1 squared: ",sumof_X1_squared)
		#print("sigma X2 squared: ",sumof_X2_squared)
		#print("sigma X1Y: ",sumof_X1Y)
		#print("sigma X2Y: ",sumof_X2Y)
		#print("sigma X1X2: ",sumof_X1X2)
		#print()

		b1 = ( (sumof_X2_squared * sumof_X1Y ) - ( sumof_X1X2 * sumof_X2Y ) ) / ( sumof_X1_squared * sumof_X2_squared  - sumof_X1X2**2)
		b2 = ( (sumof_X1_squared * sumof_X2Y ) - ( sumof_X1X2 * sumof_X1Y ) ) / ( sumof_X1_squared * sumof_X2_squared  - sumof_X1X2**2)

		print("coefficient of X1: ",b1)
		print("coefficient of X2: ",b2)
		print()

		self.coefficients=[b1,b2]

	def find_intercept(self):
		
		b1 = self.coefficients[0]
		b2 = self.coefficients[1]
		
		mean_Y  = find_mean(self.Y)
		mean_X1 = find_mean(self.X1)
		mean_X2 = find_mean(self.X2)

		b0 = mean_Y - (b1*mean_X1) - (b2*mean_X2)
		print(self.Y)
		print(self.X1)
		print(self.X2)

		print("mean of y: ",mean_Y)
		print("mean of x1: ",mean_X1)
		print("mean of x2: ",mean_X2)
		print("intercept: ",b0)
		self.intercept=b0

	def create_regression_line():
		pass

	def predict_Y(self,x1,x2):
		pY=self.intercept + (self.coefficients[0] * x1) + (self.coefficients[1] * x2)
		print(pY)
		return pY

Y =  [-3.7,3.5,2.5,11.5,5.7]
X1 = [3,4,5,6,2]
X2 = [8,5,7,3,1]

X=[X1,X2]

model = LinearRegressionModel()
model.find_coeffs(X,Y)
model.find_intercept()
model.predict_Y(3,8)