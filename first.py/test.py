#    sumof(x- (mean of x) ) *  (y - (mean of y) )
#m = --------------------------------------------------
#	        sum of square of(x - (mean of x))


import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("HeightWeight.csv")

#x = [ 1.47  , 1.5   , 1.52  , 1.55  , 1.57 , 1.6   , 1.63  , 1.65  , 1.68  , 1.7   , 1.73  , 1.75 , 1.78  , 1.8   , 1.83 ]
#y = [ 52.21 , 53.12 , 54.48 , 55.84 , 57.2 , 58.57 , 59.93 , 61.29 , 63.11 , 64.47 , 66.28 , 68.1 , 69.92 , 72.19 , 74.46 ]
x = data["Height(Inches)"].values
y = data["Weight(Pounds)"].values

def findMean (num):
	mean=0
	sumation=0
	for n in num:
		sumation+=n
	mean = sumation / len(num)
	return mean

#printing mean

def findSlope(dvx,dvy):
	meanX = findMean(dvx)
	meanY = findMean(dvy)
	total = 0
	totalX_square = 0

	for x,y in zip(dvx,dvy):
		total += (x - meanX) * (y - meanY)
		totalX_square += (x - meanX)**2

	m= total/totalX_square
	print("total: ",total)
	print("totalX_square: ",totalX_square)
	return m



def calculate_M_C(x,y):

	X = findMean(x)
	Y = findMean(y)

	m = findSlope(x,y)
	c = Y - (m*X)
	return ([m,c])

#finding regretion array
def find_Reg_Line(X,M,C):
	reg_line=[]
	for x in X:
		reg_line.append( M*x + C)
	return reg_line

def plot_reg(X,Y,M,C):

	reg_line=find_Reg_Line(X,M,C)
	#print(reg_line)

	#ploting actual data of data sets
	plt.plot(X,Y,"bo")
	#ploting regression line
	plt.plot(X,reg_line,color="orange")
	plt.title("Example 1")
	plt.ylabel("weight")
	plt.xlabel("height")
	plt.draw()
	plt.show()


#for calculating R^2
def calculate_R_Square(Y,reg_line):
	sumof_predicted = 0
	sumof_actual = 0
	meanY=findMean(Y)
	for y,yp in zip(Y,reg_line):
		sumof_predicted += (yp - meanY)**2
		sumof_actual += (y - meanY)**2

	return sumof_predicted/sumof_actual

m,c=calculate_M_C(x,y)


#claculating R^2
print( calculate_R_Square(y,find_Reg_Line(x,m,c)) )

#print(m,c)
plot_reg(x,y,m,c)
