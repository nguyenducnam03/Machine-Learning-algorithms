from sklearn import datasets
import numpy as np
import math
import operator

#Code by class KNN algorithm
class KNN:
	def __init__(self,k):
		self.k = k	

	def fit(self,X_train,y_train):
		self.X_train = X_train
		self.y_train = y_train

	#distance between two points
	def calculate_distance(self,p1,p2):
		dimension = len(p1)
		distance = 0
		for i in range(dimension):
			distance += (p1[i] - p2[i])**2
		return math.sqrt(distance)	
	
	#get k nearest neighbors
	def get_k_neighbors(self,X_train,y_train,point,k):
		distances = []
		neighbors = []

		for i in range(len(X_train)):
			distance = self.calculate_distance(X_train[i],point)
			distances.append((distance, y_train[i]))

		#sort by distance
		distances.sort(key=operator.itemgetter(0))
		for i in range(k):
			neighbors.append(distances[i][1])
		return neighbors
	
	#find the label with the most votes
	def highest_votes(self, neighbors_labels):
		labels_count = [0,0,0]
		for label in neighbors_labels:
			labels_count[label] += 1
		return labels_count.index(max(labels_count))

	#predict the label of a point
	def predict(self,X_train,y_train,point,k):
		neighbors_labels = self.get_k_neighbors(X_train,y_train,point,k)
		return self.highest_votes(neighbors_labels)

#calculate accuracy
def accuracy_score(predict,ground_truth):
	total = len(predict)
	correct_count = 0
	for i in range(total):
		if predict[i] == ground_truth[i]:
			correct_count += 1
	return correct_count/total

##=================Data====================##
iris = datasets.load_iris()
#include sepal length, sepal width, petal length,petal width
#include iris.data and iris.target
#Before divide into training and predict, we need random it, using shuffle
iris_X = iris.data
iris_y = iris.target
#shuffle by index
randIndex = np.arange(iris_X.shape[0])
np.random.shuffle(randIndex)
iris_X = iris_X[randIndex]
iris_y = iris_y[randIndex]
#X_train, y_train, test
X_train = iris_X[:100,:]
X_test = iris_X[100:,:]
y_train = iris_y[:100]
y_test = iris_y[100:]
 
y_predict = []
k = 5

#KNN algorithm
knn_classifier = KNN(k)
knn_classifier.fit(X_train,y_train)
for test in X_test:
	y_predict.append(knn_classifier.predict(X_train,y_train,test,k))
	
#calculate accuracy
acc = accuracy_score(y_predict, y_test)
print(acc)
