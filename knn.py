import cv2
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)

trainData = np.random.randint(0, 100, (25, 2)).astype(np.float32)
colors = np.random.randint(0, 2, (25, 1)).astype(np.float32)
red = trainData[colors.ravel()==1]
blue = trainData[colors.ravel()==0]
newData = np.random.randint(0, 100, (1, 2)).astype(np.float32)

plt.scatter(red[:,0], red[:,1], 100, 'r', 's')
plt.scatter(blue[:,0], blue[:,1], 100, 'b', '^')
plt.scatter(newData[:,0], newData[:,1], 100, 'g', 'o')
plt.legend(["1", "0"])

knn = cv2.ml.KNearest_create()
knn.train(trainData, 0, colors)
temp, results, neighbors, distances = knn.findNearest(newData, 3)

color = ""
if results[0][0] == 1:
    color = "red"
elif results[0][0] == 0:
    color = "blue"
print("Color: ", color)

print ("Results: ", results)
print ("Neighbors: ", neighbors)
print ("Distances: ", distances)

plt.show()