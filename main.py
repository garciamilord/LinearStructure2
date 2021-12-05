

import numpy as np
from sklearn.cluster import KMeans
import json
from sklearn.model_selection import train_test_split

class q1:
    '''Part 1 q#2'''

with open('reg_dataset.json') as f:
    r = json.load(f)
y = np.array(r['y'])
X = np.array(r['X'])
print("Part 1 q#2")
def regression(X,y):
    # (X^T X)
    a = X.T @ X
    # X(X.T @ X)^-1
    inv = X @ np.linalg.inv(a)
    # y
    b = y
    # yX(X.T @ X)^-1
    theta = b @ inv
    thetaNot = theta.T
    return thetaNot
prediction = np.dot(X,regression(X,y))
print('Prediction: '+str(prediction[-5:]).format(prediction))
print()
print('#################################################################################################################')

print('Part 2 q#2')
class q2:
    '''Part 2 q#2'''
with open('iris_dataset.json') as f:
    i = json.load(f)
y = np.array(i['target'])
X = np.array(i['data'])

X_train, X_test, y_train, y_test = train_test_split(X, y)


model = KMeans(n_clusters=3).fit(X_train, y_train)
model.predict(X_train, y_train)
print()
print('Train data: ')
print(model.predict(X_train, y_train))
print('----------------------------------------------------------------------------------------------------------------')
print('Train accurracy: '+ str(model.score(X_train, y_train)))
print()
print('----------------------------------------------------------------------------------------------------------------')
print('Original accurracy: ' + str(model.score(X_test,y_test)))

# print('------------------------------------------------------------------------')
# print(f"Train labels:\n{y_train}")
# print(f"Test labels:\n{y_test}")
# print('------------------------------------------------------------------------')
# print(f"Train labels:\n{X_train}")
# print(f"Test labels:\n{X_test}")