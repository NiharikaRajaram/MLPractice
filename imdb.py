# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 20:14:58 2019

@author: Rajaram
"""


from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)
print(X.shape())
print(Y.shape())