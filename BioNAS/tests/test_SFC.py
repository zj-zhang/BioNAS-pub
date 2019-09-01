# -*- coding: UTF-8 -*-
import unittest
from keras.models import Sequential
from keras.layers import Dense, Conv1D
from BioNAS.operators.convolutional import SeparableFC



class SFC_Test(unittest.TestCase):

	def test_SFC_build(self):
		model = Sequential()
		model.add(Conv1D(filters=3, kernel_size=4, input_shape=(200,4)))
		model.add(SeparableFC(output_dim=10, symmetric=True))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', metrics=['acc'])