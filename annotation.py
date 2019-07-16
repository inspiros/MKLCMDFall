import configs
import numpy as np
import stringutils
import csv
import re

class Annotation:
	def __init__(self):
		self.annotation = list()
		with open(configs.ANNOTATION_FILE, 'r') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=',')

			def validate(row):
				if row[2] != str(configs.KINECT):
					return None
				if not row[3].isdigit():
					row[3] = re.sub('[^0-9]','', row[3])
				return row

			for row in spamreader:
				line = validate(row)
				if line is not None:
					self.annotation.append(line)

	# Data manipulation
	def getSubjectList(self, id, startFrame = None, endFrame = None):
		ret = list()
		for row in self.annotation:
			if int(row[1]) == id or row[1] == id:
				if startFrame != None and endFrame != None:
					if (row[4] == startFrame or int(row[4]) == startFrame) and (row[5] == endFrame or int(row[5]) == endFrame):
						ret.append(row)
				else:
					ret.append(row)
		return ret

	def getClassList(self, cls, startFrame = None, endFrame = None):
		ret = list()
		for row in self.annotation:
			if int(row[3]) == cls or row[3] == cls:
				if startFrame != None and endFrame != None:
					if (row[4] == startFrame or int(row[4]) == startFrame) and (row[5] == endFrame or int(row[5]) == endFrame):
						ret.append(row)
				else:
					ret.append(row)
		return ret

	def getKinectList(self, kinect, startFrame = None, endFrame = None):
		ret = list()
		for row in self.annotation:
			if int(row[2]) == kinect or row[2] == kinect:
				if startFrame != None and endFrame != None:
					if (row[4] == startFrame or int(row[4]) == startFrame) and (row[5] == endFrame or int(row[5]) == endFrame):
						ret.append(row)
				else:
					ret.append(row)
		return ret

	def getClass(list, frame):
		for row in list:
			if (stringutils.numericStringInRange(frame, row[4], row[5])):
				return row[3]

	def classOf(id, startFrame, endFrame = None):
		for row in self.annotation:
			if int(row[1]) == id or row[1] == id:
				if stringutils.numericStringEquals(row[4] == startFrame):
					if(endFrame != None and stringutils.numericStringEquals(row[5], startFrame)):
						return row[3]
					return row[3]


def fix_label(label):
	if label == 1:
		return 9
	elif label == 2:
		return 10
	elif label == 3:
		return 17
	elif label == 4:
		return 14
	elif label == 5:
		return 15
	elif label == 6:
		return 16
	elif label == 7:
		return 11
	elif label == 8:
		return 1
	elif label == 9:
		return 2
	elif label == 10:
		return 3
	elif label == 11:
		return 4
	elif label == 12:
		return 12
	elif label == 13:
		return 18
	elif label == 14:
		return 13
	elif label == 15:
		return 7
	elif label == 16:
		return 8
	elif label == 17:
		return 19
	elif label == 18:
		return 20
	elif label == 19:
		return 5
	elif label == 20:
		return 6

def to_6_groups(label):
	if label in range(1, 5):
		return 1
	elif label in range(5, 7):
		return 2
	elif label in range(7, 9):
		return 3
	elif label in range(9, 14):
		return 4
	elif label in range(14, 17):
		return 5
	elif label in range(17, 21):
		return 6

def to_2_groups(label):
	if label in range(1, 9):
		return 1
	elif label in range(9, 21):
		return 2

def fix_labelset(labelset):
	for i in range(0, labelset.shape[0]):
		labelset[i] = fix_label(labelset[i])
	return labelset

def to_20_groups_labelset(labelset):
	return labelset

def to_6_groups_labelset(labelset):
	temp = np.zeros(labelset.shape)
	for i in range(0, labelset.shape[0]):
		temp[i] = to_6_groups(labelset[i])
	return temp

def to_2_groups_labelset(labelset):
	temp = np.zeros(labelset.shape)
	for i in range(0, labelset.shape[0]):
		temp[i] = to_2_groups(labelset[i])
	return temp
