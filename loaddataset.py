import configs
from annotation import Annotation
import blob
import fileutils
import numpy as np
import pickle
import re
import os


class Record:
	def __init__(self, data = None, label = None, frame = None):
		self.data = data
		self.label = label
		self.frame = frame

class Subject:
	def __init__(self, id, name = None):
		self.id = id
		self.name = name
		self.records = list()

class SingleDataset:
	def __init__(self):
		self.subjects = list()

	def extract(self, identities):
		X = np.array([[]])
		y = np.array([])
		z = []

		for identity in identities:
			subject_id = re.findall(r'^\d+', identity)[0]
			frame = re.findall(r'\d+$', identity)[0]
			subject_name = identity[len(subject_id) + 2:len(identity) - len(frame) - 2]
			for subject in self.subjects:
				if subject.name == subject_name and subject.id == subject_id:
					for record in subject.records:
						if int(frame) in record.frame:
							if y.shape != np.array([]).shape:
								X = np.concatenate((X, np.expand_dims(record.data, axis = 0)), axis = 0)
								y = np.concatenate((y, np.array([record.label])), axis = 0)
								z.extend([identity])
							else:
								X = np.expand_dims(record.data, axis = 0)
								y = np.array([record.label])
								z = [identity]
		return X, y, z

	def getAll(self):
		X = np.array([[]])
		y = np.array([])
		z = []

		for subject in self.subjects:
			for record in subject.records:
				if y.shape != np.array([]).shape:
					X = np.concatenate((X, np.expand_dims(record.data, axis = 0)), axis = 0)
					y = np.concatenate((y, np.array([record.label])), axis = 0)
					z.extend([str(subject.id) + '__' + str(subject.name) + '__' + str(record.frame)])
				else:
					X = np.expand_dims(record.data, axis = 0)
					y = np.array([record.label])
					z = [str(subject.id) + '__' + str(subject.name) + '__' + str(record.frame)]
		return X, y, z

	def getEvenIds(self):
		X = np.array([[]])
		y = np.array([])
		z = []

		for subject in self.subjects:
			if subject.id % 2 == 0:
				for record in subject.records:
					if y.shape != np.array([]).shape:
						X = np.concatenate((X, np.expand_dims(record.data, axis = 0)), axis = 0)
						y = np.concatenate((y, np.array([record.label])), axis = 0)
						z.extend([str(subject.id) + '__' + str(subject.name) + '__' + str(record.frame)])
					else:
						X = np.expand_dims(record.data, axis = 0)
						y = np.array([record.label])
						z = [str(subject.id) + '__' + str(subject.name) + '__' + str(record.frame)]
		return X, y, z

	def getUnevenIds(self):
		X = np.array([[]])
		y = np.array([])
		z = []

		for subject in self.subjects:
			if subject.id % 2 == 1:
				for record in subject.records:
					if y.shape != np.array([]).shape:
						X = np.concatenate((X, np.expand_dims(record.data, axis = 0)), axis = 0)
						y = np.concatenate((y, np.array([record.label])), axis = 0)
						z.extend([str(subject.id) + '__' + str(subject.name) + '__' + str(record.frame)])
					else:
						X = np.expand_dims(record.data, axis = 0)
						y = np.array([record.label])
						z = [str(subject.id) + '__' + str(subject.name) + '__' + str(record.frame)]
		return X, y, z


# Loaders
class DatasetLoader:
	def __init__(self, directory):
		subject_mapping = {}
		with open(configs.MAPPING_FILE) as mappping_file:
			mappping_file = mappping_file.read().strip().split('\n')
			for line in mappping_file:
				id = re.findall(r'^(\d+)\s', line)[0]
				name = line[len(id) + 1:]
				subject_mapping[name] = id

		def getSubject(name):
			return int(subject_mapping[name])

		#
		self.dataset = SingleDataset()

		subject_dirs = fileutils.listdir(directory)
		ann = Annotation()


		prev_subject_id = None
		sjList = None
		for dir in subject_dirs:
			subject_name = fileutils.dirname(dir)
			subject_id = getSubject(subject_name)

			duplicate = None
			for sj in self.dataset.subjects:
				if sj.id == subject_id and sj.name == subject_name:
					duplicate = sj

			##
			if duplicate == None:
				subject = Subject(subject_id, subject_name)
			else:
				subject = duplicate

			if prev_subject_id != subject_id:
				sjList = ann.getSubjectList(subject_id)
			prev_subject_id = subject_id

			for file in fileutils.recursive_walk(dir):
				if fileutils.fileextension(file) == configs.LAYER:
					###
					filename = fileutils.filename(file)
					rc = Record()
					rc.data = blob.load_np_array(file)
					rc.label = int(Annotation.getClass(sjList, filename))
					rc.frame = filename

					subject.records.append(rc)
					print('Loading', subject_id, filename, rc.label)

			if(duplicate == None):
				self.dataset.subjects.append(subject)
	#list = ann.getSubjectList(subject_id)
	#print(subject_id, len(list))


class PklDatasetLoader:
	def __init__(self, file):
		with open(file, 'rb') as input:
			dataDict = pickle.load(input)

		self.dataset = SingleDataset()

		for key in dataDict.keys():
			string = re.findall(r'file_index__\d+__', key)[0]
			subject_id = string[len('file_index__'):len(string) - len('__')]
			string = re.findall(r'filename__.+__start', key)[0]
			subject_name = string[len('filename__'):len(string) - len('__start')]
			string = re.findall(r'action__\d+__', key)[0]
			label = int(string[len('action__'):len(string) - len('__')])
			string = re.findall(r'start__\d+__', key)[0]
			start = int(string[len('start__'):len(string) - len('__')])
			string = re.findall(r'stop__\d+\.png', key)[0]
			stop = int(string[len('stop__'):len(string) - len('.png')])

			duplicate = None
			for sj in self.dataset.subjects:
				if sj.id == subject_id and sj.name == subject_name:
					duplicate = sj

			if duplicate == None:
				subject = Subject(subject_id, subject_name)
				self.dataset.subjects.append(subject)
			else:
				subject = duplicate

			rc = Record()
			rc.data = dataDict.get(key)
			rc.label = label
			rc.frame = range(start, stop + 1)

			subject.records.append(rc)
			print('Loading', subject_id, start, '-', stop)
			# 	id = re.findall(r'^(\d+)\s', line)[0]
			# 	name = line[len(id) + 1:]
