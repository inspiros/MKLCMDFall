import os
import sys
import argparse
import numpy as np
from kernel import Kernel
from sklearn.preprocessing import normalize

from datapreparation import prepare_data
from annotation import to_20_groups_labelset, to_6_groups_labelset, to_2_groups_labelset

from sklearn.svm import SVC
from MKL.algorithms import EasyMKL
from MKL.multiclass import OneVsRestMKLClassifier

from evaluate import *

from configs import RGB_DATASET_ROOT, OF_DATASET_ROOT, DEPTH_DATASET_FILE, CONFIGS

def parse_args():
	parser = argparse.ArgumentParser(description='CMDFall MKL Classification')

	parser.add_argument('--confusion_matrix', type=bool, default=True,
	                        help='Confusion Matrix.')
	parser.add_argument('--classification_report', type=bool, default=True,
	                        help='Classification Report.')

	parser.add_argument('--streams', type=int, default=3,
	                        help='Number of streams: 2 for RGB-OF and 3 for RGB-OF-Depth.')

	parser.add_argument('--num_classes', type=int, default=None,
							help='Number of classes: 20/6/2, leave None for all.')

	parser.add_argument('--kernels', type=str, default=None,
	                        help='Kernel.', required=True)
	return parser.parse_args()


args = parse_args()

CONFUSION_MATRIX = args.confusion_matrix
CLASSIFICATION_REPORT = args.classification_report

KERNEL_TYPE = args.kernels

STREAMS = args.streams

LABELING_TYPES = [to_20_groups_labelset, to_6_groups_labelset, to_2_groups_labelset]
if args.num_classes == 20:
	LABELING_TYPES = [to_20_groups_labelset]
elif args.num_classes == 6:
	LABELING_TYPES = [to_6_groups_labelset]
elif args.num_classes == 2:
	LABELING_TYPES = [to_2_groups_labelset]


def get_prepared_data():
	global RGB_DATASET_ROOT, OF_DATASET_ROOT, DEPTH_DATASET_FILE, STREAMS
	return prepare_data(RGB_DATASET_ROOT, OF_DATASET_ROOT, DEPTH_DATASET_FILE, streams=STREAMS)


def train_test_evaluation(datasets, kernels, kernels_concate=None, C_mkl=None, C_svms=None, C_concatenate=None, lam_mkl=None):
	global CONFUSION_MATRIX
	global CLASSIFICATION_REPORT
	global STREAMS
	global LABELING_TYPES
	SAVE_SUB_DIR = os.path.join('2 streams' if STREAMS==2 else '3 streams', KERNEL_TYPE.lower())

	((Xtrs, ytr_ori), (Xtes, yte_ori)) = datasets

	# Compute Kernel of concatenated normalized features
	Xtrain_concate = np.concatenate([normalize(Xtr) for Xtr in Xtrs], axis=1)
	Xtest_concate = np.concatenate([normalize(Xte) for Xte in Xtes], axis=1)

	# Compute Kernels of each modality for MKL
	Ktr_concate = kernels_concate.apply(Xtrain_concate)
	Kte_concate = kernels_concate.apply(Xtest_concate, Xtrain_concate)

	KLtr = [None for i in Xtrs]
	KLte = [None for i in Xtes]
	overall_scores = {}
	scores = {}
	for i in range(len(Xtrs)):
		KLtr[i] = kernels[i].apply(Xtrs[i])
		KLte[i] = kernels[i].apply(Xtes[i], Xtrs[i])

	print('')
	print('EVALUATION PROCEDURE BEGINS')
	for label_set in LABELING_TYPES:
		# set prefix
		if label_set is to_20_groups_labelset:
			prefix = '20 classes'
		elif label_set is to_6_groups_labelset:
			prefix = '6 classes'
		else:
			prefix = '2 classes'
		# set labels 
		ytr = label_set(ytr_ori)
		yte = label_set(yte_ori)

		# Initialize MKL
		base_learner = SVC(C=C_mkl, tol=0.0001, kernel='precomputed')
		clf = EasyMKL(estimator=base_learner, lam=lam_mkl, max_iter=1000, verbose=True)
		mkl = OneVsRestMKLClassifier(clf)

		# Fit and eval MKL
		mkl.fit(KLtr, ytr)
		y_pred = mkl.predict(KLte)
		mkl_score = f1(yte, y_pred)
		scores.update({prefix + ' EasyMKL':mkl_score})

		if CONFUSION_MATRIX:
			# save confusion matrix of MKL
			print('MKL')
			confuse(yte, y_pred, weights=mkl.weights, title='EasyMKL ' + Kernel.stringigfy(kernels, STREAMS), save=True, subdir=os.path.join(SAVE_SUB_DIR, prefix))
		if CLASSIFICATION_REPORT:
			# save classification report of MKL
			report(yte, y_pred, title='EasyMKL ' + Kernel.stringigfy(kernels, STREAMS), save=True, subdir=os.path.join(SAVE_SUB_DIR, prefix))


		# Initialize, fit and eval SVM on concatenated features
		clf_concatenate = SVC(C=C_concatenate, tol=0.0001, kernel='precomputed')
		clf_concatenate.fit(Ktr_concate, ytr)
		y_pred = clf_concatenate.predict(Kte_concate)
		svm_concate_score = f1(yte, y_pred)
		scores.update({prefix + ' SVM':svm_concate_score})

		if CONFUSION_MATRIX:
			# save confusion matrix of SVM on concatenated features
			print('SVM on concatenated features')
			confuse(yte, y_pred, title='SVM ' + kernels_concate.tostring(STREAMS), save=True, subdir=os.path.join(SAVE_SUB_DIR, prefix))
		if CLASSIFICATION_REPORT:
			# save classification report of SVM on concatenated features
			report(yte, y_pred, title='SVM ' + kernels_concate.tostring(STREAMS), save=True, subdir=os.path.join(SAVE_SUB_DIR, prefix))

		'''
		SVM on each modality
		'''
		for i in range(len(Xtrs)):
			# Fit and eval SVM on single modality
			clf = SVC(C=C_svms[i], tol=0.0001, kernel='precomputed')
			clf.fit(KLtr[i], ytr)
			y_pred = clf.predict(KLte[i])
			svm_score = f1(yte, y_pred)
			scores.update({prefix + ' SVM ' + kernels[i].name:svm_score})

			if CONFUSION_MATRIX:
				# save confusion matrix of SVM on single modality
				print('SVM on ' + kernels[i].name)
				confuse(yte, y_pred, title='SVM ' + kernels[i].tostring(STREAMS), save=True, subdir=os.path.join(SAVE_SUB_DIR, prefix))
			if CLASSIFICATION_REPORT:
				# save classification report of SVM on single modality
				report(yte, y_pred, title='SVM ' + kernels[i].tostring(STREAMS), save=True, subdir=os.path.join(SAVE_SUB_DIR, prefix))

		summary(scores, os.path.join(SAVE_SUB_DIR, prefix))
		overall_scores.update(scores)
		scores = {}

	print('')
	print('')
	print('SUMARIZING OVERALL RESULTS')
	summary(overall_scores, None, save=False)
	return overall_scores


'''
Program begins
'''
for config in CONFIGS:
	if config.is_assignable(KERNEL_TYPE):
		print('[CMDFall MKL Classification]', config.name)
		result = train_test_evaluation(get_prepared_data(), *config.to_params())
		break
