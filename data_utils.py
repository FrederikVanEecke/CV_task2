"""
	Dataset classes for interaction with the data
"""
from albumentations import (
	Compose, HorizontalFlip, CLAHE, HueSaturationValue,
	RandomBrightness, RandomContrast, RandomGamma,
	ToFloat, ShiftScaleRotate, RandomBrightnessContrast, RandomCrop)

import numpy as np
import pandas as pd
import os 
import cv2
import matplotlib.pyplot as plt 
from keras.utils import to_categorical

class Dataset_Classification(object): 
	"""
	 A dataset class usefull when training a classification model. 
	"""
	def __init__(self, config): 
		self.config = config
		
		# set labels in a usefull format
		self._classification_labels()
		
		# initialize
		self.initialize()
		
		# to keep track of sampling
		self.sampling_check = np.zeros(20)
		self.times_sampled = 0
		
		
	def initialize(self):
		"""
			Performs necessary thingss
		"""
		self.train_data_path = '../input/kul-h02a5a-computervision-groupassignment1/train/img'
		self.test_data_path = '../input/kul-h02a5a-computervision-groupassignment1/test/img'
		
		# count nbr of files within data set. 
		self.nbr_of_train_images = len(os.listdir(self.train_data_path))
		self.nbr_of_test_images = len(os.listdir(self.test_data_path))
		
		# prepare train/validation split 
		train_fraction = self.config['train_fraction']
		r_idx=np.random.permutation(self.nbr_of_train_images)
		
		self.train_indices = r_idx[:int(train_fraction*self.nbr_of_train_images)]
		self.train_sample_probs = self.probabilities[self.train_indices]/np.sum(self.probabilities[self.train_indices])
		
		self.validation_indices = r_idx[int(train_fraction*self.nbr_of_train_images):]
		self.validation_sample_probs = self.probabilities[self.validation_indices]/np.sum(self.probabilities[self.validation_indices])
		
		print('Found {} train images'.format(self.nbr_of_train_images))
		print('- {} used for training, {} used for validating'.format(len(self.train_indices), len(self.validation_indices)))
		print('Found {} test images'.format(self.nbr_of_test_images))
		
		if self.config['augmentation']: 
			print('Including augmentation when training data is generated')
		self.augment = Compose([
						#RandomCrop(width=self.config['input_shape'][0], height=self.config['input_shape'][0]),
						HorizontalFlip(p=0.5),
						RandomContrast(limit=0.1,p=0.25),
						#RandomGamma(gamma_limit=(80, 120), p=0.5),
						RandomBrightness(limit=0.15, p=0.5),
#                         HueSaturationValue(hue_shift_limit=1.5, sat_shift_limit=5,
#                                            val_shift_limit=2.5, p=.7),
						# CLAHE(p=1.0, clip_limit=2.0),
						ShiftScaleRotate(
							shift_limit=0.1, scale_limit=0.1, 
							rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.8), 
					])
	
		self.preprocessor = lambda x: x #defualt preprocessor does nothing 
		
	def get_test_set_size(self): 
		return self.nbr_of_test_images
	def get_train_set_size(self): 
		return self.nbr_of_train_images
		
	def reshape(self,im): 
		return cv2.resize(im, self.config['input_shape'])
		
	def _classification_labels(self): 
		"""
			Get the classification labels 
		"""
		# label names 
		train_df = pd.read_csv('../input/kul-h02a5a-computervision-groupassignment1/train/train_set.csv', index_col="Id")
		self.label_names = train_df.columns.to_numpy()
		
		# get rid of pandas frame 
		self.labels = train_df.to_numpy() # each row corresponds to label
		
		# instances per class
		absolute_nbr_of_instances_per_class = np.sum(self.labels,axis=0)
		# get array with label name for each image 
		self.class_name_per_image = list()
		probabilities = list()
		total_class_prob = 1/20*np.ones(20)
		#total_class_prob[14]=0
		for row in self.labels: #can probably be done more eligant
			idx = np.where(row==1)[0]
			if 14 in idx:
				idx=14
			else:
				idx=idx[0]
			probabilities.append(total_class_prob[idx]/absolute_nbr_of_instances_per_class[idx])
			self.class_name_per_image.append(self.label_names[idx])
		
		
		# make sure total probability sums to 1
		if self.config['uniform_sample_probabilities']:
			self.probabilities = np.ones(np.array(probabilities).shape)/len(probabilities) # uniform sampling
		else:
			self.probabilities = np.array(probabilities)/np.sum(probabilities) # sampling based on distribution of classes in trainning data
			
			
	def feed_preprocess_function(self, preprocessor): 
		"""
			Each network needs it's batches preprocessed in some manner. Feed this function to the Dataset object 
			who will call it when asking for batches.
			
			The preprocessor takes 
		"""
		self.preprocessor = preprocessor
	
	def prepare_image(self, image): 
		"""
			Function that performs all necessary steps from input image to image passed during trainnig. 
			This method should be overwritten depending on the model used.
		"""
		h,w,c=image.shape
		# resize manually, when augmentation is turned on a random crop will be done 100% of times.
		#if not self.config['augmentation'] or h < self.config['input_shape'][0]or w < self.config['input_shape'][1]:
		image = self.reshape(image)

		# augment if augmentation is turned on 
		if self.config['augmentation']:
#             print('Augmentation enabled: check if combination of preprocessor and augmentation makes sence')
			image=self.augment(image=image)["image"]
		
		# preprocess 
		image = self.preprocessor(image)
		return image
	
	def prepare_test_image(self, image): 
		"""
			 Same as prepare_image but without augmentation
		"""
		h,w,c=image.shape
		# resize manually, when augmentation is turned on a random crop will be done 100% of times.
		#if not self.config['augmentation'] or h < self.config['input_shape'][0]or w < self.config['input_shape'][1]:
		image = self.reshape(image)
		
		# preprocess 
		image = self.preprocessor(image)
		return image
	
	def view_preprocessed_image(self, image_id, option='train'): 
		"""
			Shows an image as it is passed during training/testing of the network. 
			image_id 
		"""
		assert hasattr(self, 'preprocessor'), 'set a preprocessor function before using this.'
		
		# get image 
		if option=='train':
			# get image 
			real_image = np.load('../input/kul-h02a5a-computervision-groupassignment1/train/img/train_{}.npy'.format(image_id) )
			label = self.class_name_per_image[image_id]
		else: 
			real_image = np.load('../input/kul-h02a5a-computervision-groupassignment1/test/img/test_{}.npy'.format(image_id) )
			label = 'unknown'
			
		image=np.copy(real_image)
		
		image = self.prepare_image(image)
		
	
	
		# print some info 
		print('original image:')
		print('-original_shape:', real_image.shape)
		print('-dtype:', real_image.dtype)
		print('-min value:', np.min(real_image))
		print('-max value:', np.max(real_image))
		
		print('final image:')
		print('-final shape:', image.shape)
		print('-dtype:', image.dtype)
		print('-min value:', np.min(image))
		print('-max value:', np.max(image))
		
		
		# show figure 
		fig, axes = plt.subplots(1,2, figsize=(30,15))
		axes[0].imshow(real_image)
		axes[0].set_title('Original image', fontsize=50)
		
		axes[1].imshow(image) # clip it to [0,1] range
		axes[1].set_title('preprocessed_image', fontsize=50)
		
		plt.suptitle('label: {}'.format(label), fontsize=50)
		fig.show()
		
	def view_possible_augmentations(self, image_id): 
		fig, axes = plt.subplots(4,4, figsize=(60,30))
		real_image = np.load('../input/kul-h02a5a-computervision-groupassignment1/train/img/train_{}.npy'.format(image_id) )
		plt.suptitle('original image is on the top left', fontsize=50)
		for ax in axes.flat: 
			ax.imshow(self.augment(image=real_image)["image"])
			ax.axis('off')
		axes[0,0] = plt.imshow(real_image)
		fig.show()
		
		
	
	def train_generator(self,batch_size):
		"""
			generator that will feed training batches during training 
		"""
		inputs = []
		targets = []
		batchcount = 0
		while True:
#             for image_id in self.train_indices:
			image_id = np.random.choice(self.train_indices, p=self.train_sample_probs)
			# sample real image
			real_image = np.load('../input/kul-h02a5a-computervision-groupassignment1/train/img/train_{}.npy'.format(image_id) )

			image = self.prepare_image(np.copy(real_image))
			inputs.append(image)

			# get corresponding label 
			targets.append(self.labels[image_id])
			
			self.sampling_check+=self.labels[image_id]
			self.times_sampled+=1
			

			batchcount += 1
			if batchcount >= batch_size:
				X = np.array(inputs)
				y = np.array(targets, dtype=np.uint8)
				yield (X, y)
				inputs = []
				targets = []
				batchcount = 0

	def validation_generator(self,batch_size):
		"""
			generator that will feed validation batches during training 
		"""
		inputs = []
		targets = []
		batchcount = 0
		while True:
			#for image_id in self.validation_indices:
			# sample real image
			image_id = np.random.choice(self.validation_indices,p=self.validation_sample_probs)
			real_image = np.load('../input/kul-h02a5a-computervision-groupassignment1/train/img/train_{}.npy'.format(image_id) )

			image = self.prepare_test_image(np.copy(real_image))

			inputs.append(image)

			# get corresponding label 
			targets.append(self.labels[image_id])

			batchcount += 1
			if batchcount >= batch_size:
				X = np.array(inputs)
				y = np.array(targets, dtype=np.uint8)
				yield (X, y)
				inputs = []
				targets = []
				batchcount = 0           
	
	def test_generator(self,batch_size):
		"""
			generator for feeding test data to model for prediction
		"""
		inputs = []
		img_id = 0
		while img_id < self.nbr_of_test_images:
			raw_image = np.load('../input/kul-h02a5a-computervision-groupassignment1/test/img/test_{}.npy'.format(img_id))
			image = self.prepare_test_image(np.copy(raw_image))
			inputs.append(image)
			img_id += 1
			if img_id%batch_size == 0:
				X = np.array(inputs)
				yield X
				inputs = []
		if len(inputs) > 0:
			return np.array(inputs)
	
	def show_class_distribution(self): 
		fig,axes=plt.subplots(figsize=(30,15))
		class_probs=np.mean(self.labels, axis=0)
		axes.bar(self.label_names,  class_probs)
		axes.tick_params(axis='both', which='major', labelsize=30)
		for tick in axes.xaxis.get_major_ticks():
			tick.label.set_rotation('vertical')
		plt.suptitle('Class distribution within the training data.', fontsize=50)
		fig.show()
		
	def show_training_sampling_distribution(self): 
		fig,axes=plt.subplots(figsize=(30,15))
		class_probs=np.mean(self.labels, axis=0)
		axes.bar(self.label_names,  self.sampling_check/self.times_sampled)
		axes.tick_params(axis='both', which='major', labelsize=30)
		for tick in axes.xaxis.get_major_ticks():
			tick.label.set_rotation('vertical')
		plt.suptitle('Number of times each class was sampled during training.', fontsize=50)
		fig.show()
	
	def get_class_distribution(self):
		return np.mean(self.labels, axis=0)



class Dataset_Segmentation(Dataset_Classification): 
	"""
	 A dataset class usefull when training a classification model. 
	"""
	def __init__(self, config): 
		self.config = config
	
		
		# set labels in a usefull format
		self._classification_labels()
		
		 # initialize
		self.initialize()
	  
	def prepare_image(self, image, mask): 
		"""
			Function that performs all necessary steps from input image to image passed during trainnig. 
			This method should be overwritten depending on the model used.
		"""
		h,w,c=image.shape
		# resize manually, when augmentation is turned on a random crop will be done 100% of times.
		#if not self.config['augmentation'] or h < self.config['input_shape'][0]or w < self.config['input_shape'][1]:
		image = self.reshape(image)
		mask=self.reshape(mask)

		# augment if augmentation is turned on 
		if self.config['augmentation']:
#             print('Augmentation enabled: check if combination of preprocessor and augmentation makes sence')
			transformed=self.augment(image=image, mask=mask)
			image=transformed['image']
			mask=transformed['mask']
		# preprocess 
		image = self.preprocessor(image)
		return image, mask
	

	def prepare_test_image(self, image, mask=None): 
		"""
			 Same as prepare_image but without augmentation
		"""
		h,w,c=image.shape
		# resize manually, when augmentation is turned on a random crop will be done 100% of times.
		#if not self.config['augmentation'] or h < self.config['input_shape'][0]or w < self.config['input_shape'][1]:
		image = self.reshape(image)
		if mask != None:
			mask=self.reshape(mask)
		
		# preprocess 
		image = self.preprocessor(image)
		return image, mask 
	
	def view_preprocessed_image(self, image_id, option='train'): 
		"""
			Shows the original, preprocessed and final mask used for training.
		"""
		assert hasattr(self, 'preprocessor'), 'set a preprocessor function before using this.'
		
		# get image 
		if option=='train':
			# get image 
			real_image = np.load('../input/kul-h02a5a-computervision-groupassignment1/train/img/train_{}.npy'.format(image_id) )
			mask = np.load('../input/kul-h02a5a-computervision-groupassignment1/train/seg/train_{}.npy'.format(image_id))
			label_name=self.class_name_per_image[image_id]
		else: 
			real_image = np.load('../input/kul-h02a5a-computervision-groupassignment1/test/img/test_{}.npy'.format(image_id) )
			mask = np.load('../input/kul-h02a5a-computervision-groupassignment1/test/seg/test_{}.npy'.format(image_id))
			label_name='Not labelled'
		image=np.copy(real_image)
		
		image, mask = self.prepare_image(image, mask)
		
	
	
		# print some info 
		print('original image:')
		print('-original_shape:', real_image.shape)
		print('-dtype:', real_image.dtype)
		print('-min value:', np.min(real_image))
		print('-max value:', np.max(real_image))
		
		print('final image:')
		print('-final shape:', image.shape)
		print('-dtype:', image.dtype)
		print('-min value:', np.min(image))
		print('-max value:', np.max(image))
		
		
		# show figure 
		fig, axes = plt.subplots(1,3, figsize=(30,15))
		axes[0].imshow(real_image)
		axes[0].set_title('Original image', fontsize=50)
		
		axes[1].imshow(image) # clip it to [0,1] range
		axes[1].set_title('preprocessed_image', fontsize=50)
		
		axes[2].imshow(mask) # clip it to [0,1] range
		axes[2].set_title('Segmentation mask', fontsize=50)
		
		plt.suptitle('label: {}'.format(label_name), fontsize=50)
		fig.show()
		
	def view_possible_augmentations(self, image_id): 
		fig, axes = plt.subplots(4,4, figsize=(60,30))
		real_image = np.load('../input/kul-h02a5a-computervision-groupassignment1/train/img/train_{}.npy'.format(image_id) )
		mask = np.load('../input/kul-h02a5a-computervision-groupassignment1/train/seg/train_{}.npy'.format(image_id))
			
		plt.suptitle('original image is on the bottom right', fontsize=50)
		for ax in axes.flat: 
			transformed = self.augment(image=real_image, mask=mask)
			new_image=transformed['image']
			new_mask=transformed['mask']
			ax.imshow(new_image, alpha=1)
			ax.imshow(new_mask, alpha=0.2)
			ax.axis('off')
		axes[0,0] = plt.imshow(real_image)
		fig.show()

	def train_generator(self, batch_size): 
		inputs = []
		targets = []
		batchcount = 0
		while True: 
			for image_id in self.train_indices: 
				real_image = np.load('../input/kul-h02a5a-computervision-groupassignment1/train/img/train_{}.npy'.format(image_id))
				mask = np.load('../input/kul-h02a5a-computervision-groupassignment1/train/seg/train_{}.npy'.format(image_id))
				image, segmask = self.prepare_image(real_image, mask)
				inputs.append(image)
				targets.append(segmask)
				batchcount += 1
				if batchcount >= batch_size:
					X = np.array(inputs)
					y = np.array(targets)
					# new_y=list()
					# for yy in y: 
					# 	new_y.append(to_categorical(yy, num_classes=21))
					# y = np.array(new_y)
					yield (X, y)
					inputs = []
					targets = []
					batchcount = 0			
	
	def validation_generator(self,batch_size):
		"""
			generator that will feed validation batches during training 
		"""
		inputs = []
		targets = []
		batchcount = 0
		while True:
			for image_id in self.validation_indices:
				# sample real image
				real_image = np.load('../input/kul-h02a5a-computervision-groupassignment1/train/img/train_{}.npy'.format(image_id))
				mask = np.load('../input/kul-h02a5a-computervision-groupassignment1/train/seg/train_{}.npy'.format(image_id) )
				image, segmask = self.prepare_test_image(real_image, mask)
				inputs.append(image)
				targets.append(segmask)
				batchcount += 1
				if batchcount >= batch_size:
					X = np.array(inputs)
					y = np.array(targets)
					# new_y=list()
					# for yy in y: 
					# 	new_y.append(to_categorical(yy, num_classes=21))
					# y = np.array(new_y)
					yield (X, y)
					inputs = []
					targets = []
					batchcount = 0           


	def test_generator(self,batch_size):
		"""
			generator for feeding test data to model for prediction
		"""
		inputs = []
		img_id = 0
		while img_id < self.nbr_of_test_images:
			raw_image = np.load('../input/kul-h02a5a-computervision-groupassignment1/test/img/test_{}.npy'.format(img_id))
			image,_ = self.prepare_test_image(np.copy(raw_image), np.zeros(raw_image.shape))
			inputs.append(image)
			img_id += 1
			if img_id%batch_size == 0:
				X = np.array(inputs)
				yield X
				inputs = []
		if len(inputs) > 0:
			return np.array(inputs)
					
	def show_class_distribution(self): 
		fig,axes=plt.subplots(figsize=(30,15))
		class_probs=np.mean(self.labels, axis=0)
		axes.bar(self.label_names,  class_probs)
		axes.tick_params(axis='both', which='major', labelsize=30)
		for tick in axes.xaxis.get_major_ticks():
			tick.label.set_rotation('vertical')
		plt.suptitle('Class distribution within the training data.', fontsize=50)
		fig.show()
	
	def get_class_distribution(self):
		return np.mean(self.labels, axis=0)
