import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from keras import backend as K
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os 
import multiprocessing
import wandb
# !pip install wandb -qqq
from wandb.keras import WandbCallback
import kerastuner as kt #!python3.x -m pip install keras-tuner
import cv2
from ipywidgets import fixed, interact 
import ipywidgets
from albumentations import (
	Compose, HorizontalFlip, CLAHE, HueSaturationValue,
	RandomBrightness, RandomContrast, RandomGamma,
	ToFloat, ShiftScaleRotate, RandomBrightnessContrast, RandomCrop)
from data_utils import Dataset_Classification, Dataset_Segmentation
import sys

# sys.path.append('./deeplab')
# from deeplabv3p import Deeplabv3
# from utils import SegModel, get_VOC2012_classes, Jaccard, sparse_accuracy_ignoring_last_label, sparse_crossentropy_ignoring_last_label
import keras

## base classification dataset
class RandomClassificationModel:
	"""
	Random classification model: 
		- generates random labels for the inputs based on the class distribution observed during training
		- assumes an input can have multiple labels
	"""
	def fit(self, X, y):
		"""
		Adjusts the class ratio variable to the one observed in y. 

		Parameters
		----------
		X: list of arrays - n x (height x width x 3)
		y: list of arrays - n x (nb_classes)

		Returns
		-------
		self
		"""
		self.distribution = np.mean(y, axis=0)
		print("Setting class distribution to:\n{}".format("\n".join(f"{label}: {p}" for label, p in zip(labels, self.distribution))))
		return self
		
	def predict(self, X):
		"""
		Predicts for each input a label.
		
		Parameters
		----------
		X: list of arrays - n x (height x width x 3)
			
		Returns
		-------
		y_pred: list of arrays - n x (nb_classes)
		"""
		np.random.seed(0)
		return [np.array([int(np.random.rand() < p) for p in self.distribution]) for _ in X]
	
	def __call__(self, X):
		return self.predict(X)
	
 

## classification dataset
class ClassifactionModel(RandomClassificationModel): 
	"""
		Main class implementing all functions necessary to train and/or use a classification model 
		This class has to be overwritten for each specific model of interest, where the base model should be implemented.
	"""
	def __init__(self, config): 
		self.config = config 
		self.config_head = config['head_model']
		
		# initialize dataset
		self.dataset = Dataset_Classification(config['dataset'])
			  
		# check if some configurations make sense 
		assert len(self.config_head['head_model_units']) == len(self.config_head['add_dropout']), 'head_models_units and add_dropout list should have same size'
	
	
	def set_config(self, config):
		self.config = config 
		self.config_head = config['head_model']
		self.dataset = Dataset_Classification(config['dataset'])
		assert len(self.config_head['head_model_units']) == len(self.config_head['add_dropout']), 'head_models_units and add_dropout list should have same size'
		
	def predict(self, X):
		# 
		
		if len(X.shape) == 1: 
			# X is a batch of images prepare all of them and create batch. 
			batch = np.array([self.dataset.prepare_test_image(im) for im in X])
			y = model.predict(batch)
		else: 
			# X is a single image 
			batch = self.dataset.prepare_test_image(X)
			batch = np.expand_dims(batch, axis=0)
			y = model.predict(batch)
			
		y = np.squeeze(y)

		label_idx = np.where(y==1.)

		return self.dataset.label_names[label_idx]
			
	def build(self): 
		"""
			Builds the model 
		"""
#       self.base_model = resnet50
		
		# define a head model
		head_model=keras.layers.GlobalAveragePooling2D()(self.base_model.output)

		for (nbr_units, dropout) in zip(self.config_head['head_model_units'], self.config_head['add_dropout']): 
			head_model=tf.keras.layers.Dense(nbr_units, activation=self.config_head['activation'])(head_model)
			if dropout:
				head_model=tf.keras.layers.Dropout(0.4)(head_model)
		
		head_model=keras.layers.Dense(20, activation=self.config_head['output_activation'])(head_model)
		#self.config['nbr_classes']
		self.head_model = head_model
				  
		# combine both models 
		self.model = keras.Model(self.base_model.input, head_model)


#         avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
#         output = keras.layers.Dense(20, activation='softmax')(avg)
#         model=keras.Model(inputs=base_model.input, outputs=output)
		   
	def compile_model(self): 
		# optimizer
		if self.config['train_parameters']['optimizer'] == 'SGD':
			optimizer = tf.keras.optimizers.SGD(
					learning_rate=self.config['train_parameters']['learning_rate'], momentum=0.9,
					nesterov=False, name="SGD"
				)
		elif self.config['train_parameters']['optimizer'] == 'ADAM':
			optimizer = tf.keras.optimizers.Adam(lr=self.config['train_parameters']['learning_rate'])

		# metric
		metrics = [tf.keras.metrics.CategoricalAccuracy(),
				  tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top 3 categorical acccuracy'), 
				  tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top 5 categorical acccuracy'),
				  tfa.metrics.FBetaScore(num_classes=20, beta=2., average='weighted')
				  ]
		
		# loss
		if self.config['train_parameters']['loss'] == 'focal':
			loss=tfa.losses.SigmoidFocalCrossEntropy(reduction='auto')
		else:
			loss='categorical_crossentropy'
		self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
		
		
	def train(self, name_run, notes, tags):
		gpus = tf.config.list_physical_devices('GPU')
		if gpus:
			try:
				# Currently, memory growth needs to be the same across GPUs
				for gpu in gpus:
					tf.config.experimental.set_memory_growth(gpu, True)
				logical_gpus = tf.config.experimental.list_logical_devices('GPU')
				print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
			except RuntimeError as e:
				# Memory growth must be set before GPUs have been initialized
				print(e)
			
		# setup logging
		if self.config['logging_wandb']:
			# w&b 
			wandb.init(name=name_run, 
				   project=self.project_name,
				   notes=notes, 
				   tags=tags,
				   entity='cv-task-2')

			# save usefull config to w&b
			wandb.config.learning_rate = self.config['train_parameters']['learning_rate']
			wandb.config.batch_size = self.config['train_parameters']['batch_size']
			wandb.config.epochs = self.config['train_parameters']['epochs']
			wandb.config.steps_per_epoch = self.config['train_parameters']['steps_per_epoch']
			 
		# build model 
		self.build()

		# set model parts trainable or not
		if self.config['train_base_model'] == False: 
			print('freezing base model layers')
			for layer in self.base_model.layers:
				layer.trainable = False
		if self.config['train_head_model'] == False: 
			print('freezing head model layers')
			for layer in self.head_model.layers:
				layer.trainable = False
		
		
		# compile model
		self.compile_model()
		
		if self.config['logging_wandb']:
			# set save_model true if you want wandb to upload weights once run has finished (takes some time)
			clbcks = [WandbCallback(save_model=False)]
		else: 
			clbcks = []

		
		# start training 
		history=self.model.fit(
					x = self.dataset.train_generator(batch_size=self.config['train_parameters']['batch_size']),
					steps_per_epoch = self.config['train_parameters']['steps_per_epoch'],
					epochs=self.config['train_parameters']['epochs'], 
					validation_data=self.dataset.validation_generator(batch_size=self.config['train_parameters']['batch_size']),
					validation_steps=20, 
					callbacks=clbcks
		)
		
		#workers=multiprocessing.cpu_count(),
		#use_multiprocessing=True,
	
	def prepare_for_inference(self, model_weights_path, force_cpu=False): 
		# if not force_cpu:
		# 	gpus = tf.config.list_physical_devices('GPU')
		# 	if gpus:
		# 		try:
		# 			# Currently, memory growth needs to be the same across GPUs
		# 			for gpu in gpus:
		# 				tf.config.experimental.set_memory_growth(gpu, True)
		# 			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		# 			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		# 		except RuntimeError as e:
		# 			# Memory growth must be set before GPUs have been initialized
		# 			print(e)
		# else:
		# 	my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
		# 	tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
		self.build()
		self.model.load_weights(model_weights_path)
	
	def show_heatmap_prediction(self, image_id):
		LAYER_NAME=self.heatmap_layer_name
		im = np.load('../input/kul-h02a5a-computervision-groupassignment1/test/img/test_{}.npy'.format(image_id))
		pre_im = self.dataset.prepare_test_image(im)
		batch = np.expand_dims(pre_im, axis=0)
	
		pred = self.model.predict(batch)
		idx=np.argmax(pred)
		score = np.round(pred[0][idx]/np.sum(pred),4)
		label=self.dataset.label_names[idx]

		grad_model = tf.keras.models.Model([self.model.inputs], [self.model.get_layer(LAYER_NAME).output, self.model.output])

		with tf.GradientTape() as tape:
			conv_outputs, predictions = grad_model(batch)
			loss = predictions[:, idx]

		output = conv_outputs[0]
		grads = tape.gradient(loss, conv_outputs)[0]

		gate_f = tf.cast(output > 0, 'float32')
		gate_r = tf.cast(grads > 0, 'float32')
		guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

		weights = tf.reduce_mean(guided_grads, axis=(0, 1))

		cam = np.ones(output.shape[0: 2], dtype = np.float32)

		for i, w in enumerate(weights):
			cam += w * output[:, :, i]

		cam = cv2.resize(cam.numpy(), (224, 224))
		cam = np.maximum(cam, 0)
		heatmap = (cam - cam.min()) / (cam.max() - cam.min())

		cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
		og_im = cv2.cvtColor(im.astype('uint8'), cv2.COLOR_RGB2BGR)

		og_im = cv2.resize(og_im, (224, 224))


		output_image = cv2.addWeighted(og_im, 0.7, cam, 1, 0)


		fig, axes = plt.subplots(1,2, figsize=(30,15))
		axes[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
		axes[0].imshow(im)
		axes[0].set_title('prediction: {}, score: {}'.format(label, np.round(100*score,2)), fontsize=25)
		plt.show()

class ResNetClassifactionModel(RandomClassificationModel): 
    def __init__(self, config): 
        self.config = config 
        self.config_head = config['head_model']
        self.project_name = 'resnet_50'
        
        # initialize dataset
        self.dataset = Dataset_Classification(config['dataset'])
        
        # feed preprocessor to dataset
        print('Feeding resnet50 preprocess function to dataset class')
        self.dataset.feed_preprocess_function(tf.keras.applications.resnet_v2.preprocess_input)
        
        # check if some configurations make sense 
        assert len(self.config_head['head_model_units']) == len(self.config_head['add_dropout']), 'head_models_units and add_dropout list should have same size'
    
    
    def set_config(self, config):
        self.config = config 
        self.config_head = config['head_model']
        self.dataset = Dataset_Classification(config['dataset'])
        assert len(self.config_head['head_model_units']) == len(self.config_head['add_dropout']), 'head_models_units and add_dropout list should have same size'
        
    def predict(self, X):
        # 
        
        if len(X.shape) == 1: 
            # X is a batch of images prepare all of them and create batch. 
            batch = np.array([self.dataset.prepare_test_image(im) for im in X])
            print(batch.shape)
            
            y = model.predict(batch)
        else: 
            # X is a single image 
            batch = self.dataset.prepare_test_image(X)
            batch = np.expand_dims(batch, axis=0)
            print(batch.shape)
            y = model.predict(batch)
            
        y = np.squeeze(y)
        print(y)
        print(self.dataset.label_names)
        label_idx = np.where(y==1.)
        print(label_idx)
        return self.dataset.label_names[label_idx]
            
            
        
        
        
        
    def build(self): 
        """
            Builds the model 
        """
        # define a resnet50 base model
        resnet50 = tf.keras.applications.ResNet50V2(
                    include_top=False,
                    weights=self.config['weights'],
                    input_shape=self.config['input_shape'],
                        )
        resnet50.trainable = False
        self.base_model = resnet50
        
        # define a head model
        head_model=resnet50.output
        head_model=tf.keras.layers.AveragePooling2D(pool_size=(5,5))(head_model)
        head_model=tf.keras.layers.Flatten()(head_model)
        
        for (nbr_units, dropout) in zip(self.config_head['head_model_units'], self.config_head['add_dropout']): 
            head_model=tf.keras.layers.Dense(nbr_units, activation=self.config_head['activation'])(head_model)
            if dropout:
                head_model=tf.keras.layers.Dropout(0.4)(head_model)
        
        head_model=tf.keras.layers.Dense(self.config['nbr_classes'], activation='sigmoid')(head_model)
        
        self.head_model = head_model
                
            
        # combine both models 
        self.model = tf.keras.Model(self.base_model.input, self.head_model)
        
    def weighted_categorical_crossentropy(self,weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy

        Variables:
            weights: numpy array of shape (C,) where C is the number of classes

        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """

        weights = K.variable(weights)

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss

        return loss
        
    def compile_model(self): 
        # optimizer
#         optimizer = tf.keras.optimizers.Adam(
#                                 learning_rate=self.config['train_parameters']['learning_rate'],
#                                 beta_1=0.9,
#                                 beta_2=0.999,
#                                 epsilon=1e-07,
#                                 amsgrad=False,
#                                 name="Adam",
#                             )
        optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.config['train_parameters']['learning_rate'], momentum=0.9,
                nesterov=False, name="SGD"
            )

        # metric
        metrics = [tf.keras.metrics.CategoricalAccuracy(),
                  tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top 3 categorical acccuracy'), 
                  tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top 5 categorical acccuracy')
                  ]
    
    

#         loss = tf.keras.losses.CategoricalCrossentropy(
#                 from_logits=True,
#                 label_smoothing=0,
#                 reduction="auto",
#                 name="categorical_crossentropy_loss",
#              )
        #loss='sparse_categorical_crossentropy'
        #oss = 'categorical_crossentropy'
        
        #loss = self.weighted_categorical_crossentropy(np.ones(20)/20)
        
        #loss=tf.keras.losses.KLDivergence(reduction="auto", name="kl_divergence")
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0,name='binary_crossentropy')


        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        
    def train(self, name_run, notes, tags):
        # build generator and discriminator models 
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
            
        # setup logging
        if self.config['logging_wandb']:
            # w&b 
            wandb.init(name=name_run, 
                   project=self.project_name,
                   notes=notes, 
                   tags=tags,
                   entity='cv-task-2')

            # save usefull config to w&b
            wandb.config.learning_rate = self.config['train_parameters']['learning_rate']
            wandb.config.batch_size = self.config['train_parameters']['batch_size']
            wandb.config.epochs = self.config['train_parameters']['epochs']
            wandb.config.steps_per_epoch = self.config['train_parameters']['steps_per_epoch']
            
            
        # build model 
        self.build()
        #self.model.summary()
        
        # set model parts trainable or not
        if self.config['train_base_model'] == False: 
            print('freezing base model layers')
            for layer in self.base_model.layers:
                layer.trainable = False
        if self.config['train_head_model'] == False: 
            print('freezing head model layers')
            for layer in self.head_model.layers:
                layer.trainable = False
        
        
        # compile model
        self.compile_model()
        
        if self.config['logging_wandb']:
            # set save_model true if you want wandb to upload weights once run has finished (takes some time)
            clbcks = [WandbCallback(save_model=False)]
        else: 
            clbcks = []

        
        # start training 
        history=self.model.fit(
                    x = self.dataset.train_generator(batch_size=self.config['train_parameters']['batch_size']),
                    steps_per_epoch = self.config['train_parameters']['steps_per_epoch'],
                    epochs=self.config['train_parameters']['epochs'], 
                    validation_data=self.dataset.validation_generator(batch_size=self.config['train_parameters']['batch_size']),
                    validation_steps=20, 
                    callbacks=clbcks
        )
        
        #workers=multiprocessing.cpu_count(),
        #use_multiprocessing=True,
