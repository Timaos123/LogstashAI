import os
import sys
import json
import time
import shutil
import pickle
import logging
import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN
from sklearn.model_selection import train_test_split
import csv
import tqdm

logging.getLogger().setLevel(logging.INFO)

def getRP(preY,testY):
	preY=np.array(preY)
	testY=np.array(testY)
	yP=list(zip(preY,testY))
	tp=0
	fp=0
	tn=0
	fn=0
	for yPItem in yP:
		if yPItem[0]==yPItem[1] and yPItem[1]==1:
			tp=tp+1
		if yPItem[0]!=yPItem[1] and yPItem[1]==1:
			fp=fp+1
		if yPItem[0]==yPItem[1] and yPItem[1]==0:
			tn=tn+1
		if yPItem[0]!=yPItem[1] and yPItem[1]==0:
			fn=fn+1
	recall=tp/(tp+fp+1)
	precision=tp/(tp+fn+1)
	return recall,precision

def train_cnn_rnn():
	input_file = "logstashTemp.dat"		
	output_file="wcData85_1.csv"
	
# 	with open(input_file,"r",encoding="utf8") as datFile:
# 		jsonDict=json.loads(datFile.readline())
# 	with open(input_file,"r",encoding="utf8") as datFile:
# 		jsonDf=pd.DataFrame([],columns=list(jsonDict.keys()))
# 		rowNO=0
# 		for row in datFile.readlines():
# 			try:
# 				jsonDf.loc[rowNO]=list(json.loads(row).values())
# 			except json.decoder.JSONDecodeError as ex:
# 				print(ex.tostring)
# 			rowNO+=1
# 		jsonDf.to_csv(output_file)
	
	print("loading data...")
	x_, y_, vocabulary, vocabulary_inv, df, labels = data_helper.load_data3(output_file,["crit","err"],10000)
# 	print("y_:",y_)
	training_config = "training_config.json"
	params = json.loads(open(training_config).read())
 
	# Assign a 300 dimension vector to each word
	word_embeddings = data_helper.load_embeddings(vocabulary)
	embedding_mat = [word_embeddings[word] for index, word in enumerate(vocabulary_inv)]
	embedding_mat = np.array(embedding_mat, dtype = np.float32)
 
	# Split the original dataset into train set and test set
	x, x_test, y, y_test = train_test_split(x_, y_, test_size=0.1)
 
	# Split the train set into train set and dev set
	x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1)
 
	logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
	logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))
 
	# Create a directory, everything related to the training will be saved in this directory
	timestamp = str(int(time.time()))
	trained_dir = './trained_results_' + timestamp + '/'
	print(trained_dir)
	if os.path.exists(trained_dir):
		shutil.rmtree(trained_dir)
	os.makedirs(trained_dir)
 
	graph = tf.Graph()
	with graph.as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn_rnn = TextCNNRNN(
				embedding_mat=embedding_mat,
				sequence_length=x_train.shape[1],
				num_classes = y_train.shape[1],
				non_static=params['non_static'],
				hidden_unit=params['hidden_unit'],
				max_pool_size=params['max_pool_size'],
				filter_sizes=map(int, params['filter_sizes'].split(",")),
				num_filters = params['num_filters'],
				embedding_size = params['embedding_dim'],
				l2_reg_lambda = params['l2_reg_lambda'])
 
			global_step = tf.Variable(0, name='global_step', trainable=False)
			#global_step will control the changes of grads_and_vars with 
			#	the change of itself which caused by optimizer.apply_gradients()
			optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3, decay=0.9)
			#initiate the optimizer whose learning_rate is firstly 1e-3
			# but it will be decreased along with the change of decay in the folume below:
			# decayed_learning_rate = learning_rate*decay_rate^(global_step/decay_steps)
			grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
			#compute gradients of loss
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			#apply the gradients to variables and change them
 
			# Checkpoint files will be saved in this directory during training
			checkpoint_dir = './checkpoints_' + timestamp + '/'
			if os.path.exists(checkpoint_dir):
				shutil.rmtree(checkpoint_dir)
			os.makedirs(checkpoint_dir)
			checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
 
			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]
 
			def train_step(x_batch, y_batch):
				feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.input_y: y_batch,
					cnn_rnn.dropout_keep_prob: params['dropout_keep_prob'],
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
				_, step, loss, accuracy = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy], feed_dict)
				print(step,"trainAccuracy",accuracy)
				with open("trainLogCsv.txt","a+",encoding="utf8") as trainLogFile:
					trainLogFile.write("========="+str(step)+"=========\n")
					trainLogFile.write("acc:"+str(accuracy)+"\n")
					trainLogFile.write("loss:"+str(loss)+"\n")
			def dev_step(x_batch, y_batch):
				feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.input_y: y_batch,
					cnn_rnn.dropout_keep_prob: 1.0,
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
				step, loss, accuracy, num_correct, predictions = sess.run(
					[global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions], feed_dict)
				return accuracy, loss, num_correct, predictions
 
			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())
			filter_writer=tf.summary.FileWriter('/path/to/logs', sess.graph)
			# Training starts here
			train_batches = data_helper.batch_iter(list(zip(x_train, y_train)), \
												params['batch_size'], \
												params['num_epochs'])
			best_accuracy, best_at_step = 0, 0
 
			# Train the model with x_train and y_train
			for train_batch in train_batches:
				if len(train_batch)>0:
					x_train_batch, y_train_batch = zip(*train_batch)
					train_step(x_train_batch, y_train_batch)
					current_step = tf.train.global_step(sess, global_step)
	 
					# Evaluate the model with x_dev and y_dev
					if current_step % params['evaluate_every'] == 0:
						dev_batches = data_helper.batch_iter(list(zip(x_dev, y_dev)), params['batch_size'], 1)
	 
						total_dev_correct = 0
						y_dev=[]
						y_pre=[]
						for dev_batch in dev_batches:
							if len(dev_batch)>0:
								x_dev_batch, y_dev_batch = zip(*dev_batch)
								acc, loss, num_dev_correct, predictions = dev_step(x_dev_batch, y_dev_batch)
								y_pre+=predictions.tolist()
								y_dev+=list(y_dev_batch)
								total_dev_correct += num_dev_correct
						y_devs=[y_devItem.tolist().index(max(y_devItem.tolist())) for y_devItem in y_dev]
# 						print("y_pre:",y_pre)
# 						print("y_devs:",y_devs)
						devRecall,devPrecision=getRP(y_pre, y_devs)
						logging.info('Recall and precision of dev set: {},{}'.format(devRecall,devPrecision))
						accuracy = float(total_dev_correct) / len(y_dev)
						logging.info('Accuracy on dev set: {}'.format(accuracy))
	 
						lossItem=loss
						accuracyItem=accuracy
	                    	
						with open("devCsv.csv","a+",encoding="utf8") as csvFile:
							myWriter=csv.writer(csvFile)
							myWriter.writerow([lossItem,accuracyItem,devRecall,devPrecision])
						
						if accuracy >= best_accuracy:
							best_accuracy, best_at_step = accuracy, current_step
							path = saver.save(sess, checkpoint_prefix, global_step=current_step)
							logging.critical('Saved model {} at step {}'.format(path, best_at_step))
							logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
			logging.critical('Training is complete, testing the best model on x_test and y_test')
 
			# Save the model files to trained_dir. predict.py needs trained model files. 
			saver.save(sess, trained_dir + "best_model.ckpt")
 
			# Evaluate x_test and y_test
			saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
			test_batches = data_helper.batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1, shuffle=False)
			total_test_correct = 0
			for test_batch in test_batches:
				if len(test_batch)>0:
					x_test_batch, y_test_batch = zip(*test_batch)
					acc, loss, num_test_correct, predictions = dev_step(x_test_batch, y_test_batch)
					total_test_correct += int(num_test_correct)
			logging.critical('Accuracy on test set: {}'.format(float(total_test_correct) / len(y_test)))
 
	# Save trained parameters and files since predict.py needs them
	with open(trained_dir + 'words_index.json', 'w') as outfile:
		json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
	with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
		pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
	with open(trained_dir + 'labels.json', 'w') as outfile:
		json.dump(labels, outfile, indent=4, ensure_ascii=False)
 
	params['sequence_length'] = x_train.shape[1]
	with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
		json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)

if __name__ == '__main__':
	train_cnn_rnn()
	# python3 train.py ./data/train.csv.zip ./training_config.json