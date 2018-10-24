#coding:utf8
'''
Created on 2018年3月30日

@author: 20143
'''
import tensorflow as tf

if __name__ == '__main__':
    merged_summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)
    total_step = 0
    while training:
      total_step += 1
      session.run(training_op)
      if total_step % 100 == 0:
        summary_str = session.run(merged_summary_op)
        summary_writer.add_summary(summary_str, total_step)