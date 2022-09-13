import tensorflow as tf


sess=tf.Session()

data1=tf.placeholder(tf.float32,[1,2])
q=tf.FIFOQueue(100,[tf.float32])
enqueue=q.enqueue(data1)

data=q.dequeue()
data.set_shape([1,2])

feed_map={data1:[[3,4]]}
sess.run(enqueue,feed_dict=feed_map)
print(enqueue)

feed_map={data1:[[5,6]]}
sess.run(enqueue,feed_dict=feed_map)


