from __future__ import print_function
import tensorflow as tf
from rexgen_direct.core_wln_global.nn import linearND, linear
# from rexgen_direct.core_wln_global.mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
from rexgen_direct.core_wln_global.models import *
from rexgen_direct.core_wln_global.ioutils_direct import *
import math, sys, random
from collections import Counter
from optparse import OptionParser
from functools import partial
import threading
from multiprocessing import Queue
from functools import reduce

tf.compat.v1.disable_v2_behavior()

'''
Script for training the core finder model

Key changes from NIPS paper version:
- Addition of "rich" options for atom featurization with more informative descriptors
- Predicted reactivities are not 1D, but 5D and explicitly identify what the bond order of the product should be
'''

NK = 10
NK0 = 5

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path", default="../data/train.txt.proc")
parser.add_option("-u", "--train_ul", dest="train_ul_path", default="../data/precursors-train.txt20.proc")
parser.add_option("-m", "--save_dir", dest="save_path", default="./model")
parser.add_option("-b", "--batch", dest="batch_size", default=20)
parser.add_option("-w", "--hidden", dest="hidden_size", default=300)
parser.add_option("-d", "--depth", dest="depth", default=3)
parser.add_option("-l", "--max_norm", dest="max_norm", default=5.0)
parser.add_option("-r", "--rich", dest="rich_feat", default=False)
opts,args = parser.parse_args()


batch_size = int(opts.batch_size)
half_batch_size = int(batch_size/2)
hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
max_norm = float(opts.max_norm)
if opts.rich_feat:
    from rexgen_direct.core_wln_global.mol_graph_rich import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
else:
    from rexgen_direct.core_wln_global.mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g

smiles2graph_batch = partial(_s2g, idxfunc=lambda x:x.GetIntProp('molAtomMapNumber') - 1)

gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
_input_atom = tf.compat.v1.placeholder(tf.compat.v1.float32, [batch_size, None, adim])
_input_atom_p = tf.compat.v1.placeholder(tf.compat.v1.float32, [batch_size, None, adim]) 
_input_bond = tf.compat.v1.placeholder(tf.compat.v1.float32, [batch_size, None, bdim])
_atom_graph = tf.compat.v1.placeholder(tf.compat.v1.int32, [batch_size, None, max_nb, 2])
_bond_graph = tf.compat.v1.placeholder(tf.compat.v1.int32, [batch_size, None, max_nb, 2])
_num_nbs = tf.compat.v1.placeholder(tf.compat.v1.int32, [batch_size, None])
_atom_degree=tf.compat.v1.placeholder(tf.compat.v1.float32, [batch_size, None, None]) 
_node_mask = tf.compat.v1.placeholder(tf.compat.v1.float32, [batch_size, None])
_src_holder = [_input_atom, _input_atom_p, _input_bond, _atom_graph, _bond_graph, _num_nbs, _atom_degree, _node_mask]   # 新增degree
_label = tf.compat.v1.placeholder(tf.compat.v1.int32, [batch_size, None])
_binary = tf.compat.v1.placeholder(tf.compat.v1.float32, [batch_size, None, None, binary_fdim])

# Queueing system allows CPU to prepare a buffer of <100 batches
q = tf.compat.v1.FIFOQueue(100, [tf.compat.v1.float32, tf.compat.v1.float32, tf.compat.v1.float32, tf.compat.v1.int32, tf.compat.v1.int32, tf.compat.v1.int32, tf.compat.v1.float32, tf.compat.v1.float32, tf.compat.v1.int32, tf.compat.v1.float32])
enqueue = q.enqueue(_src_holder + [_label, _binary])
input_atom, input_atom_p,input_bond, atom_graph, bond_graph, num_nbs, atom_degree, node_mask, label, binary = q.dequeue()

input_atom.set_shape([batch_size, None, adim])
input_atom_p.set_shape([batch_size, None, adim])
input_bond.set_shape([batch_size, None, bdim])
atom_graph.set_shape([batch_size, None, max_nb, 2])
bond_graph.set_shape([batch_size, None, max_nb, 2])
num_nbs.set_shape([batch_size, None])
atom_degree.set_shape([batch_size, None, None]) 
node_mask.set_shape([batch_size, None])
label.set_shape([batch_size, None])
binary.set_shape([batch_size, None, None, binary_fdim])

tea_label = label[:]    #
tea_flat_label = tf.compat.v1.reshape(tea_label, [-1])     #
tea_bond_mask = tf.compat.v1.to_float(tf.compat.v1.not_equal(tea_flat_label,INVALID_BOND))   #

label = label[:half_batch_size]          #
node_mask = tf.compat.v1.expand_dims(node_mask, -1)
flat_label = tf.compat.v1.reshape(label, [-1])
bond_mask = tf.compat.v1.to_float(tf.compat.v1.not_equal(flat_label, INVALID_BOND))
flat_label = tf.compat.v1.maximum(0, flat_label)

# Perform the WLN embedding 
graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs, atom_degree, node_mask, binary)
graph_inputs_p = (input_atom_p, input_bond, atom_graph, bond_graph, num_nbs, atom_degree, node_mask, binary)
with tf.compat.v1.variable_scope("stu_encoder"):
    stu_pair_hidden, _ = rcnn_wl_last(graph_inputs_p, batch_size=batch_size, hidden_size=hidden_size, depth=depth)

with tf.compat.v1.variable_scope("tea_encoder"):
    tea_pair_hidden, _ = rcnn_wl_last(graph_inputs, batch_size=batch_size, hidden_size=hidden_size, depth=depth)

# tea_pair_hidden = tea_pair_hidden[half_batch_size:]
tea_score = linearND(tea_pair_hidden, 5, scope="tea_scores")
tea_score = tf.compat.v1.reshape(tea_score, [batch_size, -1])
tea_flat_score = tf.compat.v1.reshape(tea_score, [-1])
tea_score = tea_score[:half_batch_size]

score = linearND(stu_pair_hidden, 5, scope="scores")
score = tf.compat.v1.reshape(score, [batch_size, -1])
stu_score = score[:]
score = score[:half_batch_size]


# Mask existing/invalid bonds before taking topk predictions
bmask = tf.compat.v1.to_float(tf.compat.v1.equal(label, INVALID_BOND)) * 10000
_, topk_tea = tf.compat.v1.nn.top_k(tea_score - bmask, k=NK)
_, topk = tf.compat.v1.nn.top_k(score - bmask, k=NK)
flat_score = tf.compat.v1.reshape(score, [-1])
stu_flat_score = tf.compat.v1.reshape(stu_score, [-1])

# Train with categorical crossentropy
tea_loss = tf.compat.v1.losses.mean_squared_error(stu_flat_score,tea_flat_score)
tea_loss = tf.compat.v1.reduce_sum(tea_loss * tea_bond_mask)
loss = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=flat_score, labels=tf.compat.v1.to_float(flat_label))
loss = tf.compat.v1.reduce_sum(loss * bond_mask)
loss = loss+0.001*tea_loss

stu_var = [tensor for tensor in tf.compat.v1.trainable_variables() if "stu_encoder" in tensor.name]
tea_var = [tensor for tensor in tf.compat.v1.trainable_variables() if "tea_encoder" in tensor.name]


# Use Adam with clipped gradients
_lr = tf.compat.v1.placeholder(tf.compat.v1.float32, [])
_delay = tf.compat.v1.placeholder(tf.compat.v1.float32, [])
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=_lr)
param_norm = tf.compat.v1.global_norm(stu_var)
grads_and_vars = optimizer.compute_gradients(loss / half_batch_size) #+ beta * param_norm)
grads, var = zip(*grads_and_vars)
grad_norm = tf.compat.v1.global_norm(grads)
new_grads, _ = tf.compat.v1.clip_by_global_norm(grads, max_norm)
grads_and_vars = zip(new_grads, var)
backprop = optimizer.apply_gradients(grads_and_vars)


tf.compat.v1.global_variables_initializer().run(session=session)
size_func = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
n = sum(size_func(v) for v in tf.compat.v1.trainable_variables())
print("Model size: %dK" % (n/1000,))

# Multiprocessing queue to run in parallel to Tensorflow queue, contains aux. information
queue = Queue()

def count(s):
    c = 0
    for i in range(len(s)):
        if s[i] == ':':
            c += 1
    return c



def read_data(path, path1, coord):
    '''Process data from a text file; bin by number of heavy atoms
    since that will determine the input sizes in each batch'''
    bucket_size = [10,20,30,40,50,60,80,100,120,150]
    buckets = [[] for i in range(len(bucket_size))]
    buckets_ul = [[] for i in range(len(bucket_size))]
    with open(path, 'r') as f, open(path1,'r') as f1:
        for line in f:
            r,e = line.strip("\r\n ").split()
            c = count(r)
            for i in range(len(bucket_size)):
                if c <= bucket_size[i]:
                    buckets[i].append((r,e))
                    break

        for line in f1:
            r = line.strip("\r\n ").split()[0]
            c = count(r)
            for i in range(len(bucket_size)):
                if c <= bucket_size[i]:
                    buckets_ul[i].append((r,'-1'))
                    break

    for i in range(len(buckets)):
        random.shuffle(buckets[i])
        random.shuffle(buckets_ul[i])   #
    
    head = [0] * len(buckets)
    head_ul = [0] * len(buckets)   #
    avil_buckets = [i for i in range(len(buckets)) if len(buckets[i]) > 0]
    while True:
        src_batch, edit_batch = [], []
        src_batch_ul = []    #
        bid = random.choice(avil_buckets)
        bucket = buckets[bid]
        bucket_ul = buckets_ul[bid]    #
        it_ul = head_ul[bid]   #
        data_len_ul = len(bucket_ul)   #
        it = head[bid]
        data_len = len(bucket)

        for i in range(half_batch_size):
            react = bucket[it][0].split('>')[0]
            src_batch.append(react)
            edits = bucket[it][1]
            edit_batch.append(edits)
            it = (it + 1) % data_len
        head[bid] = it

        for i in range(half_batch_size):
            react_ul = bucket_ul[it_ul][0]  #
            src_batch_ul.append(react_ul)  #
            edits = bucket_ul[it_ul][1]
            edit_batch.append(edits)
            it_ul = (it_ul + 1) % data_len_ul  #
        head_ul[bid] = it_ul   #

        # Prepare batch for tf.compat.v1
        src_tuple = smiles2graph_batch(src_batch+src_batch_ul)     # 传进来数据
        # src_tuple_ul = smiles2graph_batch(src_batch_ul)  #
        cur_bin, cur_label, sp_label = get_all_batch(zip(src_batch+src_batch_ul, edit_batch))
        feed_map = {x:y for x,y in zip(_src_holder, src_tuple)}
        feed_map.update({_label:cur_label, _binary:cur_bin})
        session.run(enqueue, feed_dict=feed_map)
        # print(feed_map)
        queue.put(sp_label)

    coord.request_stop()

coord = tf.compat.v1.train.Coordinator()
t = threading.Thread(target=read_data, args=(opts.train_path, opts.train_ul_path, coord))
t.start()

saver = tf.compat.v1.train.Saver(max_to_keep=None)
it, sum_acc, sum_err, sum_gnorm = 0, 0.0, 0.0, 0.0
sum_acc_tea, sum_err_tea=0.0, 0.0
lr = 0.001
delay = 0.1
ema=tf.compat.v1.train.ExponentialMovingAverage(_delay)
ema_op=ema.apply(stu_var)

def ema_update(stu,tea):
    for i,j in zip(stu,tea):
        tf.compat.v1.assign(j,ema.average(i))
    return tea

update=ema_update(stu_var,tea_var)

try:
    while not coord.should_stop():
        it += 1
        # print(it)
        # Run one minibatch
        _, cur_topk, cur_topk_tea, pnorm, gnorm = session.run([backprop, topk, topk_tea, param_norm, grad_norm], feed_dict={_lr:lr})
        session.run(update,feed_dict={_delay:delay})
        # print('-------------------------------------------------------')
        # print(session.run(tea_var))
        sp_label = queue.get()
        # Get performance
        for i in range(half_batch_size):
            pre_tea=0
            pre = 0
            for j in range(NK):
                if cur_topk[i,j] in sp_label[i]:
                    pre += 1
                if cur_topk_tea[i,j] in sp_label[i]:
                    pre_tea += 1
            if len(sp_label[i]) == pre: sum_err += 1
            if len(sp_label[i]) == pre_tea: sum_err_tea += 1

            pre = 0
            pre_tea = 0
            for j in range(NK0):
                if cur_topk[i,j] in sp_label[i]:
                    pre += 1
                if cur_topk_tea[i,j] in sp_label[i]:
                    pre_tea += 1
            if len(sp_label[i]) == pre: sum_acc += 1
            if len(sp_label[i]) == pre_tea: sum_acc_tea += 1

        sum_gnorm += gnorm

        if it % 50 == 0:
            print("Acc@5: %.4f, Acc@10: %.4f, tea:Acc@5: %.4f, tea:Acc@10: %.4f, Param Norm: %.2f, Grad Norm: %.2f ,it term: %d" % (sum_acc / (50 * half_batch_size), sum_err / (50 * half_batch_size), sum_acc_tea / (50 * half_batch_size), sum_err_tea / (50 * half_batch_size), pnorm, sum_gnorm / 50, it) )
            sys.stdout.flush()
            sum_acc, sum_err, sum_gnorm ,sum_acc_tea , sum_err_tea= 0.0, 0.0, 0.0 ,0.0 ,0.0
        if it % 10000 == 0:
            lr *= 0.95
            delay = min(delay*1.1,0.999)
            saver.save(session, opts.save_path + "/model.ckpt", global_step=it)
            print("Model Saved!")
except Exception as e:
    print(e)
    coord.request_stop(e)
finally:
    saver.save(session, opts.save_path + "/model.final")
    coord.request_stop()
    coord.join([t])
