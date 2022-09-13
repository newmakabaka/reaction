import tensorflow as tf
from mol_graph import max_nb
from nn import *

tf.compat.v1.disable_v2_behavior()

def rcnn_wl_last(graph_inputs, batch_size, hidden_size, depth, training=True):
    '''This function performs the WLN embedding (local, no attention mechanism)'''
    input_atom, input_bond, atom_graph, bond_graph, num_nbs, atom_degree, node_mask, binary = graph_inputs
    # 原子特征    键特征       原子图2层    键图2层    每个原子连了多少个键   掩码：有多少个原子就有多少个1
    atom_features = tf.compat.v1.nn.relu(linearND(input_atom, hidden_size, "atom_embedding", init_bias=None))
    layers = []
    for i in range(depth):
        with tf.compat.v1.variable_scope("WL", reuse=(i>0)) as scope:
            fatom_nei = tf.compat.v1.gather_nd(atom_features, atom_graph)    # 给节点上标签
            fbond_nei = tf.compat.v1.gather_nd(input_bond, bond_graph)      # 给边上标签
            h_nei_atom = linearND(fatom_nei, hidden_size, "nei_atom", init_bias=None)  # 邻居节点
            h_nei_bond = linearND(fbond_nei, hidden_size, "nei_bond", init_bias=None)  # 边
            h_nei = h_nei_atom * h_nei_bond
            mask_nei = tf.compat.v1.reshape(tf.compat.v1.sequence_mask(tf.compat.v1.reshape(num_nbs, [-1]), max_nb, dtype=tf.compat.v1.float32), [batch_size,-1,max_nb,1])
            f_nei = tf.compat.v1.reduce_sum(h_nei * mask_nei, -2)
            f_self = linearND(atom_features, hidden_size, "self_atom", init_bias=None)
            layers.append(f_nei * f_self * node_mask)  # output
            l_nei = tf.compat.v1.concat([fatom_nei, fbond_nei], 3)    # [hu(l-1),fuv]
            nei_label = tf.compat.v1.nn.relu(linearND(l_nei, hidden_size, "label_U2"))    # V[hu(l-1),fuv]
            nei_label = tf.compat.v1.reduce_sum(nei_label * mask_nei, -2)
            # new_label = f_self + nei_label     # 改动的
            new_label = tf.compat.v1.concat([atom_features, nei_label], 2)
            new_label = linearND(new_label, hidden_size, "label_U1")
            # new_label = tf.compat.v1.matmul(atom_degree, new_label)      # 新增度矩阵
            atom_features = tf.compat.v1.nn.relu(new_label) # updated atom features    hv(l)
    #kernels = tf.compat.v1.concat(1, layers)
    atom_hiddens = layers[-1] # atom FPs are the final output after "depth" convolutions
    fp = tf.compat.v1.reduce_sum(atom_hiddens, 1) # molecular FP is sum over atom FPs

    # Calculate local atom pair features as sum of local atom features
    atom_hiddens1 = tf.compat.v1.reshape(atom_hiddens, [batch_size, 1, -1, hidden_size])
    atom_hiddens2 = tf.compat.v1.reshape(atom_hiddens, [batch_size, -1, 1, hidden_size])
    atom_pair = atom_hiddens1 + atom_hiddens2

    # Calculate attention scores for each pair o atoms
    att_hidden = tf.compat.v1.nn.relu(
        linearND(atom_pair, hidden_size, scope="att_atom_feature", init_bias=None) + linearND(binary, hidden_size,
                                                                                              scope="att_bin_feature"))
    att_score = linearND(att_hidden, 1, scope="att_scores")
    att_score = tf.compat.v1.nn.sigmoid(att_score)

    # Calculate context features using those attention scores
    att_context = att_score * atom_hiddens1
    att_context = tf.compat.v1.reduce_sum(att_context, 2)

    # Calculate global atom pair features as sum of atom context features
    att_context1 = tf.compat.v1.reshape(att_context, [batch_size, 1, -1, hidden_size])
    att_context2 = tf.compat.v1.reshape(att_context, [batch_size, -1, 1, hidden_size])
    att_pair = att_context1 + att_context2

    # Calculate likelihood of each pair of atoms to form a particular bond order
    pair_hidden = linearND(atom_pair, hidden_size, scope="atom_feature", init_bias=None) + linearND(binary, hidden_size,
                                                                                                    scope="bin_feature",
                                                                                                    init_bias=None) + linearND(att_pair, hidden_size, scope="ctx_feature")
    pair_hidden = tf.compat.v1.nn.relu(pair_hidden)
    pair_hidden = tf.compat.v1.reshape(pair_hidden, [batch_size, -1, hidden_size])

    return pair_hidden, fp

