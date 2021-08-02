import os
import numpy as np
from collections import Counter
import scipy.sparse as sp
# import networkx as nx

adj_entity_gb_p = 'adj_entity_gb'


def load_data(args):
    train_data, eval_data, test_data, user_history_dict, n_user, n_item = load_rating(
        args)
    n_entity, n_relation, kg, adj_enlc, adj_relc, adj_engb = load_kg(
        args, n_item)
    user_history_dict = fix_userhist(args, user_history_dict)
    print('Loaded data')
    return train_data, eval_data, test_data, n_entity, n_relation, n_user, n_item, kg, adj_enlc, adj_relc, user_history_dict, adj_engb


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    train_file = '../data/' + args.dataset + '/train_data'
    eval_file = '../data/' + args.dataset + '/eval_data'
    test_file = '../data/' + args.dataset + '/test_data'
    if os.path.exists(train_file + '.npy') and os.path.exists(test_file + '.npy') and os.path.exists(eval_file + '.npy'):
        train_np = np.load(train_file + '.npy')
        test_np = np.load(test_file + '.npy')
        eval_np = np.load(eval_file + '.npy')
    else:
        train_np = np.loadtxt(train_file + '.txt', dtype=np.int32)
        test_np = np.loadtxt(test_file + '.txt', dtype=np.int32)
        eval_np = np.loadtxt(eval_file + '.txt', dtype=np.int32)

    user_history_dict = dict()
    for i in train_np:
        user = i[0]
        item = i[1]
        if user not in user_history_dict:
            user_history_dict[user] = set()
        user_history_dict[user].add(item)
    for user, history_item in user_history_dict.items():
        user_history_dict[user] = list(history_item)

    n_user = max(set(train_np[:, 0]) | set(
        test_np[:, 0]) | set(eval_np[:, 0])) + 1
    n_item = max(set(train_np[:, 1]) | set(train_np[:, 2]) | set(
        test_np[:, 1]) | set(test_np[:, 2]) | set(eval_np[:, 1]) | set(eval_np[:, 2])) + 1
    interaction = int(len(train_np) / 5) + len(test_np) + len(eval_np)

    print('n_user=%d, n_item=%d' % (n_user, n_item))
    print('n_interaction=%d' % interaction)
    return train_np, eval_np, test_np, user_history_dict, n_user, n_item


def load_kg(args, n_item):
    print('reading KG file ...')
    # reading kg file
    kg_file = '../data/' + args.dataset + '/kg_final'
    if os.path.exists(kg_file + '.npy'):
        kg_np = np.load(kg_file + '.npy')
    else:
        kg_np = np.loadtxt(kg_file + '.txt', dtype=np.int32)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    kg_entity, kg_relation, adj_sp = construct_kg(
        args, kg_np, n_relation, n_entity)
    adj_entity, adj_relation = construct_adj(
        args, kg_entity, kg_relation, n_item)
    adj_enbro = construct_adj_gb(
        args, kg_entity, n_item, adj_sp)
    kg_train = '../data/' + args.dataset + '/kg_train'
    if os.path.exists(kg_train + '.npy'):
        kg_np_ne = np.load(kg_train + '.npy')
    else:
        tail_entity = kg_np[:, 2]
        neg_entity = np.array([np.random.choice(np.delete(np.arange(
            n_entity), tail_entity[i], axis=0), 1) for i in range(len(tail_entity))])  # xiugai
        kg_np_ne = np.concatenate((kg_np, neg_entity), 1)
        np.save(kg_train + '.npy', kg_np_ne)
    print('number of entity ', n_entity)
    print('number of relations', n_relation)
    print('number of triples', len(kg_np))
    return n_entity, n_relation, kg_np_ne, adj_entity, adj_relation, adj_enbro


def fix_userhist(args, user_history_dict):
    for user in user_history_dict:
        n_history = len(user_history_dict[user])
        replace = n_history < args.n_memory
        sampled_indices = np.random.choice(
            n_history, size=args.n_memory, replace=replace)
        user_history_dict[user] = [user_history_dict[user][i]
                                   for i in sampled_indices]
    return user_history_dict


def construct_kg(args, kg_np, num_relation, n_entity):
    print('constructing knowledge graph...')
    kg_entity = dict()
    kg_relation = dict()
    for head, relation, tail in kg_np:
        if head not in kg_entity:
            kg_entity[head] = []
            kg_relation[head] = []
        kg_entity[head].append(tail)
        kg_relation[head].append(relation)
        if tail not in kg_entity:
            kg_entity[tail] = []
            kg_relation[tail] = []
        kg_entity[tail].append(head)
        kg_relation[tail].append(relation + num_relation)
    file = '../data/' + args.dataset + '/' + adj_entity_gb_p
    if os.path.exists(file + '.npy'):
        adj = None
    else:
        a_rows = kg_np[:, 0]
        a_cols = kg_np[:, 2]
        a_vals = [1] * len(a_rows)
        c_vals = [1] * n_entity
        c_rows = [i for i in range(n_entity)]
        c_cols = c_rows
        a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)),
                              shape=(n_entity, n_entity))
        b_adj = sp.coo_matrix((a_vals, (a_cols, a_rows)),
                              shape=(n_entity, n_entity))
        c_adj = sp.coo_matrix((c_vals, (c_rows, c_cols)),
                              shape=(n_entity, n_entity))
        adj = (a_adj + b_adj + c_adj).tolil()
    return kg_entity, kg_relation, adj


def construct_adj(args, kg_entity, kg_relation, item_num):
    # each line of adj_entity stores the sampled neighbor entities for a given entity
    # each line of adj_relation stores the corresponding sampled neighbor relations
    print('constructing local neighbor entities of items...')
    adj_entity = np.zeros(
        [item_num, args.n_neighbor], dtype=np.int64)
    adj_relation = np.zeros(
        [item_num, args.n_neighbor], dtype=np.int64)
    for entity in range(item_num):
        neighbors = kg_entity[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.n_neighbor:
            sampled_indices = np.random.choice(
                list(range(n_neighbors)), size=args.n_neighbor, replace=False)
        else:
            sampled_indices = np.random.choice(
                list(range(n_neighbors)), size=args.n_neighbor, replace=True)
        adj_entity[entity] = np.array(
            [kg_entity[entity][i] for i in sampled_indices])
        adj_relation[entity] = np.array(
            [kg_relation[entity][i] for i in sampled_indices])

    return adj_entity, adj_relation


def construct_adj_gb(args, kg_entity, item_num, adj_sp):
    print('constructing global neighbor entities of items...')
    file = '../data/' + args.dataset + '/' + adj_entity_gb_p
    if os.path.exists(file + '.npy'):
        adj_entity_gb = np.load(file + '.npy')
        adj_entity_gb = adj_entity_gb[:, -args.n_neighbor:]
    else:
        number = 50
        adj_entity_gb = np.zeros([item_num, number], dtype=np.int64)
        for item in range(item_num):
            adj_entity_gb[item] = depth_search(
                args, kg_entity, item, adj_sp, number)
        np.save(file + '.npy', adj_entity_gb)
        adj_entity_gb = adj_entity_gb[:, -args.n_neighbor:]
    return adj_entity_gb


def depth_search(args, kg_entity, entity, adj_sp, number):
    adj_entity = []
    adj_entity.append(entity)
    for i in range(15):  # 2 5 10 15 20 25
        temp = entity
        for j in range(8):  # 4 8 12 16 20 24
            neighbors = np.array(kg_entity[temp])
            # neighbors = np.random.choice(neighbors, size=)
            if j == 0:
                probly = np.ones(len(neighbors)) * (1 / len(neighbors))
            else:
                probly = np.ones(len(neighbors)) * 0.8
                index = adj_sp[adj_entity[-2], neighbors].nonzero()[1]
                probly[index] = 0.2
            probly = probly / np.sum(probly)
            pick_nei = np.random.choice(neighbors, size=1, p=list(probly))[0]
            adj_entity.append(pick_nei)
            temp = adj_entity[-1]
    adj_entity = adj_entity[1:]
    a = Counter(adj_entity)
    if [entity] != list(a):
        del a[entity]
    b = a.most_common(number)
    adj_entity = np.array([i[0] for i in b])
    leng = len(adj_entity)
    if leng < number:
        sampled_indices = np.random.choice(
            list(range(leng)), size=number - leng, replace=True)
        adj_entity = np.hstack((adj_entity, adj_entity[sampled_indices]))
    return np.array(adj_entity)[::-1]
