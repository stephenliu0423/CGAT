import numpy as np
from model import CGAT
import torch as t
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import time
import os
import pickle


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def minibatch_rs(args, data, user_history, kg, adj_enre, train=True):
    adj_enlc, adj_relc, adj_engb = adj_enre[0], adj_enre[1], adj_enre[2]
    data_size = len(data)
    l1 = int(data_size / args.batch_size) + 1
    for i in range(l1):
        start = args.batch_size * i
        end = min(args.batch_size * (i + 1), data_size)
        user_indices = data[start:end, 0]  # (batch,)
        pos_item_indices = data[start:end, 1]  # (batch,)
        neg_item_indices = data[start:end, 2]

        user_his_batch = np.array(
            [user_history[user] for user in user_indices])  # (batch, n_memory)
        entity_his_lc, relation_his_lc = get_neighbors(
            args, np.reshape(user_his_batch, -1), adj_enlc, adj_relc)
        entity_his_gb = get_neighbors(
            args, np.reshape(user_his_batch, -1), adj_engb, None)

        # (batch, n_memory, 1), (batch, n_memory, n_neigh) the neighbors of user historical items
        entity_his_lc = [np.reshape(
            entity, [len(user_indices), args.n_memory, -1]) for entity in entity_his_lc]
        relation_his_lc = np.reshape(
            relation_his_lc, [len(user_indices), args.n_memory, args.n_neighbor])
        entity_his_gb = [np.reshape(
            entity, [len(user_indices), args.n_memory, -1]) for entity in entity_his_gb]

        # [(batch, 1),(batch, n_neigh)]; (batch, n_neigh)  the neighbor of candidate items
        pos_entity_lc, pos_relation_lc = get_neighbors(
            args, pos_item_indices, adj_enlc, adj_relc)
        neg_entity_lc, neg_relation_lc = get_neighbors(
            args, neg_item_indices, adj_enlc, adj_relc)
        pos_entity_gb = get_neighbors(
            args, pos_item_indices, adj_engb, None)
        neg_entity_gb = get_neighbors(
            args, neg_item_indices, adj_engb, None)

        entity_lc = (entity_his_lc, pos_entity_lc, neg_entity_lc)
        relation_lc = (relation_his_lc, pos_relation_lc, neg_relation_lc)
        entity_gb = (entity_his_gb, pos_entity_gb, neg_entity_gb)
        if train:
            if end > len(kg):
                start = len(kg) - args.batch_size
                end = len(kg)
            kg_de = kg[start:end]
            head_indices = kg_de[:, 0]
            relation_indices = kg_de[:, 1]
            tail_indices = kg_de[:, 2]
            tail_indices_ne = kg_de[:, 3]
        else:
            head_indices, relation_indices = None, None
            tail_indices, tail_indices_ne = None, None
        kg_indice = (head_indices, relation_indices,
                     tail_indices, tail_indices_ne)
        yield user_indices, entity_lc, relation_lc, entity_gb, kg_indice


def train(args, data_info, show_loss, show_topk):
    print('starting train...')
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    n_relation = data_info[4]
    n_users = data_info[5]
    n_items = data_info[6]
    adj_enre = [data_info[8], data_info[9], data_info[11]]
    kg = data_info[7]
    user_history = data_info[10]
    weights_save_path = '../../result_w/' + args.dataset + '/cgat/'

    ensureDir(weights_save_path)

    model = CGAT(args, n_entity, n_relation, n_users, n_items)
    optimizer = optim.Adam(model.parameters(), lr=args.lr_rs,
                           weight_decay=args.l2_weight_rs)
    user_list, train_record, test_record, item_set, k_list = topk_settings(
        show_topk, train_data, test_data, n_items)
    stopping_step = 0
    should_stop = False
    cur_best_pre_0 = 0
    if args.use_cuda:
        model.cuda()
    for epoch in range(args.n_epochs):
        n = 0
        model.train()
        np.random.shuffle(train_data)
        np.random.shuffle(kg)
        t1 = time.time()
        loss_r = t.zeros(1)
        for user_indices, entity_lc, relation_lc, entity_gb, kg_indice in minibatch_rs(args, train_data, user_history, kg, adj_enre, True):
            user_indices = t.LongTensor(user_indices).cuda()
            entity_lc_cuda = []
            relation_lc_cuda = []
            entity_gb_cuda = []
            kg_cuda = []
            for i in range(len(entity_lc)):
                entity_lc_cuda.append([t.LongTensor(entity).cuda()
                                       for entity in entity_lc[i]])
                relation_lc_cuda.append(t.LongTensor(relation_lc[i]).cuda())
                entity_gb_cuda.append([t.LongTensor(entity).cuda()
                                       for entity in entity_gb[i]])
            for i in range(len(kg_indice)):
                kg_cuda.append(t.LongTensor(kg_indice[i]).cuda())

            score1, score2, all_loss = model(
                user_indices, entity_lc_cuda, relation_lc_cuda, entity_gb_cuda, kg_cuda)
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            loss_r += all_loss
            n += 1
        t2 = time.time()
        if show_loss:
            print('time: {:.4f}'.format(t2 - t1))
            print('ave_loss: %.4f' % (loss_r.data.cpu().numpy() / n))
        # result used to tune hyperparameters
        # eval_auc = evaluation(
        #     model, eval_data, user_history, kg, adj_enre, args)
        # test_auc = evaluation(
        #     model, test_data, user_history, kg, adj_enre, args)
        # print('epoch %d    eval auc: %.4f  test auc: %.4f'
        #       % (epoch + 1, eval_auc, test_auc))

        # Top evaluation
        if show_topk and (epoch + 1) % 5 == 0:
            precision, recall, ndcg, hit, result, ave_time = topk_eval(
                args, model, user_list, user_history, train_record, test_record, item_set, k_list, adj_enre)
            print('precision: ', end='')
            for i in precision:
                print('%.4f\t' % i, end='')
            print()
            print('recall: ', end='')
            for i in recall:
                print('%.4f\t' % i, end='')
            print()
            print('ndcg: ', end='')
            for i in ndcg:
                print('%.4f\t' % i, end='')
            print()
            print('hit: ', end='')
            for i in hit:
                print('%.4f\t' % i, end='')
            print('\n')
            # print('averaged time of predicting for each user:', ave_time)
            cur_best_pre_0, stopping_step, should_stop = early_stopping(recall[1], cur_best_pre_0,
                                                                        stopping_step, expected_order='acc', flag_step=2)
            if should_stop is True:
                break
            # *********************************************************
            # save the user & item embeddings for pretraining.
            # if recall[3] == cur_best_pre_0:
                # t.save(model.state_dict(), weights_save_path + 'weights.pth')
                # pickle.dump(result, open(weights_save_path + 'results', 'wb'))
                # print('save the weights in path: ', weights_save_path)

# local evaluation to tune parameters


def evaluation(model, data, user_history, kg, adj_enre, args):
    model.eval()
    auc_list = []
    for user_indices, entity_lc, relation_lc, entity_gb, kg_indice in minibatch_rs(args, data, user_history, kg, adj_enre, False):
        user_indices = t.LongTensor(user_indices).cuda()
        entity_lc_cuda = []
        relation_lc_cuda = []
        entity_gb_cuda = []
        # kg_cuda = []
        for i in range(len(entity_lc)):
            entity_lc_cuda.append([t.LongTensor(entity).cuda()
                                   for entity in entity_lc[i]])
            relation_lc_cuda.append(t.LongTensor(relation_lc[i]).cuda())
            entity_gb_cuda.append([t.LongTensor(entity).cuda()
                                   for entity in entity_gb[i]])

        score1, score2, all_loss = model(
            user_indices, entity_lc_cuda, relation_lc_cuda, entity_gb_cuda, kg_indice)
        score1 = score1.data.cpu().numpy()
        score2 = score2.data.cpu().numpy()
        scores = np.concatenate((score1, score2))
        label1 = np.ones(len(score1))
        label2 = np.zeros(len(score2))
        labels = np.concatenate((label1, label2))
        auc = roc_auc_score(y_true=labels, y_score=scores)
        auc_list.append(auc)
    model.train()
    return float(np.mean(auc_list))


def get_user_record(data):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        if user not in user_history_dict:
            user_history_dict[user] = set()
        user_history_dict[user].add(item)
    return user_history_dict


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        # user_num = 100
        k_list = [10, 20, 50]
        train_record = get_user_record(train_data)
        test_record = get_user_record(test_data)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        # if len(user_list) > user_num:
        #     user_list = np.random.choice(
        #         user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def topk_eval(args, model, user_list, user_history, train_record, test_record, item_set, k_list, adj_enre):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}
    hit_list = {k: [] for k in k_list}
    result = dict()
    adj_enlc, adj_relc, adj_engb = adj_enre[0], adj_enre[1], adj_enre[2]
    all_time = 0
    for user in user_list:
        # time0 = time.time()
        test_item_list = list(item_set - set(train_record[user]))
        item_score_map = dict()
        start = 0
        while start < len(test_item_list):
            end = min(start + args.batch_size, len(test_item_list))
            item_indices = test_item_list[start:end]
            # user_indices = [user] * (end - start)
            user_indices = [user]
            user_his_batch = np.array(
                [user_history[user] for user in user_indices])  # (batch, n_memory)
            entity_his_lc, relation_his_lc = get_neighbors(
                args, np.reshape(user_his_batch, -1), adj_enlc, adj_relc)
            entity_his_gb = get_neighbors(
                args, np.reshape(user_his_batch, -1), adj_engb, None)

            # (batch, n_memory, 1), (batch, n_memory, n_neigh).
            entity_his_lc = [np.reshape(
                entity, [len(user_indices), args.n_memory, -1]) for entity in entity_his_lc]
            relation_his_lc = np.reshape(
                relation_his_lc, [len(user_indices), args.n_memory, args.n_neighbor])
            entity_his_gb = [np.reshape(
                entity, [len(user_indices), args.n_memory, -1]) for entity in entity_his_gb]
            # (batch, 1),(batch, n_neigh)
            pos_entity_lc, pos_relation_lc = get_neighbors(
                args, item_indices, adj_enlc, adj_relc)
            pos_entity_gb = get_neighbors(
                args, item_indices, adj_engb, None)

            entity_his_lc = [t.LongTensor(
                i).cuda() for i in entity_his_lc]
            pos_entity_lc = [t.LongTensor(i).cuda() for i in pos_entity_lc]
            relation_his_lc = t.LongTensor(relation_his_lc).cuda()
            pos_relation_lc = t.LongTensor(pos_relation_lc).cuda()
            user_indices = t.LongTensor(user_indices).cuda()
            entity_his_gb = [t.LongTensor(
                i).cuda() for i in entity_his_gb]
            pos_entity_gb = [t.LongTensor(i).cuda() for i in pos_entity_gb]

            neg_entity_lc = None
            neg_relation_lc = None
            neg_entity_gb = None, None
            head_indices, relation_indices = None, None
            tail_indices, tail_indices_ne = None, None
            entity_lc_cuda = [entity_his_lc, pos_entity_lc, neg_entity_lc]
            relation_lc_cuda = [relation_his_lc,
                                pos_relation_lc, neg_relation_lc]
            entity_gb_cuda = [entity_his_gb, pos_entity_gb, neg_entity_gb]
            kg_cuda = [head_indices, relation_indices,
                       tail_indices, tail_indices_ne]
            time1 = time.time()
            score1, score2, all_loss = model(
                user_indices, entity_lc_cuda, relation_lc_cuda, entity_gb_cuda, kg_cuda)
            all_time += (time.time() - time1)
            for item, score in zip(item_indices, list(score1.data.cpu().numpy())):
                item_score_map[item] = score
            start += args.batch_size
        item_score_pair_sorted = sorted(
            item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]
        label = np.zeros(len(item_sorted))
        for i, item in enumerate(item_sorted):
            if item in test_record[user]:
                label[i] = 1
        for k in k_list:
            hit_num = np.sum(label[:k])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))
            ndcg_list[k].append(comp_ndcg(label, k))
            if np.sum(label[:k]) > 0:
                hit_list[k].append(1)
            else:
                hit_list[k].append(0)
    result['precision'] = np.array([precision_list[k] for k in k_list])
    result['recall'] = np.array([recall_list[k] for k in k_list])
    result['ndcg'] = np.array([ndcg_list[k] for k in k_list])
    result['hit'] = np.array([hit_list[k] for k in k_list])
    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]
    hit = [np.mean(hit_list[k]) for k in k_list]
    model.train()
    return precision, recall, ndcg, hit, result, all_time / len(user_list)


def comp_ndcg(label, k):
    topk = label[:k]
    dcg = np.sum(topk / np.log2(np.arange(2, topk.size + 2)))
    dcg_max = np.sum(sorted(label, reverse=True)[
                     :k] / np.log2(np.arange(2, topk.size + 2)))
    if dcg_max == 0:
        return 0
    else:
        return dcg / dcg_max


def get_neighbors(args, seeds, adj_entity, adj_relation):
    seeds_size = len(seeds)  # (batch, 1)
    seeds = np.expand_dims(seeds, axis=1)
    entities = [seeds]
    if adj_relation is not None:
        neighbor_entities = np.reshape(
            adj_entity[np.reshape(entities[0], -1)], [seeds_size, -1])
        neighbor_relations = np.reshape(
            adj_relation[np.reshape(entities[0], -1)], [seeds_size, -1])  # (batch, neighbor)
        entities.append(neighbor_entities)
        # relations.append(neighbor_relations)
        return entities, neighbor_relations
    else:
        neighbor_entities = np.reshape(
            adj_entity[np.reshape(entities[0], -1)], [seeds_size, -1])
        entities.append(neighbor_entities)
        return entities


def early_stopping(log_value, best_value, stopping_step, expected_order='acc', flag_step=2):
    # early stopping strategy:
    assert expected_order in ['acc', 'dec']

    if (expected_order == 'acc' and log_value >= best_value) or (expected_order == 'dec' and log_value <= best_value):
        stopping_step = 0
        best_value = log_value
    else:
        stopping_step += 1

    if stopping_step >= flag_step:
        print("Early stopping is trigger at step: {} log:{}".format(
            flag_step, log_value))
        should_stop = True
    else:
        should_stop = False
    return best_value, stopping_step, should_stop
