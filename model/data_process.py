import argparse
import numpy as np
import os

RATING_FILE_NAME = dict(
    {'movie': 'ratings.dat', 'book': 'BX-Book-Ratings.csv', 'news': 'ratings.txt', 'music': 'user_artists.dat'})
SEP = dict({'movie': '::', 'book': ';', 'music': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'news': 0, 'music': 0})


def read_item_index_to_entity_id_file():
    file = '../data/' + DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def convert_rating():
    print('reading rating file ...')
    file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]
    user_pos_ratings = dict()

    for line in open(file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
    print('converting rating file ...')

    user_cnt = 0
    user_index_old2new = dict()
    rating_all = []
    # store the pairs of user and positive or negetive(unwatched) item
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]
        for item in pos_item_set:
            rating_all.append([user_index, item])

    np.save('../data/' + DATASET + '/ratings_final.npy', rating_all)
    item_set = set(item_index_old2new.values())
    data_split(item_set)
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def data_split(item_set):
    rating_file = '../data/' + DATASET + '/ratings_final.npy'
    rating_np = np.load(rating_file)
    user_positive_items = dict()
    for rate in rating_np:
        user = rate[0]
        item = rate[1]
        if user not in user_positive_items:
            user_positive_items[user] = []
        user_positive_items[user].append(item)
    eval_ratio = 0.2
    test_ratio = 0.2
    n_ratings = rating_np.shape[0]

    eval_indices = np.random.choice(n_ratings, size=int(
        n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)
    test_indices = np.random.choice(list(left), size=int(
        n_ratings * test_ratio), replace=False)
    train_indices = list(left - set(test_indices))
    user_history_dict = dict()
    for i in train_indices:
        user = rating_np[i][0]
        item = rating_np[i][1]
        if user not in user_history_dict:
            user_history_dict[user] = []
        user_history_dict[user].append(item)
    train_indices = [i for i in train_indices if rating_np[i]
                     [0] in user_history_dict]
    eval_indices = [i for i in eval_indices if rating_np[i]
                    [0] in user_history_dict]
    test_indices = [i for i in test_indices if rating_np[i]
                    [0] in user_history_dict]
    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    np.save('../data/' + DATASET + '/user_history_dict.npy', user_history_dict)

    negative_sample('train', train_data, user_history_dict,
                    user_positive_items, item_set, ratio=5)
    negative_sample('eval', eval_data, user_history_dict,
                    user_positive_items, item_set)
    negative_sample('test', test_data, user_history_dict,
                    user_positive_items, item_set)


def negative_sample(datype, data, user_history_dict, user_positive_ratings, item_set, ratio=1):
    if datype == 'train':
        history = user_history_dict
    else:
        history = user_positive_ratings
    split_data = []
    for i in data:
        user_index = i[0]
        pos_item_index = i[1]
        negative_set = item_set - set(history[user_index])
        for neg_item_index in np.random.choice(list(negative_set), size=ratio, replace=False):
            split_data.append([user_index, pos_item_index, neg_item_index])
    np.save('../data/' + DATASET +
            '/' + datype + '_data.npy', split_data)


def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    # writer = open('../data/' + DATASET +
    #               '/kg_final.txt', 'w', encoding='utf-8')

    files = []
    files.append(open('../data/' + DATASET +
                      '/kg.txt', encoding='utf-8'))
    kg_final = []

    for file in files:
        for line in file:
            array = line.strip().split('\t')
            head_old = array[0]
            relation_old = array[1]
            tail_old = array[2]

            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]

            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]

            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            kg_final.append([head, relation, tail])
        file.close()

    np.save('../data/' + DATASET + '/kg_final.npy', kg_final)

    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str,
                        default='book', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.dataset
    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()
    read_item_index_to_entity_id_file()
    convert_kg()
    convert_rating()
    print('done')
