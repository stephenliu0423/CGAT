import torch as t
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np


class CGAT(nn.Module):
    def __init__(self, args, n_entity, n_relation, n_users, n_items):
        super().__init__()
        self.args = args
        self.entity_embedding = nn.Embedding(n_entity, args.dim)
        self.relation_embedding = nn.Embedding(
            2 * n_relation + 1, args.dim)
        self.user_embedding = nn.Embedding(n_users, args.dim)
        # rs_model
        self.linear_u_mlp = nn.Linear(3 * args.dim, args.dim)
        self.weight_re = nn.Linear(2 * args.dim, args.dim, bias=False)
        self.weight_agg = nn.Linear(2 * args.dim, args.dim)
        self.gate = nn.Parameter(t.FloatTensor(args.dim))
        self.graph_att = nn.Linear(2 * args.dim, args.dim)
        self.item_att = nn.Linear(4 * args.dim, 1)
        self.rnn = nn.GRU(args.dim, args.dim, batch_first=True)
        self.trans_att = nn.Linear(args.dim, args.dim)
        self.user_specific = True
        if not self.user_specific:
            self.weight_nouser = nn.Linear(args.dim, 1, bias=False)

        self.Dropout = nn.Dropout(args.dropout)
        init.zeros_(self.gate.data)
        init.xavier_uniform_(self.relation_embedding.weight)

    # (batch, n_memories, dims)
    def forward(self, user_indices, entity_lc, relation_lc, entity_gb, kg):
        # set_trace()
        user_global_em = self.user_embedding(
            user_indices)  # (batch, embeddings)  user global embedding
        entity_hislc_em = self.entity_agg(
            entity_lc[0], relation_lc[0], user_global_em)  # local embedding of user historical items
        # global embedding of items (batch, n_memory, embeddings)
        entity_hisgb_em = self.entity_agg(entity_gb[0], None, None)
        # aggregate the local embedding and global embedding of items
        gate = t.sigmoid(self.gate)
        item_history_em = gate.expand_as(
            entity_hislc_em) * entity_hislc_em + (1 - gate).expand_as(entity_hisgb_em) * entity_hisgb_em
        item_history_em = t.cat((self.self_vectors, item_history_em), -1)

        if len(item_history_em) == 1:
            item_history_em = item_history_em.expand(
                len(entity_lc[1][0]), item_history_em.shape[1], item_history_em.shape[2])
            user_global_em = user_global_em.expand(
                len(entity_lc[1][0]), user_global_em.shape[1])
        pos_enlc_em = self.aggregate_lc(
            entity_lc[1], relation_lc[1], user_global_em)  # (batch, dims)  local embedding of candidate items
        pos_engb_em = self.aggregate_gb(entity_gb[1])
        pos_item_em = gate.expand_as(pos_enlc_em) * pos_enlc_em + (
            1 - gate).expand_as(pos_engb_em) * pos_engb_em
        pos_item_em = t.cat((self.self_vectors, pos_item_em), -1)
        userp_lc_em = self.rs_attention(
            item_history_em, pos_item_em)  # (batch, dims)
        userp_em = t.cat(
            (user_global_em, userp_lc_em), 1)  # (batch, 2*dims)
        userp_em = t.relu(
            self.linear_u_mlp(self.Dropout(userp_em)))
        userp_em = self.Dropout(userp_em)
        userp_em = F.normalize(userp_em, dim=-1)
        userp_em = t.cat((user_global_em, userp_em),
                         1)  # (batch, 2 * dims)
        pos_score = t.sum(userp_em * pos_item_em, 1)  # (batch,)

        if entity_lc[2] is not None:
            neg_enlc_em = self.aggregate_lc(
                entity_lc[2], relation_lc[2], user_global_em)  # (batch, dims)
            neg_engb_em = self.aggregate_gb(entity_gb[2])
            neg_item_em = gate.expand_as(neg_enlc_em) * neg_enlc_em + (
                1 - gate).expand_as(neg_engb_em) * neg_engb_em
            neg_item_em = t.cat((self.self_vectors, neg_item_em), -1)
            usern_lc_em = self.rs_attention(
                item_history_em, neg_item_em)  # (batch, dims)
            usern_em = t.cat(
                (user_global_em, usern_lc_em), 1)  # (batch, 2*dims)
            usern_em = t.relu(
                self.linear_u_mlp(self.Dropout(usern_em)))  # (batch, dims)
            usern_em = self.Dropout(usern_em)
            usern_em = F.normalize(usern_em, dim=1)
            usern_em = t.cat((user_global_em, usern_em), 1)
            neg_score = t.sum(usern_em * neg_item_em, 1)  # (batch,)
        else:
            neg_score = t.zeros(1).cuda()
        rs_loss = -t.mean(t.log(t.sigmoid(pos_score - neg_score)))

        # knowledge graph loss for regularization
        if kg[0] is not None:
            head_em = self.entity_embedding(kg[0])
            relation_em = self.relation_embedding(kg[1])
            tail_em = self.entity_embedding(kg[2])
            tail_ne_em = self.entity_embedding(kg[3])

            nere_vectors = t.cat((tail_em, relation_em), 1)
            nere_ne_vectors = t.cat(
                (tail_ne_em, relation_em), 1)  # (batch, 2dims)
            # (batch, dims)
            nere_vectors = self.weight_re(nere_vectors)
            nere_ne_vectors = self.weight_re(nere_ne_vectors)

            score_kge = t.sum((head_em - nere_vectors).pow(2), 1)
            score_kge_ne = t.sum((head_em - nere_ne_vectors).pow(2), 1)
            kg_loss = -t.mean(t.log(t.sigmoid(score_kge_ne - score_kge)))
        else:
            kg_loss = t.zeros(1).cuda()
        all_loss = rs_loss + self.args.kg_weight * kg_loss

        return pos_score, neg_score, all_loss

    # aggregate local embeddings of items
    def entity_agg(self, entity_his, relation_his, user_global_em):
        batch = entity_his[0].shape[0]
        # (batch * n_memories, 1), (batch * n_memories, neighbor)
        entity_his = [i.reshape([-1, i.shape[-1]]) for i in entity_his]
        if relation_his is not None:
            relation_his = relation_his.reshape(
                [-1, self.args.n_neighbor])  # (batch*n_memories, neighbor)
            user_em_att = user_global_em.unsqueeze(1).expand(
                batch, self.args.n_memory, self.args.dim)  # (batch, n_memories, dims)
            # (batch*n_memories, dims)
            user_em_att = user_em_att.contiguous().view(-1, self.args.dim)
            entity_his_vec = self.aggregate_lc(
                entity_his, relation_his, user_em_att)  # (batch*n_memories, dims)
            self.self_vectors = self.self_vectors.contiguous().view(
                [batch, self.args.n_memory, self.args.dim])
        else:
            entity_his_vec = self.aggregate_gb(entity_his)
        entity_his_vec = entity_his_vec.contiguous().view(
            [batch, self.args.n_memory, self.args.dim])  # (batch, n_memories, dims)
        return entity_his_vec

    def aggregate_lc(self, entities, relations, users_embedding):
        # (batch, 1, dims), (batch, neigh, dims)...
        entity_vectors = [self.entity_embedding(i) for i in entities]
        # (batch, neigh, dims)
        relation_vectors = self.relation_embedding(relations)
        self.self_vectors = entity_vectors[0].squeeze(1)  # (batch, dims)
        # (batch, neigh, dims)
        neighbor_vectors = entity_vectors[1]
        # (batch, neigh, 2dims)
        nere_vectors = t.cat((neighbor_vectors, relation_vectors), 2)
        # (batch, neigh, dims)
        nere_vectors = self.weight_re(nere_vectors)
        user_att = t.relu(self.trans_att(users_embedding))
        vector = self.SumAttention(self_vectors=self.self_vectors, neres=nere_vectors,
                                   user_embeddings=user_att, neighbors=neighbor_vectors, user_specific=self.user_specific)
        vector = self.Dropout(vector)
        vector = F.normalize(vector)
        return vector

    def aggregate_gb(self, entities):
        # (batch, 1, dims), (batch, neigh, dims)
        entity_vectors = [self.entity_embedding(i) for i in entities]
        self_vectors = entity_vectors[0].squeeze(1)  # (batch, dims)
        # (batch, -1, neigh, dims)
        neighbor_vectors = entity_vectors[1]  # (batch, neigh, dims)
        output, h0 = self.rnn(neighbor_vectors)
        # (batch, -1, dims)
        agg_vectors = output[:, -1, :]  # (batch, dims)
        agg_vectors = t.tanh(self.weight_agg(
            t.cat((self_vectors, agg_vectors), 1)))  # (batch, dims)
        agg_vectors = self.Dropout(agg_vectors)
        agg_vectors = F.normalize(agg_vectors)
        return agg_vectors

    # aggregate the item embeddings to obation user local preference
    def rs_attention(self, item_history_embedding, item_pre_embedding):
        # (batch, n_memories)
        item_pre_embedding = item_pre_embedding.unsqueeze(
            1).expand_as(item_history_embedding)
        logits = t.tanh(self.item_att(
            t.cat((item_history_embedding, item_pre_embedding), 2))).squeeze(2)
        attention = t.softmax(logits, 1)
        user_embedding = t.matmul(attention.unsqueeze(
            1), item_history_embedding).squeeze(1)  # (batch, dims)
        return user_embedding

    # user-specific GAT
    def SumAttention(self, self_vectors, neres, user_embeddings, neighbors, user_specific):
        # batch = len(self_vectors)
        user_embeddings = user_embeddings.unsqueeze(1)  # (batch, 1, dims)
        # (batch, -1, neighbor+1, dim)
        if user_specific:
            user_embeddings = user_embeddings.expand_as(
                neighbors)  # (batch, neighbor, dims)
            self_vectors1 = self_vectors.unsqueeze(1).expand_as(neighbors)
            # (batch, neighbor, 2dims)
            cat_vectors = t.cat((self_vectors1, neres), 2)
            trans_vectors = t.tanh(self.graph_att(cat_vectors))
            logits = t.sum(user_embeddings * trans_vectors,
                           2)  # (batch, neighbor)
            attention = t.softmax(logits, 1)  # (batch, neighbor)
            new_vector = t.matmul(attention.unsqueeze(
                1), neighbors).squeeze(1)  # (batch, dim)
        else:
            self_vectors1 = self_vectors.unsqueeze(1).expand_as(neighbors)
            cat_vectors = t.cat((self_vectors1, neres), 2)
            trans_vectors = t.tanh(self.graph_att(cat_vectors))
            logits = self.weight_nouser(trans_vectors).squeeze(-1)
            attention = t.softmax(logits, 1)
            new_vector = t.matmul(attention.unsqueeze(
                1), neighbors).squeeze(1)  # (batch, dim)
        new_vector = t.tanh(self.weight_agg(
            t.cat((self_vectors, new_vector), 1)))  # (batch, dims)
        return new_vector
