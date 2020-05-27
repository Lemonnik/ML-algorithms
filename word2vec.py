import re, string
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.spatial import distance
from tqdm import tqdm


class SkipGram:

    def __init__(self, filename, pattern=None, window=7, embedding_dim=25, min_freq=3):

        self.window = window
        self.context = window - 1
        self.embedding_dim = embedding_dim
        self.min_freq = min_freq

        tokens = self.tokenize(filename, pattern)
        self.vocab = self.make_vocab(tokens)
        self.word2idx = self.make_idx()
        self.idx2word = self.make_word()
        self.pairs = np.array(self.make_context_pairs(tokens))

        # center
        self.V = np.random.rand(len(self.vocab), self.embedding_dim)
        # context
        self.U = np.random.rand(self.embedding_dim, len(self.vocab))
    

    def tokenize(self, filename, pattern=None):
        pattern=f'[{string.punctuation} ]+' if pattern is None else pattern
        regex = re.compile(pattern)
        with open(filename) as f:
            tokens = [regex.split(line.lower().rstrip()) for line in f.readlines()]
        return tokens


    def make_vocab(self, tokens):
        vocab = defaultdict(int)
        for line in tokens:
            for word in line:
                vocab[word] += 1
        return {word: freq for word, freq in vocab.items() if freq >= self.min_freq}


    def make_idx(self):
        return {word: idx for idx, word in enumerate(self.vocab.keys())}

    
    def make_word(self):
        return dict(zip(self.word2idx.values(), self.word2idx.keys()))

    
    def make_context_pairs(self, tokens):
        pairs = []
        dx = self.window // 2
        zeros = [0] * (self.window - 1)
        ones = [1] * (self.window - 1)
        for line in tokens:
          idx = [self.word2idx[word] for word in line if self.word2idx.get(word, 0)]
          for i in range(dx, len(idx) - dx):
              # context words
              u_pos = idx[i-dx:i] + idx[i+1:i+dx+1]
              target = coo_matrix((ones, (zeros, u_pos)), 
                                  shape=(1, len(self.vocab)), 
                                  dtype=np.uint64)
              pairs.append([idx[i], target])
        return pairs


    def forward(self, v_pos):
        # [B x Emb]
        v_c = self.V[v_pos, :]
        # [B x Voc] <- [B x Emb * Emb x Voc]
        out = v_c.dot(self.U)
        return out, self.softmax(out)


    def softmax(self, logit):
        e = np.exp(logit)
        return e  / np.sum(e, axis=1, keepdims=True)


    def cost(self, u_r, u_c, uv):
        # indices of non-zero targets
        u_r = u_r.reshape(-1, self.context)
        u_c = u_c.reshape(-1, self.context)
        prob_o = np.sum(uv[u_r, u_c], axis=1)
        return np.mean(-prob_o + self.context * np.log(np.sum(np.exp(uv), axis=1)))         


    def grad_out(self, y_pred, target):
        # [B x Voc] -> [1 x Voc]
        return np.mean(self.context * y_pred - target, axis=0).reshape(1, -1)


    def backward(self, v_pos, y_pred, target, lr):
        delta = self.grad_out(y_pred, target)
        # batch mean -> [1 x Emb] 
        v_mean = np.mean(self.V[v_pos, :], axis=0)

        # [1 x Emb * 1 x Voc] -> [Emb x Voc]
        U_grad = np.outer(v_mean, delta)
        # [Emb x Voc * Voc x 1]
        V_grad = self.U.dot(delta.T)

        self.U -= lr * U_grad
        # repeat grad batch times (only these embeddings should be updated)
        self.V[v_pos, :] -= lr * np.repeat(V_grad, v_pos.shape[0], axis=1).T


    @staticmethod
    def batch(data, size=1):
        l = len(data)
        idx = np.arange(l)
        np.random.shuffle(idx)
        for i in range(0, l, size):
            yield data[idx[i:i+size]] 


    def train(self, epochs=10, batch_size=8, lr=1e-3):
        cost_hist = []
        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
              cost = []
              for data in self.batch(self.pairs, size=batch_size):
                  v_pos = np.array(data[:, 0], dtype=np.uint64)
                  target = vstack(data[:, 1])
                  uv, pred = self.forward(v_pos)
                  self.backward(v_pos, pred, target, lr)
                  cost.append(self.cost(target.row, target.col, uv))
              cost_hist += [np.mean(cost)]
              log = '{}/{} epochs: loss {:.5f}'.format(epoch + 1, epochs, cost_hist[-1])
              pbar.set_description(log)
              pbar.update(1)
        return cost_hist


    def get_word_vector(self, word):
        v_pos = self.vocab.get(word, None)
        return self.V[v_pos] if v_pos else None


    def find_context(self, word):
        v_pos = self.vocab.get(word, None)
        if v_pos:
            logits, probs = self.forward([v_pos])
            idxs = np.argsort(-probs)[0][:self.context]
            return list(map(lambda i: self.idx2word[i], idxs))
        return None


    @staticmethod
    def similarity(v1, v2):
        return 1 - distance.cosine(v1, v2)


    def find_most_similar(self, word, n=5):
        idx = self.word2idx.get(word, None)
        if idx: 
            v = self.V[idx]
            cos_sim = np.apply_along_axis(lambda x: self.similarity(v, x), 1, self.V)
            most_similar = np.argsort(-cos_sim)[:n]
            return list(map(lambda i: self.idx2word[i], most_similar))
        return None
