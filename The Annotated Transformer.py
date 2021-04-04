import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")


# 3 Model Architecture
r"""
# encoder的输入是 $x_1, \ldots, x_n$；输出是 $z_1, \ldots, z_n$
# decoder的输出是 $y_1, \ldots, y_m$，一次输出一个element；
# decoder的输入除了encoder的输出外，额外增加前面step生成的所有element，用于生成next
"""


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):  # 没用到generator？
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)  # 此处使用log_softmax（$\text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)$）作为激活函数的目的是什么？


# 3.1 Encoder and Decoder Stacks
# Encoder

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # 为什么在经过了N个identical的layer后要加上一个LayerNorm层？（后面解答了）


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2  # a_2和b_2的意义是什么？
        # 查看其文档字符串貌似是以该方式加入的tensor可作为模型参数参与训练更新所以就是类似Wx+b？
        # 这里是a_2 x + b_2, 其中，x是减去均值除去标准差之后的规范化后的x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
        # 该层不是@sublayer+add@+norm，而是@norm+sublayer+dropout+add@，
        # 所以Encoder要在N个identical的layer之后加上一个LayerNorm


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


# Decoder

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

# 图示mask
# plt.figure(figsize=(5,5))
# plt.imshow(subsequent_mask(20)[0])
# None


# 3.2 Attention
# 3.2.1 Scaled Dot-Product Attention
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
# attention函数是将一个query映射为一个output，即output的数量取决于query的数量
# 这里的q k v都默认是行向量（与"The Illustrated Transformer"相同，而李宏毅举例用的是列向量），打包即为从上至下摞起来
# 由后面MultiHeadedAttention的计算结果推断传输过来的query, key, value的dim=4，
# 即(batch_size, head_num, token_num_in_one_sentence, vector_size )


# 3.2.2 Multi-Head Attention
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 函数subsequent_mask产生的mask的dim=3，这里为了一次性用于多头，增加了一个维度
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)  # 其中-1维度代表的是一句话的token数量
             for l, x in zip(self.linears, (query, key, value))]
        # l代表的linear层的维度是512*512，所以是将8个头的W^Q（W^K，W^V）放在一起计算，并没有进行区分W^Q_i
        # transpose 将多头的维度提前，起到了分割的作用
        # 计算QKV分别用了1个大linear，余下的一个大linear用在最后将拼接后的8头输出向量线性变换

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# 3.2.3 Applications of Attention in our Model

# 3.3 Position-wise Feed-Forward Networks
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
# fully connected feed-forward network


# 3.4 Embeddings and Softmax
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# 3.5 Positional Encoding
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model) # 存储全部的position encoding（一次性算出）
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        # 此步骤计算的就是$\frac{1}{10000^{\frac{2i}{d_{model}}}}$，其中$i \in \{0,1,2, \ldots, 255\}$
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # 不会作为model parameter，但是会默认作为module's state的一部分

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

# 可视化
# plt.figure(figsize=(15, 5))
# pe = PositionalEncoding(20, 0)
# y = pe.forward(Variable(torch.zeros(1, 100, 20)))
# plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
# plt.legend(["dim %d" % p for p in [4,5,6,7]])
# None


# Full Model
def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for i, p in enumerate(model.parameters()):
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    # print(i)
    return model


# Small example model.
# tmp_model = make_model(10, 10, 2)  # N增加1，model.parameters()数量增加42（7、49、91、133）
# None


# 5 Training

# quick interlude for some of the tools
# Batches and Masking
class Batch:
    "Object for holding a batch of data with mask during training."

    r"""
    参数：
        src (torch.Tensor)：模型的输入（英译法任务中输入的英文句子），shape=(batch，ntokens)
        trg (torch.Tensor)：模型的输出（ground truth，英译法任务中标准答案的法语句子），训练时可指定，预测时为None。
        pad (int)：padding的填充值？但是前面构造数据的时候明明在每个sample的第0个token填充的1
    """

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None: # 训练时trg不是None
            self.trg = trg[:, :-1]   # 除去最后一列？因为shifted right？预测第i个token输入的只是前（i-1）个token，只有这（i-1）个token需要做mask？
            self.trg_y = trg[:, 1:]  # 除去第一列？因为预测第一个token不计算loss？还是压根不需要预测第一个token？
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()  # 是整个batch全部的token数量，也没算每行的第一个token

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))  # 此处&操作尺寸不匹配可以广播
        return tgt_mask


# Training Loop
def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):  # 为何没有参数更新的操作？
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)  # out没有输出最终的预测结果啊，后面如何用它来计算loss？
        loss = loss_compute(out, batch.trg_y, batch.ntokens) # 当然loss_compute函数也没有写明
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            # 输出当前epoch的step(batch)序号，当前step(batch)平均预测每个token的loss，近50个step(batch)平均预测每个token用时
            start = time.time()
            tokens = 0
    return total_loss / total_tokens  # 当前epoch平均预测每个token的loss


# 5.1 Training Data and Batching
global max_src_in_batch, max_tgt_in_batch


def batch_size_fn(new, count, sofar):  # sofar 参数没用到？
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)  # 为何要加2？<BOS><EOS>？
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


# 5.2 Hardware and Schedule

# 5.3 Optimizer
class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate  # 意思是模型中每个参数的学习率可以单独设置？
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))  # 论文中计算学习率的公式(3)，但是多乘了一个factor是为何？


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))  # factor是2，其余的超参数都与原文一致


# Example of the curves of this model for different model sizes and for optimization hyperparameters.
# Three settings of the lrate hyperparameters.
# opts = [NoamOpt(512, 1, 4000, None),
#         NoamOpt(512, 1, 8000, None),
#         NoamOpt(256, 1, 4000, None)]
# plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
# plt.legend(["512:4000", "512:8000", "256:4000"])
# plt.show()
# None


# 5.4 Regularization
# Label Smoothing
# implement label smoothing using the KL div loss

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size  # 这里x.size(1)不是每句话的token数量么？
        # 下面的例子里面传入的x貌似是模型decoder的预测结果，是已经从num_token中选出了作为预测结果的那个，
        # 所以x.size是（batch_size,vocab_size)，且其中的值不是概率而是概率的log值，这点似乎与Generator类中linear后面接log_softmax层保持了某种程度上的统一性
        # 这里self.size也应是vocab_size，不是d_model哦
        # 而例子中target是个1d列表，其中每个值表示正确答案在vocab中的序号，其size应该是batch_size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # 为何减去2？<BOS><EOS>？
        # 在例子中正好减去了对应预测结果的每行头尾的两个概率0，不知道为什么预测输出的概率会是这种形式
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # torch.Tensor.scatter_()函数的作用数学公式上理解了，但是如何利用它进行label smoothing还是没有从物理意义上理解？
        # 看了例子貌似理解了，他就是将target对应的正确答案的index对应的entity的概率设为confidence
        # 但是存在问题是，smooth分配的时候减去了2个元素，虽然正确答案本身重新赋值confidence（scatter_()的参数reduce并没有设置为add）
        # 所以每一行的概率和不是1，多出一个（smoothing / (size - 2)），如果按照（正确答案的smooth应该被所有候选包括正确答案自己平分的话，scatter_()的参数reduce设置为add）
        # 这样就会多出两个值使得概率和大于1（2 * smoothing / (size - 2)）
        true_dist[:, self.padding_idx] = 0
        # 哦，这里动手了，将padding_idx的列设为0，就减去了多余的那一个值，现在每一行的概率和是1了
        # 但是为什么呢？因为这列是padding？
        # 所以是例子简化了，正常smooth均分的时候size需要减去（padding个数+1）？
        mask = torch.nonzero(target.data == self.padding_idx)
        # mask指示target中正确答案其实是padding的index
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)  # 将mask指示的行的概率全部置为0
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
        # 返回值是预测值x与label smooth后的ground truth值target之间的KLDivLoss损失值


# # Example of label smoothing.
# crit = LabelSmoothing(5, 0, 0.4)
# predict = torch.FloatTensor([[0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0],
#                              [0, 0.2, 0.7, 0.1, 0]])
# v = crit(Variable(predict.log()),
#          Variable(torch.LongTensor([2, 1, 0])))
#
# # Show the target distributions expected by the system.
# plt.imshow(crit.true_dist)
# plt.show()
# # 此时crit.true_dist =
# # tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
# #         [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
# #         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
# # 所以图示第一列和最后一行都是紫色代表0，正确答案对应的点是黄色0.6，其余是smooth的蓝色0.13
# # None

# # label smoothing的另一个例子
# crit = LabelSmoothing(5, 0, 0.1)
# def loss(x):
#     d = x + 3 * 1
#     predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d],
#                                  ])
#     #print(predict)
#     return crit(Variable(predict.log()),
#                  Variable(torch.LongTensor([1]))).data
# plt.plot(np.arange(1, 100), [loss(x) for x in range(1, 100)])
# # 这个例子里面只有一个sample，且预测的答案是正确的，随着x的增大，即模型对正确答案越来越确定，loss极速下降，而后缓慢回升
# # 说明标签平滑实际上开始惩罚模型如果它对一个给定的选择非常自信。
# plt.show()
# None


# A First Example-----------------------------------------
# 复制任务，输入来自小词汇表的随机符号，希望模型输出相同的符号

# Synthetic Data
def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."

    r"""
    参数：
        V (int)：词汇表的size（词的数量）
        batch (int)：每个batch中sample（sentence）的数量
        nbatches (int)：batch的数量
    """

    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1  # 为何要将第一个token统一设置为1？
        # src = Variable(data, requires_grad=False)
        # tgt = Variable(data, requires_grad=False)
        src = Variable(data, requires_grad=False).long()  # size=(30,10)
        tgt = Variable(data, requires_grad=False).long()
        yield Batch(src, tgt, 0)


# Loss Computation
class SimpleLossCompute:
    "A simple loss compute and train function."

    r"""
    参数：
        generator：
        criterion：
        opt：
    """

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # return loss.data[0] * norm
        return loss.data * norm


# Greedy Decoding
# Train the simple copy task.
V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model,
              SimpleLossCompute(model.generator, criterion, model_opt))
    model.eval()
    print(run_epoch(data_gen(V, 30, 5), model,
                    SimpleLossCompute(model.generator, criterion, None)))


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]) )
src_mask = Variable(torch.ones(1, 1, 10) )
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

# A First Example END----------------------------------------
