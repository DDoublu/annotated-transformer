r"""
paper:
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia
    Polosukhin. 2017. Attention Is All You Need. arXiv:1706.03762 [cs] (December 2017). Retrieved June 11,
    2020 from http://arxiv.org/abs/1706.03762
注释数量符号说明：
    epoch: 一个epoch，所有的数据循环训练一遍
    nbatches: 一个epoch中batch的数量（下面代码中有时指数据迭代器生成的batch的数量）
    batch_size: 一个batch中sample（sentence）的数量，即训练数据的数量=nbatches*batch_size
    src_vocab: 源语言（encoder的input）词汇表的大小
    tgt_vocab：目标语言（decoder的output）词汇表的大小
    token: 英文的 word 或者 word piece（与具体应用和算法的设计有关），中文分词后的词组或者短语（不是单个汉字字符character）
    sample（sentence）：翻译任务的话就是一对英译法的句子，（英，法）
    ntokens: 一个sentence中输入或者输出的token的数量，一般数量不统一，可能会利用padding进行长度统一（在下面代码中有时候用来
    指一个batch中需要预测的token总数量）
    src_ntokens: 一个sample中encoder输入的token的数量
    tgt_ntokens: 一个sample中decoder预测输出的token的数量
"""
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
encoder的输入是 $x_1, \ldots, x_n$；输出是 $z_1, \ldots, z_n$
decoder的输入除了encoder的输出外，额外增加前面step生成的所有element，用于生成next
decoder的输出是 $y_1, \ldots, y_m$，一次输出一个element；（实际输出和输入的数量也是相同的，只不过用于预测只选择最后一个而已）
"""


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        r"""Transformer的基础架构
        :param encoder: (Encoder) 实现Transformer的encoder
        :param decoder: (Decoder) 实现Transformer的decoder
        :param src_embed: (Sequential) 实现Transformer encoder前的embedding层以及add Positional Encoding操作
        :param tgt_embed: (Sequential) 实现Transformer decoder前的embedding层以及add Positional Encoding操作
        :param generator: (Generator) 实现将Transformer decoder的输出转换成预测的操作
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        r"""将decoder输出的d_model长度向量转换成目标词表的vocab长度向量，即预测值
        :param d_model: (int) 模型的size
        :param vocab: (int) tgt_vocab的size
        """
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
        # 此处使用log_softmax（$\text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)$）
        # 作为激活函数的目的是什么？（A: 使概率在对数空间？）


# 3.1 Encoder and Decoder Stacks
# Encoder

def clones(module, N):
    r"""
    "Produce N identical layers."
    :param module: 需要复制的层
    :param N: (int) 复制的个数
    :return: (ModuleList) 含有N个module的ModuleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        r"""Encoder是由N个完全相同的layer组成，然后加上一个norm层
        :param layer: 完全相同的layer
        :param N: (int) 相同的layer的个数
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # 为什么在经过了N个identical的layer后要加上一个LayerNorm层？
        # （A:为了代码的简洁性，该实现代码并没有严格按照论文中所述在子层操作后进行Add&Norm，
        # 而是先进行Norm然后子层操作后再Add，这样导致最后一层的输出并没有经过Norm操作，
        # 所以最后加上一个Norm操作）


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
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        # a_2和b_2的意义是什么？
        # 查看其文档字符串貌似是以该方式加入的tensor可作为模型参数参与训练更新所以就是类似Wx+b？
        # 这里是a_2 y + b_2, 其中，y是减去均值除去标准差之后的规范化的x


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
    r"""
    "Encoder is made up of self-attn and feed forward (defined below)"
    每个EncoderLayer包含两个子层，分别是self_attn自注意力子层和feed_forward前馈网络子层
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第一个子层是self_attn
        # 这里第二个参数不应该是层对象么？为何同时传输参数，得到的不就是MultiHeadAttention forward函数的返回值，
        # 是torch.Tensor了么？与下一句的传输参数类型不同啊？
        # nonono，上面理解错了，这里的第二个参数是函数本身，而不是函数的返回值
        return self.sublayer[1](x, self.feed_forward)
        # 第二个子层是feed_forward
        # sublayer 是 SublayerConnection 对象的 ModuleList，而子层中核心是什么取决于调用SublayerConnection 的 forward函数时，
        # 传输的第二个参数（该参数是一个继承nn.Model的层）


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
    r"""
    "Mask out subsequent positions."
    :param size: (int) 需要产生的mask的尺寸边长
    :return: (torch.Tensor) 返回shape=(1,size,size)的tensor，其元素mask[0]是方形的下三角（包含主对角线）全True其余全False的矩阵
    """
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
    r"""
    "Compute 'Scaled Dot Product Attention'"
    :param query: (torch.Tensor) shape=(batch_size,nheads,q.ntokens,d_model/nheads)
    :param key: (torch.Tensor) shape=(batch_size,nheads,k.ntokens,d_model/nheads)
    :param value: (torch.Tensor) shape=(batch_size,nheads,k.ntokens,d_model/nheads)
    k.ntokens=v.ntokens，就是MultiHeadAttention中传入的k和v的size(1)，就是一个sample中src或者tgt的token的数量；q.token同理
    :param mask: (torch.Tensor) mask是和kv相关的mask，而不是与q相关。所以若是src_mask，其shape=(batch_size,1,k.ntokens);
    若是tgt_mask，其shape=(batch_size,k.ntokens,k.ntokens)
    :param dropout: (nn.Dropout) nn.Dropout层
    :return: (torch.Tensor) 返回的是计算后加权和向量shape=(batch_size,nheads,q.ntokens,d_model/nheads),
    和注意力权重shape=(batch_size,nheads,q.ntokens,k.ntokens)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # 在decoder的masked Multi-Head Attention模块中该操作使得信息永远不能向左传播，其作用具体体现在两个层面：
        # 1）预测第i个token的时候，因为训练模式是将ground truth一次性输入，mask使得无法看到第i个token及之后的信息，
        # 防止利用自己预测自己，这点在测试模式下并没有意义，因为即将预测的token根本不知道，也并没有输入；
        # 2）当预测第i个token时，前面token的加权和计算中也都值能attend到自己之前的token，
        # 这一点无论是在训练模式还是测试模式下都存在。具体举一个例子，当预测第6个token时，计算第3个token只能attend到0-3个token，
        # 第5个token的加权和只能attend0-5个token，这样的编码结果有点像单向的lstm，你可能会说反正最后用于预测第6个token利用的也是
        # 第5个token的加权和，其余的无所谓，但我们考虑最后一层N=6时第5个token的加权和attend的是上一层N=5得出的0-5的token的加权和，
        # 这样前面加权和的计算就对其有影响了。但我觉得第一个层面的mask是有意义的，第二个层面的mask真的需要么？
        # 或者这就是所谓的auto-regressive property，那BiLSTM就没有保持这种属性了？但因为训练时就是这样mask的，
        # 所以也无法单独在测试时让mask=NONE？这是否只是该版本代码实现方面存在的问题？论文中的原意是怎样呢？具体到预测效果会有影响么？
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
        self.attn = None  # 存储返回的attention权重
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 函数subsequent_mask产生的mask的dim=3，这里为了一次性用于多头，增加了一个维度
        nbatches = query.size(0)  # 这里nbatches指的是每个batch的sample的数量，即batch_size

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
        # self.attn的size是(batch_size,nheads,q.size(1),k.size(1))， q.size(1)是作为q的ntokens，k.size(1)是作为k的ntokens，
        # 存储的是注意力权重

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

    def __init__(self, src, trg=None, pad=0):
        r"""

        :param src: (torch.Tensor) 模型的输入（英译法任务中输入的英文句子），shape=(batch，ntokens)
        :param trg: (torch.Tensor) 模型的输出（ground truth，英译法任务中标准答案的法语句子），训练时可指定，预测时为None。
        :param pad: (int) padding的填充值？但是前面构造数据的时候明明在每个sample的第0个token填充的1，
        可能1不是padding，但是LabelSmoothing类实例化criterion时padding_idx参数赋值是0，而数据第0个位置就是1，如何解释？
        greedy_decode函数传输参数start_symbol=1，所以0和1到底谁是padding？
        """
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
        tgt_mask = (tgt != pad).unsqueeze(-2)  # 隐藏padding
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        # padding mask 和 sequence mask 联合，此处&操作尺寸不匹配可以广播
        return tgt_mask
        # 训练模式tgt_mask需要两部分相与，padding mask和sequence mask；
        # 若是测试模式，sequence mask是不需要的，而padding mask貌似也不需要？【待确定】
        # 我理解的是这样，但是该版本代码并不是这样执行的，具体可见attention函数中有关mask的使用部分的说明

# Training Loop
def run_epoch(data_iter, model, loss_compute):
    r"""

    Standard Training and Logging Function

    :param data_iter: (generator) 数据生成迭代器
    :param model: (EncoderDecoder) Transformer 模型
    :param loss_compute: (SimpleLossCompute) loss计算，反向传播，优化器更新参数等
    :return: (torch.Tensor) 当前epoch平均预测每个token的loss
    """

    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):  # 为何没有参数更新的操作？
        out = model.forward(batch.src, batch.trg,
                            batch.src_mask, batch.trg_mask)
        # out是decoder的输出，不是最终的预测结果啊，后续在SimpleLossCompute中输出预测，具体其实是利用Generator层输出对应tgt_vocab的概率
        loss = loss_compute(out, batch.trg_y, batch.ntokens)  # 该batch全部token的loss值
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            # 输出当前epoch的step(batch)序号，当前step(batch)平均预测每个token的loss，单位时间平均预测的token数量
            start = time.time()  # 计时器归零
            tokens = 0  # token计数器归零
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
        r"""

        :param size:
        :param padding_idx:
        :param smoothing:
        """
        super(LabelSmoothing, self).__init__()
        # self.criterion = nn.KLDivLoss(size_average=False)
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        r"""

        :param x: (torch.Tensor) Generator生成的log_softmax概率结果，shape=(batch_size*tgt_ntokens, tgt_vocab)
        :param target:(torch.Tensor) ground truth，具体是正确答案在tgt_vocab中的index，shape=(batch_size*tgt_ntokens)
        :return: (torch.Tensor) 预测值x与label Smoothing后的target值之间的KLDivLoss损失
        """
        assert x.size(1) == self.size
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
        # 【0407更新】最新的理解是，smooth均分的时候减去的分别是 正确答案自己+padding
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
    r"""

    "Generate random data for a src-tgt copy task."

    :param V: (int) 词汇表的size（词的数量）
    :param batch:  (int) 每个batch中sample（sentence）的数量
    :param nbatches: (int)batch的数量
    :return:
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
    "generator生成预测值，criterion loss计算，反向传播，优化器opt更新参数"

    def __init__(self, generator, criterion, opt=None):
        r"""

        :param generator: (Generator) Linear+log_softmax，将decoder的输出转换成预测结果
        :param criterion: (LabelSmoothing) 标签平滑，计算KLDivLoss
        :param opt: (NoamOpt) 优化器，计算学习率
        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        r"""
        generator生成预测值，criterion loss计算，反向传播，优化器opt更新参数
        :param x: (torch.Tensor) 模型decoder的输出，shape=(batch_size, tgt_ntokens, d_model)
        :param y: (torch.Tensor) 模型输出的ground truth，shape=(batch_size, tgt_ntokens)，元素y[i,j]代表的是正确答案在tgt_vocab中的index
        :param norm: (torch.Tensor) 这里传输进来的是该batch预测的token的总数量，即 (batch_size * tgt_ntokens)
        :return: (torch.Tensor)  返回该batch的全部token（即为batch_size * tgt_ntokens个token）预测的loss值
        """
        x = self.generator(x)
        # 经过Generator层后shape=(batch_size, tgt_ntokens, tgt_vocab)，元素x[i,j]是长度为tgt_vocab的向量，
        # 其每一维度代表对应tgt token的概率，一般选取最高的作为预测值
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        # self.criterion是定义的loss函数，调用其forward方法，计算x和y之间具体的loss值，除以norm是预测的数量，得到平均每个token的loss
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # return loss.data[0] * norm
        return loss.data * norm  # 上面除以norm，这里又乘以norm，是为何？


# Greedy Decoding
# Train the simple copy task.
V = 11  # 这里是src_vocab=tgt_vocab=V
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)  # 构建loss函数
model = make_model(V, V, N=2)  # 构建模型
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))  # 构建优化器

# 开始训练，每个epoch包含{20 batch的训练epoch和5 batch的测试epoch}
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
    for i in range(max_len):
        out = model.decode(memory, src_mask,
                           Variable(ys),
                           Variable(subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        # Important！ 预测就是利用的输入的最后一个词作为q所对应的输出，那其实前面词的注意力得分、权重、加权和本没不必要计算啊？
        # 其实不是的，因为N=6，每一层要在上一层的基础上进行attend计算，如果说没必要的话，
        # 只能在最后一层（最后一个DecoderLayer的第一个sublayer的MultiHeadAttention中只计算最后一个词作为q的加权和，
        # 第二个子层的一般注意力是attend的memory，不受影响，或许过一些线性层本身就是并列通过的也没有影响）的计算中予以省略，
        # 但是这样增加了算法复杂度实在没必要了
        _, next_word = torch.max(prob, dim = 1)  #返回的是（最大的值，最大值的index）
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys
# 模型训练完成，开始测试评估
model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]))
src_mask = Variable(torch.ones(1, 1, 10))
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))

# A First Example END----------------------------------------
