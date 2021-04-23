import time
import numpy as np
import tensorflow as tf
import random
from collections import Counter

with open('/Users/shengliwang/zhihu/skip_gram/data/Javasplittedwords') as f:
    text = f.read()


def preprocess(text, freq=5):
    text = text.lower()
    text = text.replace('.', '<PERIOD>')
    text = text.replace(',', '<COMMA>')
    text = text.replace('"', '<QUOTATION_MARK>')
    text = text.replace(';', '<SEMICOLON>')
    text = text.replace('!', '<EXCLAMATION_MARK>')
    text = text.replace('?', '<QUESTION_MARK>')
    text = text.replace('(', '<LEFT_PAREN>')
    text = text.replace(')', '<RIGHT_PAREN>')
    text = text.replace('--', '<HYPHENS>')
    text = text.replace(':', '<COLON>')
    words = text.split(' ')
    return words


words = preprocess(text)
words_count = Counter(words)
words = [w for w in words if words_count[w] > 50]
# vocab 是整个非重复词的词表，任何词的id对应的词表都要基于这个。
vocab = set(words)
# 下面这个才是整儿八经带id的词表
vocab_to_int = {w: c for c, w in enumerate(vocab)}
int_to_vocab = {c: w for c, w in enumerate(vocab)}

print("total words: {}".format(len(words)))
print("unique words: {}".format(len(set(words))))

# 对原文本进行vocab到int的转换   包含重复词，所以是一个包含重复词id的一个list
int_words = [vocab_to_int[w] for w in words]

t = 1e-5 # t值
threshold = 0.9 # 剔除概率阈值

# 统计单词出现频次
int_word_counts = Counter(int_words)
total_count = len(int_words)
# 计算单词频率
word_freqs = {w: c/total_count for w, c in int_word_counts.items()}
# 计算被删除的概率
prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
# 对单词进行去一些高频词  然后保留剩下的词   但是有重复  同时存储的是id
train_words = [w for w in int_words if prob_drop[w] < threshold]

print(len(train_words))


def get_targets(words, idx, window_size=5):
    '''
    获得input word的上下文单词列表

    参数
    ---
    words: 单词列表
    idx: input word的索引号
    window_size: 窗口大小
    '''
    target_window = np.random.randint(1, window_size + 1)
    # 这里要考虑input word前面单词不够的情况
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    # output words(即窗口中的上下文单词)
    # 看了数据集我明白了，为什么取窗口的词是这样子取的，那是因为在数据集里面相邻的词是在一起，是上下文的词。当然不同文本之间是
    # 词有衔接和切换的，但是毕竟数量不多，所以可以采用这种方式进行处理。
    targets = set(words[start_point: idx] + words[idx + 1: end_point + 1])
    return list(targets)


def get_batches(words, batch_size, window_size=5):
    '''
    构造一个获取batch的生成器
    '''
    n_batches = len(words) // batch_size

    # 仅取full batches
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx: idx + batch_size]
        # 通过这种方式，不存在B的维度，因为全部的数据放在了一起。序列长度就是S，x和y都是。
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_targets(batch, i, window_size)
            # 由于一个input word会对应多个output word，因此需要长度统一
            x.extend([batch_x] * len(batch_y))
            y.extend(batch_y)
        yield x, y


# 模型定义就是从下面这句开始的，凡是模型内部的部分，都需要放到model.as_default()下面存储。
train_graph = tf.Graph()
with train_graph.as_default():
    # 占位符  作为数据初始化入口  模型这里只定义不初始化  初始化是在see那里通过feed进行初始化。
    inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
    labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')


vocab_size = len(int_to_vocab)
embedding_size = 200  # 嵌入维度

with train_graph.as_default():
    # 嵌入层权重矩阵  6846 * 200
    embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    print('enbedding shape: ', embedding.shape)
    # 实现lookup   ? * 200
    embed = tf.nn.embedding_lookup(embedding, inputs)
    print('enbedding shape: ', embed.shape)

n_sampled = 100

with train_graph.as_default():
    softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
    softmax_b = tf.Variable(tf.zeros(vocab_size))

    # 计算negative sampling下的损失
    loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)

    cost = tf.reduce_mean(loss)
    optimizer = tf.train.AdamOptimizer().minimize(cost)


with train_graph.as_default():
    # 随机挑选一些单词
    ## From Thushan Ganegedara's implementation
    valid_size = 7  # Random set of words to evaluate similarity on.
    valid_window = 100
    # pick 8 samples from (0,100) and (1000,1100) each ranges. lower id implies more frequent
    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(valid_examples,
                               random.sample(range(1000, 1000 + valid_window), valid_size // 2))
    valid_examples = [vocab_to_int['word'],
                      vocab_to_int['ppt'],
                      vocab_to_int['熟悉'],
                      vocab_to_int['java'],
                      vocab_to_int['能力'],
                      vocab_to_int['逻辑思维'],
                      vocab_to_int['了解']]

    valid_size = len(valid_examples)
    # 验证单词集
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 计算每个词向量的模并进行单位化
    norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
    normalized_embedding = embedding / norm
    # 查找验证单词的词向量
    valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
    # 计算余弦相似度
    similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))

epochs = 10  # 迭代轮数
batch_size = 1000  # batch大小
window_size = 10  # 窗口大小

with train_graph.as_default():
    saver = tf.train.Saver()  # 文件存储

with tf.Session(graph=train_graph) as sess:
    iteration = 1
    loss = 0
    sess.run(tf.global_variables_initializer())

    for e in range(1, epochs + 1):
        batches = get_batches(train_words, batch_size, window_size)
        start = time.time()
        #
        for x, y in batches:
            # 通过下面这一步就转化为了Tensor，由原始的List以及numpy转化为了Tensor，转化的原因就是提前在最开始定义了inputs以及labels的占位符。
            feed = {inputs: x,
                    labels: np.array(y)[:, None]}
            train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

            loss += train_loss

            if iteration % 100 == 0:
                end = time.time()
                print("Epoch {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Avg. Training loss: {:.4f}".format(loss / 100),
                      "{:.4f} sec/batch".format((end - start) / 100))
                loss = 0
                start = time.time()

            # 计算相似的词
            if iteration % 1000 == 0:
                # 计算similarity
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_vocab[valid_examples[i]]
                    top_k = 8  # 取最相似单词的前8个
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log = 'Nearest to [%s]:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_vocab[nearest[k]]
                        log = '%s %s,' % (log, close_word)
                    print(log)

            iteration += 1

    save_path = saver.save(sess, "checkpoints/text8.ckpt")
    embed_mat = sess.run(normalized_embedding)


# https://blog.csdn.net/zyq11223/article/details/90302186   图
# 采样损失函数  https://zhuanlan.zhihu.com/p/141421400  讲解的非常详细，特别好
# 学到的东西：tensorflow1.0搭建模型；  同时skip-gram的代码以及图像对上了；  通过skip-gram学习到了什么，学习带了最原始的那个大矩阵。