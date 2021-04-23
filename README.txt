
补充：上述代码只有word2vec.c是有效文件，可以在CLion中直接运行，即可获取到exe文件，即可进行模型训练。

1.上述代码需要编译，因为是c文件，所以编译后展现为exe文件才可以运行。编译最好使用CLion，是JetBrains全家桶系列，用的比较舒服。因此后期更改完成之后，可以进行编译后再次运行。
2.对于代码的负采样公式的解读：https://blog.csdn.net/google19890102/article/details/51887344
3.下面对于源码进行解读：

//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include <pthread.h>
#include <thread>
// 单词的最大字节长度
#define MAX_STRING 100
// exp查找表长度
#define EXP_TABLE_SIZE 1000
// exp幂的上下限，控制权重向量和词向量的点积大小。
// sigmoid(x)的x定义域理论上是无穷，但因为x在[-6，6)时，y已极度接近(0,1)，故简化计算，用MAX_EXP取近似
#define MAX_EXP 6
// 最大句子的单词个数
#define MAX_SENTENCE_LENGTH 1000
// huffman树的最大高度
#define MAX_CODE_LENGTH 40
// 哈希散列表的大小：语料词典使用hash索引
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  // 单词在语料中的词频
  long long cn;
  // point：根节点到叶子节点的索引（只计偏移）路径（从根节点开始: 下标：[0,codelen])
  int *point;
  // word：文本；
  // code：根节点到叶子节点的huffman编码（左子树0，右子树1）；(不计根节点：下标：[0, codelen-1])
  // codelen：word在huffman数的深度，即point和code的最大下标
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
// min_count：读取词典时，首先抛弃的词频下限
// min_reduce：语料单次数超过上限时（21M），缩减语料单词的阈值，每缩减一次则递增
int binary = 0, cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
// vocab_max_size：训练语料词典最大数目，超出会自动扩展；vocab_size：当前词典大小；layer1_size：词向量维数
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
// 语料词典的内存形态，大小为vocab_size，最大可容纳vocab_max_size
struct vocab_word *vocab;
// 语料词典的hash散列表，大小为vocab_hash_size
int *vocab_hash;
// 训练语料的单词个数，
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0, classes = 0;
// alpha：学习率
real alpha = 0.025, starting_alpha, sample = 1e-3;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

// 按论文的描述，生成采样序列
// 1, 划分长度为1的线段为table_size（table_size应远大于词典大小，原因是这样就可以在一个采样段内打更多的点，采样性能得到保障，或个图就明白了，如图一）；
// 2, 按0.75的幂经验值，依据词频，将每个单词分配至若干段上（词频越大，分配到的段越多），则随机取若干个段，则必能取到若干个词；
// 3, 幂0.75的一个解释：假设，is在语料库的词频是0.9，Constitution是0.09，bombastic是0.01，其各自的3/4次幂分别是0.92，0.16和0.032；
// 4, 因此，使用3/4次幂负采样到bombastic的概率相比之前增大3倍。即，在词频越大，分配到的段阅读的基础上，低频词更容易被采样到。
void InitUnigramTable() {   // 这个采样方法在文件tf_skip_gram内部采用python进行了实现，可以对照着c的版本进行研究，面试常问，又称  齐夫采样。
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  // 整个词典的词频大小
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  // 词典第i个单词的词频大小
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  // 分段打点，将一个单词打点到多个段上
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {   // 进行了归一化之后映射到 0-1空间进行比较  构造打点表  然后进行取样映射
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;  
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// 挨个读取文件内包含的单词：默认空格+tab+行结尾符(\n)是单词分界
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);    // 在mac里面这种函数的f去掉才能运行的=通，函数变成了  getc，表示从fin文件流里面获取到一个字符
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      // 单词后有分隔符，取出单词，回退换行
	  if (a > 0) {
		// 最后一个字符是换行符号，回退一格
        if (ch == '\n') ungetc(ch, fin);    // 将字符回退到流中，或者说将字符回退到文件中，同时当前文件的指针回退一个字符
        break;
      }
	  // 如果仅为换行，填充</s>
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
	// 截断超过MAX_STRING个字节的单词   好像没有必要，但是还没有看出来原因
    if (a >= MAX_STRING - 1) a--;
  }
  word[a] = 0;
}

// 计算单词的hash函数
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// 返回单词所在的词典位置，若没找到，返回-1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
	// hash->下标->单词->位置
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// 挨个单词读取语料，并返回其在词典中的位置
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// 将训练语料的词典计算hash后，将hash值存入词典
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  // word被载入vocab_size[vocab]
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // 如果词典超出最大数目，自动扩展后，realloc拷贝内存
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  // word的下标vocab_size，通过vocab_hash表来索引
  // word计算出hash，hash索引vocab_hash得到下标vocab_size，vocab通过下标得到word，闭环
  hash = GetWordHash(word);
  // hash冲突，后移一位：线性探测
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  // 构建word的hash和下标vocab_size的对应关系
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// 以词频计数排序词典
void SortVocab() {
  int a, size;
  unsigned int hash;
  // 起始</s>位置不变，正常单词按词频由大到小排列
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  // 抛弃低频词的同时，重新计算hash值索引
  for (a = 0; a < size; a++) {
    // 删除语料种词频小于min_count的单词
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // 排序和缩减词典后，重新计算hash索引
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  // 紧凑内存
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // 单词对应huffman树的编码和指针
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// 通过移除低频词，来缩小词典（词典最大vocab_hash_size * 0.7）
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  // 缩减后的词典大小
  vocab_size = b;
  // 重新计算hash值索引
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  // 每一轮缩减后，升级词频下限，等待下一次缩减
  min_reduce++;
}

// 使用词频创建二叉huffman树，高频词拥有更短的二进制huffman编码
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  // count：整个huffman树的词频（真正的节点数：vocab_size * 2 - 1，因此，根节点的下标：vocab_size * 2 - 2）
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  // binary：二进制class，左子树=0负类，右子树=1正类
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  // parent_node：存储叶子节点（真实词频）->汇总节点 之间下标映射，以count的下标为索引
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  // 因为vocab已由大到小排序，所以从后开始倒数，一定是由小到大
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // 挨个节点构建huffman树
  for (a = 0; a < vocab_size - 1; a++) {
    // 首先，'min1i, min2i'：最小词频节点下标
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // 接着，将构建好的huffman树的结构赋值到词典元素
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
	  // vocab[a].word的huffman编码（二进制串）：从叶子节点vocab[a]开始到根节点
      code[i] = binary[b];
	  // vocab[a].word的根节点到叶子节点的索引路径：从叶子节点vocab[a]开始到根节点
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
	// vocab[a]在huffman树中的深度
    vocab[a].codelen = i;
	// point表示父节点下标对词典大小的偏移
    vocab[a].point[0] = vocab_size - 2;
	// 逆序赋值：由根节点到叶子节点的编码和路径
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  // 在词典首位置新增</s>
  AddWordToVocab((char *)"</s>");
  while (1) {
	 memset(word, 0x00, MAX_STRING);
	// 读取单词
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
	// 检索单词位置
    i = SearchVocab(word);
	// 加入词典，更新数据
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
	// 如果训练语料太大超限，从内存词典中移除低频词
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  // 词频由大到小排序词典
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}


// int posix_memalign(void **memptr,size_t alignment,size_t size);
// allocate 1 KB along a 256-byte boundary
// ret = posix_memalign(&buf, 256, 1024);

// syn0：词向量矩阵   
// syn1：层次softmax的权重矩阵
// syn1neg：负采样的权重矩阵

// 对word2vec而言，选择一种模型后（cbow/skip-gram），对任意训练方法（hs/neg），有且只有一个权重矩阵
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;
  // 按每个词向量layer1_size维，元素用float表示，分配vocab_size个单词的字节空间。syn0 = 内存首地址 = 能被128字节整除的最小地址，即128字节对齐
  //a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  syn0 = (real*)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  // 如果使用层次softmax，相同方式，继续分配，得到syn1，全局初始化为0
  // syn1：层次softmax模型中的参数矩阵
  if (hs) {
    //a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
	syn1 = (real*)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1[a * layer1_size + b] = 0;
  }
  // 如果使用负采样，相同方式，继续分配，得到syn1，全局初始化为0
  // syn1neg：负采样模型中的参数矩阵
  if (negative>0) {
    //a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
	syn1neg = (real*)_aligned_malloc((long long)vocab_size * layer1_size * sizeof(real), 128);
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
     syn1neg[a * layer1_size + b] = 0;
  }
  // 初始化词向量矩阵（行:单词；列：元素，即，词向量用行向量表示）
  for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
	// 一种随机算法，一次初始化，在next_random的生命周期内，可持续产生随机数
    next_random = next_random * (unsigned long long)25214903917 + 11;
	// k1 = (next_random & 0xFFFF)：保留next_random的低16位（最大65535）；
	// k2 = k1/65536：[0,1)：（是否可以闭区间？）
	// k3 = (k2-0.5) / layer1_size： （[-0.5,0.5) / 词向量维数）
    syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
  }
  // 创建huffman树
  CreateBinaryTree();
}
void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  // layer1_size是词向量的维数
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  // 多线程训练，等分每个线程训练的语料
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  // 一个while，三种用途：
  // 1，一个窗口的训练，完成一个窗口后，继续下一个窗口
  // 1, 一个句子的训练，完成一个句子后，继续下一个句子
  // 2，一次迭代的训练，完成一次迭代后，继续下一次迭代
  while (1) {
	// 每完成一次迭代，训练步长逐渐减小，防止一开始设置的大步长，导致收敛时，跳过最低点，直至震荡。
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
	  // 更新步长
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
	  // 兜底，防止过小
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
	// 当完成一个句子的计算后，取出顺序语料中，经过缩减（读取词典时）和采样（开始训练前）后的单词，在词典中的下标序列
	// 注：缩减 和 采样，可以理解为对语料的预处理，毕竟词频太高或太低都不是典型语境用法
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
		// 语料单词不在词典中，跳过。说明该单词词频太低，在前期已被删除，或者，词典超限，被强制缩减
        if (word == -1) continue;
		// 一个句子的实际单词数（包括\n）
        word_count++;
		// 位置0是写死的单词</s>，表明已读取完一个句子（按\n划分）
        if (word == 0) break;
		// 对一个自然句，在保证顺序不变的基础上，随机丢弃高频词(相对整个语料库来说)
        if (sample > 0) {
		  // 词频cn越高，ran越小，在随机k2的情况下，越高的词频越有可能被跳过；
		  // sample越大，k2越大，在给定ran的情况下，较低的词频越有可能被过滤。
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
		  // 一种随机算法，一次初始化，在next_random的生命周期内，可持续产生随机数
          next_random = next_random * (unsigned long long)25214903917 + 11;
		  // k1 = (next_random & 0xFFFF)：保留next_random的低16位（最大65535）；
		  // k2 = k1/65536：[0,1)：（是否可以闭区间?）
		  // (ran < k2)时，跳过。
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
		// 保存一个句子，过滤低频词，跳过高频词后，的下标，最长不超过MAX_SENTENCE_LENGTH，否则截断
        sen[sentence_length] = word;
		// 一个句子的训练单词数
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
	  // 新读取一个句子，开始训练
      sentence_position = 0;
    }
	// 语料读完/读取单词数超过分配单词数，完成一次迭代，继续训练
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
	// 如果是新句子，从0开始。如果已经在一个句子中，则为对应中心词在词典内的位置
    word = sen[sentence_position];
    if (word == -1) continue;
	// 隐含层：cbow是窗口均值，skip-gram是其本身。隐含层节点与权重向量的乘积，作为隐含层激活函数的输入
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
	// 隐含层误差梯度：从输出层，经由隐含层，传递到输入层的，词向量的更新梯度
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
	// 参考文档：
	// 1, https://www.slideshare.net/lewuathe/drawing-word2vec
	// 2, http://www.cnblogs.com/pinard/p/7249903.html
	// 3, http://www.cnblogs.com/pinard/p/7243513.html
	// cbow模型训练：以窗口内的单词预测中心词
    if (cbow) {  
      // 输入层->隐含层
      cw = 0;
	  // 所以word2vec源码实现，并不是死板的看前后各window个单词，而是在window内随机训练一个子窗口，加速迭代？
	  for (a = b; a < window * 2 + 1 - b; a++)
	  {
		  // a==window时，表示窗口的中心位置，即中心词本身。
		  // a表示窗口序，以5为中心
		  if (a != window) 
		  {
			  // 窗口下标 -> 句子内的单词下标
			  c = sentence_position - window + a;
			  if (c < 0) continue;
			  if (c >= sentence_length) continue;
			  // 句子内的单词在词典中的下标
			  last_word = sen[c];
			  if (last_word == -1) continue;
			  // 以中心词为基准，加总窗口内单词的词向量
			  for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
			  cw++;
		  }
	  }
	  // 损失函数：对数似然函数：Loss=sum(ylogp(x)+(1-y)*log(1-p(x)))；
	  // 其中：
	  // 1，sum是元素求和 -> hs：根节点到叶子节点路径上的元素，高频词浅，低频词深。negtive：1个中心词 + neg个采样负例
	  // 2，y是lable -> hs：左右子树的label。negtive：正负例的lable
	  // 3，p(x)是二分类逻辑回归sigmoid(x·w)

	  // 隐含层->输出层，根据输入算梯度(cw:真正的窗口大小，并不是参数window，而是受限于b)
      if (cw) {
		// 隐含层neu1：cbow中的隐含层求均值操作
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
		// 层次softmax训练
		// 基本原理：
	    // 1，huffman树内部节点为简化的隐含层，每个节点都被划分为正负样例（binary:0,1），并用逻辑回归来训练，即权重矩阵syn1；
		// 2，如果我们期望语料中的单词，是模型的正常输出，则目标是从根节点开始，直到该叶子节点间的路径上，各内部节点属于各自类别的概率最大；
		if (hs)
		{
			for (d = 0; d < vocab[word].codelen; d++)
			{
				f = 0;
				// huffman树，内部路径上的词向量，权重参数入口（从根节点到叶子节点，但point记的是偏移，没问题？）
				l2 = vocab[word].point[d] * layer1_size;
				// 隐含层->输出层
				for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
				// 下面2行f应该有问题，除非对收敛结构无影响？ sigmoid在定义域上下限的取值
				if (f <= -MAX_EXP) continue;	// f = 0
				else if (f >= MAX_EXP) continue;// f = 1
				// sigmoid逆运算查表
				else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				// f：模型中的z=x1*w1+x2*w2+....；
				// g：误差率，由层次softmax模型推导而来，代码中直接使用结论，g*syn1和g*neu1[c]才分别构成中心词和权重参数的真正梯度
				g = (1 - vocab[word].code[d] - f) * alpha;
				// 1，这里仅仅是汇总neu1的梯度而已，因为neu1的梯度是一个和式
				for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
				// 2，但是，这里是对每一个采样的样本，直接梯度下降，更新参数
				for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
			}

			// neu1真正的梯度，在此时才汇总产生
		}
			
        // 负采样训练：word2vec源码和cs224n的介绍不太一样，虽然思想一致
		// cs224n：中心词不变，随机采样窗口内的其他词，组成负样本
		// wor2vec源码：窗口不变，随机采样中心词
		// 基本原理：语料记为正样本，随机采样记为负样本。构建损失函数，在最大化正样本概率的同时，要最小化负样本概率。
		if (negative > 0)
		{
			// 对一个中心词对应的一个窗口 -> 1个正例样本：d=0 和 negative个负例样本：d=1...neg，构建对数似然函数，进行负采样
			for (d = 0; d < negative + 1; d++) 
			{
				// label=1，标识正样本，word即为正样本中心词（原始语料）
				if (d == 0) {
					target = word;
					label = 1;
				}
				// label=0，标识负样本，target为负样本中心词（随机采样）
				else {
					// 使用InitUnigramTable后的码表随机采样中心词
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % table_size];
					if (target == 0) target = next_random % (vocab_size - 1) + 1;
					if (target == word) continue;
					label = 0;
				}
				// 单词target对应的权重矩阵入口
				f = 0;
				l2 = target * layer1_size;
				// syn1neg：模型参数，即，神经网络中同样需要迭代的权重向量w；
				// alpha：学习率；
				// label：正负样本，即模型中的yi；

				// f：模型中的z=x1*w1+x2*w2+....；
				// g：误差率，由负采样模型推导而来，代码中直接使用结论，g*syn1neg和g*neu1[c]才分别构成中心词和权重参数的真正梯度
				
				// 隐含层->输出层
				for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
				// f超限，直接取sigmoid的上下限值，即1和0
				if (f > MAX_EXP) g = (label - 1) * alpha;
				else if (f < -MAX_EXP) g = (label - 0) * alpha;
				// 即，首先逆运算 f=[e^-6,e^6) ->i，然后查表expTable[i]得到sigmoid值
				else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
				

				// attention！！！ 衷心建议！！！踩坑血泣！！！
				// 在这里看再多的文档，都比不上自己在纸上用举例的方式，手推一遍！！！
				// neu1*(d == 0):正例；neu1*(d != 0):负例

				// 千万不要被下面2个循环的长相蒙骗了，一个是计算梯度，而另一个是梯度下降更新参数！

				// 1，这里仅仅是汇总neu1的梯度而已，因为neu1的梯度是一个和式
				for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
				// 2，但是，这里是对每一个采样的样本，直接梯度下降，更新参数
				for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
			}

			// neu1真正的梯度，在此时才汇总产生
		}
			
        // 隐含层->输入层，根据梯度，反向传播
		// cbow的原理是，根据周围的词预测中心词，在此基础上构建损失函数，因此，反向传播后更新的是周围的词
		for (a = b; a < window * 2 + 1 - b; a++)
		{
			if (a != window)
			{
				c = sentence_position - window + a;
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = sen[c];
				if (last_word == -1) continue;
				// 反向传播，根据隐含层传递的梯度，迭代更新输入窗口的词向量
				for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
			}
		}
      }
    } 
	// skip-gram模型
	else {
		// 每次迭代的每个窗口，都将窗口内的非中心词都更新一把，这和原始skip-gram理论，只更新中心词不同？可能是一个变种，加快迭代速度？
		// 一个可能的解释：如果期望p(o/c)最大，那么是否可以反过来，期望p(c/o)最大呢？其中：o是窗口词，c是中心词。
		// 此时，代码上统一cbow和skip-gram的同时，一次窗口迭代可更新多个词向量，加快速度。
		// 唯一的区别是，cbow隐含层是窗口词均值，而skip-gram的隐含层是窗口词本身。
		for (a = b; a < window * 2 + 1 - b; a++)
		{
			if (a != window)
			{
				c = sentence_position - window + a;
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = sen[c];
				if (last_word == -1) continue;
				l1 = last_word * layer1_size;
				for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
				// HIERARCHICAL SOFTMAX
				if (hs)
				{
					for (d = 0; d < vocab[word].codelen; d++)
					{
						f = 0;
						l2 = vocab[word].point[d] * layer1_size;
						// Propagate hidden -> output
						for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
						if (f <= -MAX_EXP) continue;
						else if (f >= MAX_EXP) continue;
						else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
						// 'g' is the gradient multiplied by the learning rate
						g = (1 - vocab[word].code[d] - f) * alpha;
						// Propagate errors output -> hidden
						for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
						// Learn weights hidden -> output
						for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
					}
				}
					
				// NEGATIVE SAMPLING
				if (negative > 0)
				{
					for (d = 0; d < negative + 1; d++)
					{
						if (d == 0) {
							target = word;
							label = 1;
						}
						else {
							next_random = next_random * (unsigned long long)25214903917 + 11;
							target = table[(next_random >> 16) % table_size];
							if (target == 0) target = next_random % (vocab_size - 1) + 1;
							if (target == word) continue;
							label = 0;
						}
						l2 = target * layer1_size;
						f = 0;
						for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
						if (f > MAX_EXP) g = (label - 1) * alpha;
						else if (f < -MAX_EXP) g = (label - 0) * alpha;
						else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
						for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
						for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
					}
				}
					
				// Learn weights input -> hidden
				for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
			}
		}
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  //pthread_exit(NULL);
  return NULL;
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  //pthread_t* pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  std::thread* pt = new std::thread[num_threads];

  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pt[a] = std::thread(TrainModelThread, (void *)a); //pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pt[a].join(); //pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  // 词向量训练完成，如果不需聚类，直接保存原始词向量
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      fprintf(fo, "\n");
    }
  // 如果需要聚类，则根据单词词向量，按k均值聚类单词
  } else {
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
        centcn[cl[c]]++;
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  /*
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
    return 0;
  }
 
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  */

  // test params
  layer1_size = 200;
  strcpy(train_file, "F:/linux/TafProject/JustTest/JustTest/word2vec/english.txt");
  cbow = 1;
  strcpy(output_file, "F:/linux/TafProject/JustTest/JustTest/word2vec/english_word2vec.txt");
  window = 5;
  sample = 1e-4;
  hs = 0;
  num_threads = 1;
  iter = 3;
  min_count = 0;
  window = 2;

  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  // 初始化和预先计算exp查找表，并预先计算exp/(1+exp)，后期只需要根据x刻度和增减正负号，即可完成sigmoid函数的梯度计算，加速迭代
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
	// 预计算exp: 
	// k = i / (real)EXP_TABLE_SIZE * 2 - 1 的值域是[-1,1)；
    // e = exp(k*MAX_EXP) 的值域是[e^-6,e^6)，划分为EXP_TABLE_SIZE等分。即sigmoid函数的近似有效定义域
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    // 预计算exp/(1+exp)。即，sigmoid函数
	expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
