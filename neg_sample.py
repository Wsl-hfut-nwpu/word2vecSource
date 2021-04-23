if __name__ == "__main__":
    # 依据频率得到List的len表
    pinlv = [0.8, 1.3, 0.6, 1.5, 2.7, 1.6, 3.5, 3, 3]

    def negSample(pinlv, M=6):
        # 依据频率表得到求和分区sum_qu
        if not pinlv:
            return
        val = 0
        sum_qu = [0]
        for i in pinlv:
            val += i
            sum_qu.append(val)
        print(sum_qu)
        # M 是分的段数
        Di = 1
        Qi = 1
        dieta = sum_qu[-1] // M
        cur = dieta
        Table = {}
        while cur <= sum_qu[-1]:
            while sum_qu[Qi-1] < cur and cur <= sum_qu[Qi]:
                Table[Di] = Qi
                Di += 1
                cur += dieta
            Qi += 1
        print(Table)
        return Table
    # 听过sample利用一次循环构造出了依据频率的映射表，后面只要在1-M随机生成一个随机数，就可以打点了。
    Table = negSample([])
    print(Table[3])

# https://zhuanlan.zhihu.com/p/153502072  代码和这个链接里面的sample对应

# 上述负采样表的构建是自己构建的，那么通过阅读源码可以看到C++表的构建方法，其实和上面的大同小异吧。

# // 按论文的描述，生成采样序列
# // 1, 划分长度为1的线段为table_size（table_size应远大于词典大小）；   知道原因
# // 2, 按0.75的幂经验值，依据词频，将每个单词分配至若干段上（词频越大，分配到的段越多），则随机取若干个段，则必能取到若干个词；
# // 3, 幂0.75的一个解释：假设，is在语料库的词频是0.9，Constitution是0.09，bombastic是0.01，其各自的3/4次幂分别是0.92，0.16和0.032；
# // 4, 因此，使用3/4次幂负采样到bombastic的概率相比之前增大3倍。即，在词频越大，分配到的段阅读的基础上，低频词更容易被采样到。
# void InitUnigramTable() {
#   int a, i;
#   double train_words_pow = 0;
#   double d1, power = 0.75;
#   table = (int *)malloc(table_size * sizeof(int));
#   // 整个词典的词频大小
#   for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
#   i = 0;
#   // 词典第i个单词的词频大小
#   d1 = pow(vocab[i].cn, power) / train_words_pow;
#   // 分段打点，将一个单词打点到多个段上
#   for (a = 0; a < table_size; a++) {
#     table[a] = i;
#     if (a / (double)table_size > d1) {
#       i++;
#       d1 += pow(vocab[i].cn, power) / train_words_pow;
#     }
#     if (i >= vocab_size) i = vocab_size - 1;  // 就是需要注意最后这一步，感觉是没必要  但是还没认真对的确认过。
#   }
# }


