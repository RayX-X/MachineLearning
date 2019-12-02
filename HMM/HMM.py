# -*- coding:utf-8 -*-
# @Time    : 2019/12/1 10:47
# @Author  : Ray.X
"""
具体到分词系统，可以将天气当成“标签”，活动当成“字或词”。

词性标注：给定一个词的序列（也就是句子），找出最可能的词性序列（标签是词性）。如ansj分词和ICTCLAS分词等。

分词：给定一个字的序列，找出最可能的标签序列（断句符号：[词尾]或[非词尾]构成的序列）。
    结巴分词目前就是利用BMES标签来分词的，B（开头）,M（中间),E(结尾),S(独立成词）

命名实体识别：给定一个词的序列，找出最可能的标签序列（内外符号：[内]表示词属于命名实体，[外]表示不属于）。
    如ICTCLAS实现的人名识别、翻译人名识别、地名识别都是用同一个Tagger实现的。
"""


# 打印路径概率表
def print_dptable(V):
    print("    ", end=' ')
    for i in range(len(V)):
        print("%7d" % i, end=' ')
    print()

    for y in list(V[0].keys()):
        print("%.5s: " % y, end=' ')
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][y]), end=' ')
        print()


class HMM:
    def __init__(self, obs, states, start_p, trans_p, emit_p):
        """
        :param obs:     观测序列
        :param states:  隐含状态
        :param start_p: 初始状态
        :param trans_p: 转移概率
        :param emit_p:  发射概率
        :return:
        """
        self.obs = obs
        self.states = states
        self.start_p = start_p
        self.trans_p = trans_p
        self.emit_p = emit_p

    def viterbi(self):
        """
        一种动态规划法，解决给定一个逆向各某个特定的输出序列，求取最可能上传这个输出的隐序列
        应用于语音识别、机器翻译、拼音转汉字、分词等
        :return:
        """
        # 路径概率表 V[时间][隐状态] = 概率
        V = [{}]
        # 一个中间变量，代表当前状态是哪个隐状态
        path = {}

        # 初始化初始状态 (t == 0)
        for y in self.states:
            V[0][y] = self.start_p[y] * self.emit_p[y][self.obs[0]]
            path[y] = [y]

        # 对t>1的所有节点计算viterbi
        for t in range(1, len(self.obs)):
            V.append({})
            print(self.obs[t])
            newpath = {}

            for y in self.states:
                # (最大概率， 对应隐状态)= max(前隐状态是y0的概率 * y0转移到y的概率 * y表现为当前显状态的概率), y0 留下这天是y的最大概率
                (prob, state) = max([(V[t - 1][y0] * self.trans_p[y0][y] * self.emit_p[y][self.obs[t]], y0) for y0 in self.states])
                # 记录最大概率
                V[t][y] = prob
                # 记录路径
                newpath[y] = path[state] + [y]
            # 不需要保留旧路径
            path = newpath

        print_dptable(V)
        # 计算最大概率，及最优路径
        (prob, state) = max([(V[len(self.obs) - 1][y], y) for y in self.states])
        return prob, path[state]


if __name__ == '__main__':
    """
    发推 “啊，我前天公园散步、昨天购物、今天清理房间了！”，推断这三天的天气 rainy OR sunny
    """
    observation = ["walk", "shop", "clean"]
    Hiddenstates = ["rainy", "sunny"]
    start_probability = {"rainy": 0.6, "sunny": 0.4}
    trans_probability = {"rainy": {"rainy": 0.7, "sunny": 0.3}, "sunny": {"rainy": 0.4, "sunny": 0.6}}
    emit_probability = {"rainy": {"walk": 0.1, "shop": 0.4, "clean": 0.5}, "sunny": {"walk": 0.6, "shop": 0.3, "clean": 0.1}}

    fun = HMM(observation, Hiddenstates, start_probability, trans_probability, emit_probability)
    res = fun.viterbi()
    print(res)
