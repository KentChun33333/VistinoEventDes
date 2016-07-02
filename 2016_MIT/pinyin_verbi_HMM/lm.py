#!/usr/bin/env python
# coding:utf-8

class LanguageModel(object):

  def __init__(self, corpus_file):
    self.words = []
    self.freq = {}
    print 'loading words'
    with open(corpus_file,'r') as f:
      line = f.readline().strip()
      while line:
        self.words.append('<s>')
        self.words.extend([w.encode('utf-8') for w in list(line.decode('utf-8')) if len(w.strip())>0])
        self.words.append('</s>')
        line = f.readline().strip()

    print 'hashing bi-gram keys'
    for i in range(1,len(self.words)):
      # 条件概率
      key = self.words[i] + '|' + self.words[i-1]
      if key not in self.freq:
        self.freq[key] = 0
      self.freq[key] += 1

    print 'hashing single word keys'
    for i in range(0,len(self.words)):
      key = self.words[i]
      if key not in self.freq:
        self.freq[key] = 0
      self.freq[key] += 1

  def get_trans_prop(self, word, condition):
    """获得转移概率"""
    key = word + '|' + condition
    if key not in self.freq:
      self.freq[key] = 0
    if condition not in self.freq:
      self.freq[condition] = 0
    C_2 = (float)(self.freq[key] + 1.0)
    C_1 = (float)(self.freq[condition] + len(self.words))
    return C_2/C_1

  def get_init_prop(self, word):
    """获得初始概率"""
    return self.get_trans_prop(word,'<s>')

  def get_prop(self, *words):
    """获得指定序列的概率"""
    init = self.get_init_prop(words[0])
    product = 1.0
    for i in range(1,len(words)):
      product *= self.get_trans_prop(words[i],words[i-1])
    return init*product

def main():
    lm = LanguageModel('RenMinData.txt')
    print 'total words: ', len(lm.words)
    print 'total keys: ', len(lm.freq)
    print 'P(结|团) = ', lm.get_trans_prop('结','团')
    print 'P(斗|奋) = ', lm.get_trans_prop('斗','奋')
    print 'P(法|入) = ', lm.get_trans_prop('法','入')
    print 'P(发|入) = ', lm.get_trans_prop('发','入')
    print 'P(奋斗|团结) = ', lm.get_trans_prop('奋斗','团结')


if __name__ == '__main__':
    main()
