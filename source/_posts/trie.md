---
title: 支持中文的基于词为基本粒度的前缀树（prefix trie）实现
date: 2014-10-28 15:07:30
tags: [trie,python]
categories: [程序设计,python]
---
Trie树，也叫字典树、前缀树。可用于"predictive text"和"autocompletion"，亦可用于统计词频（边插入Trie树边更新或添加词频）。

在计算机科学中，trie，又称前缀树或字典树，是一种有序树，用于保存关联数组，其中的键通常是字符串。与二叉查找树不同，键不是直接保存在节点中，而是由节点在树中的位置决定。一个节点的所有子孙都有相同的前缀，也就是这个节点对应的字符串，而根节点对应空字符串。一般情况下，不是所有的节点都有对应的值，只有叶子节点和部分内部节点所对应的键才有相关的值。

本文实现了基于中文分词为粒度的前缀树。
<!-- more -->

```python
#!/usr/bin/python  
# -*- coding:utf-8 -*-  
# * trie, prefix tree, can be used as a dict  
# * author: yangxudongsuda@gmail.com  
import sys  
reload(sys)  
sys.setdefaultencoding("utf-8")  
  
# Singleton sentinel - works with pickling  
class NULL(object):  
  pass  
  
class Node:  
  def __init__(self, value = NULL):  
    self.value = value  
    self.children = {}  
  
class Trie(object):  
  def __init__(self):  
    self.root = Node()  
  
  def insert(self, key, value = None, sep = ' '):  # key is a word sequence separated by 'sep'  
    elements = key if isinstance(key, list) else key.split(sep)  
    node = self.root  
    for e in elements:  
      if not e: continue  
      if e not in node.children:  
        child = Node()  
        node.children[e] = child  
        node = child  
      else:  
        node = node.children[e]  
    node.value = value  
  
  def get(self, key, default = None, sep = ' '):  
    elements = key if isinstance(key, list) else key.split(sep)  
    node = self.root  
    for e in elements:  
      if e not in node.children:  
        return default  
      node = node.children[e]  
    return default if node.value is NULL else node.value  
  
  def delete(self, key, sep = ' '):  
    elements = key if isinstance(key, list) else key.split(sep)  
    return self.__delete(elements)  
  
  def __delete(self, elements, node = None, i = 0):  
    node = node if node else self.root  
    e = elements[i]  
    if e in node.children:  
      child_node = node.children[e]  
      if len(elements) == (i+1):  
        if child_node.value is NULL: return False # not in dict  
        if len(child_node.children) == 0:  
          node.children.pop(e)  
        else:  
          child_node.value = NULL  
        return True  
      elif self.__delete(elements, child_node, i+1):  
        if len(child_node.children) == 0:  
          return node.children.pop(e)  
        return True  
    return False  
  
  def shortest_prefix(self, key, default = NULL, sep = ' '):  
    elements = key if isinstance(key, list) else key.split(sep)  
    results = []  
    node = self.root  
    value = node.value  
    for e in elements:  
      if e in node.children:  
        results.append(e)  
        node = node.children[e]  
        value = node.value  
        if value is not NULL:  
          return sep.join(results)  
      else:  
        break  
    if value is NULL:  
      if default is not NULL:  
        return default  
      else:  
        raise Exception("no item matches any prefix of the given key!")  
    return sep.join(results)  
  
  def longest_prefix(self, key, default = NULL, sep = ' '):  
    elements = key if isinstance(key, list) else key.split(sep)  
    results = []  
    node = self.root  
    value = node.value  
    for e in elements:  
      if e not in node.children:  
        if value is not NULL:  
          return sep.join(results)  
        elif default is not NULL:  
          return default  
        else:  
          raise Exception("no item matches any prefix of the given key!")  
      results.append(e)  
      node = node.children[e]  
      value = node.value  
    if value is NULL:  
      if default is not NULL:  
        return default  
      else:  
        raise Exception("no item matches any prefix of the given key!")  
    return sep.join(results)  
  
  def longest_prefix_value(self, key, default = NULL, sep = ' '):  
    elements = key if isinstance(key, list) else key.split(sep)  
    node = self.root  
    value = node.value  
    for e in elements:  
      if e not in node.children:  
        if value is not NULL:  
          return value  
        elif default is not NULL:  
          return default  
        else:  
          raise Exception("no item matches any prefix of the given key!")  
      node = node.children[e]  
      value = node.value  
    if value is not NULL:  
      return value  
    if default is not NULL:  
      return default  
    raise Exception("no item matches any prefix of the given key!")  
  
  def longest_prefix_item(self, key, default = NULL, sep = ' '):  
    elements = key if isinstance(key, list) else key.split(sep)  
    node = self.root  
    value = node.value  
    results = []  
    for e in elements:  
      if e not in node.children:  
        if value is not NULL:  
          return (sep.join(results), value)  
        elif default is not NULL:  
          return default  
        else:  
          raise Exception("no item matches any prefix of the given key!")  
      results.append(e)  
      node = node.children[e]  
      value = node.value  
    if value is not NULL:  
      return (sep.join(results), value)  
    if default is not NULL:  
      return (sep.join(results), default)  
    raise Exception("no item matches any prefix of the given key!")  
  
  def __collect_items(self, node, path, results, sep):  
    if node.value is not NULL:  
      results.append((sep.join(path), node.value))  
    for k, v in node.children.iteritems():  
      path.append(k)  
      self.__collect_items(v, path, results, sep)  
      path.pop()  
    return results    
  
  def items(self, prefix, sep = ' '):  
    elements = prefix if isinstance(prefix, list) else prefix.split(sep)  
    node = self.root  
    for e in elements:  
      if e not in node.children:  
        return []  
      node = node.children[e]  
    results = []  
    path = [prefix]  
    self.__collect_items(node, path, results, sep)  
    return results  
  
  def keys(self, prefix, sep = ' '):  
    items = self.items(prefix, sep)  
    return [key for key,value in items]
```

以下是测试代码：
```python
if __name__ == '__main__':  
  trie = Trie()  
  trie.insert('happy 站台', 1)  
  trie.insert('happy 站台 xx', 10)  
  trie.insert('happy 站台 xx yy', 11)  
  trie.insert('happy 站台 美食 购物 广场', 2)  
  trie.insert('sm')  
  trie.insert('sm 国际', 22)  
  trie.insert('sm 国际 广场', 2)  
  trie.insert('sm 城市广场', 3)  
  trie.insert('sm 广场', 4)  
  trie.insert('sm 新生活 广场', 5)  
  trie.insert('sm 购物 广场', 6)  
  trie.insert('soho 尚都', 3)  
  
  print trie.get('sm')  
  print trie.longest_prefix([], default="empty list")  
  print trie.longest_prefix('sm')  
  print trie.shortest_prefix('happy 站台')  
  print trie.shortest_prefix('happy 站台 xx')  
  print trie.shortest_prefix('sm')  
  print trie.longest_prefix('sm xx', sep = '&', default = None)  
  print 'sm 广场 --> ', trie.get('sm 广场')  
  print trie.get('sm 广场'.split(' '))  
  print trie.get('神马')  
  print trie.get('happy 站台')  
  print trie.get('happy 站台 美食 购物 广场')  
  print trie.longest_prefix('soho 广场', 'default')  
  print trie.longest_prefix('soho 尚都 广场')  
  print trie.longest_prefix_value('soho 尚都 广场')  
  print trie.longest_prefix_value('xx 尚都 广场', 90)  
  print trie.longest_prefix_value('xx 尚都 广场', 'no prefix')  
  print trie.longest_prefix_item('soho 尚都 广场')  
  
  print '============== keys ================='  
  print 'prefix "sm": ', ' | '.join(trie.keys('sm'))  
  print '============== items ================='  
  print 'prefix "sm": ', trie.items('sm')  
  
  print '================= delete ====================='  
  print trie.delete('sm 广场')  
  print trie.get('sm 广场')  
  print trie.delete('sm 国际')  
  print trie.get('sm 国际')  
  print trie.delete('sm xx')  
  print trie.delete('xx')  
  
  print '====== no item matches any prefix of given key ========'  
  print trie.longest_prefix_value('happy')  
  print trie.longest_prefix_value('soho xx')  
```
运行结果如下：
> 运行结果：
None
empty list
sm
happy 站台
happy 站台
sm
None
sm 广场 -->  4
4
None
1
2
default
soho 尚都
3
90
no prefix
('soho \xe5\xb0\x9a\xe9\x83\xbd', 3)
============== keys =================
prefix "sm":  sm | sm 新生活 广场 | sm 城市广场 | sm 广场 | sm 购物 广场 | sm 国际 | sm 国际 广场
============== items =================
prefix "sm":  [('sm', None), ('sm \xe6\x96\xb0\xe7\x94\x9f\xe6\xb4\xbb \xe5\xb9\xbf\xe5\x9c\xba', 5), ('sm \xe5\x9f\x8e\xe5\xb8\x82\xe5\xb9\xbf\xe5\x9c\xba', 3), ('sm \xe5\xb9\xbf\xe5\x9c\xba', 4), ('sm \xe8\xb4\xad\xe7\x89\xa9 \xe5\xb9\xbf\xe5\x9c\xba', 6), ('sm \xe5\x9b\xbd\xe9\x99\x85', 22), ('sm \xe5\x9b\xbd\xe9\x99\x85 \xe5\xb9\xbf\xe5\x9c\xba', 2)]
================= delete =====================
True
None
True
None
False
False
====== no item matches any prefix of given key ========
Traceback (most recent call last):
  File "./word_based_trie.py", line 225, in <module>
    print trie.longest_prefix_value('happy')
  File "./word_based_trie.py", line 128, in longest_prefix_value
    raise Exception("no item matches any prefix of the given key!")
Exception: no item matches any prefix of the given key!
