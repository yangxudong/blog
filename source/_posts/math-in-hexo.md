---
title: 在Hexo中渲染MathJax数学公式
date: 2016-12-29 18:38:46
tags: [hexo,mathjax]
categories: 技术工具&平台
---

在用markdown写技术文档时，免不了会碰到数学公式。常用的Markdown编辑器都会集成[Mathjax](https://www.mathjax.org/)，用来渲染文档中的类Latex格式书写的数学公式。基于Hexo搭建的个人博客，默认情况下渲染数学公式却会出现各种各样的问题。

##原因
Hexo默认使用"hexo-renderer-marked"引擎渲染网页，该引擎会把一些特殊的markdown符号转换为相应的html标签，比如在markdown语法中，下划线'_'代表斜体，会被渲染引擎处理为`<em>`标签。

因为类Latex格式书写的数学公式下划线 '_' 表示下标，有特殊的含义，如果被强制转换为`<em>`标签，那么MathJax引擎在渲染数学公式的时候就会出错。例如，$x_i$在开始被渲染的时候，处理为$x`<em>`i`</em>`$，这样MathJax引擎就认为该公式有语法错误，因为不会渲染。

类似的语义冲突的符号还包括'*', '{', '}', '\\'等。

<!-- more -->

##解决方法

解决方案有很多，可以网上搜下，为了节省大家的时间，这里只提供亲身测试过的最靠谱的方法。

更换Hexo的markdown渲染引擎，[hexo-renderer-kramed](https://github.com/sun11/hexo-renderer-kramed)引擎是在默认的渲染引擎[hexo-renderer-marked](https://github.com/hexojs/hexo-renderer-marked)的基础上修改了一些bug，两者比较接近，也比较轻量级。
```
npm uninstall hexo-renderer-marked --save
npm install hexo-renderer-kramed --save
```
执行上面的命令即可，先卸载原来的渲染引擎，再安装新的。

然后，跟换引擎后行间公式可以正确渲染了，但是这样还没有完全解决问题，行内公式的渲染还是有问题，因为[hexo-renderer-kramed](https://github.com/sun11/hexo-renderer-kramed)引擎也有语义冲突的问题。接下来到博客根目录下，找到node_modules\kramed\lib\rules\inline.js，把第11行的escape变量的值做相应的修改：
```
//  escape: /^\\([\\`*{}\[\]()#$+\-.!_>])/,
  escape: /^\\([`*\[\]()#$+\-.!_>])/,
```
这一步是在原基础上取消了对\\,\{,\}的转义(escape)。
同时把第20行的em变量也要做相应的修改。
```
//  em: /^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
  em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
```
重新启动hexo（先clean再generate）,问题完美解决。哦，如果不幸还没解决的话，看看是不是还需要在使用的主题中配置mathjax开关。

## 在主题中开启mathjax开关

如何使用了主题了，别忘了在主题（Theme）中开启mathjax开关，下面以next主题为例，介绍下如何打开mathjax开关。

进入到主题目录，找到_config.yml配置问题，把mathjax默认的false修改为true，具体如下：
```
# MathJax Support
mathjax:
  enable: true
  per_page: true
```
别着急，这样还不够，还需要在文章的Front-matter里打开mathjax开关，如下：
```
---
title: index.html
date: 2016-12-28 21:01:30
tags:
mathjax: true
--
```
不要嫌麻烦，之所以要在文章头里设置开关，是因为考虑只有在用到公式的页面才加载 Mathjax，这样不需要渲染数学公式的页面的访问速度就不会受到影响了。
