---
title: hexo博客迁移流程
date: 2017-02-04 19:55:44
tags: hexo
categories: 技术工具&平台
---

本文并不是从头开始搭建hexo博客的教程，而是因为最近个人换了电脑，需求迁移原来搭建好的hexo博客，为了防止将来还需要迁移，特记录下操作流程，以便将来查看。

迁移博客的工作量要比从头开始搭建简单很多，很多插件的服务端配置都不需要重新设置，只需要在本地做相应的操作即可。

<!-- more -->

## 安装hexo

第一步当然是安装相应的软件和配置好环境。需要安装的软件有Node.js和Git,去官网下载安装即可。

当Node.js和Git都安装好后就可以正式安装Hexo了，终端执行如下命令：
```
sudo npm install -g hexo
```
终端cd到一个你选定的目录，执行hexo init初始化命令：
```
hexo init <folder>
```
终端cd到目录下，安装npm
```
npm install
```
此时开启hexo服务就可以在本地（本地预览地址http://localhost:4000 ）预览博客主页了
```
hexo s
```

##下载主题：
```
cd your-hexo-site
git clone https://github.com/iissnan/hexo-theme-next themes/next
```

##拷贝文件
拷贝原来搭建好的博客的站点配置文件和主题配置文件_config.yml并在对应的配置目录下做替换。

如何还有其他修改过的配置文件，也一并替换掉。比如我的主题目录下layout/_partials/header.swig就有部分修改，因而需要替换。

最重要的是拷贝并替换数据文件，即站点更目录下的source文件夹，所有博文的原始文件都在这个目录下。

## 配置MathJax

配置方法参考我的另一篇博文：{% post_link math-in-hexo 在Hexo中渲染MathJax数学公式 %}

## 安装RSS插件

```
npm install hexo-generator-feed --save
```
## 安装Git插件
```
npm install hexo-deployer-git --save
```

## 安装hexo-asset-image的插件来处理图片

```
npm install https://github.com/CodeFalling/hexo-asset-image --save
```
确认_config.yml 中有 post_asset_folder:true，设置post_asset_folder为true参数后，在建立文件时，Hexo
会自动建立一个与文章同名的文件夹，您可以把与该文章相关的所有资源都放到那个文件夹，如此一来，您便可以更方便的使用资源。结构如下：
```
本地图片测试
├── apppicker.jpg
├── logo.jpg
└── rules.jpg
本地图片测试.md
```
这样的目录结构（目录名和文章名一致），只要使用 <pre> ![logo](本地图片测试/logo.jpg)</pre> 就可以插入图片。其中[]里面不写文字则没有图片标题。

## 配置站内搜索
安装Algolia
```
npm install hexo-algolia --save
```
因为hexo-algolia的作者并没有把post.path进行index，所以data.path是undefined，即搜索出的内容点击之后跳转链接不正确，解决方案是在node_modules/hexo-algolia/lib， 找到command.js，打开文件，在storedPost变量里加path:
```
var storedPost = _.pick(data, ['title', 'date', 'slug', 'path', 'content', 'excerpt', 'objectID']);
```
上传数据到Algolia引擎服务
```
hexo algolia
```

## 提交搜索引擎
分别用下面两个命令来安装针对谷歌和百度的插件
```
npm install hexo-generator-sitemap --save
npm install hexo-generator-baidu-sitemap --save
```

## 大功告成
执行如下命令就可以预览博客了。
```
hexo g && hexo s
```
最后要部属到远程服务器，如github或coding上，别忘了在新电脑上生成ssh公钥，并注册到远程服务器。

PS: 为了兼容coding，需要在source文件夹下touch一个空的文件：Staticfile.

因为博客同时部署到github和coding两个平台，但配置文件里智能配置一个root路径，所以生成的sitemap.xml或者baidu-sitemap.xml中有一个网站的路径是错误的，需要在部署之前手动修改。

博客预览效果没问题的话，就可以部属到远程服务器了。
```
hexo d
```
