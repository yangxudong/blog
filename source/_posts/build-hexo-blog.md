---
title: hexo博客迁移流程
date: 2017-02-04 19:55:44
tags: hexo
categories: 技术工具&平台
comments: true
toc: true
---

本文并不是从头开始搭建hexo博客的教程，而是因为最近个人换了电脑，需求迁移原来搭建好的hexo博客，为了防止将来还需要迁移，特记录下操作流程，以便将来查看。

迁移博客的工作量要比从头开始搭建简单很多，很多插件的服务端配置都不需要重新设置，只需要在本地做相应的操作即可。

<!-- more -->

## 安装hexo

第一步当然是安装相应的软件和配置好环境。需要安装的软件有Node.js和Git,去官网下载安装即可。

当Node.js和Git都安装好后就可以正式安装Hexo了，终端执行如下命令：
```
sudo npm install -g hexo-cli
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
git clone https://github.com/theme-next/hexo-theme-next.git themes/next
```

在 NexT 主题目录下的 languages/{language}.yml （{language} 为你所使用的语言）为添加的菜单项设置对于的语言内容。比如添加自定义菜单`essays: 随笔`。

##拷贝文件
拷贝原来搭建好的博客的站点配置文件和主题配置文件_config.yml并在对应的配置目录下做替换。

如何还有其他修改过的配置文件，也一并替换掉。比如我的主题目录下layout/_partials/header.swig就有部分修改，因而需要替换。

最重要的是拷贝并替换数据文件，即站点更目录下的source文件夹，所有博文的原始文件都在这个目录下。

如果原博客的配置和文章等信息托管在github上，则可以使用下面的方法从远程下载文件到本地。首先需要切换到博客目录，执行git init命令把当前目录设置为git托管目录，并添加远程仓库，如下所示，需按照自己的远程仓库替换地址。接着就可以从远程仓库拉取数据了，如果拉取的过程中提升本地文件已存在，则需要先重命名本地文件或删除本地文件。

```
git init
git remote add origin https://github.com/xxxx/blog.git
git pull origin master
```

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

或者配置站内搜索
```
npm install hexo-generator-searchdb --save
```

## 开启评论功能

```
npm install gitment --save
```

## 开启字数统计功能

```
npm install hexo-symbols-count-time --save
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
