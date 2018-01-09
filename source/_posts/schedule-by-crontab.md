---
title: 使用crontab调度hadoop任务和机器学习任务的正确姿势
date: 2017-10-28 15:07:30
tags: [调度,hadoop,hive,crontab]
categories: [程序设计,shell]
---
![icon](schedule-by-crontab/flow.jpg)

虽然现在越来越多的开源机器学习工具支持分布式训练，但分布式机器学习平台的搭建和运维的门槛通常是比较高的。另外一方面，有一些业务场景的训练数据其实并不是很大，在一台普通的开发机上训练个把小时足矣。单机的机器学习工具使用起来通常要比分布式平台好用很多。

特征工程在机器学习任务中占了很大的一部分比重，使用hive sql这样的高级语言处理起来比较方面和快捷。因此，通常特征工程、样本构建都在离线分布式集群（hive集群）上完成，训练任务在数据量不大时可以在gateway机器上完成。这就涉及到几个问题：

1. gateway机器上的daily训练任务什么时候开始执行？
2. 模型训练结束，并对新数据做出预测后如何把数据上传到分布式集群？
3. 如何通知后置任务开始执行？
<!-- more -->
对于第一个问题，理想的解决方案是公司大数据平台的调度系统能够调度某台具体gateway上部署的任务，并且可以获取任务执行的状态，在任务执行成功后可以自动调度后置任务。然而，有时候调度系统还没有这么智能的时候，就需要我们自己想办法解决了。Crontab是Unix和类Unix的操作系统中用于设置周期性被执行的指令的工具。使用Crontab可以每天定时启动任务。美中不足在于必须要自己检查前置任务是否已经结束，也就是说我们要的数据有没有产出，同时还要有一个让后置任务等待当前任务结束的机制。

## 检查前置任务是否已经结束

如果前置任务是hive任务，那么结束标志通常是一个hive表产生了特定分区，我们只需要检查这个分区是否存在就可以了。有个问题需要注意的是，可能在hive任务执行过程中分区已经产生，但任务没有完全结束前数据还没有写完，这个时候启动后续任务是不正确。解决办法就是在任务结束时为当前表添加一个空的“标志分区”，比如原来的分区是“pt=20170921”，我们可以添加一个空的分区“pt=20170921.done”（分区字段的类型为string时可用），或者“pt=-20170921”（分区字段的类型为int时可用）。然后，crontab调度的后置任务需要检查这个“标志分区”是否存在。

```bash
function log_info()
{
    if [ "$LOG_LEVEL" != "WARN" ] && [ "$LOG_LEVEL" != "ERROR" ]
    then
        echo "`date +"%Y-%m-%d %H:%M:%S"` [INFO] ($$)($USER): $*";
    fi
}

function check_partition() {
   log_info "function [$FUNCNAME] begin"
   #table,dt
   temp=`hive -e "show partitions $1"`
   echo $temp|grep -wq "$2"
   if [ $? -eq 0 ];then
       log_info "$1 parition $2 exists, ok"
       return 0
   else
       log_info "$1 parition $2 doesn't exists"
       return 1
   fi  
   log_info "function [$FUNCNAME] end"
}
```

如果前置任务是MapReduce或者Spark任务，那么结束标志通常是在HDFS上产出了一个特定的路径，后置任务只需要检查这个特定路径是否存在就可以。
```bash
## 功能: 检查给定的文件或目录在hadoop上是否存在
## $1 文件或者目录, 不支持*号通配符
## $? return 0 if file exist, none-0 otherwise
function hadoop_check_file_exist()
{
    ## check params
    if [ $# -ne 1 ]
    then
        log_info "Unexpected params for hadoop_check_file_exist() function! Usage: hadoop_check_file_exist <dir_or_file>";
        return 1;
    fi

    ## do it
    log_info "${HADOOP_EXEC} --config ${HADOOP_CONF} fs -test -e $1"
    ${HADOOP_EXEC} --config ${HADOOP_CONF} fs -test -e "$1"
    local ret=$?
    if [ $ret -eq 0 ]
    then
        log_info "$1 does exist on Hadoop"
        return 0;
    else
        log_info "($ret)$1 does NOT exist on Hadoop"
        return 2;
    fi
    return 0;
}
```
其实，hive任务的表的内容也是存储在HDFS上，因此也可以用检查HDFS路径的方法，来判断前置hive任务是否已经结束。可以用下面命令查看hive表对应的HDFS路径。
```
hive -e "desc formatted $tablename;"
```

## 循环等待前置任务结束

当前置任务还没有结束时，需要循环等待。有两种方法，一种是自己在Bash脚本里写代码，如下：
```bash
  hadoop_check_file_exist "$hbase_dir/$table_name/pt=-$bizdate"
  while [ $? -ne 0 ] 
  do  
    local hh=`date '+%H'`
    if [ $hh -gt 23 ]
    then
        echo "timeout, partition still not exist"
        exit 1
    fi  
    log_info "$hbase_dir/$table_name/pt=-$bizdate doesn't exist, wait for a while"
    sleep 5m
    hadoop_check_file_exist "$hbase_dir/$table_name/pt=-$bizdate"
  done 
```

第二种方法，是利用crontab的周期性调度功能。比如可以让crontab每隔5分钟调度一次任务。这个时候需要注意的是，可能前一次调度的进程还没有执行结束，后一次调度就已经开始。这个时候可以使用linux flock文件锁实现任务锁定，解决冲突。
```
flock [-sxon][-w #] file [-c] command
-s, --shared:    获得一个共享锁
-x, --exclusive: 获得一个独占锁
-u, --unlock:    移除一个锁，通常是不需要的，脚本执行完会自动丢弃锁
-n, --nonblock:  如果没有立即获得锁，直接失败而不是等待
-w, --timeout:   如果没有立即获得锁，等待指定时间
-o, --close:     在运行命令前关闭文件的描述符号。用于如果命令产生子进程时会不受锁的管控
-c, --command:   在shell中运行一个单独的命令
-h, --help       显示帮助
-V, --version:   显示版本
```
其中，file是一个空文件即可。比如，crontab文件可以这样写：
```
*/5 6-23 * * * flock -xn /tmp/pop_score.lock -c 'bash /home/xudong.yang/pop_score/train/main.sh -T -p -c >/dev/null 2>&1'
```
如果使用这种方法，脚本里面检查前置任务没有结束时就直接退出当前进程即可，像下面这样：
```
  hadoop_check_file_exist "$hbase_dir/$table_name/pt=-$bizdate"
  if [ $? -ne 0 ]; then
    log_info "$hbase_dir/$table_name/pt=-$bizdate doesn't exist, wait for a while"
    exit 1
  fi
```
虽然文件锁能解决crontab调度冲突的问题，但是我们只希望脚本被成功执行一次，任务结束之后，后续的调度直接退出。还有一种情况需要考虑，有可能crontab调度的任务的正在运行，这个时候我们自己也手动启动了同样的任务，如何避免这样的冲突呢？

无非就是要有个标记任务已经成功运行或者正在运行标识能够在脚本里读取，如何做到呢？对就是在指定目录下建立特定名称的空文件。在脚本开始的时候坚持标记文件是否存在，存在就直接退出。在任务正常运行结束的时候touch成功执行的标记。结构大概如下：
```bash
# 变量定义等
......
[ -f $data_home/$bizdate/DONE ] && { log_info "DONE file exists, exit" >> $log_file_path; exit 0; }
[ -f $data_home/$bizdate/RUNNING ] && { log_info "RUNNING file exists, exit" >> $log_file_path; exit 0; }

touch $data_home/$bizdate/RUNNING
trap "rm -f $data_home/$bizdate/RUNNING; echo Bye." EXIT QUIT ABRT INT HUP TERM
# do something here
......
if [ -f $data_home/$bizdate/RUNNING ]
then
    mv $data_home/$bizdate/RUNNING $data_home/$bizdate/DONE
else
    touch $data_home/$bizdate/DONE
fi
exit 0;
```
有了RUNNING标记就不怕手动执行任务时和crontab调度冲突了；有了DONE标记就不怕每天的任务被调度多次了。

## 从分布式集群下载数据
从hdfs下载数据的函数：
```bash
## 功能: 将hadoop上的文件下载到本地并merge到一个文件中
## $1 hadoop叶子目录 或 文件名--支持通配符 (*)
## $2 本地文件名
## $? return 0 if success, none-0 otherwise
function hadoop_getmerge()
{
    ## check params
    if [ $# -ne 2 ]
    then
        log_info "Unexpected params for hadoop_getmerge() function! Usage: hadoop_getmerge <hadoop_file> <local_file>";
        return 1;
    fi

    if [ -f $2 ]
    then
        log_info "Can not do hadoop_getmerge because local file $2 already exists!"
        return 2;
    fi

    ## do it
    ${HADOOP_EXEC} --config ${HADOOP_CONF} fs -getmerge $1 $2;
    if [ $? -ne 0 ]
    then
        log_info "Do hadoop_getmerge FAILED! Source: $1, target: $2";
        return 3;
    else
        log_info "Do hadoop_getmerge OK! Source: $1, target: $2";
        return 0;
    fi

    return 0;
}
```
HIVE表里的数据也可以先找到对应的HDFS目录，然后用上面的函数下载数据，唯一需要注意的是，HIVE表必须stored as textfile，否则下载下来的数据不可用。
万一hive表不是已文本文件的格式存储的怎么办呢？不要紧，还是有办法的，如下：
```
  mkdir -p $data_home/$bizdate/raw
  declare sql="
    set hive.support.quoted.identifiers=None;
    insert overwrite local directory '$data_home/$bizdate/raw'
    row format delimited fields terminated by '\t'
    select \`(pt)?+.+\` from $table_name where pt=$bizdate; 
  "
  log_info $sql
  $hive -e "$sql"
```

## 上传数据到分布式集群

模型训练和预测之后，必须把预测数据上传到分布式集群，以便后续处理。
```
  local create_table_sql="
    create table if not exists $target_table_name (
        ......
    )
    partitioned by (pt int)
    row format delimited fields terminated by '\t' 
    lines terminated by '\n' 
    stored as textfile;
  "
  log_info $create_table_sql
  $hive -e "$create_table_sql"

  local upload_sql="load data local inpath '$data_home/$bizdate/$predict_file' into table $target_table_name partition(pt=${bizdate});"
  log_info $upload_sql
  $hive -e "$upload_sql"
```

## 通知后置任务开始执行

其实crontab没法通知后置任务当前任务已经结束，那怎么办呢？

把真正的后置任务加一个前置依赖任务，而这个依赖任务是部署在调度系统上的一个shell任务，该任务的前置任务是crontab调度任务的前置任务，并且这个任务做的唯一一件事情就是循环检查crontab调度任务的数据有没有产出，已经产出就结束，没有产出就sleep一小段时间之后再继续检查。

```bash
check_partition $table_name $bizdate
while [ $? -ne 0 ] 
do
  sleep 5m
  hh=`date '+%H'`
  if [ $hh -gt 23 ]
  then
      echo "timeout, partition still not exist"
      exit 1
  fi  
  check_partition $table_name $bizdate
done
```

## 那些年，我们踩过的坑

一、crontab调度任务不会自动export环境变量

开始的时候，手动执行脚本正常运行，但是crontab调度每次都会在`hadoop fs -test -e $path`命令执行出错，表现为明明`$path`已经存在，但指令总是返回1，而不是0 。经过苦苦排查之后才发现，hadoop依赖的环境变量JAVA_HOME和HADOOP_HOME没有在脚本里导入。而用户在终端里登录到服务器上时，这2个环境变量是自动导入的。所以，务必记得在脚本开始的时候就导入环境变量：

```
#!/bin/bash
export JAVA_HOME=/usr/local/jdk
export HADOOP_HOME=...
```

二、crontab调度任务不能写太多标准输出，否则任务会在某个时刻自动中断

这个也挺坑的，务必记得在crontab的指令里重定向标准输出和标准错误到一个文件里，或者重定向到unix的黑洞`/dev/null`里。

```
*/5 6-23 * * * flock -xn /tmp/pop_score.lock -c 'bash /home/xudong.yang/pop_score/train/main.sh -T -p -c >/dev/null 2>&1'
```

这里推荐在脚本使用tee命令同时输出日志到终端和文件，这样用户手动执行的时候可以直接在终端里看到程序的执行情况，crontab调度里可以查看日志文件来排查问题。当然，输出到终端的部分，在使用crontab调度时需要重定向到黑洞里。
```
main | tee -a $log_file_path 2>&1
if [ ${PIPESTATUS[0]} -ne 0 ]
then
    log_error "run failed."
    exit 1;
fi
```
