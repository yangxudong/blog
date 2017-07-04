---
title: JVM类加载机制剖析
date: 2017-07-04 22:35:15
tags: [JVM,ClassLoader]
categories: [程序设计,java]
---
# 一、何为类加载器

我们编写的.java文件经过编译器编译之后，生成.class文件，即字节码文件，类加载器就是负责加载字节码文件到JVM中，并将字节码转换成为java.lang.class类的实例，这个实例便是我们编写的类，通过class实例的newInstance方法，便可以得到java类的对象。

类加载器是类加载过程中的关键角色，他存在于「类加载Class load」过程的「加载」阶段中，在这个阶段，JVM虚拟机完成了三件事情:

1. 通过一个类的全限定名(包名称+类名称)获取定义此类的二进制字节流（类的权限定名可以映射到文件系统中的文件路径）；
2. 将这个字节流所代表的静态存储结构转化为方法区的运行时数据结构；
3. 在内存中生成一个代表这个类的java.lang.Class对象，作为方法区这个类的各种数据的访问入口；
<!-- more -->
## java.lang.ClassLoader
1. loadClass(String className): 加载className类，返回java.lang.class类的实例，异常则抛出ClassNotFoundException
2. defineClass(String name, byte[] b, int off, int len): 加载字节码，返回class类的实例，异常则抛出NoClassDefFoundError

# 二、类加载器的体系结构

![](http://img1.tbcdn.cn/L1/461/1/5194b58d3bfaf478db6570afc17e8f9916f702fd.png)

## 1. 启动类加载器「Bootstrap ClassLoader」

处于最顶端的类加载器，主要负责JAVA_HOME/jre/lib目录下的核心jar或者由-Xbootclasspath选项指定的jar包的装入工作。深入分析下Launcher的源码，发现Bootstrap ClassLoader其实加载的是System.getProperty("sun.boot.class.path")定义下的类包路径。

查看JVM启动后Bootstrap ClassLoader具体加载了哪些jar：
```java
URL[] bootUrls = sun.misc.Launcher.getBootstrapClassPath().getURLs();
for (URL url : bootUrls) {
    System.out.println(url.toExternalForm());
}
```
 
Bootstrap ClassLoader是由C++编写的并且内嵌于JVM中，该加载器是无法被java程序直接引用的。比如，java.util.ArrayList类处于rt.jar包下，该包是由Bootstrap ClassLoader负责加载，所以下面这段代码打印出来就是null了。

```java
ArrayList list = new ArrayList();
System.out.println("list的类加载器为:"+list.getClass().getClassLoader());
```
 
## 2. 扩展类加载器「Extension ClassLoader」

扩展类加载器是由sun.misc.Launcher$ExtClassLoader实现，顾名思义这个类加载器主要负责加载JAVA_HOME\lib\ext目录中或者被java.ext.dirs系统变量定义的路径下的所有类库。

## 3. 应用程序类加载器「App ClassLoader」

应用程序类加载器是由sun.misc.Launcher$AppClassLoader实现，通过源码发现，该类加载器负责加载System.getProperty("java.class.path")也就是classpath下的类库。该类加载器又可以称为**系统类加载器**，在用户没有明确指定类加载器的情况下，系统默认使用AppClassLoader加载类。

## 4. 自定义类加载器「Custom ClassLoader」

自定义类加载器是提供给用户自定义「加载哪里的类」而产生的，当初虚拟机在定义「通过一个类的全限定名(包名称+类名称)获取定义此类的二进制字节流」并没有把获取方式限定死，提供了灵活的方式给用户使用，被加载的类可以来自于数据库、可以来自本地文件、可以来自云存储介质等等，用户所需要的就是自定义类加载器并且继承ClassLoader,最后重写「findClass」方法，ClassLoader为我们提供了defineClass方法可以方便的加载源码的二进制字节流。

```java
/*
 * for example, an application could create a network class loader to
 * download class files from a server.  Sample code might look like:
 */

ClassLoader loader = new NetworkClassLoader(host,port);
Object main = loader.loadClass("Main", true).newInstance();

/*
 *The network class loader subclass must define the methods {@link
 * #findClass <tt>findClass</tt>} and <tt>loadClassData</tt> to load a class
 * from the network.  Once it has downloaded the bytes that make up the class,
 * it should use the method {@link #defineClass <tt>defineClass</tt>} to
 * create a class instance.  A sample implementation is:
 */

class NetworkClassLoader extends ClassLoader {
 String host;
 int port;

 public Class findClass(String name) {
    byte[] b = loadClassData(name);
    return defineClass(name, b, 0, b.length);
 }
 private byte[] loadClassData(String name) {
    // load the class data from the connection
 }
}
```

![](http://upload-images.jianshu.io/upload_images/3901673-decfb670e9357fef.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


# 三、双亲委派模型

在同一个JVM中，一个类加载器和一个类的全限定名共同唯一决定一个类Class的实例。也就是说判定两个类相等法则：类的全名（包名+类名）相同+类加载器相同。

类加载器在加载类的过程中，会先代理给它的父加载器，以此类推。这样的好处是能够保证Java核心库的类型安全，例如java.lang.String类，如果不存在代理模式，则不同的类加载，根据判定两个类相等的法则，会导致存在不同版本的String类，会导致不兼容问题。

核心代码：
```
protected Class<?> loadClass(String name, boolean resolve)
        throws ClassNotFoundException
    {
        synchronized (getClassLoadingLock(name)) {
            //  [首先，检查该类是否被当前类加载器加载]
            Class c = findLoadedClass(name);
            if (c == null) {
                long t0 = System.nanoTime();
                try {
                    if (parent != null) {
            // [调用父类加载器的loadClass方法，实现了自底向上的检查类是否被加载的功能]
                        c = parent.loadClass(name, false);
                    } else {
             // [父类加载器为null，也就是去调用BootClassLoader加载]
                        c = findBootstrapClassOrNull(name);
                    }
                } catch (ClassNotFoundException e) {
                    // ClassNotFoundException thrown if class not found
                    // from the non-null parent class loader
                }
                if (c == null) {
                    long t1 = System.nanoTime();
                    // [调用当前类加载器findClass方法实现了的自顶向下的类加载功能：ExtClassLoader.findClass(name) -> AppClassLoader.findClass(name) -> CustomClassLoader.findClass(name)]
                    c = findClass(name);
                    // this is the defining class loader; record the stats
                    sun.misc.PerfCounter.getParentDelegationTime().addTime(t1 - t0);
                    sun.misc.PerfCounter.getFindClassTime().addElapsedTimeFrom(t1);
                    sun.misc.PerfCounter.getFindClasses().increment();
                }
            }
            if (resolve) {
                resolveClass(c);
            }
            return c;
        }
    }
```

![](http://img2.tbcdn.cn/L1/461/1/47e33802ae2f51d7452645ff811a50fce349b119.png)

# 四、类加载过程

在前面介绍类加载器的代理模式的时候，提到过类加载器会首先代理给其它类加载器来尝试加载某个类。这就意味着真正完成类的加载工作的类加载器和启动这个加载过程的类加载器，有可能不是同一个。真正完成类的加载工作是通过调用 defineClass来实现的；而启动类的加载过程是通过调用 loadClass来实现的。前者称为一个类的定义加载器（defining loader），后者称为初始加载器（initiating loader）。在 Java 虚拟机判断两个类是否相同的时候，使用的是类的定义加载器。也就是说，哪个类加载器启动类的加载过程并不重要，重要的是最终定义这个类的加载器。两种类加载器的关联之处在于：一个类的定义加载器是它引用的其它类的初始加载器。如类 com.example.Outer引用了类com.example.Inner，则由类 com.example.Outer的定义加载器负责启动类 com.example.Inner的加载过程。

类加载器在成功加载某个类之后，会把得到的 java.lang.Class类的实例缓存起来。下次再请求加载该类的时候，类加载器会直接使用缓存的类的实例，而不会尝试再次加载。也就是说，对于一个类加载器实例来说，相同全名的类只加载一次，即 loadClass方法不会被重复调用。

参考资料：
http://www.ibm.com/developerworks/cn/java/j-lo-classloader/
