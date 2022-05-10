# HPC & Parallel Computation Resources



## Major knowlege overview


1. Parallel Computation
2. GPU CUDA programming 
3. CPU ARM/X86 Intrinsic programming
4. Common Kernel Opimization
   1. GEMM
   2. Conv
5. Computer Architecture & System
6. Compiler
7. Operating System



### Computer Architecture

* Book
  1. Computer Architecture: A Quantitative Approach, David Patterson & John Hennessy 计算机系统结构 量化研究方法 [Link](https://book.douban.com/subject/1175378/)
     1. Berkeley CS 252A required textbook
     2. CMU 18-842 required textbook
  2. Computer Organization and Design: The Hardware/Software Interface (ARM), David Patterson & John Hennessy 计算机组成与设计：硬件/软件接口
     1. Berkeley CS 252A recommended textbook
  3. CSAPP Computer Systems: A Programmer's Perspective 深入理解计算机系统 [Link](https://book.douban.com/subject/1896753/)



* Course
  * Computer Architecture, ETH, Onur Mutulu, fall 2020 [homepage have video recording](https://safari.ethz.ch/architecture/fall2020/doku.php?id=schedule) 
  * Graduate Computer Architecture ETH, fall 2019 [homepage have video recording](https://safari.ethz.ch/architecture/fall2019/doku.php?id=start)
  * Stanford CS107 Computer Organization and System
  * Berkeley CS 252A [spring 2021 homepage have no video recording](https://inst.eecs.berkeley.edu/~cs152/sp21/)
  * CMU 15-213 CSAPP [bilibili](https://www.bilibili.com/video/BV1iW411d7hd?from=search&seid=13353650717207742800d%3Ffrom%3Dsearch&seid=13353650717207742800)
  * CMU 18-842 Computer Architecture and Systems [homepage没have video recording](https://course.ece.cmu.edu/~ece742/S21/index.html)



### Compiler

* Book
  * Principle of Compiler [Link](https://book.douban.com/subject/3296317/)



* Course
  * Stanford CS143 Compiler



### OS

* Book
  * Operating Systems: Three easy pieces



* Course
  * Stanford CS240 Advanced Topic in Operating System [link of Stanford online](https://online.stanford.edu/courses/cs240-advanced-topics-operating-systems)
  * MIT 6.828 operating system engineering. 知乎上有人推荐上这门课，说是上了这么课以后感觉之前的OS都白学了



### Parallel Computation

* Book
  * Introduction to HPC scientists and engineer 高性能科学与工程计算
    * 是一个LBL出的系列的书中的一本。
    * 中文一般，读原版
  * 并行算法设计与性能优化 知乎上得分比较高。看了一些实际上写的很粗糙
  * Parallel Processing for Scientific Computing 劳伦斯实验室推荐
  * Introduction to High-Performance Scientific Computing Victor 豆瓣上有人提过这本书，但是说“高性能科学与工程计算“更有深度



* 知乎
  * [并行计算入门 专栏文章](https://zhuanlan.zhihu.com/p/181669611)
  * [知乎 个人专栏 严忻恺](https://www.zhihu.com/people/yan-xin-kai-38/posts?page=3)



* Course
  * MIT Performance Engineering of Software Systems 6.172 [Link](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-172-performance-engineering-of-software-systems-fall-2018/index.htm) 
    * 知乎上人推荐是最好的HPC入门课
    
  * Berkeley CS 267 [Link](https://sites.google.com/lbl.gov/cs267-spr2021/home)
  
  * CMU 15-418/618 [Course Website](http://15418.courses.cs.cmu.edu/spring2016/ ), [bilibili page](https://www.bilibili.com/video/BV1aa4y1s7EH?from=search&seid=5014171813592522248&spm_id_from=333.337.0.0)
  
  * Stanford CS149 Parallel Computing, [Course Website](http://cs149.stanford.edu/fall20), [bilibili page](https://www.bilibili.com/video/BV1aa4y1s7EH?from=search&seid=10326830494415144772&spm_id_from=333.337.0.0)
    * 同CMU 15-618 课程有改进
    
  * Stanford CME 213 Intro to Parallel Computing using MPI, openMP, CUDA [Course Website](https://ericdarve.github.io)  [bilibili page](https://www.bilibili.com/video/BV1Fv411j7GP?from=search&seid=11467285219799944839)
    * Stanford的学生觉得149比213的效果要更好, 因为213的知识太偏向于上手了
    
  * SSSIHL CS1207 Programming For Performance [link](http://progforperf.github.io/lectures.html) 
    * 使用了全世界很多课程的slides，组合出来的课程

  * Stanford CS 74: An Introduction to Building High-Performance Computing Systems (by Stanford HPC center)
  
  * Stanford ME 344: Introduction to High-Performance Computing. Steve Jones (Stanford HPC manager)
    * 更偏向于系统方向的hpc，像是cluster，numa，legion
    * 这门课好像很久没有开过了

  * IIT HPC for science & engineer [video](https://nptel.ac.in/courses/112/105/112105293/#) 是配套上面书的内容，九坤的比赛推荐用这个

  * ETH Design of Parallel and High-Performance Computing [homepage](https://acl.inf.ethz.ch/teaching/dphpc/)
    * 看了一下课程大纲，感觉整体内容讲的比较浅，不如cmu的好
    
    



### ML-Sys

* Course
  * Stanford CS217 Hardware Acceleration for ML [当前课程没有公开slides](https://online.stanford.edu/courses/cs217-hardware-accelerators-machine-learning)
  * UC Berkeley, CS 294 System for AI, Fall 2019, [homepage](https://ucbrise.github.io/cs294-ai-sys-fa19/)
  * UMich, EECS 598, System for AI, Winter 2021 有github的 page有课程的slides
  * UW CS 559W system for ML [slides](http://dlsys.cs.washington.edu/schedule)
  * Hardware for Machine Learning [link](https://inst.eecs.berkeley.edu/~ee290-2/sp21/)
  * EPFL CCS 723 [link](https://parsa.epfl.ch/course-info/cs723/index.php?page=readings.php)



### GPU CUDA

* Course
  * UIUC ECE 508, Parallel Algorithm Techniques, Wen-mei Hwu
  * UIUC ECE 408, Applied Parallel Programming.
  * BU CUDA
  * Caltech CS 179 GPU Programming [website](http://courses.cms.caltech.edu/cs179/)
  * Heterogeneous Parallel Programming, OpenCourse, Wen-mei Hwu
    * 内容与UIUC ECE相同



* 书
  * Programming Massively Parallel Processors, Second Edition & Third edition 大规模并行处理器编程实战
  * CUDA C++ Best Practices Guide [Link](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
  * CUDA C Programming Guide [Link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)



### Architecture

* Book
  * 设计密集型数据应用。
    * 知乎上推荐说看完这本书以后，拥有成为合格架构师的基础知识。但是缺点是这本书没有分析一个完整真是的框架。真实的分析各个分布式系统可以让我站在巨人的肩膀上，可以通过看论文来学习。



* Course
  * MIT 6.824 parallel computation. 
    * [知乎上关于课程的讲解](https://zhuanlan.zhihu.com/p/432761838)
    * 偏向于大型系统的，还是不是单机kernel level优化。
    * 每节课都会学习经典论文，讲授论文的核心思想和难点，是学习理论知识很好的资料。
    * 实验部分四课程精华，实现mapreduce，实现raft





## Degree & Program

* Tsinghua Master HPC

> http://ac.cs.tsinghua.edu.cn/curriculum.html

1. Introduction to the extensive data system
2. introduction to a distributed system



* Chalmers University of Technology

> https://www.chalmers.se/en/education/programmes/masters-info/Pages/High-Performance-Computer-Systems.aspx

1. Computer architecture
   1. Computer architecture is the engineering discipline on computer design, which conveys principles for converting the raw speed of transistors into application software performance through computational structures that exploit the parallelism in software. This course covers the essential principles for designing a computer that offers high performance to the application software.
2. high-performance parallel programming
   1. In this course, the students will not only learn the parallel programming paradigms, languages, and development environments for High-performance computer systems (e.g., OpenMP, MPI, CUDA) but also be exposed to performance problem-solving techniques and methodologies to improve the execution time of parallel programs and exploit the full potential of the underlying parallel systems.
3. Real-time systems
4. Parallel computer architecture



* UChicago

> https://masters.cs.uchicago.edu/page/high-performance-computing

1. MPCS 51087 - High-Performance Computing
2. MPCS 52060 - Parallel Programming
3. MPCS 53120 - Applied Data Analysis, 
4. MPCS 51300 - Compilers
5. MPCS 52011 - Introduction to Computer Systems
6. MPCS 52015 - Advanced Computer Systems
7. MPCS 52010 - Computer Architecture
8. MPCS 52060 - Parallel Programming
9. MPCS 52030 - Operating Systems
10. MPCS 52040 - Distributed Systems
11. MPCS 51040 - C Programming
12. MPCS 51100 - Advanced Programming
13. MPCS 51044 - C++ for Advanced Programmers
14. MPCS 51045 - Advanced C++



* USC MS HPC

> https://www.cs.usc.edu/academic-programs/masters/high-performance-computing-simulations/

1. CSCI 558 - Internetworking and Distributed Systems Lab (3)
2. CSCI 653 – High PerformanceHigh-Performance Computing and Simulations (4)
3. EE 653 - Multithreaded Architectures and Data-Flow Computing (3
4. EE 657 - Parallel and Distributed Computing (3)



* GW certificate in HPC 

> https://www.ece.seas.gwu.edu/certificate-high-performance-computing

1. Introduction to High-Performance Computing
2. Parallel Computer Architecture
3. Grid and Network Computing



* Trinity college DublinDublin MS HPC

> https://maths.tcd.ie/postgraduate/masters/modules/

1. MA5611 High-Performance Computing Software (10 ECTS)
2. MA5612 High-Performance Computing Systems and Hardware (10 ECTS)

