## CPU




### Von-Neuman model

世界上的计算机都是下面这个模型的instance

<img src="Note.assets/Screen Shot 2021-11-07 at 1.18.02 PM.png" alt="Screen Shot 2021-11-07 at 1.18.02 PM" style="zoom:30%;" />

* instruction processing 过程
1. fetch instruction
2. decode instruction
3. execute
4. memory write back result
* instruction 类型
1. operate 计算
   1. ADD R1 R2 R3
      1. read r2 r3
      2. add them 
      3. write result to r1
2. data transfer 
   1. ldr r4 r6 3
      1. read r6
      2. add 3 to r6
      3. load memory at address = r6
      4. store that memory in r4
3. control flow / branching




### Simple CPU Arch

> 参考
> 
> 1. CMU 15.418 L2
> 2. 并行算法设计与性能优化 chapter 2.1
> 3. Berkeley CS267 L2

* basic processor

<img src="Note.assets/Screen Shot 2021-10-16 at 11.27.57 PM.png" alt="Screen Shot 2021-10-16 at 11.27.57 PM" style="zoom:30%;" />

* superscalar processor

是什么：一个clock内同时执行没有数据以来或者控制依赖的指令。

硬件实现：通过多条pipeline做到ILP

由hardware detect&实现

Superscalar execution:  processor dynamically finds independent instructions in an instruction sequence and executes them in parallel

<img src="Note.assets/Screen Shot 2021-10-16 at 11.28.14 PM.png" alt="Screen Shot 2021-10-16 at 11.28.14 PM" style="zoom:30%;" />

* multi-core processor 

<img src="Note.assets/Screen Shot 2021-10-16 at 11.28.51 PM.png" alt="Screen Shot 2021-10-16 at 11.28.51 PM" style="zoom:30%;" />

* SIMD

same instruction broadcast to all ALUs.

Amortize cost/complexity of managing an instruction stream.

<img src="Note.assets/Screen Shot 2021-10-16 at 11.29.32 PM.png" alt="Screen Shot 2021-10-16 at 11.29.32 PM" style="zoom:30%;" />

* multithread processor

interleavr processing of multiple thread on same core to hide stall. Latency hiding, not reducing latency.

Many context storage.

在单核处理器上使用多线程可以通过线程交换来使核心处于繁忙状态，从而hide latency

<img src="Note.assets/Screen Shot 2021-10-16 at 11.32.41 PM.png" alt="Screen Shot 2021-10-16 at 11.32.41 PM" style="zoom:40%;" />

* thread

user level thread : 由线程库控制，无需操作系统支持。

kernel level thread : 由OS操控的线程库

pthread是用户级线程，但是linux使用了一对一的内核级线程实现（NPTL后版本）

多线程需要注意

1. 线程过多
2. 数据竞争
3. 死锁
* hyper-threading

each clock, core choose instructions from multiple threads to run on ALUs.

* pipeline

通过增加流实现长度。实现流水线多个stage都处于运行状态，从而在一个clock cycle中可以同时进行IF ID OF IF WB的操作

代码中指令并行是有限的，一旦达到限制，再增加流水线不会有好处。

<img src="Note.assets/Screen Shot 2021-10-22 at 6.56.24 PM.png" alt="Screen Shot 2021-10-22 at 6.56.24 PM" style="zoom:30%;" />

speed up < number of pipeline stage (pipeline带来的速度提升upper bound by number of stage)

<img src="./Note.assets/Screen Shot 2022-01-26 at 7.47.49 PM.png" alt="Screen Shot 2022-01-26 at 7.47.49 PM" style="zoom:33%;" />

硬件中多个部分都有pipeline

1. 整体cpu运行是pipeline
2. ALU内部也有pipeline
3. memory访问也有pipeline
* 分支预测

x86

导致分支预测不好的情况

1. 循环较小
2. 经常使用虚函数




### SIMD

> Berkeley CS267 L2

* x86

avx2 : 256 bits register, 16 register 

avx512 : 512 btis register, 32 register

AVX2 与 AVX512 register底层是同一个物理寄存器。之所以AVX2只有16个是因为出AVX2指令集的时候只出了这么多（当时没有512的硬件支持）

AVX2 AVX512 SSE的向量化操作有些需要内存对齐(e.g. load store有对其版本和不对齐版本，速度差异会很大)

x86里向量化操作的clock cycle是与scalar一致的

* arm

TODO 有空补充

arm里向量化操作的clock cycle不一定与scalar一致，经常比scalar要大一点




## Memory




#### Locality

> Berkeley CS 267 L2

1. spatial locality : 访问临近的物体
2. temporal locality : 重复访问一个物体




#### Two Cost

> Berkeley CS 267 L2

1. latency : cost to load 1 word $\alpha$
   1. latency to computation ratio 变得越来越小，以后latency是问题。这个差距比bandwidth还严重
2. Bandwidth : average rate (bytes/sec) to load/store large chunk. ($\beta$ : time/bytes)
   1. bandwidth to computation ratio 变得越来越小，以后bandwidth是问题

time to load n words = $\alpha + \beta * n$




#### Handling Memory Latency

> Berkeley CS 267 L2

1. reuse value in fast memory : 增加CompInten, 减少bandwidth负担
2. move larger chunks 一次性取很多数据，而不是一个数据，充分利用bandwidth
   1. 例子是多个线程取数据
3. 使用SIMD issue multiple read & write in single instruction. 来重温利用bandwidth
4. 使用多线程 issue multiple read & write in parallel. 来hide latency




### Simple Memory Design

> 参考
> 
> 1. CMU 15.618 Lecture 23




#### memory system

<img src="Note.assets/Screen Shot 2021-10-22 at 10.39.12 AM.png" alt="Screen Shot 2021-10-22 at 10.39.12 AM" style="zoom:50%;" />




#### DRAM array

* 发生一次内存请求的时候
1. Precharge row buffer。把当前row buffer的数据放回到对应的row，prechagre来准备下一个row传输信息进来。
2. row activation 选择对应的row，链接circuits。transfer对应的row到row buffer中
3. column selection选择对应的column
4. transfer data onto bus 数据传输走

注意这里data pins只有8 bits，也就是一次最多可以传输8 bits

<img src="Note.assets/Screen Shot 2021-10-22 at 10.41.27 AM.png" alt="Screen Shot 2021-10-22 at 10.41.27 AM" style="zoom:30%;" />

* 发生第二次内存请求的时候。如果是同一个DRAM row
1. column selection
2. transfer data onto bus

不再发生precharge和

* DRAM Latency

DRAM访问的latency是不一样的。这里提到的latency与cache没有关系

最慢：Precharge (PRE) + row activate (RAS) + column access (CAS)

最快：CAS




#### 加速DRAM数据传输

* DRAM burst mode

每次PRE RAS后传输更多的数据。也就是选择多个column传输出去。一般在连续内存访问或者prefetch比较有用。

amortize latency over larger transfer

<img src="Note.assets/Screen Shot 2021-10-22 at 10.47.19 AM.png" alt="Screen Shot 2021-10-22 at 10.47.19 AM" style="zoom:30%;" />

<img src="Note.assets/Screen Shot 2021-10-22 at 10.47.01 AM.png" alt="Screen Shot 2021-10-22 at 10.47.01 AM" style="zoom:30%;" />

* DRAM multiple bank

多个bank (dram 2d array)，一个data pins，从而以pipeline的形式组织memory

<img src="Note.assets/Screen Shot 2021-10-22 at 10.48.55 AM.png" alt="Screen Shot 2021-10-22 at 10.48.55 AM" style="zoom:30%;" />

* DIMM 

多个dram chip （每一个chip是上面multiple bank）

一次性从多个chip上读取，每个chip读取8个bit，组合成一次读取。增加了有效带宽

需要由memory controller控制内存如何储存

<img src="Note.assets/Screen Shot 2021-10-22 at 5.51.03 PM.png" alt="Screen Shot 2021-10-22 at 5.51.03 PM" style="zoom:30%;" />

* memory controller scheduler

memory controller 维持一个 queue for each bank. FR-FCFS first ready, first come first sever queue. 

如果来的数据与当前在active row上的是同一个row，就优先处理

如果来的数据与当前active row的不是同一个row，则放到queue中

<img src="Note.assets/Screen Shot 2021-10-22 at 5.52.57 PM.png" alt="Screen Shot 2021-10-22 at 5.52.57 PM" style="zoom:30%;" />

* dual channel memory system 

each channel can issue independent request 

也就是有两个memory controller

* 3d stacking & HBC/HBM

HBC : hybrid memory cube 以3D的形式组织dram，从而减小dram到cpu的距离

灰色的部分是interposer 用于链接cpu和不同的memory的。与传统的cpu不同的是，memory与cpu使用interprocesor链接，而不是电路板。

<img src="Note.assets/Screen Shot 2021-10-22 at 5.54.38 PM.png" alt="Screen Shot 2021-10-22 at 5.54.38 PM" style="zoom:30%;" />

MCDRAM 

intel Xeon Phi上提供的片上16GB内存。使用3d stack的方法。

```cpp
float* foo = hbw_malloc( sizeof(float) * 1024 );
```




#### Bulk copy

memcpy会发生如下情况。产生bandwidth问题。但是no computation is performed

<img src="Note.assets/Screen Shot 2021-10-22 at 5.57.26 PM.png" alt="Screen Shot 2021-10-22 at 5.57.26 PM" style="zoom:30%;" />

通过改变memory构造，让memcoy避免走到cpu (CMU的一项active research)

<img src="Note.assets/Screen Shot 2021-10-22 at 6.03.49 PM.png" alt="Screen Shot 2021-10-22 at 6.03.49 PM" style="zoom:30%;" />




#### Frame buffer compression in GPU

现代GPU有looselessly compressing frame buffer

使用大数值储存p，delta x, delta y. 使用小数值储存c1, c2, ... c12 (correction value)

<img src="Note.assets/Screen Shot 2021-10-22 at 6.06.57 PM.png" alt="Screen Shot 2021-10-22 at 6.06.57 PM" style="zoom:30%;" />




#### Non-uniform memory access NUMA

> 参考
> 
> 1. CMU 15.418 l3

<img src="Note.assets/Screen Shot 2021-10-16 at 11.48.29 PM.png" alt="Screen Shot 2021-10-16 at 11.48.29 PM" style="zoom:50%;" />

优点

1. 更scalable

缺点

1. 每个processor访问不同内存时间不同
2. 很大程度受到core通信的interconnect的限制
   1. Intel quickpath QPI




#### Little's law for bandwidth concurrency

Concurrency = latency * bandwidth

latency = 10 sec, bandwidth = 2 bytes/sec, concurrency = 20 bytes in flight to hit bandwidth speeds

<img src="./Note.assets/Screen Shot 2022-01-26 at 7.37.41 PM.png" alt="Screen Shot 2022-01-26 at 7.37.41 PM" style="zoom:50%;" />

<img src="./Note.assets/Screen Shot 2022-01-26 at 7.38.37 PM.png" alt="Screen Shot 2022-01-26 at 7.38.37 PM" style="zoom:50%;" />



### Communication Lower Bound on Nested Loop

> Ref
>
> Berkeley CS 267 Lectuer 6

* 是什么

在nested loop情况下的算法，communication lower bound是什么，在什么情况下才能达到communication lower bound



* 为什么

memory movement is the most expensive in terms of enegery and speed in computation.

Data movement 的时间很久

data movement占用chip上的大小很大

data movement consume最多的能量



#### N-body

* 是什么

```cpp
for i = 1 : n
  for j = 1 : n
    F(i) = F(i) + force(A(i), A(j))
```

可以简化为two nested loop

```cpp
for i = 1 : n
  for j = 1 : n
    access A(i) and B(j)
```



* 利用cache进行数据reuse

如果有M大小的cache。则可以读取 M/2 的A， M/2的B，计算(M/2)^2 = M^2/4 次iteration

<img src="Note.assets/Screen Shot 2022-02-10 at 10.52.28 AM.png" alt="Screen Shot 2022-02-10 at 10.52.28 AM" style="zoom:50%;" />



* communication lower bound

每一次使用cache，可以处理 M^2/4 次iteration。这里的最优解是A B读取的数据一样多，这样映射得到的是square

为了处理 n^2 个iteration，需要读取 n^2 / (M^2/4)  = 4(n/M)^2 次数据从slow memory 到 fast memory

也就需要读取 4(n/M)^2 * M = 4n^2/M 的数据

lower bound = $\Omega (n^2/M)$



#### GEMM

* 从3D的角度理解GEMM

如果要计算一个C(i, j) 则需要对应的全部的 A B block

<img src="Note.assets/Screen Shot 2022-02-10 at 11.01.24 AM.png" alt="Screen Shot 2022-02-10 at 11.01.24 AM" style="zoom:50%;" />



* 利用cache进行数据reuse

把对应的A B C都放到cache里

能够同时处理的iteration/3D空间中的cube大小，upper bound by A B C block的面积的square

<img src="Note.assets/Screen Shot 2022-02-10 at 11.02.57 AM.png" alt="Screen Shot 2022-02-10 at 11.02.57 AM" style="zoom:50%;" />



* communication lower bound

假设cache大小为M，对应的A B C block最大为 $M/3$

对应的cube大小最大为 $((M/3)^3)^{1/2} = M^{3/2}/27$

因为是3 nested loop，总的iteration 数量是 $n^3$，也就是 $n^3/ (M^{3/2}/27)) $个对应的cube。

总共需要读取的数据是 $M * n^3/ (M^{3/2}/27)) = n^3 * 27 / M^{1/2}$

lower bound是 $\Omega( n^3 ) / M^{1/2} $

最优算法对应是cube



* communication lower bound for parallel case

对于parallel来说，fast memory = local processor memory，slow memory = other processor memory

lower bound on number of read/write = number of words moved between one processor and others

<img src="Note.assets/Screen Shot 2022-02-10 at 11.29.43 AM.png" alt="Screen Shot 2022-02-10 at 11.29.43 AM" style="zoom:50%;" />



#### General

* communication lower bound

不管多少个loop，不管多少个index，只要能找到如下的映射关系就可以。

需要A的读取，B的读取，C的读取是连续的（可能需要reorder loop）

<img src="Note.assets/Screen Shot 2022-02-10 at 11.31.44 AM.png" alt="Screen Shot 2022-02-10 at 11.31.44 AM" style="zoom:50%;" />

<img src="Note.assets/Screen Shot 2022-02-10 at 11.32.28 AM.png" alt="Screen Shot 2022-02-10 at 11.32.28 AM" style="zoom:50%;" />



* 是否是实际可行的lower bound

可行，但是取决于loop reorder等性质

<img src="Note.assets/Screen Shot 2022-02-10 at 11.32.49 AM.png" alt="Screen Shot 2022-02-10 at 11.32.49 AM" style="zoom:33%;" />



#### CNN

* 是什么

CNN 可以转化为7 nested loop

<img src="Note.assets/Screen Shot 2022-02-10 at 11.34.09 AM.png" alt="Screen Shot 2022-02-10 at 11.34.09 AM" style="zoom:50%;" />



* communication lower bound

<img src="Note.assets/Screen Shot 2022-02-10 at 11.34.26 AM.png" alt="Screen Shot 2022-02-10 at 11.34.26 AM" style="zoom:50%;" />




## Cache

* memory wall

处理器的吞吐量与内存的吞吐量差异性越来越到。导致很多程序变为memory boun的




### Cache design




#### Multicore Cache Hierchy

<img src="Note.assets/Screen Shot 2021-10-17 at 12.17.10 PM.png" alt="Screen Shot 2021-10-17 at 12.17.10 PM" style="zoom:30%;" />




#### Cache line

<img src="Note.assets/Screen Shot 2021-10-17 at 12.15.10 PM.png" alt="Screen Shot 2021-10-17 at 12.15.10 PM" style="zoom:30%;" />




#### Associativity

> 参考
> 
> 1. MIT 6.172 l14

* fully associative 

cache block can reside anywher in the cache

缺点：Search entier cache for tag。速度比较慢

<img src="Note.assets/Screen Shot 2021-10-17 at 1.26.52 PM.png" alt="Screen Shot 2021-10-17 at 1.26.52 PM" style="zoom:30%;" />

* direct mapped cache 

cache block's set determines its location in cache

缺点：某些特定的pattern访问，会导致很高的eviction率

相当于set associative cache + 每个set里面只有一个cache line。

* set associative cache

cache分为多个set，memory也分为多个set，按照set的对应关系进行替换

<img src="Note.assets/Screen Shot 2021-10-17 at 1.28.36 PM.png" alt="Screen Shot 2021-10-17 at 1.28.36 PM" style="zoom:50%;" />

<img src="Note.assets/Screen Shot 2021-10-17 at 1.29.12 PM.png" alt="Screen Shot 2021-10-17 at 1.29.12 PM" style="zoom:40%;" />




#### Cache miss

> 参考
> 
> 1. MIT 6.172 l14

* cold miss 
  
  * 因为第一次访问内存所以导致的miss。无法避免/减少

* capacity miss 
  
  * 因为cache的整体大小优先导致的eviction，就算是使用了fully associative cache也会导致eviction

* conflict miss 
  
  * cache里面存放了too many block from the same set in cache. 如果使用fully associative cache的话就不会产生miss
  * 因为limited associativeity （一个set里面的cache line数量太少）导致。

* sharing miss 
  
  * 另一个processor需要exclusive acceess to cache block
1. true sharing miss 
   1. 两个processor访问same data on same cache line. 如果有一个write则需要exclusive mode，另外一个processor需要放弃。
2. false sharing miss
   1. 两个processor访问different data on same cache line. 因为是以cache line为单位调度的，依旧需要evict。




#### Cache Compression

特点：data that falls within a cache line often has low dynamic range (similar in numericle value). 

可以压缩数据来减少使用的cache

* 使用一个base

<img src="Note.assets/Screen Shot 2021-10-22 at 6.05.34 PM.png" alt="Screen Shot 2021-10-22 at 6.05.34 PM" style="zoom:30%;" />

* 实际情况

一般使用两个base




#### Write

write back : 当cache line被evicted的时候&cache line设定为dirty，才会更改对应ram数据

write through：每当缓存有write指令，都需要把数据写回内存或下一级缓存。




#### volatile

C中的volatile是让编译器不要对这个变量的读写做任何优化，也就是不把变量保存在cache中，每次读取都直接去变量的内存地址读取，每次写也都直接写到变量的地址中。

如果某个内存地址中的值可能被一个线程或者另一个设备读写，则需要用volatile保证每一个thread都可以读取到最新值。




### Prefetch

> 参考
> 
> 1. 并行算法设计与性能分析 chapter 2

预取可以通过硬件或者软件控制。

* 特点
1. 如果程序的局部性很差，prefetch反而会降低性能
   1. prefetch的数据在后面没有使用，反而增加了bandwith的负担
2. prefetch过早，prefetch的内容在用到之前被evict了
3. associativeity 太小。prefetch了以后导致还在被用到的数据被evict
* 硬件预取

cache miss而从ram载入到cache的同时，也会连续请求下一个cache line。

* 软件预取

由compiler支持，buildin_prefetch 

TODO 补充一般prefetch多少比较合适




### Ideal cache model

> 参考
> 
> 1. MIT 6.172 l14

* 是什么
1. 用于分析cache performence的模型
2. two-level hierchy
3. cache size of M bytes
4. cache-line length of B bytes
5. fully associative
   1. ideal cache model 不考虑conflict misses，因为是fully associative cache.
6. Optimal, ominiscient replacement. 当evict cache line的时候，被evicted的是最优选择，当然这样的eviction policy是不存在的。

<img src="Note.assets/Screen Shot 2021-10-17 at 1.40.19 PM.png" alt="Screen Shot 2021-10-17 at 1.40.19 PM" style="zoom:30%;" />

* performence measurement
1. work $W$
2. cache misses $Q$
* 为什么使用ideal cache model
1. 先解决ideal cache模型的performence
2. 再考虑如何解决conflict misses
* LRU Lemma

建立LRU cache miss与ideal cache miss的关系

Suppose that an algorithm incurs Q cache misses on an ideal cache of size M. Then on a fully associative cache of size 2M that uses the least-recently used (LRU) replacement policy, it incurs at most 2Q cache misses

* Cache-miss lemma

<img src="Note.assets/Screen Shot 2021-10-17 at 4.34.03 PM.png" alt="Screen Shot 2021-10-17 at 4.34.03 PM" style="zoom:30%;" />

* Tall-cache assumption

number of cache line > cache line size

<img src="Note.assets/Screen Shot 2021-10-17 at 4.34.42 PM.png" alt="Screen Shot 2021-10-17 at 4.34.42 PM" style="zoom:33%;" />

* Submatrix caching lemma

<img src="Note.assets/Screen Shot 2021-10-17 at 4.35.44 PM.png" alt="Screen Shot 2021-10-17 at 4.35.44 PM" style="zoom:33%;" />

当submatrix可以放入到cache里的时候，读取submatrix最大的cache miss是 3n^2 / B

* cilk multi-processor cache miss

在idea-cache model下。找到multiprocessor的cache miss数量与serialize code cache miss数量的关系
$$
\mathrm{Q}_{P}=Q_{1}+\mathrm{O}\left(\mathrm{S}_{\mathrm{p}} \mathcal{M} / \mathcal{B}\right)
$$
$Q_P$ : number of cache misses in cilk coputation when run on P processor

$S_P$ : number of successful steals during computation. 

一般后面由于steal导致的cache miss term不会很大。

意义：减少serialized program的cache miss数量，可以减少平行运算时的cache miss 

证明：在cilk runtime里，只有在当前processor为空的时候才会去steal work，每次steal work了以后的worst case是全部都是cold cache。当M/B个cache miss以后，就与task在之前的processor上运行是一样的（之前processor最多有这么多保存了这个被steal的task的cache）。
$$
S_P = O(P T_{\infinity})
$$
$T_\infinity$ : span of computation. 理论最小多核时间 (间mIT 6.172 l7)




### Memory/Cache Coherence

> 参考
> 
> 1. CMU 15.418 l10
> 2. https://en.wikipedia.org/wiki/Cache_coherence

* 为什么产生，是什么

In a shared memory multiprocessor system with a separate cache memory for each processor, it is possible to have many copies of shared data: one copy in the main memory and one in the local cache of each processor that requested it. When one of the copies of data is changed, the other copies must reflect that change. Cache coherence is the discipline which ensures that the changes in the values of shared operands (data) are propagated throughout the system in a timely fashion. 在一个共享内存多处理器系统中，每个处理器都有一个单独的高速缓存，可以有多个共享数据副本：一个副本在主内存中，另一个在请求它的每个处理器的本地高速缓存中。当数据的一个副本发生更改时，其他副本必须反映该更改。缓存一致性是确保共享操作数（数据）值的更改以及时的方式在整个系统中传播的原则。

Memory coherence problem exists because there is both global storage (main memory) and per-processor local storage (processor caches) implementing the abstraction of a single shared address space. 内存一致性问题的存在是因为存在全局存储（主存）和每处理器本地存储（处理器缓存），实现了单个共享地址空间的抽象。也就是对于programmer而言，并不存在cache，只有一个大的shared address space。

In other words: it is possible to put operations involving X (注意，是同一个variable/在同一个cache line上的variable) on a timeline such that the observations of all processors are consistent with that timeline

目的：The goal of cache coherence is to ensure that the memory system in a parallel computer behaves as if the caches were not there

现代计算机的cache coherence是以cache block为单位的。也就可能会发生fasle sharing miss. 

<img src="Note.assets/Screen Shot 2021-10-17 at 1.54.47 PM.png" alt="Screen Shot 2021-10-17 at 1.54.47 PM" style="zoom:30%;" />

* requirement / 另一种定义方式
1. write propagation
   1. Notification of a write must eventually get to the other processors. Note that precisely when information about the write is propagated is not specified in the definition of coherence.
2. write serialization
   1. Writes to the same address are serialized: two writes to address X by any two processors are observed in the same order by all processors.
   2. 之所以需要让其余的processor看到相同的write内容，是因为其余的processor可能在两次write中间read，如果看到的write的内容不同的话，会导致read不同
* 实现方法
1. software ( heavy overhead )
   1. OS use page fault
2. hardware ( light overhead )
   1. snooping
   2. directory




#### Snooping-Based

想法: all coherence-related activity is broadcast to all processors in the system (more specifically: to the processor’s cache controllers)。cache会告诉别的cache它发生了设么，别的cache也会告诉当前cache别的cache发生了什么。

main idea：to write to a line, cache must obtain exclusive access to it. The other cache invalid their copy of the line.

cache controller工作

1. load store by local processor
2. message it received from other cache on coherence-related activity

分类

1. invalid based
2. update based （当前并不常用）

缺点：Scalability of snooping implementations is limited by ability to broadcast coherence messages to all caches! 因为broadcast消息到其余的processor是有限的，也就限制了num processor可以放在一起。

##### MSI Write back invalidation protocol

目的：当一个processor需要exclusive的时候（write），其余的processor会放弃用用的数据。

states

1. invalid I ：数据不在cache中
2. shared S：cacheline在一个或者多个processor中被read使用到，但是对于当前processor来说并不确定其余processor是否有。
3. modified M：cache line在exclusive/dirty state

cache之间的数据传输经常是通过cache to cache transfers来实现的。尽管这让chip变得更加复杂，但是reduce latency, reduce memory bandwith required by application(否则全部的cache交换信息都要经过memory，也就是要求很高的bandwidth)

<img src="Note.assets/Screen Shot 2021-10-17 at 3.46.06 PM.png" alt="Screen Shot 2021-10-17 at 3.46.06 PM" style="zoom:50%;" />

##### MESI write back invalid protocol

增加了exclusive clean state, 让整体变得更加高效

<img src="Note.assets/Screen Shot 2021-10-17 at 3.48.21 PM.png" alt="Screen Shot 2021-10-17 at 3.48.21 PM" style="zoom:30%;" />

##### MOESI

intel/AMD使用的方法

由一个processor把cache line直接给其余需要的processor

##### Dragon write-back update protocal

when write to a line, immediately update value in all other cache line (in other processor). so that other processor don't need to drop and reload the cache line.

* 缺点
1. 已经不再使用，因为导致了high traficc给memory bandwidth带来了压力。
2. 如果不断的给一个cache line进行更新，也就代表不断的进行value update on other core. 导致了很大的bandwidth。
3. 有些时候p0更新了以后，其实p123并不需要，但是依旧被更新。

<img src="Note.assets/Screen Shot 2021-10-17 at 3.56.48 PM.png" alt="Screen Shot 2021-10-17 at 3.56.48 PM" style="zoom:50%;" />




#### Directory-Based

TODO 还需要补充




### Memory Consistency

> 参考
> 
> 1. CMU 15.618 Lecture 13

* 是什么，为什么产生

目的：obtain higher performance by allowing recording of memory operations in parallel system.

注意

1. 与是否有cache无关
2. 变量在不同的variable上




#### Sequentially consistent memory system

满足下面这4点要去的memory system。

定义：any parallel execution is the same as if all the memory operations were executed in some sequential order. 可以理解为memory像是一个switch一样在不同的processor之间选择，所以最后产生的一定是某一个顺序的。

1. W→R: write to X must commit before subsequent read from Y * 
2. R→R: read from X must commit before subsequent read from Y 
3. R→W: read to X must commit before subsequent write to Y 
4. W→W: write to X must commit before subsequent write to Y




#### Relaxed memory consistency model

允许certain memory ordering to be violeted. 目的是为了hiding latency

##### W->R constrain relaxed

下面的部分只relax了W->R, 并没有relax W->W(也就是不可以reorder pair of write, 不可以reorder pair of read).

常见的对比是TSO与PC产生的结果是否与sequential consistency的结果一致。

* write buffer

write buffer holds writes that have been issued by the processor, but not yet committed in the system(including cache)

是一种常见的optimization allow read processd writes.

<img src="Note.assets/Screen Shot 2021-10-18 at 2.29.41 PM.png" alt="Screen Shot 2021-10-18 at 2.29.41 PM" style="zoom:30%;" />

* Total store ordering TSO
1. Processor P can read B before its write to A is seen by all processors. (processor can move its own reads in front of its own writes)
2. Reads by other processors cannot return new value of A until the write to A is observed by **all processors**. 如果有一个processor看到A改变了，则全部processor都会见到改变。
* Processor consistency PC
1. Any processor can read new value of A before the write is observed by all processor. 如果有一个processor看到A改变了，不保证其余processor都看到改变。

##### W->W constrian relaxed

* Partial Store Ordering PSO

Execution may not match sequential consistency on program 1

##### Intel ARM

Arm : very relaxed consistency model

Intel: 提供了sunc instruction if software requires a specific instruction ordering. 

##### c++11 atomic

atomic 默认使用 sequential consistency.  为了保证正确定。

可以使用std::memory_order 告诉 compiler 不用 strong order，允许re-order




### Reduce cache miss

> 参考
> 
> 1. MIT 6.172 l14




#### conflict miss

* submatrix

submatrix对于一个column的数据访问会导致conflict misses。因为可能一个column的全部数据都碰巧在一个set里面。

解决方案：拷贝内存到一个临时的matrix里，然后再读取这个临时的matrix。

<img src="Note.assets/Screen Shot 2021-10-17 at 1.38.38 PM.png" alt="Screen Shot 2021-10-17 at 1.38.38 PM" style="zoom:33%;" />




#### false sharing miss

> 参考
> 
> 1. CMU 15.418 l10
> 2. Berkeley CS 267 L4 关于效果，关于本质原因

* multiple thread update close value on same cache line

本质上是由于cache coherence导致的。如果更改了某一个cache line上的数据，需要让相关联的core知道对cache line的改变，增加bandwidth的压力，让perf降低。

下面的例子通过把per thread variable放在不同的cache line上（pad var to cache line size），从而加快速度。避免了fasle sharing miss. 

由于有cache coherence protocol的限制，如果一直改变一个cache line的值，会不断的导致cache miss，drop cache line, communication. 

```cpp
// 全部的variable都在一个cache line上
int per_thread_counter[NUM_THREAD];

// variable在不同的cache line上
struct per_thread_state
{
  int counter;
  char padding[CACHE_LINE_SIZE - sizeof(int)];
};

per_thread_state per_thread_counter[NUM_THREAD];
```

* 实际效果

取决于how cache is implemented & cache coherence protocal. 减少false sharing带来的效果可能是小/大。如果避免了false sharing了以后带来优化不是很大，不用太惊讶，可能是bottleneck不是false sharing




#### cache-aware algorithm

> TODO 需要增加




#### cache-oblivious algorithm




#### cache efficiency analysis



