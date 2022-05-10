## Basic Knowlege


#### Impediments to speedup

> Reference
> 
> 1. MIT 6.172 lecture15

1. insifficient parallesim
2. scheduling overhead 
3. lack of memory bandwidth
4. true/false sharing




#### Arithmetic intensiry/computation intensity

> Reference
> 
> 1. CMU 15.418 L2
> 2. Berkeley CS267 L2

计算访存比

ratio of math operations to data access operations in a instruction stream. 

因为parallel program的时候，bandwidth是critical resource。大部分cpu op都是bandwidth bound application。

![Screen Shot 2022-01-26 at 8.04.09 PM](/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.04.09 PM.png)

actual time to run program
$$
f * t_f + m * t_m = f * t_f * (1 + t_m / t_f * 1 / CI)
$$
larger CI意味着time closer to $f * t_f$

machine balance $t_m/t_f$ 定义了machine的boundary

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.55.45 PM.png" alt="Screen Shot 2022-01-26 at 8.55.45 PM" style="zoom:50%;" />

* potential CI for vector matrix

matrix vector计算的效率很低，因为CI的理论上线只有2

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.08.00 PM.png" alt="Screen Shot 2022-01-26 at 8.08.00 PM" style="zoom:50%;" />

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.07.52 PM.png" alt="Screen Shot 2022-01-26 at 8.07.52 PM" style="zoom:30%;" />

* potential CI for matrix matrix

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.09.07 PM.png" alt="Screen Shot 2022-01-26 at 8.09.07 PM" style="zoom: 33%;" />

naive的方法CI=2

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.09.46 PM.png" alt="Screen Shot 2022-01-26 at 8.09.46 PM" style="zoom:33%;" />

blocked的方法CI=n/N

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.14.23 PM.png" alt="Screen Shot 2022-01-26 at 8.14.23 PM" style="zoom:33%;" />

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.15.44 PM.png" alt="Screen Shot 2022-01-26 at 8.15.44 PM" style="zoom:50%;" />

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.21.47 PM.png" alt="Screen Shot 2022-01-26 at 8.21.47 PM" style="zoom:50%;" />




#### Communication Lower Bound

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.38.25 PM.png" alt="Screen Shot 2022-01-26 at 8.38.25 PM" style="zoom:50%;" />

当使用straness的时候，减少了总的数据移动量

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.42.45 PM.png" alt="Screen Shot 2022-01-26 at 8.42.45 PM" style="zoom:50%;" />




#### Parallel Program Process

> Reference
> 
> 1. CMU 15.418 l4

* Four phases
1. decomposition : divide problem into subproblems
   1. identify dependency
   2. create **at least enouh tasks** to keep all execution unit on machine busy. 
   3. **paralel slack**: 一般创建的task数量是execution capacity的4/8/10倍，为了使更充分的利用硬件资源（thread scheduling)
   4. 但是也不希望创建过多的task，因为过多的task有较多的overhead（steal，queue）
   5. programmer负责
2. Assignment : assign aubproblem to workers ( parallel threads )
   1. goal: Balance workload, reduce communication cost
   2. static assignment, dynamic assignment
   3. programmer/compiler负责
   4. 更多信息见【work assignment部分】
3. Orchestration: coordinate between worker, communication between thread，atomic，synchronization
4. mapping: map worker to hardware

<img src="Note.assets/Screen Shot 2021-10-17 at 10.26.27 AM.png" alt="Screen Shot 2021-10-17 at 10.26.27 AM" style="zoom:30%;" />




#### Amdahl's law

> Reference
> 
> 1. CMU 15.418 l4

dependency limit max speedup due to parallisim

speed up will be limited by the fraciton that's not able to parallel

S = fraction of sequential execution

max speed up < $1 \over S$




#### fork-join model

> Reference
> 
> 1. CMU 15.418 l6

<img src="Note.assets/Screen Shot 2021-10-17 at 11.34.36 AM.png" alt="Screen Shot 2021-10-17 at 11.34.36 AM" style="zoom:33%;" /> 

* 实现

使用thread pool的方法实现。create exactly as many worker threads as execution contexts in the machine. 

实际的runtime会在第一次调用cilk_spwan的时候启动对应数量的thread。是一种lazy initialization

<img src="Note.assets/Screen Shot 2021-10-17 at 11.40.47 AM.png" alt="Screen Shot 2021-10-17 at 11.40.47 AM" style="zoom:30%;" />

会先run child，然后再run parent(cilk的实现方法)，这样能够避免for loop里启动很多的child，让processor0的queue包含全部的task，其余processor都需要从processor0里steal

cilk plus runtime always run spwawned child. 




#### Parallel  & Other

> Berkeley CS 267 L1

##### Concurrent & Parallel

concurrent : 多个任务逻辑上（编程抽象上）平行运算，但是实际上可能只是serial exec (multi thread on single core single thread block)

parallel : 多个任务在物理硬件上同时active



##### Distributed & Parallel

distributed : 一个master，多个client，一起工作。但是client并不一定一起计算

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 7.06.49 PM.png" alt="Screen Shot 2022-01-26 at 7.06.49 PM" style="zoom:50%;" />




#### Supercomputer Today

现在的大多是petaflop，正在努力搭建exaflop

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 7.07.01 PM.png" alt="Screen Shot 2022-01-26 at 7.07.01 PM" style="zoom:50%;" />




#### Moore's Law

2x transistors/chip every 1.5 years

clock rate goes up by x -> raw computing power goes up by $x^4$

但是增加transistor是有限的，因为有物理限制，还有heat density限制

唯一可行的方案是增加parallel




#### Roofline Model

> Berkeley CS 267 L3

Developed by Sam Williams @ Berkeley, lots of citations, become a verb `rooflien that model`

Idea: application is limited by (1) computation (2) memory bandwidth

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.50.47 PM.png" alt="Screen Shot 2022-01-26 at 8.50.47 PM" style="zoom:30%;" />

* arith perf

横向的是计算intensity，使用SIMD，更好的ILP，balance mul add的目的是为了提升横向线的高度

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.51.27 PM.png" alt="Screen Shot 2022-01-26 at 8.51.27 PM" style="zoom:30%;" />

* machine balance

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.53.11 PM.png" alt="Screen Shot 2022-01-26 at 8.53.11 PM" style="zoom:30%;" />

machine balance的数值越低，同样的算法适配到硬件上就更容易memory bound

machine balance是computration intensity的threshold

横轴是computational intensiry. 

Haswell : 10 Flops/Byte

KNL : 7 Flops/Bytes

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.57.15 PM.png" alt="Screen Shot 2022-01-26 at 8.57.15 PM" style="zoom:30%;" />

* data movement

使用compulsory data movement，也就是size of input output作为datam ovement, 不管reuse的问题

是一个upper bound not to exceed

* CI

<img src="/Users/xiaosx/Library/Mobile Documents/com~apple~CloudDocs/HPC/Note.assets/Screen Shot 2022-01-26 at 8.55.23 PM.png" alt="Screen Shot 2022-01-26 at 8.55.23 PM" style="zoom:30%;" />

* memory bandwidth

使用prefetch，临近内存访问，是为了让bandwidth往左边倾斜，从而更容易comp bound

<img src="Note.assets/Screen Shot 2022-01-26 at 8.59.26 PM.png" alt="Screen Shot 2022-01-26 at 8.59.26 PM" style="zoom:30%;" />

<img src="Note.assets/Screen Shot 2022-01-26 at 9.00.39 PM.png" alt="Screen Shot 2022-01-26 at 9.00.39 PM" style="zoom:33%;" />

* Bounding

<img src="Note.assets/Screen Shot 2022-01-26 at 9.00.03 PM.png" alt="Screen Shot 2022-01-26 at 9.00.03 PM" style="zoom:33%;" />

* common result

<img src="Note.assets/Screen Shot 2022-01-26 at 9.02.58 PM.png" alt="Screen Shot 2022-01-26 at 9.02.58 PM" style="zoom:50%;" />




#### Theory Peak Performence

performence in GFlops = cpu speed in GHz * number of cpu core * CPU instructions per cycle * number of double float operation per FMA instruction.

Flops : floating point operation, usually double precision. 也就是 number of operations

Flop/s : floating point operations per second.

Bytes : size of data ( double precision float is 8 bytes )



* KNL Cori

1.4 GHz per core

2 512 bits vector unit ( 2 instrction per cycle )

GFlops = 1.4 (GHz) * 2 (instruction per cycle) * 8 (8 double float per instriction) * 2 (fma count as 2 instruction)



#### Performence Model : Latency and Bandwith Model

> Ref
>
> Berkeley CS267 Lecture 9

$\alpha$ : latency

$\beta$ : inverse of bandwith

$time = \alpha + n * \beta$



特点：

1. long msgch eaper than many shorter one
2. one msg time cost = thousands of flops



注意：对于multi-core cpu来说并非是准确的模型，因为NIC会是bottleneck，但是这个模型没有考虑到NIC Bottleneck

 
