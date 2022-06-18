# GPU

## Questions & TODO

1. warp level atomic
2. parent grid and child grid 嵌套
3. 总结一下global memory如何声明
4. read only cache for global memory 总结 [link to read](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-3-0)
5. write的时候发生了多少，是否经过L1 cache，粗细粒度是多少
6. GPU bank的划分
7. shared meomoey的访问模式



## Design Principle

### CPU VS GPU

> Reference
>
> 1. Berkeley CS 267 Lecture 7
>
> 2. UIUC 408 L1
> 3. Programming Massively Parallel Processors 3rd chapter 1



* 什么是GPU

GPU是heterogeneous chip. 有负责不同功能的计算模块

<img src="Note.assets/Screen Shot 2022-02-10 at 11.38.25 AM.png" alt="Screen Shot 2022-02-10 at 11.38.25 AM" style="zoom:50%;" />



SMs: streaming multiprocessors

SPs: streaming processors : each SM have multiple SP that share control logic and instruction cache



* 为了设么设计

GPU design for high throughput, don't care about throughput so much

CPU design for low latency

<img src="Note.assets/Screen Shot 2022-05-21 at 11.15.12 AM.png" alt="Screen Shot 2022-05-21 at 11.15.12 AM" style="zoom:50%;" />



* CPU GPU

CPU : multicore system : latency oriented 

GPU : manycore / many-thread system : throughput oriented



#### Idea to design throuput oriented GPU

* Idea 1 ： 去除CPU中让CPU serialize code运行更快的

CPU中包含out of order execution, branch predictor, memory prefetch等机制让CPU运行serialize code fast，但是这些部分占用很大的memory和chip。

GPU去除这些部分。

<img src="Note.assets/Screen Shot 2022-02-10 at 11.45.50 AM.png" alt="Screen Shot 2022-02-10 at 11.45.50 AM" style="zoom:50%;" />



* Idea 2 ：larger number of smaller simpler core

相比起使用small number of complex core, GPU的工作经常simple core就可以处理。

但也带来了挑战，需要programmer expose large parallel从而充分利用全部的core



* idea 3：让simple core共享instruction stream，减少负责Fetch Decode的芯片面积

因为很多工作都是parallel的，所以多个small simple core共享instruction stream就可以，减少了chip上负责instruction stream的部分。

SIMT single instruction multiple threads. 

SIMT 与 SIMD 有一些不一样。SIMT可以平行thread，而SIMD只可以平行instruction



* idea 4：使用mask来解决branching

在CPU中使用branch prediction

在GPU中，使用mask来解决branching

<img src="Note.assets/Screen Shot 2022-02-10 at 11.50.08 AM.png" alt="Screen Shot 2022-02-10 at 11.50.08 AM" style="zoom:50%;" />





* idea 5：hide latency instead of reduce latency

CPU通过fancy cache + prefetch logic来avoid stall

GPU通过lots of thread来hide latency。这依赖于fast switch to other threads, 也就需要keep lots of threads alive.

<img src="Note.assets/Screen Shot 2022-02-10 at 11.51.41 AM.png" alt="Screen Shot 2022-02-10 at 11.51.41 AM" style="zoom:50%;" />



* GPU Register 特点

GPU的register通常很大，在V100里与half L1 cahce+shared memory一样大

经常也被叫做inverted memory hierchy




### Program Model

> 参考
>
> 1. UIUC ECE Lecture 2,3
> 1. Programming Massively Parallel Processors 3rd chapter 2,3

CUDA Threads: SPMD single program multiple data

CPU call GPU kernel function 是 asynchronize 的



#### Kernel function & Block

Grid -> Block -> Warp ( implementation 而不是program model ) -> threads 

每一个kernel function被map到一个grid上

threads in same block "shared memory" & synchronize

block之间的thread只能通过global memory的方法来进行信息交流

Block 被assign 到不同的streamming multi-professor (sm) 上. Threads are assigned to execution resources on a block by block basis.

block & warp execute in arbitary order

maxwell take max 32 blocks per sm

maxwell take max 2048 threads per sm

maxwell take max 1024 threads per block

一般会选择num block = multipler of number of SM

<img src="./Note.assets/Screen Shot 2022-05-29 at 5.33.49 PM.png" alt="Screen Shot 2022-05-29 at 5.33.49 PM" style="zoom:50%;" />

如果device与host同时被使用，compiler会生成两个版本的代码。



* 特点

1. cuda kernel 启动都是 async的
2. cudaMemcpy device to host 有 implicit barrier



##### 与硬件的对应关系

<img src="Note.assets/3_2.png" alt="3_2" style="zoom:50%;" />






#### SM warp scheduling

warp是sm内部的schedule unit。

size of warp 是取决于硬件的，现在的GPU硬件都是warp size32的，但是以后可能改变

最近的architecture，每一个SM可以同时运行多个warp的instruction。

warp within a block can be execute in any order w.r.t each other



* 为什么使用warp

为了share control unit



* block中thread到warp的映射

如果block是1D的，则每32个映射到一个warp上

如果block是2D/3D的，会先把2D/3D project到1D上，然后每32个thread映射到一个warp上



* 为什么每个SM中需要放多个warp，如果同一时间只能跑有限个warp？

如果一个warp的instruction无法立刻执行（等待前一个instruction的资源），则这个warp not selected for execution.

其余的resident warp nolonger waiting for resources会被运行。如果多个resident warp都可以运行，则会采用latency hiding的priority mechanism来选择先运行谁。

<img src="Note.assets/3_4.png" alt="3_4" style="zoom:50%;" />



**zero-overhead scheduling** : selection of ready warps for execution avoid introduycing idle or waisting time into execution timeline. 

如果有sufficent resident warp，则hardware will likely find warp to execute at any point in time.

能够切换到其余的warp来hide latency是GPU不需要大的cache，不需要branch prediction等硬件，可以把更多的硬件给floating point operation的原因



<img src="Note.assets/Screen Shot 2021-11-07 at 12.53.14 PM.png" alt="Screen Shot 2021-11-07 at 12.53.14 PM" style="zoom:40%;" />



* threads的状态

1. all thread inside block not scheduled on SM
2. all thread inside block scheduled on SM
   1. Warp that's SM is currently executing
   2. Warp that ready to be executed by SM
   3. Warp that not ready to be executed because of dependency (e.g. load memory not finish yet)






#### Handle Corner Case

* handle 1d corner case

```cpp
__global__ void add( int* A, int* B, int* C, int n )
{
        i = blockIdx.x * blockDim.x + threadIdx.x;
      // avoid kernel larger than data
    if ( i < n )
      C[i] = A[i] + B[i];
}
```



* handle 2d corner case

<img src="Note.assets/Screen Shot 2021-11-07 at 12.45.45 PM.png" alt="Screen Shot 2021-11-07 at 12.45.45 PM" style="zoom:30%;" />

```cpp
__global__ void RGB2GRAY(unsigned char* gray, unsigned char* rgb, int width, int height)
{
  int col = threadIdx.x + blockIdx.x * blockDim.x;
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  if ( col < width && row < height )
  {
    // do computation
  }
}
```




#### Branch Divergence

GPU对于每一个thread使用predicated execution。

如果一个warp内的多个thread会走不同的path，则multiple path被实际运行。

```cpp
if ( threadIdx.x > 2 )
{
  // path 1
}
else
{
  // path 2
}
```

如果代码中有branching (if else)，但是warp内的thread只走一个path(都走了if condition)，不会有branching

一个解决branching的常用方法就是branch granularity是warp的倍数、这样就能保证一个warp内的全部thread都只走一个brach，而且依旧two cotrol path

```cpp
if ( threadIdx.x / WARP_SIZE > 2 )
{
  // path 1
}
else
{
  // path2
}
```



#### Synchronization within block

> Programming Massively Parallel Processors 3rd edition chapter 3

`__syncthreads()` 用来synchronize all threads in block

如果有if else then statement + syncthreads 在branch内部的话，all threads inside block either all run if or all run else. 否则就会wait forever

CUDA runtime只有在确保一个block所需要的全部resource都有的时候才会run block，否则会出现跑到一半发现resource不够，某些thread无法运行，其余的thread因为synchronize等待无法运行的thread。



* transparent scalability

同样的代码，在拥有不同资源的硬件上都可以运行。

blocks can execute in any order relative to each other, which allows for transparent scalability across different devices

CUDA为了保证transparent scalability，所以不允许block之间的synchronize。因为只有block之间不synchonize，block之间没有依赖关系，cuda runtime才可以arbitrary schedule block  on different hardware.

<img src="./Note.assets/IMG_D6008918F051-1.jpeg" alt="IMG_D6008918F051-1" style="zoom:50%;" />



#### Warp level primative

> reference
>
> 1. NVIDIA TECH BLOG Using CUDA Warp-Level Primitives [link](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)
> 2. Stackoverflow activatemask vs ballot_sync [link](https://stackoverflow.com/questions/54055195/activemask-vs-ballot-sync)



* 是什么：

使用shuffle指令，threads within single warp can access register of other threads within warp. warp内的thread可以访问其余thread的寄存器。

warp内的thread叫做lane。lane number from 0 to 31



* 为什么好

1. 更大的编程flexible
2. 原来thread之间进行数据交换需要使用shared memory，latency以及bandwidth都比register要小。现在在一个warp内部可以使用register，更小的latency以及更大的bandwidth



```cpp
// warp shuffle for reduction
val += __shfl_down_sync(0xffffffff, val, 16);
val += __shfl_down_sync(0xffffffff, val, 8);
val += __shfl_down_sync(0xffffffff, val, 4);
val += __shfl_down_sync(0xffffffff, val, 2);
val += __shfl_down_sync(0xffffffff, val, 1);

#define FULL_MASK 0xffffffff
for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
```



```cpp
// Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
```





##### Independent thread scheduling

> Reference
>
> NVIDIA TECH BLOG Inside Volta: The World’s Most Advanced Data Center GPU [link](https://developer.nvidia.com/blog/inside-volta/)

* 总结

从volta版本开始的program model开始，warp within thread schedule independently，而不是lock step

 Volta GV100 is the first GPU to support independent thread scheduling, which enables finer-grain synchronization and cooperation between parallel threads in a program. 



* Pascal SIMT Model

Pascal and earlier NVIDIA GPUs execute groups of 32 threads—known as warps—in SIMT (Single Instruction, Multiple Thread) fashion. The Pascal SIMT execution model maximizes efficiency by reducing the quantity of resources required to track thread state (one program counter shared across threads in warp) and by aggressively reconverging (下图中的reconverge) threads to maximize parallelism. 

<img src="Note.assets/image15.png" alt="Figure 11: Thread scheduling under the SIMT warp execution model of Pascal and earlier NVIDIA GPUs. Capital letters represent statements in the program pseudocode. Divergent branches within a warp are serialized so that all statements in one side of the branch are executed together to completion before any statements in the other side are executed. After the else statement, the threads of the warp will typically reconverge." style="zoom:48%;" />



* Volta SIMT model

maintaining execution state per thread, including the program counter and call stack 每一个thread都有自己的program counter

threads can now diverge and reconverge at sub-warp granularity, and Volta will still group together threads which are executing the same code and run them in parallel.



<img src="Note.assets/warp_pascal_volta.png" alt="Figure 11: Volta (bottom) independent thread scheduling architecture block diagram compared to Pascal and earlier architectures (top). Volta maintains per-thread scheduling resources such as program counter (PC) and call stack (S), while earlier architectures maintained these resources per warp." style="zoom:48%;" />



execution is still SIMT: at any given clock cycle CUDA cores execute the same instruction for all active threads in a warp just as before, retaining the execution efficiency of previous architectures. 程序依旧是以SIMT来运行的，只不过现在不同branch的工作允许intreleave了。

the scheduler supports independent execution of threads(through use of SIMT), it optimizes non-synchronizing code to maintain as much convergence as possible for maximum SIMT efficiency. scheduler允许每个thread独立运行，但是会尽量converge来增加SIMT利用率。

需要注意的是下图中Z运行之前并没有进行reconverge(因为thread scheduled independently），这是因为compiler认为Z可能与X Y有某些data dependency。但是这也导致SIMT的efficency降低（本来可以一个cycle运行Z，但是现在需要两个cycle，每次跑一部分）。

<img src="Note.assets/image5.png" alt="Figure 13: Volta independent thread scheduling enables interleaved execution of statements from divergent branches. This enables execution of fine-grain parallel algorithms where threads within a warp may synchronize and communicate." style="zoom:50%;" />

可以使用cuda9的syncwarp来保证线程之间converge，从而实现更大的SIMT efficency。如果在Z之前就syncwarp，则会提高SIMT利用率。

<img src="Note.assets/image16.png" alt="Figure 14: Programs can use explicit synchronization to reconverge threads in a warp." style="zoom:48%;" />



##### primatives

1.  Synchronized data exchange: exchange data between threads in warp.
   1. 这些sync语句要求thread首先被sync（也就是不需要再单独使用syncwarp()语句了），所以在调用这些语句的时候，数据thread会被sync （好像是cc 9+以后warp内的thread不保证一起执行）
   2. The new primitives perform intra-warp thread-level synchronization if the threads specified by the mask are not already synchronized during execution.
   3. Mask 的作用：mask to mean the set of threads in the warp that should participate in the collective operation.
   4. 老版本的warp primative 不enforce synchronization。使用老版本的代码叫做implicit warp-synchronous programming，是一个危险的行为。

```cpp
__all_sync, __any_sync, __uni_sync, __ballot_sync
__shfl_sync, __shfl_up_sync, __shfl_down_sync, __shfl_xor_sync
__match_any_sync, __match_all_sync

// Each thread that calls __shfl_sync() or __shfl_down_sync() receives data from a thread in the same warp
```



`unsigned __ballot_sync(unsigned mask, int predicate);` 会首先synchronize，不同的thread之间交换一个predicate （true/false)。用于create mask for other warp operation

```cpp
// 使用ballot_sync决定warp内只有部分thread参与计算，从而允许reduction计算的时候不是32的倍数
//  __ballot_sync() is used to compute the membership mask for the __shfl_down_sync() operation. __ballot_sync() itself uses FULL_MASK (0xffffffff for 32 threads) because we assume all threads will execute it.
unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < NUM_ELEMENTS);
if (threadIdx.x < NUM_ELEMENTS) { 
    val = input[threadIdx.x]; 
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
}
```



2. Active mask query: returns a 32-bit mask indicating which threads in a warp are active with the current executing thread.
   1. 并没有强制caller thread 会进行synchronize。The CUDA execution model does not guarantee that all threads taking the branch together will execute the __activemask() together. Implicit lock step execution is not guaranteed. 也就是一步一步的lock step是不保证的
   2. activamask只是用来知道哪些thread碰巧convergent了，并不保证activemask调用后也会lock step。activemask相当于detect，但并不是set。
   3. Don’t just use __activemask() as the mask value. __activemask() tells you what threads happen to be convergent when the function is called, which can be different from what you want to be in the collective operation.

```cpp
__activemask
```



下面这个例子就是一个错误的使用activemask的例子，因为并不保证activemask是被多个thread同clock执行的

```cpp
if (threadIdx.x < NUM_ELEMENTS) { 
    unsigned mask = __activemask(); 
    val = input[threadIdx.x]; 
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
}
```



3. Thread synchronization: synchronize threads in a warp and provide a memory fence.
   1. 之所以需要使用syncwarp是因为cuda9开始，warp内的thread并不保证在一个clock内同时运行相同的instruction。
   2. memory fense的道理是和syncthread与shared memory一起使用相同的。因为warp内的thread并不能保证lock step，所以在写入/读取shared memory的时候，需要使用syncwarp来确保memory fense

```cpp
__syncwarp
```



```cpp
// 一个错误的例子
// read write to shared memory 依旧可能导致race condition, 因为 += 代表read comp write 
shmem[tid] += shmem[tid+16]; __syncwarp();
shmem[tid] += shmem[tid+8];  __syncwarp();
shmem[tid] += shmem[tid+4];  __syncwarp();
shmem[tid] += shmem[tid+2];  __syncwarp();
shmem[tid] += shmem[tid+1];  __syncwarp();

// 正确的例子
// CUDA compiler 会在针对不同版本的arch选择删除_syncwarp。 
// 如果在cuda9之前的硬件上，threads in warp run in lock step, 则会删除这些syncwarp。
unsigned tid = threadIdx.x;
int v = 0;
v += shmem[tid+16]; __syncwarp();
shmem[tid] = v;     __syncwarp();
v += shmem[tid+8];  __syncwarp();
shmem[tid] = v;     __syncwarp();
v += shmem[tid+4];  __syncwarp();
shmem[tid] = v;     __syncwarp();
v += shmem[tid+2];  __syncwarp();
shmem[tid] = v;     __syncwarp();
v += shmem[tid+1];  __syncwarp();
shmem[tid] = v;
```



注意：syncwarp + 老版本的warp primative +  syncwarp 不等于新版本的 warp primative 。因为CUDA program model不保证thread stay convergent after leaving syncwarp



* 老版本效果

可以使用下面的command编译得到老版本的lock step效果

```cpp
 -arch=compute_60 -code=sm_70
```



#### Kernel切分决定

kernel之间有implicit barrier （kernel1的代码完全运行结束后，才会运行kernel2的代码）。如果kernel小而多的话，implicit barrier会成为overhead。



#### Get device property

```cpp
cudaDevicePeop dev_prop;
cudaGetDeviceProperties(&dev_prop, 0);

dev_prop.maxThreadsPerBlock;
dev_prop.multiProcessorCount;
dev_prop.clockRate;
dev_prop.maxThreadsDim[0/1/2];
dev_prop.maxGridSize[0/1/2];
dev_prop.warpSize;
dev_prop.regsPerBlock; // 每个sm可以使用的register的个数，通常与算法中总共的thread数量一起使用，从而确定每个thread可以使用多少个register，从而dynamic select which kernel to run。
dev_prop.sharedMemPerBlock; // 每个sm可以使用的shared memory大小。希望host code根据不同的hardware从而使用不同的shared memory大小，从而充分使用硬件。
```



#### Dynamic partition of resource

SM resource是动态分配给每一个block的，resource是有限的，需要决定block大小等来within limit of resource + use as many threads as possible



常见限制

1. num threads per sm
2. Num registers per sm
3. shared memory size per sm



##### 考虑block/sm threads/sm限制

thread到硬件的映射是以block为单位的。一个SM可以包含多个（有最多限制）的block。

如果SM内的一种多种resource不够支持最多block一起运行，cuda runtime则会以block为单位减少同时在一个SM上运行的block，从而保证all resources fall into limit of SM



* 假设 (下面的数据会依据使用的不同硬件改变)
1. SM take 8 blocks 
2. SM take 1536 threads 
3. 三个block分块的方式 $8*8, 16*16, 32*32$



* $8*8$ : 

64 threads per block

each sm can take 1536 / 64 = 24 blocks

但是sm限制最多8个block，也就是最多 8 * 64 = 512 threads ( out of 1536 threads ) go into one sm

这个是不好的，我们希望SM中的thread能够多到足够掩盖memory latency （warp schedule data已经好了的）



* $16 * 16$

256 threads per block

each sm can take 1536 / 256 = 6 blocks (小于block/sm的限制)

可以在硬件上使用full threads & block capacity



* $32 * 32$

1024 threads per block

each sm can take 1 block (小于block/sm的限制)

无法使用full threads



##### 考虑shared memory threads/sm block/sm限制

* 假设
1. 64 kb shared memory per SM
2. 2048 threads / sm
3. 150 GB/s bandwidth
4. 1000 GFlops
5. square matrix computation 使用 simple shared memory 16 / 32 的tile size
6. 使用shared memory进行tilening



* TILE_WIDTH 16

each block 使用 2 (submatrix M and N) * 16 * 16 * 4 bytes (float) = 2048 bytes shared memory

150 / 4 (bytes per float) * 16 = 600 GFlops : not yet fully utilize computation resources

64 kb / 2048 bytes/block = max 32 block per sm due to share memory constrain 

2048 threads / (16*16) = 8 block per sm due to threads limit

每一个时间点会有 2 (two load / threads) * 16*16 * 8 block/sm = 4096 pendig load (通过检查pending load来判断是否有让SM busy)。这个点很重要，因为使用多个thread的目的就是让sm有很多pending load，这样才能swap between warp



* TILE_WIDTH 32

each block 使用 2 * 32 * 32 * 4 bytes (float) = 8kb bytes share memory

150 / 4 (bytes per float) * 32 = 1200 GFlops : have potential to fully utilize computation resource

64/8 = max 8 block per sm due to share memory constrain

2048 / (32*32) = 2 block per sm due to threads limit

每一个时间点会有 2 * 32 * 32 * 2 = 4096 pending loads

同样的memory parallelsim exposed. 

尽管32的复用的内存更大，memory paralle与16一样。可能会存在一些block schedule的问题，因为更少的总block个数。



#### Cooperative Groups

> Reference
>
> 1. NVDIA TECH BLOG Cooperative Groups: Flexible CUDA Thread Programming [link](https://developer.nvidia.com/blog/cooperative-groups/)

* 是什么

原来的sync是within block的，从CUDA9开始，现在支持自定义thread group，可以是smaller than block，也可以是across block，甚至across gpu。group内的thread可以进行synchonize

<img src="Note.assets/synchronize_at_any_scale.png" alt="Figure 3. Cooperative Groups enables synchronization of groups of threads smaller than a thread block as well as groups that span an entire kernel launch running on one or multiple GPUs." style="zoom:50%;" />



* question

不确定这样的group synchronize从效率上来讲是好还是坏



#### Scheduling and Concurrency

> Reference
>
> 1. CMU 15.418 Spring 2016 lecture 5 slides 56 discussion [link](http://15418.courses.cs.cmu.edu/spring2016/lecture/gpuarch/slide_056)

![slide_056](Note.assets/slide_056.jpg)



**注意：下面的讨论都是围绕着slides中特定的硬件版本**

Each clock, the GTX 980 SMM core:

1. Selects up to four unique, runnable warps to run instructions from. These four warps can come from any thread block currently active on the core. This is an instance of simultaneous multi-threading (lecture 2). 每个clock，会从64个active warp中选择4个active wap。（active warp的定义是sm maintain warp execution context)。这里的平行是**simutaneous multi-threading**。之所以能选择4个warp是因为有4个warp scheduler (以及对应的pc)
2. From each of these four warps, the clock attempts to find up to two instructions to execute. This is instruction level parallelism (lecture 1) within the warp. If independent instructions are not present in the warp's instruction stream, then only one instruction from the warp can be executed. There is no ILP in the instruction stream! 每个clock，每个warp (out of 4)，会选择两个独立的instruction来运行。如果找不到两个独立的instruction来运行的话，则运行一个instruction。这里的平行是**ILP**. 这里independent instruction指的是会使用seprate functional units in SM
   1. e.g. 如果程序中有两个独立的FMA，SM硬件中有两组FMA，则在一个clock cycle内这两个FMA会同时运行

3. Out of the eight total instructions the core tries to find (across the four threads), up to four of those can be arithmetic instructions. These instructions will be executed on the four different groups of 32-wide SIMD ALUs that the core has. To be absolutely clear, all 4x32 = 128 ALUs in the SMM execute these four 32-wide instructions at the same time -- true parallel execution. (As pointed out in the footnote, and discussed a bit in the discussion on the previous slide, other non-basic-arithmetic instructions, like loads and stores, or special arithmetic ops like sin/cos/pow can also be executed simultaneously with these four basic-SIMD-arithmetic ops, but I didn't draw the units that perform these operations on this diagram.) 从最多8个可能的instruction里，最多4个是数学计算，可以同时被4个32长度的SIMTD ALU处理（same clock)，也可以同时处理load store



Note that in the diagram above, the core has the ability to maintain execution contexts for up to 64 warps at once (if you like to think in terms of CUDA threads rather than warps, this is 64x32=2048 CUDA threads). These warps are executed concurrently by the SMM core, but in any one clock the core will only execute instructions from at most four of them. This is the idea of interleaved multi-threading as illustrated here. SM可以维持64个active warp通过维持他们的execution context。这64个active warp可以来自多个block。SM对block的schedule是使用或者不使用整个block。SM通过interleave这64个warp来hide latency。



Do not confuse the requirement that all CUDA threads (or their corresponding warps) in a thread block must be live--a.k.a. occupying an execution context on a core-- during the lifetime of the thread block (a requirement that we discuss again on slide 73) with the fact that the core can indeed run instructions from multiple threads simultaneously on its parallel execution units. It seems to me that interleaved multi-threading and simultaneous execution of instructions from multiple threads are being confused in some of the comments posted above. 一个block内thread只要运行的时候就是都active的，因为SM是以block为单位分配资源。



#### Nested Kernel Call

从Kepler开始，GPU Kernel可以启动GPU Kernel。

Kepler之前，GPU kernel只能从CPU启动



<img src="Note.assets/3_8.png" alt="3_8" style="zoom:50%;" />






### Memory Model

> 参考
>
> 1. UIUC ECE Lecture 4
> 1. Berkeley CS 267 Lecture 7 on memory colesing
> 1. PMPP 3rd chapter 5



#### Hierchy of Memory

per thread register

per block shared mempry

per block l1 cache

per grid global memory

per grid l2 cache

per grid constant memory



global memory 是 off chip的

register / shared memory是on chip的



constant memory物理上在global memory上, 但是会cache for efficency 所以正确使用的话会很快

scope of constant varaible 是 across all grids. 是valid throughout program execution.



CUDA中pointer用来指向global memory

<img src="Note.assets/Screen Shot 2021-11-06 at 10.05.13 PM.png" alt="Screen Shot 2021-11-06 at 10.05.13 PM" style="zoom:70%;" />

<img src="Note.assets/Screen Shot 2021-11-07 at 1.23.18 PM.png" alt="Screen Shot 2021-11-07 at 1.23.18 PM" style="zoom:70%;" />



* local memory

array如果fix size + small in size有可能会被compiler放在register上。否则会被放在local memory上（物理上在off-chip global memoy上）

struct如果占用空间很大的话，也有可能被放在local memory上



##### 声明使用各种array

<img src="Note.assets/Screen Shot 2022-05-29 at 10.10.41 PM.png" alt="Screen Shot 2022-05-29 at 10.10.41 PM" style="zoom:50%;" />

* automatic variable

如果没有使用 `__device__`, `__shared__`, `__constant__`的话，则是automatic variable。如果是一个变量的话会是register。如果是per thread array的话会是global memory。



* local memory

per thread array (per thread local memory) 会被储存在 gloabl memory 中，因为compiler不知道这个array会有多长，无法把array拆分后放到regsiter中。

如果per thread array fix size + small in size, 则compiler有可能把它放在register中。

per thread struct 如果大的话，compiler会把它放在local memory中



* global memory

`__device__` 从device 声明使用global memory

cudaMalloc cudaMemCpy 从host声明使用global memory



#### Memory Coalesing and Burst

* 什么是coalesing

the hardware combines, or *coalesces*, all these accesses into a consolidated access to consecutive DRAM locations / combined multiple request into a single request for consecutive locations。也就是硬件会融合多个临近的访问为一个对DRAM的访问

GPU中的cache主要作用是memory coalesing，来combine access to DRAM into burst to reduce 总的 num access to DRAM

optimization：访问内存的时候要让多个thread连续访问，尽量使用完burst内全部数据，而且访问要尽量aligned to burst size



##### DRAM 

* 原理

一个capacitor储存bit

一个select选择读取哪个capacitor

一个bit line read / write数据。每个bit line只读取一个bit的数据，也就是多个select里面只select一个

需要constantly check value / recharge value



<img src="Note.assets/Screen Shot 2022-05-30 at 11.52.45 PM.png" alt="Screen Shot 2022-05-30 at 11.52.45 PM" style="zoom:50%;" />





* 特点

bit line的capacitance很大，导致速度很慢。

bit的capacitance很小，需要使用sense amplifier来放大信号



##### Burst & DRAM Bank

> Reference
>
> 1. NVIDIA Tech Blog Coalesced Transaction Size [link](https://forums.developer.nvidia.com/t/coalesced-transaction-size/24602)
> 2. Blog CUDA基础 4.3 内存访问模式 [link](https://face2ai.com/CUDA-F-4-3-内存访问模式/)



多个bit line组成core array

数据传输分为两个部分。core array -> column latches / buffer -> mux pin interface 

**burst** 当访问一个内存位置的时候，多个bit line的数据都会从core array传输到column latches，然后再使用mux来选择传送给bus interace哪个数据 / one burst of memory access to get data that used by multiple attemps to read.

**burst size** 读取一次memory address，会有多少个数据被放到buffer中。

常见的GPU burst size是 1024 bits / 128 bytes (from Fermi). 这里的burst size经常被叫做**line size**

当L1 cache disabled at compile time (default enable), burst size是32 bytes. 

<img src="Note.assets/Screen Shot 2022-05-31 at 12.02.50 AM.png" alt="Screen Shot 2022-05-31 at 12.02.50 AM" style="zoom:50%;" />



* 充分利用burst：连续访问内存

第一次访问，全部4个数据都放到buffer里

<img src="Note.assets/Screen Shot 2022-05-31 at 12.05.15 AM.png" alt="Screen Shot 2022-05-31 at 12.05.15 AM" style="zoom:50%;" />



第二次访问连续的数据，直接从buffer里读取数据，不用再去core array

<img src="Note.assets/Screen Shot 2022-05-31 at 12.05.50 AM.png" alt="Screen Shot 2022-05-31 at 12.05.50 AM" style="zoom:50%;" />



**bursting** 每一次读取burst of data，读取的数据应该被充分使用，因为读取burst里面的两个数据的时间远远小于读取两个random address/两个burst。

<img src="Note.assets/Screen Shot 2022-05-31 at 12.06.52 AM.png" alt="Screen Shot 2022-05-31 at 12.06.52 AM" style="zoom:50%;" />



##### Multiple Bank

只用burst并不能实现processor所需要的DRAM bandwidth。

因为bank访问core array cell的时间很长（下图蓝色部分）而实际使用bus interface传输数据时间很短（下图红色部分），通常比例是20:1， 如果只使用一个bank，interface bus会idle。所以需要在一个bus 上使用多个bank，来充分利用bus bandwidth。如果使用多个bank，大家交替使用interface bus，保证bus不会idle

通过多个bank链接到interface bus，从而让interface bus充分的使用，也就保证了每个时间都有数据从interface bus传送过来。

<img src="Note.assets/Screen Shot 2022-05-31 at 12.07.59 AM.png" alt="Screen Shot 2022-05-31 at 12.07.59 AM" style="zoom:50%;" />



<img src="Note.assets/Screen Shot 2022-05-31 at 12.08.29 AM.png" alt="Screen Shot 2022-05-31 at 12.08.29 AM" style="zoom:50%;" />



* 一个bus需要多少个bank？

如果访问core array与使用bus传输数据的时间比例是20:1，那么一个bus至少需要21个bank才能充分使用bus bandwidth。



一般bus有更多的bank，不仅仅是ratio+1，原因是

1. 使用更多的bank，更能让data spread out across bank。如果一块data只在一个bank上的话，需要多个burst才能完全访问（时间很久）。如果一块data在多个bank的话，可以overlap core array access time （总时间变短）
2. 每个bank可以存储的诗句有限，否则访问一个bank的latency会很大。



* 实现max bandwidth方法

In order to achieve the memory access bandwidth specified for device, there must be a 

1. sufficient number of threads making simultaneous memory accesses. 
2. Furthermore, these memory accesses must be evenly distributed to the channels and banks. 
3. Of course, each access to a bank must also be a coalesced access,



##### Multiple Channels

modern Double data rate （DDR） bus可以传输two word of data in each clock cycle. 

假设bus clock speed是1GHz， 每秒钟只能传送 8 bytes / words * 2 words per clock * 1 GHz = 16 GB/sec. 但是一般GPU processor要求128GB/s的数据

单独一个channel/一个bus interface不足以达到processor要求DRAM bandwidth，所以需要使用多个channel。

<img src="Note.assets/Screen Shot 2022-05-31 at 2.47.07 PM.png" alt="Screen Shot 2022-05-31 at 2.47.07 PM" style="zoom:50%;" />



* interleaved data distribution 

是什么：把array spread across banks and channel in the memory system. 这样允许core array acccess time overlap, 减少总access time. 

<img src="Note.assets/Screen Shot 2022-05-31 at 2.53.55 PM.png" alt="Screen Shot 2022-05-31 at 2.53.55 PM" style="zoom:50%;" />



##### Parallel memory & Parallel therad

1. good utilization of the potential access bandwidth of the DRAM system requires that many threads simultaneously access data that reside in different banks and channels.  
   1. 为了充分利用DRAM带宽，需要让多个thread同时访问DRAM数据，并且这些访问需要连续，需要在DRAM中分散在多个bank和channel上
2. the execution throughput of the device relies on good utilization of the parallel structure of the DRAM system.
   1. 充分利用DRAM贷款，需要访问在不同bank和channel上的的数据 （连续访问一段内存）。如果访问的数据在一个channel上，则无法充分利用带宽



##### Simple CUDA code analysis

对于simple CUDA kernel来说，是使用多个thread+多个iteration的方法来遍历M和N

对于N的访问是coalesced，多个thread在一个step访问的数据是连续的，也就是属于一个burst/充分使用burst



<img src="Note.assets/Screen Shot 2022-05-31 at 12.09.35 AM.png" alt="Screen Shot 2022-05-31 at 12.09.35 AM" style="zoom:50%;" />



对于M的访问不是coalesced的，每个thread读取的数据都会导致一次memory burst。Step1的4个value需要4个burst。在进行step2的时候，step1的burst被调换，导致memory得重新传输。

需要注意的是，这里使用4个thread+多个iteration来读取数据M，也就对应着simple cuda code

<img src="Note.assets/Screen Shot 2022-05-31 at 12.10.24 AM.png" alt="Screen Shot 2022-05-31 at 12.10.24 AM" style="zoom:50%;" />



##### Tiling CUDA with shard memory analysis / corner turning

**corner turning** : Use of a transposed thread order to allow memory loads to coalesce when loading global to shared.

当使用tilnig+每个thread读取一个M N到shared memory的时候，读取M也是burst的。这是因为比起上面的simple code使用iteration读取，这里使用多个thread读取，一次burst的数据会被临近的thread使用(M00 M01分别被2个thread读取，每个thread只读取一个M elem)，而不是下一个iteration被清空。

这里对于M没有使用transpose，但是因为使用多个thread读取数据，依旧保证了burst，这与CPU代码需要使用transpose是不一样的。

同时shared memory使用的是SRAM，不像DRAM有burst的问题，所以读取M的shared memory的时候尽管不是连续读取也没有问题。shared memories are implemented as intrinsically high-speed on-chip memory that does not require coalescing to achieve high data access rate.

<img src="Note.assets/Screen Shot 2022-05-31 at 12.10.56 AM.png" alt="Screen Shot 2022-05-31 at 12.10.56 AM" style="zoom:70%;" />



##### Align 2D image

> Reference
>
> 1. PMPP Chapter 7

<img src="Note.assets/IMG_463A2479525D-1.jpeg" alt="IMG_463A2479525D-1" style="zoom:50%;" />

当读取image 文件的时候，library经常会padded width = multiply of burst size. 

如果没有padded的话，raw 1的起始位置会是misaligned from DRAM burst，导致读取的时候读取多个burst，让速度变慢

padded info叫做 `pitch` 



#### Global Memory

##### Pinned memory copy

> Reference
>
> 1. NVIDIA Tech Blog How to Optimize Data Transfers in CUDA C/C++ [link](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
>    1. 包含了一个memory bandwidth test的代码
> 2. Stackoverflow Why is CUDA pinned memory so fast? [link](https://stackoverflow.com/questions/5736968/why-is-cuda-pinned-memory-so-fast)



* synchronize host to device memory copy

Host (CPU) data allocations are pageable by default. The GPU cannot access data directly from pageable host memory, so when a data transfer from pageable host memory to device memory is invoked, the CUDA driver must first allocate a temporary page-locked, or “pinned”, host array, copy the host data to the pinned array, and then transfer the data from the pinned array to device memory. 为了避免要拷贝的数据page  out，首先会使用一个临时的pinned memory拷贝数据到那里，然后再拷贝到device上（下图左）。尽管是从CPU的memory->memory，这个过程会经过CPU core，导致内存收到限制。（CMU的最新arch研究关于如何从cpu mem直接拷贝的mem，不经过cpu）

Not-locked memory can generate a page fault on access, and it is stored not only in memory (e.g. it can be in swap), so driver need to access every page of non-locked memory, copy it into pinned buffer and pass it to DMA 如果内存不是pinned的，则访问的时候对应的内存可能在disk/ssd上，需要经过CPU进行page swap，拷贝到临时的pinned memory，再使用DMA从临时pinned memory拷贝到device global memory上

```cpp
int *h_a = (int*)malloc(bytes);
memset(h_a, 0, bytes);

int *d_a;
cudaMalloc((int**)&d_a, bytes);
// synchronize copy
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
```



* asynchronize host to device memory copy

为了避免cpu change memory page, 需要使用pinned memory

之所以async是为了overlap data transfer with computation

可以有higher bandwidth比起没有pinned的内存

因为pinned内存是有限的资源，分配pinned内存可能会失败，所以一定要检查是否有失败

You should not over-allocate pinned memory. Doing so can reduce overall system performance because it reduces the amount of physical memory available to the operating system and other programs. 不要过度使用pinned memory，这会导致系统整体速度变慢

Locked memory is stored in the physical memory (RAM), so device can fetch it w/o help from CPU (DMA, aka Async copy; device only need list of physical pages). pinned内存可以直接使用DMA拷贝到GPU，不需要经过CPU，从而有更大的bandwidth。

<img src="Note.assets/pinned-1024x541.jpg" alt="pinned-1024x541" style="zoom:50%;" />



```cpp
int *h_aPinned, d_a;
checkCuda(cudaMallocHost((int**)&h_aPinned, bytes));
memset(h_aPinned, 0, bytes);

checkCuda( cudaMalloc((void**)&d_a, bytes) );

// synchronize copy
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
```



##### Batch small transfer

host memory -> device global memory 的拷贝是有overhead的。

希望避免多个small memory copy, 希望是one large memory copy

所以要pack多个small memory copy to large memory copy



#### Shared memory

* 是什么

1. on chip (For volta use same physical resources SRAM)
2. SRAM support random access
3. don't have constrain of burst like DRAM



* 什么时候使用

1. 数据有复用的话，考虑使用shared memory



* 使用时需要注意

1. 使用shared memory一定要注意不要忘记synchronize的使用
2. shared memory时有限的resource，需要考虑使用shared memory以后一个sm能有多少个thread和block



* load from global memory to shared memory 过程

内存拷贝与CPU相似，需要经过register

global memory -> cache L1/L2 -> per thread register -> shared memory

不存在直接从global memory到shared memory的硬件




#### Constant memory

> UIUC 408 Lecture 7




##### Constant cache

CPU中的cache需要保证cache coherence

GPU中因为多线程，实现cache coherence更加复杂

GPU有两种cache，L1 cache和constant cache。



* 特点

1. cosntant cache read only
2. 被constant memory, texture memory使用
3. 比起L1 cache，有更大的throughput aceess than L1 cache。尽管latency是一样的，都是5 cycle



##### Use constant memory

constant memory物理上在global memory上，是off chip的

使用constant memory主要是为了使用constant cache，从而减少对global memory的访问

全局可见

使用例子：image filter的weight，math formula的参数

```cpp
// constant memory declared outside all function
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH];

// copy from device to constant memory
cudaMemcpyToSymbol(Mc, Mask,, MASK_WIDTH*MASK_WIDTH*sizeof(float));
```





#### Atomic

> UIUC 508 Lecture 2

##### Evolving

GPU atomic随着GPU Arch也在改进

atomic on shared memory >> atomic on global memory 



* GT200

atomic is on global memory, no L2 cache 



* Fermi to Kelpler

both atomic on L2 cache

Improve atomic by add more l2 cache buffer 



* kepler to maxwell

improve shared memory atomic through using hardware. 

Kepler use software for shared memory atomic



* after maxwell

atomic is rouphly the same

the flexibility of atomic is changed. now have atomic within warp / block.



##### Warp-aggregated atomics

> reference
>
> 1. CUDA TECH BLOG CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics [link](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)



NVCC compiler (from CUDA 9) now performs warp aggregation for atomics automatically in many cases, so you can get higher performance with no extra effort. In fact, the code generated by the compiler is actually faster than the manually-written warp aggregation code. NVCC现在支持自动编译代码使用warp aggregation而且速度比手写的warp aggregation要更快，下面的例子只是用于展示warp aggregation这个方法。



是什么：In warp aggregation, the threads of a warp first compute a total increment among themselves, and then elect a single thread to atomically add the increment to a global counter. This aggregation reduces the number of atomics performed by up to the number of threads in a warp (up to 32x on current GPUs), and can dramatically improve performance. 以warp为单位进行atomic，首先在warp内部计算出来要atomic的值，然后选出一个thread执行atomic。这样减少了atomic操作的执行次数（atomic操作导致serial execution以及更低的bandwidth，更多的atomic操作带来更低的bandwidth）。

atoimc次数与bandwidth是log的反向相关。下图中的横轴可以理解为number of atomic operation.

<img src="Note.assets/image2.png" alt="Figure 1. Performance of filtering with global atomics on Kepler K80 GPU (CUDA 8.0.61)." style="zoom:60%;" />





#### Cache

##### cache 是什么

GPU Memory line是1024 bits / 128 bytes (same as burst size)

cache由多个cache line组成，每个cache line大小与memroy line大小一致，都是128 bytes



##### cache 与Shared memory比较

* same

1. both on chip. For volta use same physical resources SRAM



* different

1. prigrammer control shared memory 
2. micro-arch determine content of cache



##### disable l1 cache

> Reference
>
> 1. NVIDIA Tech Blog Cache behavior when loading global data to shared memory in Fermi [link](https://forums.developer.nvidia.com/t/cache-behavior-when-loading-global-data-to-shared-memory-in-fermi/29259)
> 1. NVIDIA Tech Blog Coalesed Transaction Size [link](https://forums.developer.nvidia.com/t/coalesced-transaction-size/24602)

load from global memory to register will be cache at L1 at default. 

If you use the inline PTX call “ld.global.cg”, the read will be cached in L2 but not L1. “ld.global.cs” will not cache the read in L1 or L2. Alternatively, you can use a simple NVCC flag to make either of these types of reads the default by using “-Xptxas -dlcm=cg” or “-Xptxas -dlcm=cs”, though of course that then applies to ALL reads in your program.



##### Read-only cache for global memory access

> Reference
>
> 1. Blog CUDA-F-4-3-内存访问模式 [link](https://face2ai.com/CUDA-F-4-3-内存访问模式/)
> 2. Stackoverflow What is the difference between __ldg() intrinsic and a normal execution? [link](https://stackoverflow.com/questions/26603188/what-is-the-difference-between-ldg-intrinsic-and-a-normal-execution)

从computation capacity 3.5+， 可以利用read only cache来读取global memory。

read only cache的burst size是32 bytes，而不是L1 cache的128 bytes。

更适合于random access data

```cpp
__global__ void copyKernel(float * in,float* out)
{
    int idx=blockDim*blockIdx.x+threadIdx.x;
    out[idx]=__ldg(&in[idx]);

}
```



#### Register

##### Register size

> Reference
>
> 1. Saving registers with smaller data types? [link](https://forums.developer.nvidia.com/t/saving-registers-with-smaller-data-types/7376)

Registers 是 32 bit / 4 bytes 大小的 (same size as int / float)。如果数据类型是double的话，则使用2个register。

可以通过pack small data into a register (e.g. 2 short) and use bitmask + shift 来读取。从而减少register usage per thread





#### Memory Fence

> Reference
>
> 1. CUDA Toolkits document [link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)
> 2. stackoverflow [link](https://stackoverflow.com/questions/5232689/cuda-threadfence)



* 是什么

CUDA使用weakly-ordered memory model。一个thread写入shared memory, global memory, paged lock memory的顺序与另一个thread观察到的顺序是不一样的。如果两个thread一个read，一个write，没有sync的话，则行为是undefined的

通过使用memory fence，保证 (1) all write before fence对于程序(不同的scope)来说发生在all write after fence之前. (2) all read before fence对于程序(不同的scope)来说发生在all read after fence之前



* 三个方程

```cpp
// fence for all thread within one block
void __threadfence_block();

// fence for all thread within one GPU device
void __threadfence();

// fence for all thread across all GPU device
void __threadfence_system();

```



* 例子 1

下面这个例子中，不可能得到A=1,B=20。因为X=10一定发生在Y=20之前，如果observe了Y=20的话，则X=10一定运行完了

```cpp
__device__ int X = 1, Y = 2;

__device__ void writeXY()
{
    X = 10;
    __threadfence();
    Y = 20;
}

__device__ void readXY()
{
    int B = Y;
    __threadfence();
    int A = X;
}
```



* 例子 2

Imagine, that one block produces some data, and then uses atomic operation to mark a flag that the data is there. But it is possible that the other block, after seeing the flag, still reads incorrect or incomplete data.

一个block写入global memory数据以及用atomic写入flag，另一个block通过flag判断是否可以读取global memory的数据。

 If no fence is placed between storing the partial sum and incrementing the counter, the counter might increment before the partial sum is stored 

如果没有memory fence的话，可能flag会首先被atomic设置了，然后才设置global memory的数据。这样另一个block在读取到flag以后就开始读取global memmory的值可能就是不对的。

通过使用memory fence，确保在fence后面读取memory的数据确实是fence之前写入的数据



#### Volatile

> Reference
>
> 1. CUDA Toolkits Document I.4.3.3 [link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#volatile-qualifier)

* 是什么

compiler可以对global memory/shared memory的read write进行优化，例如cache在L1 cache或者register上，只要符合memory fence的要求就可以进行优化。

声明volatile以后，compiler会不optimize，全部的写入会写入到gloabl memory/shared memory上。这样另一个thread可以读取对应的内存并且得到正确的数值。



## Common Optimization Techniques

> Reference
>
> 1. Algorithm and Data Optimization Techniques for Scaling to Massively Threaded Systems

<img src="Note.assets/Screen Shot 2022-06-04 at 5.09.40 PM.png" alt="Screen Shot 2022-06-04 at 5.09.40 PM" style="zoom:50%;" />



### Data Layout Transformation

> Reference
>
> DL A Data Layout Transformation System for Heterogeneous Computing

GPU充分利用burst memory是很重要的优化方法

如果burst内的数据没有立刻被使用的话（DRAM的buffer中存放burst），则会被下一个burst代替，需要重新传输。

对于CPU来说，data layout对程序的影响没有那么显著，因为CPU有large cache per thread，可以cache部分数据，没有那么依赖于DRAM的burst data。下面的array of struct结构中，thread0的cache会储存整个struct的内容。

对于GPU来说，data layout对程序的影响很显著，因为GPU的cache比较小。GPU的cache主要适用于memory coalesing，而不是locality

SoA或者DA(discrete array)的结构对GPU有用，因为充分利用burst 的结果

<img src="Note.assets/Screen Shot 2022-06-04 at 5.00.49 PM.png" alt="Screen Shot 2022-06-04 at 5.00.49 PM" style="zoom:50%;" />



ASTA array of structures of tiled arrays 是一种 SoA的变体。相当于AoS of mini-SoA(of size coarsening factor)

1. 解决OpenCL需要对不同hw有不同数据结构的kernel的问题
2. 解决`partition camping`，也就是数据集中在某一个bank/channel上，没有充分利用DRAM aggregate bandwidth

通常`coarsening factor` (下面eg是4) at least the number of thread partitioning in memory access (num thread in block)

<img src="Note.assets/Screen Shot 2022-06-04 at 5.07.21 PM.png" alt="Screen Shot 2022-06-04 at 5.07.21 PM" style="zoom:50%;" />



在NVIDIA的arch下，DA与ASTA的perf相似

<img src="Note.assets/Screen Shot 2022-06-04 at 5.09.01 PM.png" alt="Screen Shot 2022-06-04 at 5.09.01 PM" style="zoom:50%;" />



### Scatter to Gather

> UIUC ECE 508 Lecture 2

scatter对于编程来说更加直接

GPU应该避免使用scatter，应该使用gather的方法

在GPU上的程序改变scatter为gather可以提升性能



* 是什么

scatter : parallel over input, writing value to non-contigious memory location

gather : parallel over output, reading values from non-contigious memory location。也叫做owner compoutes rules



* scatter 缺点

1. contentious write (write conflict) 需要被hardware serialize。（下图红色的arrow）。当thread多的时候会有很多conflict，write到某一个位置会被serialized
2. random write无法充分利用memory burst
3. atomic的arch直到最近才被支持



* gather 优点

1. write的时候充分利用burst
2. 没有write conflict，不需要serialize write



* 程序特点

1. input一般是irregular的，output一般是regular的。从irregular data映射到regular data是简答的，这也是为什么很多程序是scatter的
   1. input是particle coordinate(x,y,z), output是3d spatial grid
2. 有些时候each input只影响有限个output，所以conflict write的影响没有那么大



* gather缺点

1. 存在overlapping read，但是可以被hardware使用cache/shared memory来缓解



<img src="Note.assets/Screen Shot 2022-06-04 at 9.52.02 PM.png" alt="Screen Shot 2022-06-04 at 9.52.02 PM" style="zoom:50%;" />



#### Example Direct Coulombs Summation (DCS)

irregular input atom and regular output potential grid. Every input influence every output. Given a input atom array, calculate potential on each output grid location.

DCS是最精确的方法，也可以使用其余的方法从而更快，但是就没有这么精确了

<img src="Note.assets/Screen Shot 2022-06-05 at 7.06.24 PM.png" alt="Screen Shot 2022-06-05 at 7.06.24 PM" style="zoom:50%;" />



##### Scatter C Code

minimize computation by moving loop invariance out of loop

![Screen Shot 2022-06-05 at 7.11.03 PM](Note.assets/Screen Shot 2022-06-05 at 7.11.03 PM.png)



##### Scatter CUDA Code

替换C code部分的for loop为thread

需要使用CUDA atomic。由于atoimic导致serialization，会慢

<img src="Note.assets/Screen Shot 2022-06-05 at 7.34.22 PM.png" alt="Screen Shot 2022-06-05 at 7.34.22 PM" style="zoom:50%;" />



##### Gather C Code

存在重复计算varaible

存在重复访问atom array

在C的版本下，比simple scatter版本更慢

对于energygrid的写入更加连续

<img src="Note.assets/Screen Shot 2022-06-05 at 7.35.04 PM.png" alt="Screen Shot 2022-06-05 at 7.35.04 PM" style="zoom:50%;" />



##### Gather CUDA Code

替换for grid x与for grid y为2d block thread

对于一个warp内的thread来说，同步读取same atom array value，也就只访问一次atom array

写入energygrid充分利用memory burst

对于sequential来说更快的代码（simple scatter），计算量更少的代码，并不一定在parallel上更快

<img src="Note.assets/Screen Shot 2022-06-05 at 7.36.39 PM.png" alt="Screen Shot 2022-06-05 at 7.36.39 PM" style="zoom:50%;" />



##### Improved Gather C Code

对于CPU代码来说，cache的利用比computation更重要。尽管scatter C code的计算最少，但是并非是对cache利用最好的

问题中energygrid > 20 x atom array。 atom array可以被放在cache里重复读取，但是对energygrid的读取希望只读取一次，因为无法放入在cache里。

C code中更好办法在这个问题里实际上是output oriented的

<img src="Note.assets/Screen Shot 2022-06-05 at 7.43.26 PM.png" alt="Screen Shot 2022-06-05 at 7.43.26 PM" style="zoom:50%;" />



### Tiling

是什么：buffer input into on-chip storage, to be read multiple times. 

效果：reduce global memory bandwidth pressure



<img src="Note.assets/Screen Shot 2022-06-04 at 9.55.10 PM.png" alt="Screen Shot 2022-06-04 at 9.55.10 PM" style="zoom:50%;" />



* 为什么有shared memory/scratch pad

on chip storage越大（shared memory越大，tile size越大），越能减少bandwidth的压力

如果on chip storage只可被单独thread可见，则on chip storage会比较小（无法给几千个thread分配大的on chip storage）。解决方法是share on chip storage across thread通过scratchpad/shared memory



* 效果

取决于不适用tiling，只使用cache的效果怎么样。

在modern GPU上，cache相对更加复杂+大，所以使用tiling的效果就没有那么多。UIUC 408 Lecture 14里面的例子里，使用tilning只提升了40%左右的速度，原因是因为绝大多数access to global memory都是通过L1 cache的，cache hit rate有98%。



* 例子

GEMM : `DenseLinearAlgebra-CommunicationLowerBound::GEMM GPU UIUC`

Conv : `DenseLinearAlgebra-CommunicationLowerBound::Conv GPU UIUC` 



#### Example GEMM

> 参考
> 
> 1. UIUC ECE 408 Lecture4, lecture5, lecture6




##### Simple C Code

注意下面使用到的变量名，后面会都是用类似的变量名

```cpp
void GEMM(float* M, float* N, float* P, int width)
{
  for ( int i = 0; i < width; ++i )
  {
    for ( int j = 0; j < width; ++j )
    {
      float sum = 0f;
      for ( int k = 0; k < width; ++k )
      {
        float a = M[i * width + k];
        float b = N[k * width + j];
        sum += a * b;
      }
      P[i * width + j] = sum;
    }
  }
}
```

<img src="Note.assets/Screen Shot 2021-11-07 at 1.35.19 PM.png" alt="Screen Shot 2021-11-07 at 1.35.19 PM" style="zoom:40%;" />



##### Simple CUDA Code

```cpp
__global__ 
void MatrixKernel( float* d_M, float* d_N, float* d_P, int width )
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if ( ( row < width ) && ( col < width ) )
  {
    float pval = 0;
    for ( int k = 0; k < width; ++k )
    {
      pval += d_M[ row * width + k ] * d_N[ k * width + col ]; // access global memory
    }
    d_P[row * width + col ] = pval; // access global memory
  }
}
```



* bandwidth 分析知道直接访问global memory的话是不够的

Load 1 N 1 M element : 4 bytes each, total 8 bytes

进行一次计算 2 FLOP 

4 bytes memory access / FLOP 



假设GPU计算与内存

1000 GFLOP/s computation power

150 GB/s memory bandwidth

实际computation usage 150 GB/s / (4B/FLOP) = 37.5 GFLOP/s

并没有充分的利用computation

在实际的运算中，memory并不是总busy，所以实际上代码只能运行在25GFLOPs左右



上面的分析其实比较简单，更加通用的分析应该从computation intensity出发，从而知道对于每一个bytes access to global memory, 需要how many flops



##### Tile with shared memory

假设 M N P 是 square matrix

single threads for each P_ij, parallize computation of elements of P

block size in 2D of TILE_WIDTH * TILE_WIDTH

grid size in 2D of ceil( width / TILE_WIDTH ) * ceil( width / TILE_WIDTH )

<img src="Note.assets/Screen Shot 2021-11-07 at 1.38.09 PM.png" alt="Screen Shot 2021-11-07 at 1.38.09 PM" style="zoom:40%;" />



解决方法：把数据放到shared memory中（速度更快）从而做到数据复用。每一个thread block负责计算seprate ties。

<img src="Note.assets/Screen Shot 2022-05-30 at 6.14.43 PM.png" alt="Screen Shot 2022-05-30 at 6.14.43 PM" style="zoom:40%;" />

<img src="Note.assets/Screen Shot 2021-11-07 at 5.03.38 PM.png" alt="Screen Shot 2021-11-07 at 5.03.38 PM" style="zoom:40%;" />



假设square tile + square matrix的情况。每一个thread会负责load 1 M elem，load 1 N elem，write 1 P elem

当使用了shared memory的时候，第一个想法就是注意需要有synchronize

**对于每一个M N中的input数据，通过tile的方法，被复用TILE_WIDTH次。**

```cpp
__global__ void SquareMatrixKernel1( float* d_M, float* d_N, float* d_P, int width )
{
  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTilen[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // row col 对应着最后的 P 里面的index，也就对应着 M N 里面的Row Col
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;

    // 一个thread block负责多个tile block
  for ( int q = 0; q < width / TILD_WIDTH; ++q )
  {
    // load data to shared memory
    subTileM[ty][tx] = M[Row * width + q * TILE_WIDTH + tx];
    subTileN[ty][tx] = N[(q * TILE_WIDTH+ty)*Width+Col];

    // barrier, wait for all threads load finish
    __syncthreads();

    // This part require data loaded by other threads
    for ( int k = 0; k < TILE_WIDTH; ++k )
      Pvalue += subTileM[ty][k] * subTileN[k][tx];

    // barrier, wait for all threads load finish
    __syncthreads();
  }

  // write result
  P[Row*Width+Col] = Pvalue;
}
```





* bandwidth 分析

16 * 16 tiles : 对于每一个从global memory读取的数据，复用16次。

可以做到150 GB/s / 4B/FLOP * 6 = 600 GFLOP/s

32 * 32 tiles ： 对于每一个从global memory读取的数据，复用32次。

可以做到150 / 4 * 32 = 1200 GFLOPS > 1000 GFLOPS

内存带宽不再是限制。



##### handle boundary

如果data并不是TILE_WIDTH的整数倍，需要解决boundary的问题



* 两种boundary的问题
1. thread 计算有效的p value，但是load global memory的时候会访问out of data. 这个时候应该假设load上来的数据都为0

<img src="Note.assets/Screen Shot 2021-11-07 at 5.48.07 PM.png" alt="Screen Shot 2021-11-07 at 5.48.07 PM" style="zoom:40%;" />

<img src="Note.assets/Screen Shot 2021-11-07 at 5.48.52 PM.png" alt="Screen Shot 2021-11-07 at 5.48.52 PM" style="zoom:40%;" />

2. thread 没有计算有效的p value，而且load global memory的时候会访问out of data.这个时候应该假设load上来的数据都为0，并且不写入无效的output p中

<img src="Note.assets/Screen Shot 2021-11-07 at 5.52.30 PM.png" alt="Screen Shot 2021-11-07 at 5.52.30 PM" style="zoom:40%;" />

<img src="Note.assets/Screen Shot 2021-11-07 at 5.52.43 PM.png" alt="Screen Shot 2021-11-07 at 5.52.43 PM" style="zoom:40%;" />



* 解决方法

1. test during tile load

如果target within input matrix, load

如果target outside input matrix, put 0

尽管多了一个branching，但是保证了计算部分的代码不会变得复杂。而且这个branching只会影响到一个warp内的thread，对于大多数thread都是在一个warp内不会导致两个branch都运行



2. test during tile store

如果value p outiside valid range, 则不写入。



* 有关branch divergence

只针对于block on boundary, 因为在非boundary中，尽管代码中有if else，但是依旧全部的thread走一个path

```cpp
__global__ void SquareMatrixKernel1( float* d_M, float* d_N, float* d_P, int width )
{
  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTilen[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

    // row 是P M N 对应的index。
  // 所以可以被用于判断写入index，以及load index
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float Pvalue = 0;

    // 一个thread block负责多个tile block
  for ( int q = 0; q < (width-1) / TILD_WIDTH+1; ++q )
  {
    // load data to shared memory
    if ( Row < )

    subTileM[ty][tx] = M[Row * width + q * TILE_WIDTH + tx];
    subTileN[ty][tx] = N[(q * TILE_WIDTH+ty)*Width+Col];

    // barrier, wait for all threads load finish
    __syncthreads();

    for ( int k = 0; k < TILE_WIDTH; ++k )
      Pvalue += subTileM[ty][k] * subTileN[k][tx];

    // barrier, wait for all threads load finish
    __syncthreads();
  }

  // write result
  P[Row*Width+Col] = Pvalue;
}
```


#### Example Conv
> Reference
> 1. PMPP Chapter 7
> 2. UIUC 408 Lecture 7,8,9



* Defination

Convolution Masks / Filters (这里不用kernel来避免混淆)



##### Simple 1D Conv

* Math

N 是 input array

M 是mask

P 是output array

如果遇见boundary / ghost element，可以使用zero / padding的方法

<img src="Note.assets/Screen Shot 2022-05-31 at 7.48.24 PM.png" alt="Screen Shot 2022-05-31 at 7.48.24 PM" style="zoom:50%;" />



* CUDA Code

```c
__global__ 
void convolution_1D_basic_kernel(float *N, float *M, float *P, int Mask_Width, int Width)
{ 
	// every thread in charge of one output element
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float Pvalue = 0;
	int N_start_point = i - (Mask_Width/2);
	
  // iterate through filter
	for (int j = 0; j < Mask_Width; j++) 
	{
  	if (((N_start_point + j) >= 0) && ((N_start_point + j) < Width)) 
  	{
  		Pvalue += N[N_start_point + j]*M[j];
 	 	} 
 	} 
} 
```



##### Simple 2D Conv

* Math

<img src="Note.assets/Screen Shot 2022-05-31 at 7.50.55 PM.png" alt="Screen Shot 2022-05-31 at 7.50.55 PM" style="zoom:50%;" />



##### Constant Memory for Filter 

```cpp
// global variable, outside any kernel/function
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH];

// Initialize Mask
float Mask[MASK_WIDTH][MASK_WIDTH];
for(unsigned int i = 0; i < MASK_WIDTH * MASK_WIDTH; i++) {
  Mask[i] = (rand() / (float)RAND_MAX);
  if(rand() % 2) Mask[i] = - Mask[i];
}

// Copy from host to device constant memory
cudaMemcpyToSymbol(Mc, Mask, MASK_WIDTH*MASK_WIDTH*sizeof(float));

// launch kernel
// constant memory is visible to all kernel functions throughout application lifetime
ConvolutionKernel<<<dimGrid, dimBlock>>>(Nd, Pd);
```



##### 3 Strategy for Tile

input数据存在复用，所以可以使用shared memory来减少对global memory的总访问次数

<img src="Note.assets/Screen Shot 2022-05-31 at 9.50.23 PM.png" alt="Screen Shot 2022-05-31 at 9.50.23 PM" style="zoom:50%;" />

<img src="Note.assets/Screen Shot 2022-05-31 at 9.50.40 PM.png" alt="Screen Shot 2022-05-31 at 9.50.40 PM" style="zoom:50%;" />



##### Strategy 1 for 1D

<img src="Note.assets/Screen Shot 2022-05-31 at 9.51.39 PM.png" alt="Screen Shot 2022-05-31 at 9.51.39 PM" style="zoom:50%;" />

* 特点

1. 总thread个数=总output elem个数
2. 多个stage/step来load data
3. halo value放在shared memory中
4. parallel computation of each output element
5. 包含strategy1，下面的三个strategy都没有充分的利用shared memory的大小。在UIUC 408的homework里面涉及到了充分利用shared memory的部分



* adv

1. coalesce global memory access
2. no branch divergence during computation



* disadv

1. 使用更多的shared memory
2. branch divergence during reading data。只有在array两边的warp才会发生divergence的问题



* 算法 & 代码

1. all thread load data from start of input

<img src="Note.assets/Screen Shot 2022-05-31 at 9.56.52 PM.png" alt="Screen Shot 2022-05-31 at 9.56.52 PM" style="zoom:50%;" />



2. 2 * halo threads load remianing data

只有最后的一个warp才会有divergence，前面array的warp都不会走if，也就没有warp divergence

<img src="Note.assets/Screen Shot 2022-05-31 at 9.57.05 PM.png" alt="Screen Shot 2022-05-31 at 9.57.05 PM" style="zoom:50%;" />



3. no divergence during computation

<img src="Note.assets/Screen Shot 2022-05-31 at 9.58.09 PM.png" alt="Screen Shot 2022-05-31 at 9.58.09 PM" style="zoom:70%;" />



* benifit analysis

下面分析了internal tile的影响，对于含有boundary的tile影响是不一样的

<img src="Note.assets/Screen Shot 2022-05-31 at 10.51.18 PM.png" alt="Screen Shot 2022-05-31 at 10.51.18 PM" style="zoom:50%;" />

reduction ratio for different tile size and tile width

<img src="Note.assets/Screen Shot 2022-06-01 at 11.12.35 AM.png" alt="Screen Shot 2022-06-01 at 11.12.35 AM" style="zoom:50%;" />



##### Strategy 2 for 2D

<img src="Note.assets/Screen Shot 2022-05-31 at 9.53.53 PM.png" alt="Screen Shot 2022-05-31 at 9.53.53 PM" style="zoom:50%;" />



* 特点

1. 总thread个数=总input elem个数
2. halo value放在shared memory中
3. some threads (not all) compute output
4. parallel load memory from global memory to shared memory



* adv

1. coalesce global memory access
2. no divergence during read



* disadv

1. branch divergence during computation
2. 使用更多的shared memory



* 算法 & 代码 （2D)

<img src="Note.assets/Screen Shot 2022-05-31 at 10.47.49 PM.png" alt="Screen Shot 2022-05-31 at 10.47.49 PM" style="zoom:50%;" />

复杂的点在于input coordinate map to output coordinate



* benifit analysis for 2D

<img src="Note.assets/Screen Shot 2022-06-01 at 11.21.36 AM.png" alt="Screen Shot 2022-06-01 at 11.21.36 AM" style="zoom:50%;" />

<img src="Note.assets/Screen Shot 2022-06-01 at 11.21.48 AM.png" alt="Screen Shot 2022-06-01 at 11.21.48 AM" style="zoom:50%;" />



##### Strategy 3 for 1D

<img src="Note.assets/Screen Shot 2022-05-31 at 9.54.54 PM.png" alt="Screen Shot 2022-05-31 at 9.54.54 PM" style="zoom:50%;" />

* 特点

1. 总thread个数=总output elem
2. parallel computation of each output element
3. 只把internal element的部分放在shared memory上。halo value直接从gloabl memory访问 / 从 L2访问
4. 对于recent gpu arch来说，这个方法是prefered的，因为有L2 cache，halo value会被nbr block访问的时候放到L2 cache上



* Caching

总Fermi开始，就提供L1 L2 cache。L1是per sm的，L2是across SM的。

一个tile的ghost cell data对于nbr tile就是internal element。所以如果tile 0 访问ghost cell data from global memory，有很大的程度数据已经在L2 cache上了，因为tile 1做了blocking，数据被放到L2上。



* adv

1. 节省shared memory空间
2. no branch divergence during load memory



* disadv

1. branch divergence during computation。发生在array左右两边的warp上



* 算法/代码

<img src="Note.assets/Screen Shot 2022-05-31 at 9.58.41 PM.png" alt="Screen Shot 2022-05-31 at 9.58.41 PM" style="zoom:50%;" />





##### Bandwidth Computation

2022年的GPU需要很大的reuse才能克服bandwidth限制利用computation。

<img src="Note.assets/Screen Shot 2022-06-01 at 11.23.41 AM.png" alt="Screen Shot 2022-06-01 at 11.23.41 AM" style="zoom:50%;" />



需要很大的mask size才能更好的达到peak performence

<img src="Note.assets/Screen Shot 2022-06-01 at 11.24.22 AM.png" alt="Screen Shot 2022-06-01 at 11.24.22 AM" style="zoom:50%;" /><img src="Note.assets/Screen Shot 2022-06-01 at 11.24.34 AM.png" alt="Screen Shot 2022-06-01 at 11.24.34 AM" style="zoom:50%;" />



### Joint Register and Shared Memory Tiling

> Reference
>
> UIUC ECE 508 Lecture 4
>
> Benchmarking GPUs to Tune Dense Linear Algebra



* register 特点

1. low latency 
2. high throughput : per thread per clock cycle可以进行多个register访问与计算
3. load data是serial的
4. private to each thread
5. 进行register tiling需要thread coarsening



* shared memory 特点

1. comparable latency
2. lower throughput compared with register 
3. can be loaded cooperatively by multiple thread



* 为什么joint

1. hardware path是不同的，可以combine tiling for register and shared memory 来增加throughput



#### Source of reuse

在做shared memory tiling的时候，reuse来自于shared memory的数据被多个thread访问，而不是来自于一个thread内部访问一个value多次。

Tile size是T * T的话，每一个thread load一个M，一个N到shared memory，sync（确保数据都在shared memory中），然后遍历一小行M和一小列N来计算一个P，sync（确保shared memory被使用完），然后处理下一个tile

对于每一个M的值，被T（结果P中tile的num col）个thread使用。

对于每一个N的值，被T（结果P中tile的num row）个thread使用。

<img src="Note.assets/Screen Shot 2022-06-06 at 6.00.42 PM.png" alt="Screen Shot 2022-06-06 at 6.00.42 PM" style="zoom:50%;" />

从上面的分析中知道S的大小是independent of reuse factor, 所以S并不一定要等于T。

同时因为计算P的会有两个sync（load to shared memory, wait for all comp on shared memory finish)， 所以S的大小也不能太小，否则sync会占用主要的时间

同时tile size不一定是square的。

every M value reused U time

every N value reused T time

<img src="Note.assets/Screen Shot 2022-06-06 at 6.16.45 PM.png" alt="Screen Shot 2022-06-06 at 6.16.45 PM" style="zoom:50%;" />



#### Example GEMM
> Volkov and Demmel SC 08

一个dim进行register tiling。一个dim进行shared memory tiling

<img src="Note.assets/Screen Shot 2022-06-06 at 6.32.38 PM.png" alt="Screen Shot 2022-06-06 at 6.32.38 PM" style="zoom:70%;" />



1. 对tile of N (S * U)，使用shared memory tiling。thread cooperatively load S*U tile into shared memory

   1. 对于每一个thread，使用S elem of M in reg 以及 S*U shared mem，计算U elem of P
   2. 每一个elem of N被复用T time，因为已经选择了U=16（相对小的val），所以希望T可以大。选择T=64，与之前的32*32的T\*T一致
   3. load of N tile可以被coalsed。可以使用corner-turning ，因为U是16 half warp

2. 对于tile of M (S*T), 使用register tiling。每个thread load S element of M into register

   1. 在shared memory tiling的情况下，S\*T elements是储存在shared memory，可以被多个thread访问的。现在S\*T element是private to each thread (S element for each thread, total T threads)。原来每个thread负责load一个M的elem，现在每个thread负责load 一行M的elem到register

   2. 同时每个thread也需要保存U个element of P来做privitization

      1. GPU普遍有10s reg per thread, 选择U=16

   3. 对于M的访问是不coalsed的，无法进行corner-truning。多个thread同一个时间点访问的内存是不连续的。需要对M进行transpose、。

      <img src="Note.assets/Screen Shot 2022-06-06 at 6.41.33 PM.png" alt="Screen Shot 2022-06-06 at 6.41.33 PM" style="zoom:50%;" />

3. sync 来确保tile of N都在shared memory上
4. 每个thread计算出U个P结果，T个thread计算出T*U个P结果。
   1. 不希望U*S (每个thread的工作量)小，否则每两个sync之间的工作就比较少，划不来
   2. 不希望S过于大，否则造成register pressure。而且S的load是到register的，是serialize的，也不希望大。
   3. 因为tile N需要load S * U(16)个数据，一共有T(64)个thread，希望每个thread都有load elem的工作，所以选择S=T/U=4。
5. sync来确保计算都完成，shared memory可以复用
6. 进行下一个tile of M，tile of N, 对同一个T * U的tile of P的计算



### Grid-stride loop / thread granularity / thread coarsening 

> Ref
>
> 1. Berkeley CS 267 Lecture 7
> 2. PMPP Chapter 5
> 3. UIUC 508 Lecture 3
> 4. NVIDIA Tech BLOG CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops  [link](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/)



* 是什么

1. 原来需要多个thread完成的工作，现在使用一个thread完成，从而减少redundant work （parallel经常会有redundant computation在不同的thread上）
2. 两种类型：一个是让一个thread在1个iteration中完成原来多个thread的工作。另一个是让一个thread block在完成当前thread block的工作后（与原来的工作相同），再处理下一个thread block。
3. thread 0会处理 elem 0, elem 0 + num thread in grid, elem 0 * 2 * num thread in grid. 每一次的step是grid，也是为什么叫做grid stride loop的原因
   1. 对比起来，每个thread处理一个元素的loop叫做`monolithic kernel`


<img src="Note.assets/Screen Shot 2022-06-05 at 7.58.14 PM.png" alt="Screen Shot 2022-06-05 at 7.58.14 PM" style="zoom:50%;" />



* 优点

1. Eliminating redundant work can ease the pressure on the instruction processing bandwidth and improve the overall execution speed of the kernel. 对重复工作结果进行复用，从而减少instruction processing stream的贷款限制
   1. 可以理解为一些会重复的computation，现在shared through register。本来register 是local to each thread, 无法shared across thread的
   2. 访问register的throughput很大，per thread per cycle可以访问多个register file
   3. 访问register的latency很小，只有1 clock cycle

2. scalability，可以支持program size > total num thread on hardware. 

3. 可以tune code with num block = multiply of SM, 然后使用grid stride loop来支持不同大小的问题。

   1. ```cpp
      int numSMs;
      cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
      // Perform SAXPY on 1M elements
      saxpy<<<32*numSMs, 256>>>(1 << 20, 2.0, x, y);
      ```

4. Thread reuse amortizes thread creation and destruction cost along with any other processing the kernel might do before or after the loop (such as thread-private or shared data initialization).

5. 更容易debugging，可以把launch 1 thread 1 block的kernel来debug，不需要改变kernel内容

   1. ```cpp
      saxpy<<<1,1>>>(1<<20, 2.0, x, y);
      ```

6. readability as sequential code 。与sequential code一致都有for loop的存在，更好理解代码



* 缺点

1. （在情况1下）每个thread使用更多的register，可能导致一个sm内总的thread数量减少（因为register constrain）。导致insufficent amount of parallelism。
   1. not enough block per sm to keep sm busy
   2. not enough block to balance across sm (thread合并了以后，总的thread数量减小，总的block数量也就减少了，而且每个block的时间久了，容易导致imbalance)
   3. not enough thread to hide latency。通过warp间swap来hide latency，但是当总thread减少，总warp减少

2. larger computation tiles. 产生more padding and wasted computation，一般通过reduce number of thread per block 解决
   1. 如果一个thread在coarsening以后干了k*k个thread的工作，把原来的block size分别变为width/k和height/k来避免more padding and waste computation




（one output per thread的idle)

<img src="Note.assets/Screen Shot 2022-06-05 at 8.46.53 PM.png" alt="Screen Shot 2022-06-05 at 8.46.53 PM" style="zoom:50%;" />

(two output per thread的idle，更多idle)

<img src="Note.assets/Screen Shot 2022-06-05 at 8.47.17 PM.png" alt="Screen Shot 2022-06-05 at 8.47.17 PM" style="zoom:50%;" />





* 为什么使用

increase efficency outweight reduce of parallelism就可以



* 例子1

```cpp
int numSMs;
cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId);
add<<<32 * numSMs, blockSize>>>(N, x, y);

// GPU function to add two vectors
__global__
void add(int n, float *x, float *y) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  // 这里通过for loop对thread进行复用
  for (int i = index; i < n; i+=stride)
    y[i] = x[i] + y[i];
}
```



#### Example GEMM

reuse tile from matrix M for multipel tile N

原来每个thread block用一个tile M和一个tile N进行计算。

现在每个thread block用一个tile M和多个tile N进行计算，复用shared memory里面的tile M，让一个thread block进行多次计算。

`DenseLinearAlgebra-CommunicationLowerBound.md::GEMM-UIUC` 没有考虑这个部分。

<img src="Note.assets/Screen Shot 2022-05-31 at 7.07.46 PM.png" alt="Screen Shot 2022-05-31 at 7.07.46 PM" style="zoom:50%;" />



#### Example DCS

原来：each thread compute one output (见common optimization techiniques - scatter to gather - dcs example )

现在：each thread compute four output



<img src="Note.assets/Screen Shot 2022-06-05 at 9.39.43 PM.png" alt="Screen Shot 2022-06-05 at 9.39.43 PM" style="zoom:50%;" />



优点

1. reduce number of load of atoms array by factor of 4
2. eliminate redundant computation like dxx, dysqpdzsq



缺点：

1. 使用更多register，现在需要储存4个dxi, 4个energvalxi.



#### Example 7 point Stencil

stencil难点: memory intense & little computation (one fma per load value)



<img src="Note.assets/Screen Shot 2022-06-05 at 9.43.18 PM.png" alt="Screen Shot 2022-06-05 at 9.43.18 PM" style="zoom:50%;" />



* thread coarsening along Z-axis

原来：每个thread计算一个output

现在：每个thread计算pencil along Z dim

效果：

1. 减少grid index的计算。现在可以 x + 1, y + 1 而不是 thread.x block.x blockdim.x 每次重新计算x y
2. 更少的thread block，更少的thread block scheduling overhead

<img src="Note.assets/Screen Shot 2022-06-05 at 9.50.27 PM.png" alt="Screen Shot 2022-06-05 at 9.50.27 PM" style="zoom:50%;" />



* register tiling

一个thread计算pencil along z axis，z-axis上的数据复用可以使用register tilning

<img src="Note.assets/Screen Shot 2022-06-05 at 10.02.38 PM.png" alt="Screen Shot 2022-06-05 at 10.02.38 PM" style="zoom:50%;" />

<img src="Note.assets/Screen Shot 2022-06-05 at 10.03.03 PM.png" alt="Screen Shot 2022-06-05 at 10.03.03 PM" style="zoom:50%;" />



效果：

原来一个output需要访问7个input through global memory access

现在一个output需要访问5个input through globlal memory access

节省了25%的global memory bandwidth



* shared memory tilning 

对于每一个current，都会被xy dim上的4个nbr thread使用。可以使用shared memory来共享这个部分

<img src="Note.assets/Screen Shot 2022-06-05 at 10.36.28 PM.png" alt="Screen Shot 2022-06-05 at 10.36.28 PM" style="zoom:50%;" />

对于每一个block来说，现在有4 slices of data 在 on-chip memory 上。这是有限的内存，会限制block 的大小

1. prev, curr, next 的 register slices。每一个thread使用3个register，对于block负责x y dim来说，就是3 slides of memory
2. shared memory of current



<img src="Note.assets/Screen Shot 2022-06-05 at 10.38.29 PM.png" alt="Screen Shot 2022-06-05 at 10.38.29 PM" style="zoom:50%;" />



对于halo来说，使用与conv strategy3类似的方法，访问global memory。更新的gpu arch有更大的across block L2 cache，这些halo cell会被nbr block放到L2 cache里，所以导致的thread divergence不会太严重

<img src="Note.assets/Screen Shot 2022-06-05 at 10.39.40 PM.png" alt="Screen Shot 2022-06-05 at 10.39.40 PM" style="zoom:50%;" />



效果：

原来计算一个elem需要访问5个global memory

现在计算一个eleme需要访问1个global memory与4个shared memory

但是由于shared memory与register的限制，block dim不能很大，导致halo的部分影响了reduce ratio

<img src="Note.assets/Screen Shot 2022-06-05 at 10.41.50 PM.png" alt="Screen Shot 2022-06-05 at 10.41.50 PM" style="zoom:50%;" />





### Privatization

是什么：buffer output into on-chip storage (or even register), to be write multiple times. 

每个thread/group of thread有自己的local copy of output，首先写在local copy里，然后再合并到final output中。下图中显示了privitization at multiple level

效果：避免多个thread通过使用atomic同时写入一个内存地址，使用atomic会drastically decrease memory throughput。

<img src="Note.assets/Screen Shot 2022-06-04 at 9.58.57 PM.png" alt="Screen Shot 2022-06-04 at 9.58.57 PM" style="zoom:50%;" />



* GPU上使用的缺点

CPU上由于thread的数量较小，private copy of output不会是问题

GPU上由于thread的总数量很多，使用privitization需要注意

1. data foorprint of the copy。使用shared memory或者是register是否会导致 thread  per sm 减少
2. overhead of combining private copy会比较大，因为这里依旧需要atomic



解决方法是one copy for a group/block of thread in scratchpad / shared memory, 这样可以同时兼顾latency (5 cycle)与bandwidth（在shared memory上使用atomic的bandwidth依旧可以）



#### Example Histogram

> Reference
>
> 1. UIUC 408 Lecture 18

histogram中有highly contentious output conflict，每个thread都有很多的写

通过让8 thread shared private histogram on shared memory （而不是all thread within block) 来保证atomic bandwidth。

如果使用all thread wihtin block share private copy的话，使用atoimic就会是一个问题

主要高速我们privitization的coarsening是取决于程序的，并不一定总是private to thread / private to block



#### Example Reduction

> Reference
>
> 1. UIUC 408 Lecture 17
> 2. Optimizing parallel reduction in cuda by Mark Harris [link](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
> 4. Faster Parallel Reductions on Kepler NVIDIA Blog [link](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)



* 什么是reduction

Reduce a set of input to a single value with binary operator

sequential algorithm take O(n) step

parallel algorithm take O(log n) step with tree

但是使用parallel algorithm经常存在的问题是diminishing parallelisim。对于CUDA来说是不好的，因为CUDA的资源是以block为单位进行分配的。

<img src="Note.assets/Screen Shot 2022-06-07 at 5.01.48 PM.png" alt="Screen Shot 2022-06-07 at 5.01.48 PM" style="zoom:40%;" />





* work efficency

parallel algorithm 是work efficent的

<img src="Note.assets/Screen Shot 2022-06-07 at 5.02.29 PM.png" alt="Screen Shot 2022-06-07 at 5.02.29 PM" style="zoom:50%;" />



##### Simple reduction algorithm

* 特点

每个thread load 1 value from global memory to shared memory (will go through register)

使用tree based方法合并

privitization：每个block内有一个output的拷贝，最后一起放到global上



* 缺点

存在branch divergence，没有工作的warp因为divergence的原因依旧activate



<img src="Note.assets/Screen Shot 2022-06-09 at 2.38.36 PM.png" alt="Screen Shot 2022-06-09 at 2.38.36 PM" style="zoom:50%;" />

<img src="Note.assets/Screen Shot 2022-06-09 at 2.38.48 PM.png" alt="Screen Shot 2022-06-09 at 2.38.48 PM" style="zoom:50%;" />



##### Reduce branch divergence

* 特点

让activate thread是连续的，上面simple reduce里面activate thread是不连续的(e.g. step2中thread 0 4 8 12)，这样就避免了branch divergence。

在除最后5个iteration（warp内的32thread）以外，没有branch divergence。

需要注意的是， if ( index < blockDim.x ) 为false的warp依旧会被schedule运行，只不过运行完if判断为false以后就让其余的warp运行，所以这里依旧有schedule overhead。



* 缺点

对内存的访问会产生bank conflict，访问是gap of power of 2, 会导致访问的内存都在一个bank上面，无法充分利用aggregated memory bandwidth.

<img src="Note.assets/Screen Shot 2022-06-09 at 2.41.54 PM.png" alt="Screen Shot 2022-06-09 at 2.41.54 PM" style="zoom:50%;" />

<img src="Note.assets/Screen Shot 2022-06-09 at 2.41.43 PM.png" alt="Screen Shot 2022-06-09 at 2.41.43 PM" style="zoom:50%;" />



##### Reduce memory bank conflict

* 特点

使用连续的shared memory内存，从而避免bank conflict



<img src="Note.assets/Screen Shot 2022-06-09 at 2.43.56 PM.png" alt="Screen Shot 2022-06-09 at 2.43.56 PM" style="zoom:50%;" />

<img src="Note.assets/Screen Shot 2022-06-09 at 2.44.13 PM.png" alt="Screen Shot 2022-06-09 at 2.44.13 PM" style="zoom:50%;" />



##### On-the fly computation

* 特点

上面的kernel中有一半的thread会在load完数据以后就没有工作了，这是很浪费thread的，同时占用了很多shared memory。

每个thread可以load多个element from global memory to register, compute a result, and then put to shared memory.

相比起储存全部的数据到shared memory里，这个kernel可以使用更少的threads block来处理相同的数据/同样大小的thread block处理更多的数据。当threads block大小减小的时候，就减少了diminishing parallelisim的问题。

<img src="Note.assets/Screen Shot 2022-06-09 at 2.46.33 PM.png" alt="Screen Shot 2022-06-09 at 2.46.33 PM" style="zoom:50%;" />



##### Reduce instruction overhead

* 特点

程序是memory bound的，之所以没有达到memory bandwidth是因为有其余的instruction overhead（也有其余的原因，像是kernel launch overhead等），希望减少其余的instruction 大小，减小instruction pipeline pressure。

因为CUDA有对于threads per block的限制，所以可以completely unroll避免for loop

使用template实现。因为template会给出blocksize的大小，红色的if branch会在compile time决定。

<img src="Note.assets/Screen Shot 2022-06-09 at 4.39.22 PM.png" alt="Screen Shot 2022-06-09 at 4.39.22 PM" style="zoom:50%;" />



##### Avoid sync in last 5 iteration

> Reference
>
> 1. Volatile - when to use ? [link](https://forums.developer.nvidia.com/t/volatile-when-to-use-regarding-registers/21993/3)



对于最后5个iteration(within warp0)不需要sync，因为warp内的thread是concurrent executed的，所以内部一定是sync的。



* 使用shared memory

这里对shared memory的使用没有用memory fence或者sync，所以需要使用volatile来避免compiler cache shared memory data on register

volitile tells the compiler it is not to optimize variables (especially shared memory) by placing them in registers because another thread may update the variable. (The update would be ignored if the register was used instead).

volatile should be used when the data can be changed outside the current thread without memory fences (for writes) or synchronization (for reads and writes). Otherwise the compiler is free to optimize the reads/writes to the variable by caching the data in a local register.

<img src="Note.assets/Screen Shot 2022-06-09 at 4.40.12 PM.png" alt="Screen Shot 2022-06-09 at 4.40.12 PM" style="zoom:50%;" />



* 使用warp shuffle

```cpp
  // Use XOR mode to perform butterfly reduction
    for (int i=16; i>=1; i/=2)
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
```



##### Algorithm Cascading

> Reference
>
> 1. Stackoverflow CUDA algorithm cascading [link](https://stackoverflow.com/questions/23232782/cuda-algorithm-cascading)

是什么：结合sequential与parallel reduction。(sequential)每个thread首先从global memory读取多个值，sum up in register, 然后放到shared memory。(parallel) threads within block 从shared memory读取数据，parallel reduction。



为什么：保证了每个thread都有一些work来做(sequential 的部分每个threa都进行相加，相比起完全parallel的情况下只有部分thread相加)，减少shared  memory的使用



<img src="Note.assets/Screen Shot 2022-06-09 at 4.59.10 PM.png" alt="Screen Shot 2022-06-09 at 4.59.10 PM" style="zoom:50%;" />



##### Compute centric

使用thread block以后，keep thread block active。

如果有thread不再active计算的话，则使用他们load数据。

没有看到有实现使用了这个方法，不是很确定这个方法怎么做

<img src="Note.assets/Screen Shot 2022-06-07 at 8.16.48 PM.png" alt="Screen Shot 2022-06-07 at 8.16.48 PM" style="zoom:50%;" />



### Algorithm Cascading

> Reference
>
> 1. UIUC 408 Lecture 15 on reduction, algorithm cascading
> 2. PMPP Chapter 8.5 three phase



* 是什么

混合sequential 与 parallel 算法，从而让每个thread有足够的工作(sequential)来避免parallel的overhead，而且允许thread之间通过parallle来进行计算



#### Example prefix-sum / parallel scan

> Reference
>
> 1. UIUC 408 Lecture 16, 17
> 2. PMPP Chapter 8
>
> 
>
> Note: 之所以把scan放在algorithm cascading里面，是因为最重要的能够实现work efficent的three phase算法用到了Algorithm cascading的想法



* Inclusive scan defination

<img src="Note.assets/Screen Shot 2022-06-16 at 10.46.18 AM.png" alt="Screen Shot 2022-06-16 at 10.46.18 AM" style="zoom:50%;" />



* Sequential C 

```cpp
y[0] = x[0];
for (i = 1; i < Max_i; i++)
    y[i] = y[i-1] + x[i];
```



##### Kogge-Stone

<img src="Note.assets/Screen Shot 2022-06-17 at 10.26.50 PM.png" alt="Screen Shot 2022-06-17 at 10.26.50 PM" style="zoom:50%;" />



```cpp
__global__ void Kogge-Stone_scan_kernel(float *X, float *Y, int InputSize) 
{
    // num thread = SECTION_SIZE
    __shared__ float XY[SECTION_SIZE];

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    // each thread load one element from global memory to shared memory
    if (i < InputSize)
        XY[threadIdx.x] = X[i];

    // the code below performs iterative scan on XY
  	// log(n) step in total from stride 1 to stride SECTION_SIZE in step 2
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) 
    {
      	// ensure previous step all thread have finish write to shared memory 
        // and the shared memory contain new information
        __syncthreads(); 

      	int tmp;
        if (threadIdx.x >= stride)
            tmp = XY[threadIdx.x] + XY[threadIdx.x-stride];
     
        // ensure all thread have load required information from shared memory, 
      	// no data from previous step is needed (from shared memory), safe to overwrite.
        __syncthreads(); 
        if (threadIdx.x >= stride)
            XY[threadIdx.x] = tmp;
    }

    // write shared memory back to global memory
    Y[i] = XY[threadIdx.x];
}
```





* 特点

1. number of thread = number of elements
2. 处理within block scan
3. 随着iteration，第一个warp内会产生branch divergence，但是对于较大的block不是问题
4. 通过stride的方式来shared computation result
5. 需要两个syncthread在使用shared memory的时候
6. 可以使用warp内部的shuffle instruction来实现。



* speed and work efficency analysis

step : O(log n)

work : (n-1) + (n-2) + (n-4) + ... + (n-n/2) = N*log2N – (N–1) = O(n log n) work

speed up : (N*log2N)/P  where p is number of execution unit



当谈论speed up的时候，不是看step有多少，而是看speed up ratio。因为平行算法也只有有限的hardware resource (execution unit), 所以处理n个数据对于有限的execution unit需要花费多个step。

在有充分的hardware resource的时候，speedup是有效果的。

当硬件资源不充分的时候，就无法充分利用这个算法，同时更加energy consumption



##### Double buffer Kogge-Stone

<img src="Note.assets/Screen Shot 2022-06-17 at 10.36.10 PM.png" alt="Screen Shot 2022-06-17 at 10.36.10 PM" style="zoom:50%;" />



* 特点

1. 避免了第二个sync thread的使用。因为现在可以直接overwrite，不需要判断是否可以overwrite



##### Brent-Kung

<img src="Note.assets/Screen Shot 2022-06-17 at 10.43.02 PM.png" alt="Screen Shot 2022-06-17 at 10.43.02 PM" style="zoom:50%;" />

```cpp
__global__ void Brent_Kung_scan_kernel(float *X, float *Y, int InputSize) 
{
    __shared__ float XY[SECTION_SIZE];
    int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
    
    // each thread load two elem from global memory to shared memory
    if (i < InputSize) XY[threadIdx.x] = X[i];
    if (i+blockDim.x < InputSize) XY[threadIdx.x+blockDim.x] = X[i+blockDim.x];
    
    // parallel scan step method 1
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) 
    { 
        // ensure all thread from previous step have finish
        // so that shared memory contain valid data to be used
        __synchthreads(); 
        if ((threadIdx.x + 1)%(2*stride) == 0) 
        {
           XY[threadIdx.x] += XY[threadIdx.x - stride];
        }
    }

    // parallel scan step method 2
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) 
    {
        // ensure all thread from previous step have finish
        // so that shared memory contain valid data to be used
        __syncthreads();
        int index = (threadIdx.x+1) * 2* stride -1;
        if (index < SECTION_SIZE) 
        {
            XY[index] += XY[index - stride];
        }
    }

    // post scan step
    for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) 
    {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index + stride < SECTION_SIZE) 
        {
            XY[index + stride] += XY[index];
        }
    }

    // thread write result from shared memory back to global memory
    __syncthreads();
    if (i < InputSize) Y[i] = XY[threadIdx.x];
    if (i+blockDim.x < InputSize) Y[i+blockDim.x] = XY[threadIdx.x+blockDim.x];
}
```





* 特点

1. number of thread = number of element / 2
2. 处理within block scan
3. 使用reduction tree来实现work efficent （因为用了reduction tree，每个iteration的work减少了，从而work efficent了）
4. 每个iteration，只用进行一次sync，因为每个thread的dst并不被其余的thread作为src，所以不需要考虑是否可以overwrite的问题



* parallel scan step 两种方法

method 2通过计算thread到index的映射关系（不直接使用thread id作为index），从而使用了decreasing number of contigious thread 来避免branch divergence

branch divergence 除了within warp 存在，其余都不存在

第一个iteration，thread 0-7负责idx 1,3,5,7,9,11,13,15

第二个iteration，thread 0-3 负责idx 3,7,11,15



* step and work analyaia

step : 2 log n

work : 2N-2-log2(N)

算法本身是work efficent的

但是由于cuda对资源的分配是以block为单位的，所以inactive thread虽然没有进行计算，但是依旧占用资源。导致占用的资源实际上接近 (N/2)*(2*log2(N)−1). 

由于cuda资源分配的特点，最终的work efficency与kogge-stone算法相近。



##### Three phase with algorithm cascaing

<img src="Note.assets/Screen Shot 2022-06-17 at 11.10.32 PM.png" alt="Screen Shot 2022-06-17 at 11.10.32 PM" style="zoom:50%;" />



* algorithm

1. input section partition into number of threads subsection. 
   1. num thread = 4, num input = 40, each subsection contain 10 element. (in the figure 8.7 above, subsection length is 4)
2. Corner turning to memory coalesed load input section to shared memory
3. (step 1) each thread perform scan within subsection serially 
   1. thread perform scan on 10 subsection elment 
4. (step 2) Kogge–Stone/Brent–Kung to scan with last element in each subsection array.
5. (Step 3) each thread add previous subsection sum to current array.



* 特点

1. work efficent in CUDA
2. 处理within block scan
3. 能够处理的elem不受到max thread per block 的限制，只收到shared memory的限制。
   1. kogge-stone最多处理max thread per block个element
   2. brent-kung最多处理2 * max thread per block个element



* work and step analysis

Assume that we use the Kogge–Stone algorithm for phase 2. For an input list of N elements, if we use T threads, the amount of work done is N−1 for phase 1, T\*log2T for phase 2, and N−T for phase 3. If we use P execution units, the execution can be expected to take (N−1+T\*log2T+N−T)/P time units.



##### Hierarchical scan

<img src="Note.assets/Screen Shot 2022-06-17 at 11.18.29 PM.png" alt="Screen Shot 2022-06-17 at 11.18.29 PM" style="zoom:50%;" />



* 特点

1. 处理inter-block scan
2. 允许任何大小的scan，不再局限于max thread per block / max shared memory per block



* 算法

1. kernel 1 : 计算scan block的scan，可以使用kogge-stone/brent-kung/three-phase的方法。

   1. 需要多一个parameter S of dim SIZE/SECTION_SIZE，最后一个thread负责把partial sum写入到global memory array S里

   2. 需要把shared memory的partial result写入到shared memory中

      ```cpp
      __syncthreads();
      if (threadIdx.x == blockDim.x-1) {
        S[blockIdx.x] = XY[SECTION_SIZE – 1];
      }
      ```

2. kernel 2 : 可以使用kogge-stone/brent-kung/three-phase的方法，把S作为input，把S作为output。

   1. 通过kernel1运行完成，termiante kernel，从而实现全部kernel 1的synch

3. kernel 3：take S与Y（global memory array)作为input，把S对应value加到每一个Y的元素中。

   1. 需要读取global memory Y，写入global memory Y

   ```cpp
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   Y[i] += S[blockIdx.x-1];
   ```



kernel 1，kernel 2可以使用三种algo的原因是因为一般scan block大小不会太大，由scan block最后sum组成的S也不会太大




##### Stream based scan hierchical scan

> reference
>
> 1. CUDA Toolkits document [link](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions)
> 2. stackoverflow cuda threadfence [link](https://stackoverflow.com/questions/5232689/cuda-threadfence)

```cpp
// scan_value : volatile global memory
// use volatile to ensure when other block read this memory, current block have write result to memory location instead of cache / register
__shared__ float previous_sum;
if (threadIdx.x == 0)
{
  // Wait for previous flag
  while (atomicAdd(&flags[bid], 0) == 0){;}
  // Read previous partial sum from global memory
  previous_sum = scan_value[bid];
  // Propagate partial sum
  scan_value[bid + 1] = previous_sum + local_sum;
  // Memory fence
  // ensures that the partial sum is completely stored to memory (scan_value有最终结果) before the flag is set with atomicAdd()
  __threadfence();
  // Set flag
  atomicAdd(&flags[bid + 1], 1);
}
// all other threads inside block wait for thread 1
__syncthreads();
```



* 特点

1. 避免了hierchical的两次global memory读写
2. 使用adjacent block synchronization在block之间传递信息。
   1. 对于每一个block，等待左边的block传递信息给他，然后再把信息传递给右边的block
   2. 通过使用atomic的方式来实现
   3. 在block之间传递消息的时候是serialize的
3. scan_value 需要是 volatile的，从而避免compiler把数据放在register/reorder代码，从而实际上没有写入scan_value
4. 对于scan_value与flag的访问尽管是在global memory上，但是在modern gpu时是在L2 cache上的访问。



* 通过dynamic block indexing 避免deadlock

为什么有deadlock：block scheduling不按顺序，如果block i在block i+1后面运行+block i+1占用了全部的resource，导致block i+1在运行结尾等待block i，但是block i由于没有resource无法被schedule，从而造成deadlock

什么是dynamic block index：decouple usage of thread bloxk index from blockIdx.x

```cpp
__shared__ int sbid;
if (threadIdx.x == 0)
  sbid = atomicAdd(DCounter, 1);
__syncthreads();
const int bid = sbid;
```




### Binning

> 这个部分不太懂

是什么：create a map from output location to small subset of input location.

sorting of input elements into bins representing a region of space containing those input elements.



### Compaction

是什么：压缩数据中的hole，从而减少memory overhead

<img src="Note.assets/Screen Shot 2022-06-04 at 5.12.26 PM.png" alt="Screen Shot 2022-06-04 at 5.12.26 PM" style="zoom:50%;" />



### Regularization

是什么：解决thread之间的load imbalance问题



* load imbalance的问题

1. 导致thread divergence
2. 一个block内如果有load imbalance，会导致resource在整个block运行结束之前（也就是imbalance里最多的work）不会释放，导致block占用有限的resource更多的时间（尽管在imbalance的时候，block不需要这么多的resource），导致num thread per SM降低



