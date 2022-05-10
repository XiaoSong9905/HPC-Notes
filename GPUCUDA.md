
## GPU

* todo 
1. branch divergence
2. GPU 有什么类型的register



### Design Principle

> ref
>
> Berkeley CS 267 Lecture 7

* 什么是GPU

GPU是heterogeneous chip. 有负责不同功能的计算模块

<img src="Note.assets/Screen Shot 2022-02-10 at 11.38.25 AM.png" alt="Screen Shot 2022-02-10 at 11.38.25 AM" style="zoom:50%;" />



* 为了设么设计

GPU design for high throughput, don't care about throughput so much

CPU design for low latency



* CPU GPU

CPU : multicore system

GPU : manycore system



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

经常也被叫做invertyed memory hierchy




### Program Model

> 参考
> 
> 1. UIUC ECE Lecture 2,3

CUDA Threads: SPMD single program multiple data

CPU call GPU kernel function 是 asynchronize 的

Grid -> Block -> Warp ( implementation 而不是program model ) -> threads 

每一个kernel function被map到一个grid上

threads in same block share "shared memory" & synchronize

block之间的thread只能通过global memory的方法来进行信息交流

Block 被assign 到不同的streamming multi-professor (sm) 上

block & warp execute in arbitary order




#### SM warp scheduling

warp是sm内部的schedule unit

SM implement zero-overhead warp scheduling

Warps whose next instruction has its operands ready for consumption are eligible for execution

<img src="Note.assets/Screen Shot 2021-11-07 at 12.53.14 PM.png" alt="Screen Shot 2021-11-07 at 12.53.14 PM" style="zoom:40%;" />




#### Handle Corner Case

* simple add kernel

```cpp
__global__ void add( int* A, int* B, int* C, int n )
{
        i = blockIdx.x * blockDim.x + threadIdx.x;
      // avoid kernel larger than data
    if ( i < n )
      C[i] = A[i] + B[i];
}
```

* handle extra

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

GPU对于每一个thread使用predicated execution。如果一个warp内的多个thread会走不同的path，则multiple path被实际运行。

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

一个解决branching的常用方法就是branch granularity是warp的倍数

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

最新的架构中好像允许每一个thread有自己的pc (老的结构上是一个warp共享pc？)，这样的结构可以允许thread不branching。



#### Grid-stride loop

> Ref
>
> Berkeley CS 267 Lecture 7

* 是什么

1. 减少number of block in grid
2. 从而reuse threads for computaiton
3. amortize threads creation/destruction cost



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




#### Memory Bandwidth 限制



#### Kernel切分决定

kernel之间有implicit barrier。如果kernel小而多的话，implicit barrier会成为overhead。




#### Block Granlarity 决定

##### 考虑block/sm threads/sm限制

* 假设 (下面的数据会依据使用的不同硬件改变)
1. SM take 8 blocks 
2. SM take 1536 threads 
3. 三个block分块的方式 $8*8, 16*16, 32*32$
* $8*8$ : 

64 threads per block

each sm can take 1536 / 64 = 24 blocks

但是sm限制最多8个block，也就是最多使用 8 * 64 = 512 threads ( out of 1536 threads )

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
3. square matrix computation 使用 simple shared memory 16 / 32 的tile size
* TILE_WIDTH 16

each block 使用 2 * 16 * 16 * 4 bytes (float) = 2048 bytes / block

64 kb / 2048 bytes/block = max 32 block per sm due to share memory

2048 threads / (16*16) = 8 block per sm due to threads limit

每一个时间点会有 2 (two load / threads) * 16*16 * 8 block/sm = 4096 pendig load (通过检查pending load来判断是否有让SM busy)

* TILE_WIDTH 32

each block 使用 2 * 32 * 32 * 4 bytes (float) = 8kb bytes/block

64/8 = max 8 block per sm due to share memory

2048 / (32*32) = 2 block per sm due to threads limit

每一个时间点会有 2 * 32 * 32 * 2 = 4096 pending loads

同样的memory parallelsim exposed. 

尽管32的复用的内存更大，memory paralle与16一样。可能会存在一些block schedule的问题，因为更少的总block个数。




### Memory Model

> 参考
>
> 1. UIUC ECE Lecture 3,4
> 1. Berkeley CS 267 Lecture 7 on memory colesing

per thread register

per block shared mempry

per block l1 cache

per grid global memory

per grid l2 cache

per grid constant memory

<img src="Note.assets/Screen Shot 2021-11-06 at 10.05.13 PM.png" alt="Screen Shot 2021-11-06 at 10.05.13 PM" style="zoom:40%;" />

<img src="Note.assets/Screen Shot 2021-11-07 at 1.23.18 PM.png" alt="Screen Shot 2021-11-07 at 1.23.18 PM" style="zoom:30%;" />

per thread array (per thread local memory) 会被储存在 gloabl memory 中，因为compiler不知道这个array会有多长，无法把array拆分后放到regsiter中。



* 为什么使用shared memory

为了数据复用，就像是CPU里做blocking把数据放到cache上一样。



* within block communication

使用shared memory + barrier



* between block communication

使用atomic, global memory


##### Memory Coalesing
