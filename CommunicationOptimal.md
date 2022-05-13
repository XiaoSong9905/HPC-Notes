

## Communication Lower Bound on Nested Loop

> Reference
>
> Berkeley CS 267 Lectuer 6 b

* 是什么

communication = moving data (between main memory and cache, between processor and network)

在nested loop情况下的算法，communication lower bound是什么，在什么情况下才能达到communication lower bound



* 为什么关注

memory movement is the most expensive in terms of enegery and speed in computation.

Data movement 的时间很久

data movement占用chip上的大小很大

data movement consume最多的能量



### N-body

#### 是什么

```cpp
// Force角度
for i = 1 : n
  for j = 1 : n
    F(i) = F(i) + force(A(i), A(j))

// Potential角度
for i = 1 : n
  for j = 1 : n
    e = e + potential( A(i), B(j) )
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



optimal的情况下，读取多个tile，每一个tile都是square

<img src="Note.assets/Screen Shot 2022-05-13 at 10.17.27 AM.png" alt="Screen Shot 2022-05-13 at 10.17.27 AM" style="zoom:40%;" />



#### communication lower bound

每一次使用cache，可以处理 M^2/4 次iteration。这里的最优解是A B读取的数据一样多，这样映射得到的是square

为了处理 n^2 个iteration，需要读取 n^2 / (M^2/4)  = 4(n/M)^2 次数据从slow memory 到 fast memory

也就需要读取 4(n/M)^2 * M = 4n^2/M 的数据

lower bound = $\Omega (n^2/M) = \Omega( \text{\# loop iterations inside for loop } / M )$



### GEMM

* 对比n-body

在n-body里面，找到了给定cache大小，如何放输入（sqaure的放），能够让cache内数据运行的iteration数量最多。在GEMM里面同样找到如何充分利用cache的方法，从而每一次cache里运行的iteration数量最大，从而达到communication lower bound

<img src="Note.assets/Screen Shot 2022-05-13 at 10.21.19 AM.png" alt="Screen Shot 2022-05-13 at 10.21.19 AM" style="zoom:40%;" />



#### 从3D的角度理解GEMM

如果要计算一个C(i, j) 则需要对应的全部的 A B block

<img src="Note.assets/Screen Shot 2022-02-10 at 11.01.24 AM.png" alt="Screen Shot 2022-02-10 at 11.01.24 AM" style="zoom:50%;" />



* 利用cache进行数据reuse

把对应的A B C的一个square都放到cache里

一次cache能够同时处理的iteration/3D空间中的cube大小，upper bound by A B C block的面积的square。右边的图片显示了upper bound，左边的图片是equal

<img src="Note.assets/Screen Shot 2022-02-10 at 11.02.57 AM.png" alt="Screen Shot 2022-02-10 at 11.02.57 AM" style="zoom:50%;" />



#### communication lower bound

* serial case

假设cache大小为M，对应的A B C block最大为 $M/3$

对应的cube大小最大为 $((M/3)^3)^{1/2} = M^{3/2}/27$

因为是3 nested loop，总的iteration 数量是 $n^3$，也就是 $n^3/ (M^{3/2}/27)) $个对应的cube。

总共需要读取的数据是 $M * n^3/ (M^{3/2}/27)) = n^3 * 27 / M^{1/2}$

lower bound是 $\Omega( n^3 ) / M^{1/2} $，这个值与关于matmul comm lower bound Theorem (Hong & Kung 1981) 一致

最优算法对应是cube



* parallel case

需要注意的是，这里假设 M = 3n^2 / p 代表每个processor的数据一样，实际上的parallel algorithm存在数据需要共享的case

<img src="Note.assets/Screen Shot 2022-05-13 at 10.27.35 AM.png" alt="Screen Shot 2022-05-13 at 10.27.35 AM" style="zoom:50%;" />



### General

#### communication lower bound

不管多少个loop，不管多少个index，只要能找到如下的映射关系就可以。

需要A的读取，B的读取，C的读取是连续的（可能需要reorder loop）

<img src="Note.assets/Screen Shot 2022-02-10 at 11.31.44 AM.png" alt="Screen Shot 2022-02-10 at 11.31.44 AM" style="zoom:50%;" />

<img src="Note.assets/Screen Shot 2022-02-10 at 11.32.28 AM.png" alt="Screen Shot 2022-02-10 at 11.32.28 AM" style="zoom:50%;" />



* 是否是实际可行的lower bound

可行，但是取决于loop reorder，loop dependency等性质

<img src="Note.assets/Screen Shot 2022-02-10 at 11.32.49 AM.png" alt="Screen Shot 2022-02-10 at 11.32.49 AM" style="zoom:33%;" />



### CNN

* 是什么

CNN 可以转化为7 nested loop

<img src="Note.assets/Screen Shot 2022-02-10 at 11.34.09 AM.png" alt="Screen Shot 2022-02-10 at 11.34.09 AM" style="zoom:50%;" />



* communication lower bound

<img src="Note.assets/Screen Shot 2022-02-10 at 11.34.26 AM.png" alt="Screen Shot 2022-02-10 at 11.34.26 AM" style="zoom:50%;" />

