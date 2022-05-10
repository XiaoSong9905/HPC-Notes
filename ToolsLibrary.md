

## Tools




### valgrind

▪    check number of cache misses

告诉不同的cache hit rate

```shell
valgrind --tool=cachegrind ./a.out
```




### ISPC

* assignment policy
1. create many tasks (unlimited)
2. create limited worker
3. for each worker, after completeing current task, worker thread inspect list and assigns itself to the next uncompleted task. 而不是一次性启动n thread。而是有限的thread主动找work

<img src="Note.assets/Screen Shot 2021-10-17 at 10.31.13 AM.png" alt="Screen Shot 2021-10-17 at 10.31.13 AM" style="zoom:30%;" />




## Library




### Intel Math Kernel library

[link](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html#gs.dp7bp4)

* 包含
1. Linear Algebra. V2V M2V M2M
2. FFT
3. Random number generator
4. Sparse Linear Algebra Functions. 
   1. sparse matrix with dense matrix
   2. sparse matrix with dense vector
5. summary statistics




### Eigen

* 参考
1. [知乎 为什么eigen快](https://www.zhihu.com/question/28571059)
2. [Eigen开发团队的ppt](https://download.tuxfamily.org/eigen/eigen_CGLibs_Giugno_Pisa_2013.pdf)




## Hardware




#### Cori KNL Node

| term      | value                                                |
| --------- | ---------------------------------------------------- |
| processor | Intel Xeon Phi Processor 7250 (knights landing)      |
| 指令集       | avx2, avx512                                         |
| avx512    | 两个AVX512流水线，同时计算两个vector                             |
| peak      | 44.8 GFlops/core                                     |
| l1 cache  | 64 kb (32 i cache, 32 data), direct-mapped           |
| L2 cache  | each tile (2 core) share 1mb l2 cache. Direct-mapped |
| l3 cache  | no l3 cache                                          |
| speed     | 1.4 GHz = 1.4 * 10^9 clock cycle per second          |
|           |                                                      |



#### Cori V100

| term        | value |
| ----------- | ----- |
| sm          | 80    |
| warp per sm | 64    |
|             |       |
|             |       |
|             |       |
|             |       |
|             |       |
|             |       |


