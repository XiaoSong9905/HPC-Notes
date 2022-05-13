
## Optimize Serial Code

> Reference
> 
> 1. MIT 6.172 lecture2 (new bently rule), lecture2 (GEMM optimization)
> 2. 并行算法设计与性能优化 chapter 4



Work: the work of a program is the sum total of all the operations executed by the program

Optimizing work: 可以通过算法的方法来减少总的work(从time complexity角度上考虑). 但是reduce the work并不意味着reduce the run time, 因为hardware的不同部分比较复杂，需要综合考虑。




### Application level




#### Use compiler

开启简单的编译器选项就可以进行加速

```shell
-O3 -ffast-math -funroll-all-loops -mavx -mtune=native
```

-ffast-math 是为了开启float的向量化

-funroll-all-llops 允许循环展开

-mtune=native 根据当前处理器做优化




#### 去掉全局变量

全局变量（尤其在多个文件中）会影响编译器优化。因为编译器需要在多个文件间分析全局变量的使用




### Data Structure




#### packing

一个word里存放多个数据。使用更少的数据。

但是需要注意很多时候C++是以word为单位进行load store。使用packing的方法可能会有memory inconsistent的问题

```cpp
typedef struct
{
  int year: 13;
  int month: 4;
  int day: 5;
} date_t;
```




#### encoding

把数据进行压缩，使用更少的bytes来表示数据

但是encoding并不总是会提升速度，因为有些时候解码encoding信息反而会带来整体效果不好

e.g. 用int来表示DOB




#### augmentation

add information to a data structure to amke common operation do less work. 

e.g. 保留linked list end




#### precomputation

在compile time计算出一些值，避免run time计算出结果。

metaprogramming：使用一个程序，首先打印出来可以hardcode的diamante，把打印的内容放在实际的代码中

但是precomputation并不一定更快，因为precomputation的值会被放在c++的text部分，这个部分需要从disk里面load上来，所以不一定比on the run计算会好。

e.g. pascal traingle的值直接hardcode




#### caching

显性的caching一些数据




#### sparsity

常用于sparse matrix，使用sparse matrix representaiton从而来避免储存很多的0，并且进行很多的无效计算。

常用于static sparse graph, 使用sparse的方法储存图

* CSR Compressed Sparse Row 

```shell
row: 每一个row，开始位置是下面的哪个index. offset to column array
col:不同的column index
vals: 对应的值
nnz: number of non-zero entries
```

<img src="Note.assets/Screen Shot 2021-10-14 at 2.04.30 PM.png" alt="Screen Shot 2021-10-14 at 2.04.30 PM" style="zoom:50%;" />




#### struct organization

1. 大数据在前，小数据在后，可以节省空间
2. 可以使用 `__attribute_` 方法改变struct对齐方式




### Loops




#### Hoisting

避免重复计算loop-invariant变量。也就是把一些计算从loop循环体里放到loop外面只计算一次

有时候compiler会做hoisting，但是当程序复杂的时候，compiler就搞不定了

e.g. resize时候的映射关系，可以提前计算，而不是在loop里run time的计算




#### Sentinels

使用dummy value，来避免logic中需要单独解决boundary condition。

有些时候处理boundary condition需要在while里使用if，产生很多没必要的overhead，因为每一次循环都需要判断corver case




#### loop unrolling

减少total number of iteration of loop. 从而减少用于control loop的instruction的总数量。

同时因为loop body变大了，会允许compiler在loop body中做更多的optimization。这往往是更大的优化。

在for很小的时候，compiler可能会帮你做unroll

对于二层循环来说，通常建议展开外层循环。

通常建议展开小循环（loop body 小的方程)

大循环展开可能导致register overflow. 从而导致性能下降。




#### loop fusion / jamming

把两个相同index range的loop融合成一个loop，从而减少overhead of loop control

进行loop融合可以增加arithmetic intensiry

目的是为了exploit locality to avoid redudant data transfer between CPU and memory. 

```cpp
void add(int n, float* a, float* b, float* c )
{
  for ( int i = 0; i < n; i++ )
    c[i] = a[i] + b[i];
}

void mul(int n, float* a, float* b, float* c )
{
  for ( int i = 0; i < n; i++ )
    c[i] = a[i] * b[i];
}

void mul_add_fused(int n, float* a, float* b, float* c, float* d, float* e )
{
  for ( int i = 0; i < n; i++ )
    e[i] = d[i] = (a[i] + b[i]) * c[i];
}
```




#### loop splitting

* 大的loop body

如果loop body比较大存在register overflow的情况 + loop body依赖于多个input soruce，则可以拆分loop body为几个小的loop。

因为依赖多个input source，所以拆分以后不会增加IO。

因为loop body大，所以拆分以后增加的for loop branching不会是问题

* loop body 中存在branching + branching不太可能预测

如果loop中有branching的话，可以把一个for拆分为两个没有branching的loop。这样减少了分支预测失败率




#### eliminating wasted iteration

有些时候iteration是重复计算覆盖的，避免重复的iteration计算

e.g. matrix transpose中如果使用双loop的话，第二个loop的截止点应该是第一个loop的i，而不是n




#### loop reordering

通过改变for loop的访问pattern，更好使用locality

参考 "常见算法-GEMM CPU-Loop Reordering"




### Algorithm




#### tilning 分块

分块后让一个块的内容都在cache上，这样计算的时候latency就会小了




#### lookup table

通过预先计算一个表格，在runtime的时候访问这个表格，而不是在runtime的时候动态计算来加速

e.g. nn resize的一开始会计算映射




#### Utilize ILP

通过让前后代码减少依赖来增加ILP的效果。

```cpp
// 传统代码
for ( int i = 0; i < n; i += 4 )
{
  sum += a[i];
  sum += a[i+1];
  sum += a[i+2];
  sum += a[i+3];
}

// ILP友好代码
for ( int i = 0; i < n; i += 4 )
{
  sum0 += a[i];
  sum1 += a[i+1];
  sum2 += a[i+2];
  sum3 += a[i+3];
}
sum = sum0 + sum1 + sum2 + sum3;
```



* swap的优化

```cpp
// 传统写法
void swap(int& a, int& b)
{
  int tmp = a;
  a = b;
  b = tmp;
}

// 优化ILP写法
void swap(int &a, int&b)
{
  int tmpa = a;
  int tmpb = b;
  a = tmpb;
  b = tmpa;
}
```




#### arhithmetic

int add : 1 clock cycle

int mul : 3 clock cycle 

int divide : 10+ clock cycle

int module : k0 / k00 clock cycle 

减少代码中module divide的使用，用mul add进行替换




#### get memory to cache

对指针进行解引用会访问内存。可以把会重复用到的内存先放到临时变量里，然后使用临时变量。一般情况下临时变量会被储存在register里

访问内存会在200-400 cycle，访问寄存器在1个cycle

```cpp
int x = *arr;
// do some computation with x
```




#### FMA

x86中fma指令的cycle个数与+或者*单独的是一样的。可以通过使用FMA指令来减少总clock

FMA指令只会经历一轮rounding，常用于high percision algorithm

```cpp
x = y + c * z;
x = round(c * z + y)
```




### Logic




#### constant folding and propagation

在compile time的时候计算constant expression

C++中使用constexpr




#### common-subexpression elimination

avoid computing the same expression multiple times by storing the temp result for later use。避免重复计算某些值

一般由compiler来帮助做。




#### Algebraic Identities

使用cheap algebric expression代替expensive algebric expression

进行替换的时候，需要注意floating point overflow的问题。

e.g. 让等式两侧都求平方，而避免sqrt。sqrt是一个比较昂贵的操作




#### short-circuiting

stop evaluating as soon as you know the answer. 

e.g. 一串数字是否sum大于某一个值。可以在计算sum的过程中就与threshold进行对比，而不是计算完全部的sum才对比




#### ordering tests

更容易成功的test应该在不容易成功的test前面

便宜的test应该在昂贵的test前面

```cpp
// 下面的例子，' '出现的几率最大，所以放在前面
bool is_whitespace(char c)
{
  if ( c == ' ' || c == '\n' || c == '\t' || c == '\t')
    return true;
     return false;
}
```




#### fast path

某些测试可能很贵（两个球是否相交），可以先进行简单的粗略测试来排除一大部分结果（两个球的bounding box是否相交，这个测试相对便宜），如果失败的话再进行昂贵的测试。




#### combining tests

使用switch代替一大堆的if else

如果是一大堆if else嵌套的话，可以把多层的if的值pack成一个word，然后switch这个word

```cpp
void test_fast(int a, int b, int c )
{
  int pack = (a == 1) << 2 + (b == 1) << 1 + (c == 1);
  switch( pack )
  {
    case 0:
      xx;
      break;
  }
}

void test_slow(int a, int b, int c )
{
  if ( a == 0 )
  {
    if ( b == 0 )
    {
      if ( c == 0 )
      {

      }
      else
      {

      }
    }
    else
    {

    }
  }
}
```




### Functions




#### reduce parameter

函数参数优先通过register传递，超过register支持以后才会通过stack进行传递。

减少function parameter的数量 & funciton paramter variable size 小（不copy by value on struct) 可以避免使用stack传递参数




#### inlining

避免funciton call产生的overhead。并且允许compiler在inline了以后做更深一步的优化（例如数据不需要pack再unpack）

compiler可能会帮你inline。也可以自己使用inline hint

相比起marcro，inlining还可能由compiler进行common sub expression的替换

一般对小于10行并且没有代码分支的函数进行inline

如果函数过大依旧inline的话，容易造成register overflow

* calling overhead
1. save register to stack
2. pushing function argument to stack
3. increment stack pointer and jump to begining of new code




#### tail-recursion

recursion function 使用 tail recursion以后会被优化成为while loop




#### coarsening recursion

increase size of base case to avoid function overhead




#### restrict

在传入pointer to array的时候，使用restrict key word可以帮助compiler做一些深度优化。

restrict ： the only pointer point to specific data

用于明确知道两个指针指向非同一个数据

```cpp
void f(long* restrict a, long* restrict b){}
```

没有restrict的情况让编译器无法优化，因为有ptr aliasing

这个时候编译器会重新从内存读取数据，相加。

```cpp
int f(int *a, int* b)
{
  *a = 5;
  *b = 6;
  return *a + *b; // 不一定是11，如果指针指向一个地方的话是6
}
```

如果使用restrict以后，编译器知道指针不指向统一数据，在 *a + *b 的时候就直接compile time生成对应的答案。

如何理解C语言关键字restrict？ - Milo Yip的回答 - 知乎 https://www.zhihu.com/question/41653775/answer/92088248

