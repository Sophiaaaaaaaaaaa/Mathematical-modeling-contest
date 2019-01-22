from random import *
from math import *
times = 1000000
count = 0
for i in range(times):
    x = uniform(1,2)    #产生（1，2）之间的随机浮点数
    y = uniform(0,1)
    if x*x*y<1:   
        count += 1
e1 =2*count/times
print(e1)
