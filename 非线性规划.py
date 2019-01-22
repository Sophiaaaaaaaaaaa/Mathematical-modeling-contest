import numpy as np
from scipy.optimize import minimize

# 目标函数
def objective(x):
    return (x[0] - 1)**2 + (x[1] - 2.5)**2+np.log(x[0])

# 约束条件
def constraint1(x):
    return x[0] - 2 * x[1] + 2  #不等约束

def constraint2(x):
    
    return -x[0] - 2 * x[1] + 6 #不等约束

def constraint3(x):
        
    return -x[0] + 2 * x[1] + 2 #不等约束

# 初始猜想
n = 2
x0 = np.zeros(n)
x0[0] = 2
x0[1] = 0


# show initial objective
print('Initial SSE Objective: ' + str(objective(x0)))

# 边界约束
b = (0.0,None)
bnds = (b, b) # 注意是两个变量都要有边界约束

con1 = {'type': 'ineq', 'fun': constraint1} 
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3} 
cons = ([con1,con2,con3]) # 3个约束条件

# 优化计算
solution = minimize(objective,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)
x = solution.x

# show final objective
print('Final SSE Objective: ' + str(objective(x)))

# print solution
print('Solution')
print('x1 = ' + str(x[0]))
print('x2 = ' + str(x[1]))
