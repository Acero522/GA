import numpy as np
import random
import math
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt

maxIter=100     #最大迭代次数
population=50      #种群数量
lenChrom=10     #染色体长度
pc=0.8      #交叉概率
pm=0.01     #变异概率
dim=4       #变量的维度
lb=[-1, -1, -1, -1]     #最小取值
ub=[1, 1, 1, 1]         #最大取值
#初始化
def Initialization():
    pop = []
    for i in range(population):
        temp1 = []
        for j in range(dim):
            temp2 = []
            for k in range(lenChrom):
                temp2.append(random.randint(0, 1))
            temp1.append(temp2)
        pop.append(temp1)
    return pop
# 将二进制转化为十进制
def b2d( pop_binary):
    pop_decimal = []
    for i in range(len(pop_binary)):
        temp1 = []
        for j in range(dim):
            temp2 = 0
            for k in range(lenChrom):
                temp2 += pop_binary[i][j][k] * math.pow(2, k)
            temp2 = temp2 * (ub[j] - lb[j]) / (math.pow(2, lenChrom) - 1) + lb[j]
            temp1.append(temp2)
        pop_decimal.append(temp1)
    return pop_decimal
# 轮盘赌模型选择适应值较高的种子
def Roulette( fitness, pop):
    # 适应值按照大小排序
    sorted_index = np.argsort(fitness)
    sorted_fitness, sorted_pop = [], []
    for index in sorted_index:
        sorted_fitness.append(fitness[index])
        sorted_pop.append(pop[index])
    # 生成适应值累加序列
    fitness_sum = sum(sorted_fitness)
    accumulation = [None for col in range(len(sorted_fitness))]
    accumulation[0] = sorted_fitness[0] / fitness_sum
    for i in range(1, len(sorted_fitness)):
        accumulation[i] = accumulation[i - 1] + sorted_fitness[i] / fitness_sum
    # 轮盘赌
    roulette_index = []
    for j in range(len(sorted_fitness)):
        p = random.random()
        for k in range(len(accumulation)):
            if accumulation[k] >= p:
                roulette_index.append(k)
                break
    temp1, temp2 = [], []
    for index in roulette_index:
        temp1.append(sorted_fitness[index])
        temp2.append(sorted_pop[index])
    newpop = [[x, y] for x, y in zip(temp1, temp2)]
    newpop.sort()
    newpop_fitness = [newpop[i][0] for i in range(len(sorted_fitness))]
    newpop_pop = [newpop[i][1] for i in range(len(sorted_fitness))]
    return newpop_fitness, newpop_pop
# 交叉繁殖：针对每一个种子，随机选取另一个种子与之交叉。
# 随机取种子基因上的两个位置点，然后互换两点之间的部分
def Crossover( pop):
    newpop = []
    for i in range(len(pop)):
        if random.random() < pc:
            # 选择另一个种子
            j = i
            while j == i:
                j = random.randint(0, len(pop) - 1)
            cpoint1 = random.randint(1, lenChrom - 1)
            cpoint2 = cpoint1
            while cpoint2 == cpoint1:
                cpoint2 = random.randint(1, lenChrom - 1)
            cpoint1, cpoint2 = min(cpoint1, cpoint2), max(cpoint1, cpoint2)
            newpop1, newpop2 = [], []
            for k in range(dim):
                temp1, temp2 = [], []
                temp1.extend(pop[i][k][0:cpoint1])
                temp1.extend(pop[j][k][cpoint1:cpoint2])
                temp1.extend(pop[i][k][cpoint2:])
                temp2.extend(pop[j][k][0:cpoint1])
                temp2.extend(pop[i][k][cpoint1:cpoint2])
                temp2.extend(pop[j][k][cpoint2:])
                newpop1.append(temp1)
                newpop2.append(temp2)
            newpop.extend([newpop1, newpop2])
    return newpop
# 变异：针对每一个种子的每一个维度，进行概率变异，变异基因为一位
def Mutation( pop):
    newpop = copy.deepcopy(pop)
    for i in range(len(pop)):
        for j in range(dim):
            if random.random() < pm:
                mpoint = random.randint(0, lenChrom - 1)
                newpop[i][j][mpoint] = 1 - newpop[i][j][mpoint]
    return newpop
# 绘制迭代-误差图
def Ploterro( Convergence_curve):
    mpl.rcParams['font.sans-serif'] = ['Courier New']
    mpl.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(10, 6))
    x = [i for i in range(len(Convergence_curve))]
    plt.plot(x, Convergence_curve, 'r-', linewidth=1.5, markersize=5)
    plt.xlabel(u'Iter', fontsize=18)
    plt.ylabel(u'Best score', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, )
    plt.grid(True)
    plt.show()
#Fobj：适应性函数/价值函数
def Fobj(factor):
    cost = (factor[0] - 1) ** 2 + (factor[1] + 1) ** 2 + factor[2] ** 2 + factor[3] ** 2
    return 1 / cost

def Run():
    pop = Initialization()
    errolist = []
    for Current_iter in range(maxIter):
        pop1 = Crossover(pop)
        pop2 = Mutation(pop1)
        pop3 = b2d(pop2)
        fitness = []
        for j in range(len(pop2)):
            fitness.append(Fobj(pop3[j]))
        sorted_fitness, sorted_pop = Roulette(fitness, pop2)
        best_fitness = sorted_fitness[-1]
        best_pos = b2d([sorted_pop[-1]])[0]
        pop = sorted_pop[-1:-(population + 1):-1]
        errolist.append(1 / best_fitness)
        if 1 / best_fitness < 0.0001:
            print("Iter = " + str(Current_iter))
            print("Best_score = " + str(round(1 / best_fitness, 4)))
            print("Best_pos = " + str([round(a, 4) for a in best_pos]))
            break
    return best_fitness, best_pos, errolist

Best_score, Best_pos, errolist = Run()
Ploterro(errolist)
