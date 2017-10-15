#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Make a robot called myrobot that starts at
# coordinates 30, 50 heading north (pi/2).
# Have your robot turn clockwise by pi/2, move
# 15 m, and sense. Then have it turn clockwise
# by pi/2 again, move 10 m, and sense again.
#
# Your program should print out the result of
# your two sense measurements.
#
# Don't modify the code below. Please enter
# your code at the bottom.

from math import *
import random


# 用于测量的标记
landmarks  = [[20.0, 20.0], [80.0, 80.0], [20.0, 80.0], [80.0, 20.0]]
# 地图大小 world_size*world_size
world_size = 100.0


class robot:
    def __init__(self):
        self.x = random.random() * world_size
        self.y = random.random() * world_size
        self.orientation = random.random() * 2.0 * pi
        self.forward_noise = 0.0;
        self.turn_noise    = 0.0;
        self.sense_noise   = 0.0;
    # 设置位置 和 朝向
    def set(self, new_x, new_y, new_orientation):
        if new_x < 0 or new_x >= world_size:
            raise ValueError, 'X coordinate out of bound'
        if new_y < 0 or new_y >= world_size:
            raise ValueError, 'Y coordinate out of bound'
        if new_orientation < 0 or new_orientation >= 2 * pi:
            raise ValueError, 'Orientation must be in [0..2pi]'
        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)
    
    # 设置噪声
    def set_noise(self, new_f_noise, new_t_noise, new_s_noise):
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.forward_noise = float(new_f_noise);
        self.turn_noise    = float(new_t_noise);
        self.sense_noise   = float(new_s_noise);
    
    # 测量当前位置到 各个标记点的 距离（添加测量噪声）
    def sense(self):
        Z = []
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            dist += random.gauss(0.0, self.sense_noise)
            Z.append(dist)
        return Z
    
    # 移动(朝向 + 距离) 添加噪声（转向噪声 和移动噪声）
    def move(self, turn, forward):
        if forward < 0:
            raise ValueError, 'Robot cant move backwards'         
        
        # 转向噪声
        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turn_noise)
        orientation %= 2 * pi # 在0~2*pi范围内
        
        # 移动噪声  分解到 x，y轴
        dist = float(forward) + random.gauss(0.0, self.forward_noise)
        x = self.x + (cos(orientation) * dist)
        y = self.y + (sin(orientation) * dist)
        x %= world_size       # 在 地图内
        y %= world_size
        
        # set particle
        res = robot()
        res.set(x, y, orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res
    
    def Gaussian(self, mu, sigma, x):
        
        # 均值mu 标准差sigma
        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(2.0 * pi * (sigma ** 2))# 计算带有噪声的测量 在 测量数据高斯分布中的 概率大小
    
    
    def measurement_prob(self, measurement):
        
        # calculates how likely a measurement should be
        
        prob = 1.0;
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2) # 真值，不带噪声的真实值
            prob *= self.Gaussian(dist, self.sense_noise, measurement[i])                  # 测量值越准确，粒子权重越大
        return prob
    
    
    
    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))


# 粒子和 机器人位置
def eval(r, p):
    sum = 0.0;
    for i in range(len(p)): # calculate mean error
        dx = (p[i].x - r.x + (world_size/2.0)) % world_size - (world_size/2.0)
        dy = (p[i].y - r.y + (world_size/2.0)) % world_size - (world_size/2.0)
        err = sqrt(dx * dx + dy * dy)
        sum += err
    return sum / float(len(p))



####   DON'T MODIFY ANYTHING ABOVE HERE! ENTER CODE BELOW ####

myrobot = robot()
myrobot.set_noise(0.05,0.05,5.0)  # 设置噪声 前进噪声 转向噪声 测量噪声
myrobot.set(30.0,50.0,pi/2)     # 设置起点
myrobot=myrobot.move(-pi/2,15.0)# 顺时针转 pi/2 前进15米
#print myrobot
print myrobot.sense()           # 测量当前位置 到四个 标记的距离

myrobot=myrobot.move(-pi/2,10.0)# 顺时针转 pi/2 前进10米
#print myrobot 
print myrobot.sense()           # 测量当前位置 到四个 标记的距离


# 随机产生 1000个机器人 粒子
N = 1000
T =10
p = []
for i in range(N):
    ramdom_particle = robot()
    p.append(ramdom_particle)
#print len(p)	
p2 = []
for i in range(N):
	 p2.append(p[i].move(0.1,5.0))#小的移动	
#print len(p2)

# 得到每个粒子  的 权重 （看其测量值与真值的切合度）
w = []
for i in range(N):	
	w.append(p[i].measurement_prob(p[i].sense()))
	
p3 = []
index = int(random.random() * N) #随机选一个下标开始
beta = 0.0
m_w = max(w)	
for i in range(N):
	beta += random.random() * 2.0 * m_w
	while beta > w[index]:#较小权重的粒子 去除
		beta -= w[index]
		index = (index + 1) % N
 	p3.append(p[index])
p = p3
	