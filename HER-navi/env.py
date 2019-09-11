# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 19:33:29 2019
@author: Or
"""

import torch
import numpy as np
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from copy import deepcopy

class Navigate2D:
    def __init__(self,N,Nobs,Dobs,Rmin):
        self.N = N
        self.Nobs = Nobs
        self.Dobs = Dobs
        self.Rmin = Rmin
        self.state_dim = [N,N,3]
        self.action_dim = 4
        self.scale = 10.0
        # self.action_sapce = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        self.action_space = [0, 1, 2, 3]

    def get_dims(self):
        return self.state_dim, self.action_dim

    def reset(self):
        grid = np.zeros((self.N,self.N,3))
        for i in range(self.Nobs):
            center = np.random.randint(0,self.N,(1,2))
            minX = np.maximum(center[0,0] - self.Dobs,1)
            minY = np.maximum(center[0,1] - self.Dobs,1)
            maxX = np.minimum(center[0,0] + self.Dobs,self.N-1)
            maxY = np.minimum(center[0,1] + self.Dobs,self.N-1)
            grid[minX:maxX,minY:maxY,0] = 1.0

        free_idx = np.argwhere(grid[:,:,0] == 0.0)
        start = free_idx[np.random.randint(0,free_idx.shape[0],1),:].squeeze()
        while (True):
            finish = free_idx[np.random.randint(0,free_idx.shape[0],1),:].squeeze()
            if ((start[0] != finish[0]) and (start[1] != finish[1]) and (np.linalg.norm(start - finish) >= self.Rmin)):
                break
        grid[start[0],start[1],1] = self.scale*1.0
        grid[finish[0],finish[1],2] = self.scale*1.0
        done = False
        return grid, done

    def step(self,grid,action):
        max_norm = self.N

        new_grid = deepcopy(grid)
        done = False
        reward = -1.0
        act = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        pos = np.argwhere(grid[:,:,1] == self.scale**1.0)[0]
        target = np.argwhere(grid[:,:,2] == self.scale*1.0)[0]
        new_pos = pos + act[action]

        dist1 = np.linalg.norm(pos - target)
        dist2 = np.linalg.norm(new_pos - target)
        #reward = (dist1 - dist2)*(max_norm - dist2)
        #reward = -dist2
        reward = -1
        if (np.any(new_pos < 0.0) or np.any(new_pos > (self.N - 1)) or (grid[new_pos[0],new_pos[1],0] == 1.0)):
            #dist = np.linalg.norm(pos - target)
            #reward = (dist1 - dist2)
            return grid, reward, done, dist2
        new_grid[pos[0],pos[1],1] = 0.0
        new_grid[new_pos[0],new_pos[1],1] = self.scale*1.0
        if ((new_pos[0] == target[0]) and (new_pos[1] == target[1])):
            reward = 0.0
            done = True
        #dist = np.linalg.norm(new_pos - target)
        #reward = (dist1 - dist2)
        return new_grid, reward, done, dist2

    @staticmethod
    def get_tensor(grid):
        S = torch.Tensor(grid).transpose(2,1).transpose(1,0).unsqueeze(0)
        return S

    def render(self,grid):
        #imshow(grid)
        plot = imshow(grid)
        return plot

if __name__ == '__main__':
    import time
    N = 20
    Nobs = 15
    Dobs = 2
    Rmin = 10
    env = Navigate2D(N,Nobs,Dobs,Rmin)

    grid, done = env.reset()
    plt.ion()
    env.render(grid)
    input("Press <ENTER> to continue...")

    time.sleep(1)
    grid, _, _, _ = env.step(grid, 0)
    env.render(grid)
    input("Press <ENTER> to continue...")

    time.sleep(1)
    grid, _, _, _ = env.step(grid, 0)
    env.render(grid)
    input("Press <ENTER> to continue...")
