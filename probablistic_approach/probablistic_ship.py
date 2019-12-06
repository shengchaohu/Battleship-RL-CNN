import numpy as np
import matplotlib.pyplot as plt
import random
from random import randint

# random.seed(42)
test=2000
iters=[]

for _ in range(test):
    n=10
    ships=[2,3,4,5]
    mask=np.full([n,n],0)
    mask2=np.full([n,n],0)
    target_set=set() # records the positions of opponent's ships
    target_set2=dict()

    def canfit_horizontal(i,j,ship,mask):
        point=0
        for k in range(j,j+ship):
            if mask[i,k]==-1:
                return -1
            point+=mask[i,k]
        return point

    def canfit_vertical(i,j,ship,mask):
        point=0
        for k in range(i,i+ship):
            if mask[k,j]==-1:
                return -1
            point+=mask[k,j]
        return point

    def get_density(mask):
        density=np.full([n,n],0)
        for ship in ships:
            for i in range(n):
                for j in range(n-ship+1):
                    p=canfit_horizontal(i,j,ship,mask)
                    if p>=0:
                        for k in range(j,j+ship):
                            if mask[i,k] not in (-1,1): # hit cell and missed cell's density should be zero
                                # density[i,k]+=1 if p==0 else p # in target mode, rectangles with no overlapping with current hit cells should have density 0
                                density[i,k] += 1 + p
            for i in range(n-ship+1):
                for j in range(n):
                    p=canfit_vertical(i,j,ship,mask)
                    if p>=0:
                        for k in range(i,i+ship):
                            if mask[k,j] not in (-1,1):
                                # density[k,j]+=1 if p==0 else p
                                density[k,j] += 1 + p
        return density

    def return_max(mask):
        density=get_density(mask)
        indices=np.where(density==density.max())
        return (indices[0][0],indices[1][0],density)

    # randomly place ship
    for ship in ships:
        found = False
        while not found:
            i = randint(0, n-1)
            j = randint(0, n-1)
            horizontal = randint(0, 1)
            if horizontal == 1:
                if j + ship - 1 < n and canfit_horizontal(i,j,ship,mask2) >= 0:
                    found = True
                    target_set2[ship]=set()
                    for k in range(j,j+ship):
                        target_set.add((i,k))
                        target_set2[ship].add((i,k))
                        mask2[i,k]=-1
            else:
                if i + ship - 1 < n and canfit_vertical(i,j,ship,mask2) >= 0:
                    found = True
                    target_set2[ship]=set()
                    for k in range(i,i+ship):
                        target_set.add((k, j))
                        target_set2[ship].add((k,j))
                        mask2[k,j]=-1
    # print("target_set = ", target_set)

    iter = 0
    while(len(target_set) > 0):
        iter += 1
        r,c,density=return_max(mask)
        print(r,c)
        print(target_set)
        print(target_set2)
        plt.imshow(-density, cmap='gray')
        plt.show()
        if (r,c) in target_set:
            print("----hit----")
            mask[r,c] = 1
            target_set.remove((r,c))
            tmp=0
            for k in target_set2:
                if (r,c) in target_set2[k]:
                    target_set2[k].remove((r,c))
                    if not target_set2[k]:
                        ships.remove(k)
                        tmp=k
            if tmp:
                del target_set2[tmp]
        else: 
            print("----miss----")
            mask[r,c] = -1

    iters.append(iter)
    if len(iters)%100==0:
        print(len(iters))

    print("iteration -----", iter)
    plt.imshow(mask, cmap='gray')
    plt.show()

bin=max(iters)-min(iters)+1

plt.hist(iters,bins=bin,color='red',histtype='stepfilled',alpha=0.75)
plt.show()

