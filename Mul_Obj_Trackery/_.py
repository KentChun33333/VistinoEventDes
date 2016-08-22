class Customer:
    def doCustomer(self):
        print("Service")
    
    def pay(self):
        print("PAY")
    
    def accept(self, visitor): pass

class Member(Customer):
    def doMember(self):
        print("MemberService")
    
    def accept(self, visitor):
        visitor.visitMember(self)
    
class VIP(Customer):
    def doVIP(self):
        print("VIP Service")
        
    def accept(self, visitor):
        visitor.visitVIP(self)

class VisitorImpl:
    def visitMember(self, member):
        member.doMember();
    
    def visitVIP(self, vip):
        vip.doVIP()
    
class Service:
    def __init__(self):
        self.visitor = VisitorImpl()
    
    def doService(self, customer):
        customer.doCustomer()
        customer.accept(self.visitor)
        customer.pay()

service = Service()
service.doService(VIP())


def demo(N):
    count = 0
    while N > 1:
        if N%2==0:
            N=N/2
            count+=1
        else :
            N=(N+1)/2
            count+=1
    return count

def demo(N):
    initMemo = [0]*N
    for i in range(1,N):
        initMemo[i]=inintMemo[i-1]+1

f(n) = f(n/2), 
f(n) = f(3 * n + 1)

def demo(N):
    memo = 

f(n) = 1 if n ==1, 
f(n) = f(n/2) + 1 if n is even, 
f(n) = f(3 * n + 1) + 1 if n is odd

def demo(N):
    memo = [1]*N+1
    for i in range(2,N+1):
        if i %2 == 0:
            memo[i*2-1]=memo[i]+1
        else:
            memo[i/3+1]

def demo(N):
    memo=[1]*(10*N)
    for i in range(1,N+1)[::-1]:
        if i == 1 :
            return 1 
        elif i %2 == 0:
            memo[i]=memo[i/2+1]+1
        else:
            memo[i]=memo[3*i+1]+1
    return max(memo[1:N+1])


def demo(N):
    if N
    memo=[1]*(10*N)
    if N<=1 :
        return 1
    while N>1:
        if i %2 == 0:
            memo[N]=memo[N/2+1]+1
            N=N/2+1
        else:
            memo[N]=memo[3*N+1]+1
            N=3*N+1
    return max(memo[1:N+1])


def demo(N):
    count = 0
    while N > 1:
        if N%2==0:
            N=demo(N/2)+1
            count+=1
        else :
            N=demo(3*N+1)+1
            count+=1
    return count

f(n) = 1 if n ==1, 
f(n) = f(n/2) + 1 if n is even, f(n) = f(3 * n + 1) + 1 if n is odd

def base(N):
    if N==1:
        return 1
    if N%2==0:
        if memo[N/2] >1:
            return 
        return N/2
    else:
        return 3*N+1

def demo(N):
    memo=[1]*(3*N+2)
    for i in range(1,N+1)[::-1]:
        count=0
        while N>1:
            N=base(N)
            count+=1
        memo[i]=count
    return max(memo)

def demo_v2(N):
    memo=[1]*(100*N+2)
    for i in range(1,N+1):
        count=0
        while N>1 :
            if memo[N]>1:
                count+=memo[N]+1
                N=1
            else:
                N=base(N)
                count+=1
        memo[i]=count
    return max(memo)

def demo_v3(N):
    memo=[1]*(3*N+2)
    for i in range(1,N+1):
        count=0
        while N>1 :
            if memo[N]>1:
                memo[i]+=memo[N]
                N=1
            else:
                N=base(N)
                count+=1
        memo[i]=count
    return max(memo)


def demo_v4(N):
    memo = [1]*3*N+2
    for i in range(2,N+1):
        if i


memo = [0]*10000000
def base(N, memo):
    if memo[N] > 0:
        return memo[N]
    else:
        if N == 1:
            return 1
        if N % 2==0:
            return base((N / 2),memo) + 1
        else:
            return base((3 * N + 1),memo) + 1


import timeit


import cv2
import skimage

import pandas as pd
import numpy as np

cv2


