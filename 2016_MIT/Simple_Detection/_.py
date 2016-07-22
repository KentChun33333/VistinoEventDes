Codefight ...
Leetcode ...

city = [
[-1, 5, 20],
[-1,-1,10],
[-1,-1,-1]
]

def findCity(city):
    memo = []                          # memo the all possible length
    for x in range(len(city)):         # init X
        y=len(city)-1                  # init Y
        tmp = 0
        while y<=len(city)-1 and x>=0: # moving each step 
            tmp2 = city[x][y]
            print (x,y)
            print tmp2

            if tmp2>0:
                tmp+=tmp2
                y-=1
                x-=1
                print x, y
            else:# END While
                tmp+=999
                y=len(city)
        memo.append(tmp)
        print memo
    return min(memo)


city= [
 [-1,-1,19, 8,18,-1,-1,-1,-1,-1], 
 [10, 6, 4, 7, 0,10,18,-1, 0,-1], 
 [-1,-1,15,-1,17, 3,-1,14,16, 3], 
 [4 ,19, 3,15, 8, 4, 6,11, 5, 8], 
 [5 , 3,10,-1, 0,14,15, 1,16, 5], 
 [-1, 8,-1,-1, 5,-1, 5, 0, 1,-1], 
 [-1,18,-1,19, 2,-1,10,-1, 8, 6], 
 [14, 8,12,16,-1,-1, 0,16,15,17], 
 [4 , 5, 1,12, 0, 4, 8,15, 1,-1], 
 [13, 7,17,-1, 4,13,16, 3,12, 9]]

findCity_way
max_step = n-1

memo = []
island = list(range(len(city)))
for x in city[0]:



def findCity(city):
    memo = []                          # memo the all possible length
    max_step = len(city)-1
    for x in range(len(city)):         # init X
        y=len(city)-1                  # init Y
        for y in range(len(city))[::-1]:
            tmp = 0)
            while y<=len(city)-1 and x>=0: # moving each step 
                tmp2 = city[x][y]
                print (x,y)
                print tmp2    
                if tmp2>0:
                    tmp+=tmp2
                    y-=1
                    x-=1
                    print x, y
                else:# END While
                    tmp+=9999
                    y=len(city)
            if tmp==0:
            	tmp=9999
            memo.append(tmp)
            print memo
    return min(memo)



island = list(range(len(city)))
1, 2, 3, 4, 5, 6, 7, 8, 9

1->2->3->4->5->6->7->8->9
1->2->3->4->5->6->7->8->9
1->2->4->5->6->7->8->9
1->2->5->6->7->8->9
1->2->6->7->8->9
1->2->7->8->9
1->2->8->9
1->2->9

1->3->4->5->6->7->8->9
1->4->5->6->7->8->9
1->5->6->7->8->9
1->6->7->8->9
1->7->8->9
1->8->9
1->9


init_x = 0
end_y = len(city)-1

def visit_search():
	