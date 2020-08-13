# Euclidean Algorithm.
# Given a rectangle of sizes a and b, give the size of the largest square that can divide 
# all the rectangle area. 

def euclid(a,b):
    while b!=0:
        r=a%b
        a=b
        b=r
    return a

sq = euclid(1680, 640)
print('The smallest square has size', sq) # 80