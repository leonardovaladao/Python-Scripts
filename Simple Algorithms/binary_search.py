# I will use the Fibbonaci Sequence as an example of array. My binary search will 
# try to find an element in that sequence, and return it's index.
def fibbonaci(max):
    fb = [0, 1]
    for i in range(max):
        fb.append(fb[-1]+fb[-2])
    return fb

# Binary search
def binary_search(element, list):
    low = 0
    high = len(list)-1
    while low <= high:
        half_index = int((low+high)/2)
        if list[half_index] == element:
            return half_index
        elif list[half_index] > element:
            high = half_index-1
        else:
            low = half_index+1
    return None

# I will try to find the single element:
el = 139423224561697880139724382870407283950070256587697307264108962948325571622863290691557658876222521294125

# Define the size of my Fibbonaci Sequence
fibo = fibbonaci(1000)

# Find the element
print('Trying to find', el)
index = binary_search(el, fibo)
if index!=None:
    print('Found. Index:', index)
else:
    print('Not found.')