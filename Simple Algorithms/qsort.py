# Quicksort: Use functional programming to create an algorithm that order some list.
# Take the item on the middle index. Separate two lists: one for items smaller than the item,
# one for items greater than the item. Use the same function on the two lists, and return the
# separation of those lists. Repeat the process until your list has none or one item. 

def qsort(list):
    if len(list) <= 1:
        return list
    else:
        half = int(len(list)/2)
        pivot = list[half]
        minor = [i for i in list[0:] if i < pivot]
        major = [i for i in list[0:] if i > pivot]
    return qsort(minor) + [pivot] + qsort(major)

ar = [5,6,1,2,9,0,3,7,4,8]
new_list = qsort(ar)
print(new_list)