# Simple algorithm that, given a target-word and a word list, it 
# returns the most similar word relative to the target

# Find larger number:
def check_larger(cel):
  init = 0
  for i in cel:
    if max(i) > init:
      init = max(i)
  return init 

# Larger Common Substring Algorithm
def lcs(target, word_list):
  init = 0
  lcs = None
  for word in word_list:
    cel = [ [ 0 for i in range(len(target)) ] for j in range(len(word)) ] 
    for i, j in zip(range(len(target)), range(len(word))):
      if target[i]==word[j]:
        cel[i][j] = cel[i-1][j-1] + 1
      else:
        cel[i][j] = max(cel[i-1][j], cel[i][j-1])
    
    if check_larger(cel) > init:
      init = check_larger(cel)
      lcs = word 
  return lcs


word = lcs('mish', ['forth', 'wish', 'fish', 'fort', 'mink', 'mist'])
print(word) #Returns wish