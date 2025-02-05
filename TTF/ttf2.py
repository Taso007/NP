import numpy as np
import math

R = 3
C = 3
print("Enter the entries in a single line (separated by space): ")

# User input of entries in a single line separated by space
entries = list(map(int, input().split()))
userMat = np.array(entries).reshape(R, C)

def calculateNorms(mat1, mat2) :
  # first we substract the two matrixes in i and j, then we square it up and then sum it up
  frobeniusNorm = math.sqrt(sum((mat1[i][j] - mat2[i][j]) ** 2 for i in range(R) for j in range(C)))

  #first we have to calculate the sum of columns so, we have to ubstract the two matrixes in i and j find its absolute value of the substraction and then sum it up 
  sumsOfCOlumns = [sum(abs(mat1[i][j] - mat2[i][j]) for i in range(R)) for j in range(C)]
  #after doing all that we find the norm by finding the mux of sum
  norm1 = max(sumsOfCOlumns)

  # this is basically the same as the first norm but it is done on rows.
  sumsOfRows = [sum(abs(mat1[i][j] - mat2[i][j]) for j in range(C))
   for i in range(R)]
  infinityNorm = max(sumsOfRows)

  return frobeniusNorm, norm1, infinityNorm


generatedMat = [np.random.randint(0, 10, size=(R, C)) for _ in range(10)]

frobeniusDistancs = []
norm1Distance = []
infinityNormDistance = []

for i, genMatrix in enumerate(generatedMat):
    frob, first, inf = calculateNorms(userMat, genMatrix)
    frobeniusDistancs.append((i, frob))
    norm1Distance.append((i, first))
    infinityNormDistance.append((i, inf))


frobeniusDistancs.sort(key=lambda x: x[1])
norm1Distance.sort(key=lambda x: x[1])
infinityNormDistance.sort(key=lambda x: x[1])

print(f"closest frobenius matrix: {frobeniusDistancs[0][0]}")
print(f"farthest frobenius matrix: {frobeniusDistancs[-1][0]}")
print(f"closest first norm matrix: {norm1Distance[0][0]}")
print(f"farthest first norm matrix: {norm1Distance[-1][0]}")
print(f"closest infinity norm matrix: {infinityNormDistance[0][0]}")
print(f"farthest infinity norm matrix: {infinityNormDistance[-1][0]}")
