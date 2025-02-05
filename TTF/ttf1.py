import math

# L1 Norm is the sum of the magnitudes of the vectors in a space.
def calculate1Norm(vec):
    return sum(abs(x) for x in vec)

# L2 Euclidean norm is the shortest distance to go from one point to another.
def calculate2Norm(vec):
    sumOfSquares = sum(x**2 for x in vec)
    return math.sqrt(sumOfSquares)

# The infinity norm measures how large the vector is by the magnitude of its largest entry.
def infinityNorm(vec):
    return max(abs(x) for x in vec)

# p-norm on suitable real vector spaces given by the p-th root of the sum (or integral) of the p-th powers of the absolute values of the vector components
def calculatePnorm(vec, p):
    sumOfPowers = sum(abs(x)**p for x in vec)
    return sumOfPowers**(1/p)

# user input
vector_input = input("Enter vector: ")
vector = list(map(int, vector_input.split()))
# [2, 3, -4, 5]

# L1 Norm
norm1 = calculate1Norm(vector)
print(f"L1 Norm: {norm1}") # 14

# L2 Norm
norm2 = calculate2Norm(vector)
print(f"L2 Norm: {norm2}") # 7.3484692283495345

# Infinity Norm
norminfinity = infinityNorm(vector)
print(f"Infinity Norm: {norminfinity}") # 5

# p-Norm
p = float(input("Enter the value of p for the p-norm: "))
# 3

normP = calculatePnorm(vector, p)
print(f"{p}-Norm: {normP}") # 6.0731779437513245
