rows = int(input("Enter number of rows: "))
columns = int(input("Enter number of columns:"))

matrix = []
print("Enter elements of the matrix row-wise:")

for i in range(rows):
    row = list(map(int, input().split()))
    matrix.append(row)

# Transpose the matrix
transpose = [[matrix[j][i] for j in range(rows)]for i in range(columns)]

print("Transpose of the matrix:")
for row in transpose:
    print(row)