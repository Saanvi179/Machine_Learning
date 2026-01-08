# Matrix A
r1 = int(input("Enter number of rows for matrix A: "))
c1 = int(input("Enter number of columns for matrix A: "))

A = []
print("Enter the elements of matrix A row-wise:")
for i in range(r1):
    row = list(map(int, input().split()))
    A.append(row)

# Matrix B
r2 = int(input("Enter number of rows for matrix B: "))
c2 = int(input("Enter number of columns for matrix B: "))

B = []
print("Enter the elements of matrix B row-wise:")
for i in range(r2):
    row = list(map(int, input().split()))
    B.append(row)

# Check if multiplication possible
if c1 != r2:
    print("Error: Matrices cannot be multiplied")
else:
    # Initialize result matrix with zeros
    result = [[0 for j in range(c2)] for i in range(r1)]

    # Perform matrix multiplication
    for i in range(r1):
        for j in range(c2):
            for k in range(c1):
                result[i][j] += A[i][k] * B[k][j]

    
    print("Resultant Matrix after multiplication:")
    for row in result:
        print(' '.join(map(str, row)))