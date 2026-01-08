list1 = list(map(int, input().split()))
list2 = list(map(int, input().split()))

common_elements = set(list1) & set(list2)
print("Number of common elements:", len(common_elements))