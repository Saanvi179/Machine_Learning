import random
import statistics

numbers = [random.randint(100, 150) for i in range (100)]

mean = statistics.mean(numbers)
median = statistics.median(numbers)
mode = statistics.mode(numbers)

print("Random Numbers: ", numbers)
print("Mean: ", mean)
print("Median: ", median)
print("Mode: ", mode)