import csv

num_attributes = 6

print("\nThe Given Training Data Set\n")
with open(r'D:\VS\Python\ML-Lab\os\training_data.csv', 'r') as csvfile:
    data = list(csv.reader(csvfile))
    for row in data:
        print(row)

for row in data:
    if row[-1] == 'Yes':
        hypothesis = row[:num_attributes]
        break

print("\nFind S: Finding a Maximally Specific Hypothesis\n")
for row in data:
    if row[-1] == 'Yes':
        for j in range(num_attributes):
            if row[j] != hypothesis[j]:
                hypothesis[j] = '?'

print("\nThe Maximally Specific Hypothesis for the given Training Examples:")
print(hypothesis)
