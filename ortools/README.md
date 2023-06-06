To load addresses.csv:


```
import csv

adresses = []
with open('addresses.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row != []:
            adresses.append([float(row[0]), float(row[1])])
            
