# read data
import os
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
print(parent_dir)
user_feature_path = os.path.join(parent_dir,'data','user_feature.dat')
item_feature_path = os.path.join(parent_dir,'data','item_feature.dat')
shop = os.path.join(parent_dir,'data','shop.dat')

print('========read user_feature:=========')
first_five = []
total_rows = 0
with open(user_feature_path,'r') as f:
    for line in f:
        total_rows += 1
        if total_rows <= 5:
            first_five.append(line.strip())
print("First 5 rows:")
for row in first_five:
    print(row)

print("Total number of rows:", total_rows)

print('========read item_feature:=========')
first_five = []
total_rows = 0
with open(item_feature_path,'r') as f:
    for line in f:
        total_rows += 1
        if total_rows <= 5:
            first_five.append(line.strip())
print("First 5 rows:")
for row in first_five:
    print(row)
print("Total number of rows:", total_rows)

print('========read shop:=========')
first_five = []
total_rows = 0
with open(shop,'r') as f:
    for line in f:
        total_rows += 1
        if total_rows <= 5:
            first_five.append(line.strip())
print("First 5 rows:")
for row in first_five:
    print(row)
print("Total number of rows:", total_rows)