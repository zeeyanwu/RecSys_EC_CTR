import torch

# Create a tensor of shape [2, 3, 4]
a = torch.tensor([
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]],

    [[10, 20, 30, 40],
     [50, 60, 70, 80],
     [90, 100, 110, 120]]
])

print("a.shape =", a.shape)
print("a =", a)

# Sum along dim=1
b = torch.sum(a, dim=1)
print("\nAfter torch.sum(a, dim=1):")
print("b.shape =", b.shape)
print("b =", b)
