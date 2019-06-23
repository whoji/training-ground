import torch


x = torch.ones(2,2,requires_grad=True)
print(x)
print(x.requires_grad )
print(x.grad, "\n")

# by dafault the requires_grad is False
x2 = torch.ones(2,2)
print(x2)
print(x2.requires_grad )
print(x2.grad, "\n")

y = x + 2
print(y)
print(y.grad_fn, "\n")

# y*y is element-wise mul
z = y*y*3
print(z, "\n")

out = z.mean()
print(out, "\n")

print(x.grad)
out.backward()
print(x.grad, "\n")

print("------------------------")
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)


print("------------------------")

x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x)
print(x.grad)

print("------------------------")

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)