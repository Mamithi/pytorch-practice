import torch 
import torch.autograd as autograd 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim  

torch.manual_seed(1) 

#  torch.tensor(data) creates a torch.Tensor object with the given data.
V_data = [1., 2., 3.]
V = torch.tensor(V_data)
print(V)

# Creates a matrix 
M_data = [[1., 2, 3.], [4., 5., 6]]
M = torch.tensor(M_data)
print(M)

# Create a 3D tensor of size 2x2x2 
T_data = [[[1., 2.], [3, 4.]], [[5., 6.], [7., 8.]]]
T = torch.tensor(T_data)
print(T)

print(V[0])
print(V[0].item())

print(M[0][0].item())
print(T[0])

x = torch.randn(3, 4, 5)
print(x)

x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
z = x + y
print(z)

x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5) 
z_1 = torch.cat([x_1, y_1])
print(z_1)

# concatenate columns 
x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

# Reshaping tensors
x = torch.randn(2, 3, 4)
print(x)
print(x.view(2, 12)) # reshape to 2 rows 12 columns
# same as above, if one of teh dimensions is -1 its size can be inferred
print(x.view(2, -1))

x = torch.tensor([1., 2., 3.], requires_grad=True)
y = torch.tensor([4., 5., 6.], requires_grad=True)

z = x + y
print(z) 

print(z.grad_fn)

s = z.sum()
print(s)
print(s.grad_fn)

# calling .backward() on any variable will run backprop, starting from it.
s.backward()
print(x.grad)

x = torch.randn(2,2)
y = torch.randn(2,2)
print(x.requires_grad, y.requires_grad)
z=x+y
print(z.grad_fn)

x = x.requires_grad_()
y = y.requires_grad_()
z = x+y
print(z.grad_fn)
print(z.requires_grad)

new_z = z.detach()
print(new_z.grad_fn)

print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print((x**2).requires_grad)
