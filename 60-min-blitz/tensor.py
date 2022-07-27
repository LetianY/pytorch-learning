import torch
import numpy as np

########################################################################################################
# Create Tensor
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Create Tensor from numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# Create Tensor from another tensor
x_ones = torch.ones_like(x_data) # ones, retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # std normal randn, overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

########################################################################################################
# Tensor Operations
# Use GPU: We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")
  
# Change Value
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

# Join Tensors
t1 = torch.cat([tensor, tensor, tensor], dim=1)
t2 = torch.stack([tensor, tensor, tensor], dim=1)
print(t1, '\n', t2)

# Multiplying Tensors
tensor = torch.tensor(data)
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

# Matrix Multiplication
print(f"tensor:\n{tensor}\n{tensor}\n")
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# In-place operations: Operations that have a _ suffix are in-place. For example: x.copy_(y), x.t_(), will change x
print(tensor, "\n")
tensor.add_(5)
print(tensor)


########################################################################################################
# Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor: Changes in the NumPy array reflects in the tensor.
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
