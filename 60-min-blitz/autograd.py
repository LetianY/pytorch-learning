import torch, torchvision

# The model is implemented on CPU
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# Forward Pass
prediction = model(data)

# Backward pass: Autograd then calculates and stores the gradients for each model parameter in the parameter’s .grad attribute.
loss = (prediction - labels).sum()
loss.backward() # backward pass

# Load optimizer
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# Initiate gradient descent, the optimizer adjusts each parameter by its gradient stored in .grad
optim.step() #gradient descent

##############################################################################################################################
# Differentiation in Autograd
# Let’s assume a and b to be parameters of an NN, and Q to be the error
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2

# We need to explicitly pass a gradient argument in Q.backward() because it is a vector.
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

# When we call .backward() on Q, autograd calculates these gradients and stores them in the respective tensors’ .grad attribute.
print(9*a**2 == a.grad)
print(-2*b == b.grad)

##############################################################################################################################
# Exclusion from DAG
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

# parameters that don’t compute gradients are usually called frozen parameters
a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

##############################################################################################################################
# In finetuning, we freeze most of the model and typically only modify the classifier layers to make predictions on new labels. 
model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False

# In resnet, the classifier is the last linear layer model.fc. (fully connected layer)
# We can simply replace it with a new linear layer (unfrozen by default) that acts as our classifier.
model.fc = nn.Linear(512, 10)

# Now all parameters in the model, except the parameters of model.fc, are frozen.
# Optimize only the classifier
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

