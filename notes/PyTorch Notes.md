
**Tensors**
- specialized DS similar to array/matrices
- used to encode I/O of a model and model's parameters
- similar to NumPy's ndarrays BUT tensors are hardware accelerator compatible
- optimized for autograd

Tensor Initialization
	Sometimes you'll just want to fill tensors with zeros or ones.
	This happens a lot with masking (like masking some of the values in one tensor with zeros to let a model know not to learn them).
```
torch.tensor(data) #where data can be a n-dimensional list or numpy array.

torch.zeros(1,2) #create a 2d tensor containing all zeros
torch.ones(3,4,2) # [[1,1][1,1][1,1][1,1]][[1,1][1,1][1,1][1,1]][[1,1][1,1][1,1][1,1]]

torch.empty(size)
torch.rand(5,3) #for Uniform distribution (in the half-open interval [0.0, 1.0)
torch.randn(5,3) #for Standard Normal (aka. Gaussian) distribution (mean 0 and variance 1)
```

![[Pasted image 20240922203258.png]]


Tensor shape and size
`tensor.shape()` or `tensor.size`
`.ndim` gives the number of dimensions. (also, no. of brackets on one side = no. of dimensions)
![[Pasted image 20240924023345.png]]
![[Pasted image 20240924021409.png]]

Accessing tensor
```
x = torch.tensor(5,3)
x[:, 0] #all tensors in 0th column
x[2, :] #all tensors in 2nd row
x[1,1] #tensor at 1st column and 1st row
x[1,1].item() #value of the tensor at 1st col and 1st row
``` 

Attributes of a tensor
```
tensor.dtype # contains datatype; torch.float16 torch.float32 etc 
tensor.device # Device tensor is stored on
```


Tensor operations on GPU are much faster than on CPU.

==By default, tensors are created on the CPU==. We need to explicitly move tensors to the GPU using `.to` method (after checking for GPU availability).
Copying large tensors across devices can be expensive in terms of time and memory.
```
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")
```

Neural networks are full of matrix multiplications and dot products.

The [`torch.nn.Linear()`](https://pytorch.org/docs/1.9.1/generated/torch.nn.Linear.html) module, also known as a feed-forward* layer or fully connected layer, implements a matrix multiplication between an input `x` and a weights matrix `A`.

`*` 2 broad types of Artificial Neural Networks: (i)Feed Forward (ii) Recurrent


Tensor aggregation(go from more values to less values):

`tensor.max()`,`tensor.min()`, `tensor.type(torch.float32).mean()`, `tensor.sum()`
where tensor is `tensor = torch.rand(2,3)`

find the index of a tensor where the max or minimum occurs with [`torch.argmax()`](https://pytorch.org/docs/stable/generated/torch.argmax.html) and [`torch.argmin()`](https://pytorch.org/docs/stable/generated/torch.argmin.html) respectively.

This is helpful incase you just want the position where the highest (or lowest) value is and not the actual value itself (we'll see this in a later section when using the [softmax activation function](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)).


Reshaping, Stacking, Squeezing and Unsqueezing
![[Pasted image 20240924142429.png]]
