class Scalar:
    """A scalar value with automatic differentiation support.
    
    This class represents a node in the computational graph and keeps track of 
    the gradient information needed for backpropagation.
    
    Attributes:
        data: The scalar value.
        grad: The gradient of the loss with respect to this value.
        _backward: Function to compute gradients during backpropagation.
        _prev: Tuples of (input_node, gradient_function) for each input.
    """
    
    def __init__(self, data):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set()
    
    def __repr__(self):
        return f"Scalar(data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        result = Scalar(self.data + other.data)
        
        # Store references to inputs
        result._prev = {(self, 1.0), (other, 1.0)}
        
        # Define backward pass
        def _backward():
            self.grad += result.grad * 1.0
            other.grad += result.grad * 1.0
        
        result._backward = _backward
        
        return result
    
    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        result = Scalar(self.data * other.data)
        
        # Store references to inputs
        result._prev = {(self, other.data), (other, self.data)}
        
        # Define backward pass
        def _backward():
            self.grad += result.grad * other.data
            other.grad += result.grad * self.data
        
        result._backward = _backward
        
        return result
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Power must be int or float"
        result = Scalar(self.data ** power)
        
        # Store references to inputs
        result._prev = {(self, power * self.data ** (power - 1))}
        
        # Define backward pass
        def _backward():
            self.grad += result.grad * power * self.data ** (power - 1)
        
        result._backward = _backward
        
        return result
    
    def __truediv__(self, other):
        return self * other**(-1)
    
    def __rtruediv__(self, other):
        return other * self**(-1)
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __radd__(self, other):
        return self + other
    
    def __rmul__(self, other):
        return self * other
    
    def backward(self):
        """Compute gradients through backpropagation."""
        # Topological sort
        topo = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child, _ in node._prev:
                    build_topo(child)
                topo.append(node)
        
        build_topo(self)
        
        # Initialize gradient at output
        self.grad = 1.0
        
        # Backpropagate
        for node in reversed(topo):
            node._backward()