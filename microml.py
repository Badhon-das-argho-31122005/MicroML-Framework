import time
import math 
import random

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} execution time: {execution_time:.6f} seconds")
        return result
    return wrapper

class BaseLayer:
    def forward(self, input):
        raise NotImplementedError("subclass must implement forward method")
    
class ReLULayer(BaseLayer):
    def forward(self, input):
        return [[max(0.0, val) for val in row] for row in input]

class SigmoidLayer(BaseLayer):
    def forward(self, input):
        return [[1 / (1 + math.exp(-val)) for val in row] for row in input]

class LinearLayer(BaseLayer):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = [[random.uniform(-0.5, 0.5) for _ in range(out_features)] for _ in range(in_features)]
        self.biases = [random.uniform(-0.1, 0.1) for _ in range(out_features)]
        
    def forward(self, input):
        return [
            [
                sum(i * w for i, w in zip(row, col)) + b
                for col, b in zip(zip(*self.weights), self.biases)
            ]
            for row in input
        ]

class Sequential:
    def __init__(self, *layer):
        self.layer = layer
        
    def __call__(self, input, *args, **kwds):
        output = input
        for layer in self.layer:
            output = layer.forward(output)
        return output
        
    def __repr__(self):
        layer_str = [f"  ({i}): {layer.__class__.__name__}" for i, layer in enumerate(self.layer)]
        return "MicroML Sequential Model:\n" + "\n".join(layer_str)
    
X_train = [
    [0.5, -0.2, 0.1],
    [-0.5, 0.8, -0.3]
]       

model = Sequential(
    LinearLayer(in_features=3, out_features=4),
    ReLULayer(),
    LinearLayer(in_features=4, out_features=2),
    SigmoidLayer()
)

print(model)
print("-" * 40) 

@time_decorator
def runtime(nn_model, data):
    return nn_model(data)

prediction = runtime(model, X_train)

print("\nModel Predictions:")
for i, pred in enumerate(prediction):
    print(f"Data {i+1}: {pred}")
