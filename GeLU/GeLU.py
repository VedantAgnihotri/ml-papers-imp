import numpy as np
import openturns as ot
import math

def _cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

x = np.linspace(-5, 5, 1000)
sample = ot.Sample([[float(xi)] for xi in x])
kde = ot.KernelSmoothing()
pdf_estimator: ot.Distribution = kde.build(sample)

def _pdf_arr(xa: np.ndarray) -> np.ndarray:
    return np.array([pdf_estimator.computePDF([float(xi)]) for xi in xa])
    
def gelu(x):
    #return x*_cdf(x) # actual formula
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))) # computationally efficient approximation

def gelu_d(xa:np.ndarray)->np.ndarray:
    cdfs = np.array([_cdf(xi)for xi in xa])
    pdfs = _pdf_arr(xa)
    return cdfs + xa*pdfs
    

xa = np.array([-2.0, 0.0, 2.0])
gelu_values = gelu(xa)
print(gelu_values)
grads = gelu_d(xa)
print(list(zip(xa, grads)))