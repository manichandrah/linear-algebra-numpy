'''Exercise 4 — ReLU, three ways. All three agree, but they're not interchangeable in spirit:
np.where(cond, a, b) is a fully general element-wise if/else — reach for it when the two
branches are genuinely different expressions, not just "clip to zero."
np.clip(x, 0, None) says exactly what it means here: floor at 0, no ceiling.
result[result < 0] = 0 is boolean indexing as a write target (§3.3) — which is why the
function opens with result = x.copy() . Skip that and result = x is just another name for
the same array (plain Python name-binding does no indexing at all), so
result[result < 0] = 0 would silently mutate your original input too.'''
#ReLU in Conditional Method
import numpy as np
x = np.array([-1, -2, -3, -20, 2, 3, 6, 4, 9, 7, 1, -98, 29, -32, -45, 19, 67, -69])
def relu_conditional(x):
    return np.where(x>0, x, 0)
    #general syntax: np.where(condition, value_if_true, value_if_false)
#ReLU in bounding method
def relu_clipping(x):
    return np.clip(x, 0, None)
    #general syntax: np.clip(value, left_hand_limit, right_hand_limit)
#ReLU in Boolean Indexing
def relu_boolean(x):
    result = x.copy()
    result[result < 0] = 0
    return result
x = np.array([-1, -2, -3, -20, 2, 3, 6, 4, 9, 7, 1, -98, 29, -32, -45, 19, 67, -69])
print(f"ReLU in Conditional Method is {relu_conditional(x)}")
print(f"ReLU in Bounding Method is {relu_clipping(x)}")
print(f"ReLU in Boolean Indexing is {relu_boolean(x)}")
#ReLU in NumPy
def relu_numpy(x):
    return np.maximum(0, x)
print(f"ReLU in NumPy is {relu_numpy(x)}")
