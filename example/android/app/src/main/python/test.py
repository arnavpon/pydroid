def test_multiply(a, b):
    return a * b

def test_add(a, b):
    return a + b

def test_subtract(a, b):
    return a - b

def test_divide(a, b):
    if (b != 0): 
        return a / b
    

print(test_multiply(2, 3))
print(test_add(5, 3))
print(test_subtract(8, 3))