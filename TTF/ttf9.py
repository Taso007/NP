def gradient_fx(x, y):
    return 6 * x - y + 5

def gradient_fy(x, y):
    return 8 * y - x - 3

def f(x, y):
    return 3 * x**2 + 4 * y**2 - x * y + 5 * x - 3 * y + 7


x, y = 1, 1
n = 0.05

for i in range(3):
    grad_x = gradient_fx(x, y)
    x = x - n * grad_x
    
    print(f"Iteration {i + 1} - After updating x:")
    print(f"x = {x:.4f}, y = {y:.4f}, f(x, y) = {f(x, y):.4f}")

    grad_y = gradient_fy(x, y)
    y = y - n * grad_y

    print(f"Iteration {i + 1} - After updating y:")
    print(f"x = {x:.4f}, y = {y:.4f}, f(x, y) = {f(x, y):.4f}\n")
