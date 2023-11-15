def target_func(x):
    if x == 0 or x == 1:
        return x
    else:
        a, b = 0, 1
        for _ in range(2, int(x) + 1):
            a, b = b, 2 * b + a
        return b

for i in range(10):
    print(target_func(i))