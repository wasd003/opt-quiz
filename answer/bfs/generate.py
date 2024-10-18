import random

n = int(1e5)
m = n * 100

print(n, m)
st = {}
for i in range(m):
    while True:
        a, b = random.randint(1, n), random.randint(1, n)
        if a in st and b in st[a]:
            continue
        else:
            if a not in st:
                st[a] = set()
            assert b not in st[a], f"{b} is already in st[{a}]"
            st[a].add(b)
            print(a, b)
            break