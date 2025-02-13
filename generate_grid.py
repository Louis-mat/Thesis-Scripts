import numpy as np

sep = 100000
m2 = 0.1
M1 = np.linspace(0.5, 2.9, 49)

with open("grid.txt", 'w') as f:
    for m1 in M1:
        mystring = "--initial-mass-1 {:.2f} --initial-mass-2 {} -a {}\n".format(m1, m2, sep)
        f.write(mystring)
