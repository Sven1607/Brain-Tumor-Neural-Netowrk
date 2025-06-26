import numpy as np

input_arr = np.random.randn(20, 20)
kernel = np.array([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]])

def conv(I, K, S):
    final_arr = []
    for i in range(0, np.size(I, axis=0), S):
        for j in range(0, np.size(I, axis=1), S):
            try:
                a = I[j, i+2]
                a = I[j+2, i]
            except IndexError:
                continue
            selected_I = I[i:i+3, j:j+3]
            pixel = float(np.sum(np.sum(selected_I*K, axis=0)))
            final_arr.append(pixel)
    return final_arr


