import numpy as np

U = [
    [-0.6831809,  -0.04200519,  0.72904007],
    [ 0.18289107,  0.76098587, -0.62245591],
    [-0.61517874,  0.5147624,  -0.59713884],
    [-0.33867074, -0.80966534, -0.47931635],
    [-0.52629529, -0.78190408,  0.33412466]
]

v = [-0.96735714, -0.02182641,  0.25247529]
u = [-0.6831809,  -0.04200519,  0.72904007]

dot = lambda zipped: sum([i*j for i,j in zipped])

# u_o = U[3]
u_o = [-0.61517874,  0.5147624,  -0.59713884]

def softmax(v, u_o, U):
	denom = 0
	for u in U:
		denom += np.exp(dot(zip(u,v)))
	return np.exp(dot(zip(u_o,v))) / denom

sigmoid = lambda x: 1 / (1 + np.exp(-x))

