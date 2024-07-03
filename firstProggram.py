from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import UnitaryGate
import numpy as np
import sys
circuit = QuantumCircuit(1)
N = 3
pos = QuantumRegister(N,"pos")
coin = QuantumRegister(1,"coin")

c_pos = ClassicalRegister(N,"classical_pos")
c_coin = ClassicalRegister(1,"classical_coin")

circuit = QuantumCircuit(pos,coin,c_pos,c_coin)

state = Statevector([0,0,0,1,0,0,0,0])
circuit.prepare_state(state, pos)
circuit.prepare_state(Statevector([1,0]), coin)

M=[[0,1,0,0,0,0,0,0],
          [0,0,1,0,0,0,0,0],
          [0,0,0,1,0,0,0,0],
          [1,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,1],
          [0,0,0,0,1,0,0,0],
          [0,0,0,0,0,1,0,0],
          [0,0,0,0,0,0,1,0]]




gate = UnitaryGate(M)
no_steps = 3

for i in range(no_steps):
    circuit.h(3)
    circuit.append(gate, [0,1,2])

circuit.measure(pos,c_pos)
circuit.measure(coin,c_coin)



results = Sampler().run(circuit).result()
statistics = results.quasi_dists[0].binary_probabilities()


def binaryToDecimal(n):
    return int(n,2)

for state in statistics.copy():
    temp = state[1:]
    statistics[temp] = statistics.pop(state)
    statistics[binaryToDecimal(temp)] = statistics.pop(temp)

print(statistics)
    
plot_histogram(statistics)
circuit.draw("mpl")
plt.show()

heads = Statevector([1,0])
tails = Statevector([0,1])

arr = np.array([heads.data])
print(arr)
arr = arr.T
print(arr)
# arr = np.zeros(4)
# arr[3] = 1.0
# print(arr)
# mat = Statevector(arr)
# print(mat)

# arr = np.zeros(4)
# arr[3] = 1.0
# arr = np.array([arr])
# arr = arr.T
# print(arr)
# mat = Statevector(arr)
# print(mat)



def ket(N,n,coin_state):
    arr = np.zeros(N)
    arr[n] = 1.0
    return Statevector(arr)^coin_state

def bra(N,n,coins_state):
    arr = np.zeros(N)
    arr[n] = 1.0
    arr = np.array([arr])
    arr = arr.T
    mat = Statevector(arr)
    print(mat)
    coins_state = [[1],[0]] 
    return(mat^coins_state)
N = 4
M = np.array((N*2,N*2))
# for i in range(N):
#     M += np.outer(ket(N,(i+1)%N,heads),bra(N,i,heads)) + np.outer(ket(N,(i-1)%N,tails),bra(N,i,tails))
# print(bra(N,2,heads))

sys.exit()





circuit.draw("mpl")
plt.show()
results = Sampler().run(circuit).result()
statistics = results.quasi_dists[0].binary_probabilities()
plot_histogram(statistics)
plt.show()
circuit.draw("mpl")
plt.show()


# circuit.h(Y)
# circuit.h(X)


# def decimalToBinary(n): 
#     return bin(n).replace("0b", "") 

# print(decimalToBinary(15))

M=[[0,1,0,0,0,0,0,0],
          [0,0,1,0,0,0,0,0],
          [0,0,0,1,0,0,0,0],
          [1,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,1],
          [0,0,0,0,1,0,0,0],
          [0,0,0,0,0,1,0,0],
          [0,0,0,0,0,0,1,0]]
gate = UnitaryGate(M)
no_steps = 3
state = Statevector([1,0,0,0,0,0,0,0])
circuit.prepare_state(state, qr)
for i in range(no_steps):
    circuit.h(2)
    circuit.append(gate, [0,1,2])









# # circuit.cx(X, Y)
# # circuit.x(X)






# print(circuit)



circuit.measure(qr, cr)
circuit.draw("mpl")
plt.show()
results = Sampler().run(circuit).result()
statistics = results.quasi_dists[0].binary_probabilities()
plot_histogram(statistics)
plt.show()