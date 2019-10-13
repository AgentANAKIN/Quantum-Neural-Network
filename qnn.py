# This was the first neural network that I translated from Python to C to increase my understanding of both programming languages plus neural networks. The original code is from https://www.sololearn.com/learn/744/?ref=app. I have modified the code to seed the pseudo-random number generator from a quantum computer.

from qiskit import Aer, ClassicalRegister, execute, QuantumCircuit, QuantumRegister
from qiskit.tools.monitor import job_monitor



def quniform(min, max):
    range = max - min
    qaddend = range * qmeasure('sim')
    qsum = qaddend + min
    return qsum

def qquniform(min, max):
    range = max - min
    qaddend = range * qmeasure('real')
    qsum = qaddend + min
    return qsum

def qmeasure(hardware):
    if (hardware == 'real'):
        qubits = 14
        #from qiskit.providers.ibmq import least_busy
        #backend = least_busy(IBMQ.backends())
        provider = IBMQ.get_provider(hub='ibm-q')
        provider.backends()
        backend = provider.get_backend('ibmq_16_melbourne')
    else:
        qubits = 32
        backend = Aer.get_backend('qasm_simulator')
        
    q = QuantumRegister(qubits) # initialize all available quantum registers (qubits)
    c = ClassicalRegister(qubits) # initialize classical registers to measure the qubits
    qc = QuantumCircuit(q, c) # initialize the circuit

    i = 0
    while i < qubits:
        qc.h(q[i]) # put all qubits into superposition states so that each will measure as a 0 or 1 completely at random
        i = i + 1
   
    qc.measure(q, c) # collapse the superpositions and get random zeroes and ones
    job = execute(qc, backend=backend, shots=1)
    job_monitor(job)
    result = job.result()
    mraw = result.get_counts(qc)
    m = str(mraw)
    subtotal = 0
    for i in range(qubits):
        subtotal = subtotal + (int(m[i+2]) * 2**(i)) # convert each binary digit to its decimal value, but read left-to-right for simplicity
    multiplier = subtotal / (2**qubits) # convert the measurement to a value between 0 and 1
    return multiplier



from numpy import exp, array, random, dot

class neural_network:
    def __init__(self):
        self.weights = []
        self.weights.append([qquniform(-1, 1)])
        self.weights.append([qquniform(-1, 1)])
        print("self.weights ",self.weights)

    def train(self, inputs, outputs, num):
        for iteration in range(num):
            output = self.think(inputs)
            error = outputs - output
            adjustment = 0.01*dot(inputs.T, error)
            self.weights += adjustment

    def think(self, inputs):
        return (dot(inputs, self.weights))



neural_network = neural_network()

# training data
inputs = array([[2, 3], [1, 1], [5, 2], [12, 3]])
outputs = array([[10, 4, 14, 30]]).T

neural_network.train(inputs, outputs, 10000)

print(neural_network.think(array([15, 2])))
