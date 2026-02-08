import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

def quantum_circuit(x, params):
    qc = QuantumCircuit(2)

    # Data encoding
    qc.ry(x[0], 0)
    qc.ry(x[1], 1)

    # Variational parameters
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)

    # Entanglement
    qc.cx(0, 1)

    return qc

def get_expectation(qc):
    state = Statevector.from_instruction(qc)
    probs = state.probabilities()
    return probs[0] + probs[1] - probs[2] - probs[3]
