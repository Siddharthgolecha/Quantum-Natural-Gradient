#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np
import cmath

a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def state(params1):
    """Return state of the variational circuit"""
    variational_circuit(params1)
    return qml.state()

def measurement(params1, params2):
    """Calculate the measurement of the states w.r.t. to two different
    parameterized circuits.
    """
    s0 = state(params1)
    s1 = state(params2)
    val = (np.absolute(np.conj(s0)@s1))**2
    return val

def parameter_shift_term(qnode, params, i):
    """Calculate the Parameter shift term for the parameters"""
    shifted = params.copy()
    shifted[i] += np.pi/2
    forward = qnode(shifted)  # forward evaluation
    shifted[i] -= np.pi
    backward = qnode(shifted) # backward evaluation
    return 0.5 * (forward - backward)

def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    F = np.zeros((6,6))
    shift = np.eye(len(params))
    gradient = np.zeros([6])

    for i in range(len(F)):
        for j in range(len(F[i])):
            shifted = params.copy() + (np.pi/2)*(shift[i] + shift[j])
            f0 = measurement(params, shifted)
            shifted = params.copy() + (np.pi/2)*(shift[i] - shift[j])
            f1 = measurement(params, shifted)
            shifted = params.copy() + (np.pi/2)*(-shift[i] + shift[j])
            f2 = measurement(params, shifted)
            shifted = params.copy() - (np.pi/2)*(shift[i] + shift[j])
            f3 = measurement(params, shifted)
            F[i][j] = float((- f0 + f1 + f2 - f3)/8)


    for i in range(len(gradient)):
        gradient[i] = parameter_shift_term(qnode, params, i)

    F_inv = np.linalg.inv(F)
    QNG = np.dot(F_inv,gradient.T)
    natural_grad = QNG.T

    return natural_grad


def non_parametrized_layer():
    """
    A layer of fixed quantum gates.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":

    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
