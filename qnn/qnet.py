import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Function

from config import Config
from gradient_calculator import calculate_gradient_list
from input.vector_data_handler import VectorDataHandler
from quantum_network_circuit import QuantumNetworkCircuit


class QNetFunction(Function):

    @staticmethod
    def forward(ctx, input, weight, qnn: QuantumNetworkCircuit, shots, save_statevectors):
        weight_vector = torch.flatten(weight).tolist()
        batch_size = input.size()[0]

        if (input > np.pi).any() or (input < 0).any():
            logging.info('Input data to quantum neural network is outside range {0,π}. Consider using a bounded \
            activation function to prevent wrapping round of states within the Bloch sphere.')

        for i in range(batch_size):
            input_vector = torch.flatten(input[i, :]).tolist()

            if i == 0:
                logging.debug("First input vector of batch to QNN: {}".format(input_vector))

            if type(qnn.config.data_handler is VectorDataHandler):
                if qnn.input_data is None:
                    qnn.construct_network(input_vector)
                ctx.QNN = qnn
            else:
                single_input_qnn = copy.deepcopy(qnn)
                single_input_qnn.construct_network(input_vector)
                ctx.QNN = single_input_qnn

            parameter_list = np.concatenate((np.array(input_vector), weight_vector))

            result = qnn.evaluate_circuit(parameter_list, shots=shots)
            vector = torch.tensor(qnn.get_vector_from_results(result)).unsqueeze(0).float()
            if save_statevectors and result.backend_name == 'statevector_simulator':
                state = result.get_statevector(0)
                qnn.statevectors.append(state)

            if i == 0:
                output = vector
            else:
                output = torch.cat((output, vector), 0)

        ctx.shots = shots

        if cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        ctx.save_for_backward(input, weight)
        ctx.device = device
        output = output.to(device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        device = ctx.device
        weight_vector = torch.flatten(weight).tolist()
        batch_size = input.size()[0]

        for i in range(batch_size):
            input_vector = torch.flatten(input[i, :]).tolist()

            gradient = calculate_gradient_list(ctx.QNN, parameter_list=np.concatenate((input_vector, weight_vector)),
                                               method=ctx.QNN.config.grad_method, shots=ctx.shots)

            ctx.QNN.gradients.append(gradient.tolist())

            single_vector_d_out_d_input = torch.tensor(gradient[:len(input_vector)]).double().to(device)
            single_vector_d_out_d_weight = torch.tensor(gradient[len(input_vector):]).double().to(device)

            if i == 0:
                batched_d_out_d_input = single_vector_d_out_d_input.unsqueeze(0)
                batched_d_out_d_weight = single_vector_d_out_d_weight.unsqueeze(0)
            else:
                batched_d_out_d_input = torch.cat((batched_d_out_d_input, single_vector_d_out_d_input.unsqueeze(0)), 0)
                batched_d_out_d_weight = torch.cat((batched_d_out_d_weight, single_vector_d_out_d_weight.unsqueeze(0)),
                                                   0)
        batched_d_loss_d_input = torch.bmm(batched_d_out_d_input, grad_output.unsqueeze(2).double()).squeeze()
        batched_d_loss_d_weight = torch.bmm(batched_d_out_d_weight, grad_output.unsqueeze(2).double()).squeeze()
        return batched_d_loss_d_input.to(device), batched_d_loss_d_weight.to(device), None, None, None


class QNet(nn.Module):
    """
    Custom PyTorch module implementing neural network layer consisting on a parameterised quantum circuit. Forward and
    backward passes allow this to be directly integrated into a PyTorch network.
    For a "vector" input encoding, inputs should be restricted to the range [0,π) so that there is no wrapping of input
    states round the bloch sphere and extreme value of the input correspond to states with the smallest overlap. If
    inputs are given outside this range during the forward pass, info level logging will occur.
    """

    def __init__(self, n_qubits, encoding, ansatz_type, layers, sweeps_per_layer, activation_function_type, shots,
                 backend_type='qasm_simulator', save_statevectors=False):
        super(QNet, self).__init__()

        config = Config(encoding=encoding, ansatz_type=ansatz_type, layers=layers,
                        sweeps_per_layer=sweeps_per_layer,
                        activation_function_type=activation_function_type,
                        meas_method='all', backend_type=backend_type)
        self.qnn = QuantumNetworkCircuit(config, n_qubits)

        self.shots = shots

        num_weights = len(list(self.qnn.ansatz_circuit_parameters))
        self.quantum_weight = nn.Parameter(torch.Tensor(num_weights))

        self.quantum_weight.data.normal_(std=1. / np.sqrt(n_qubits))

        self.save_statevectors = save_statevectors

        logging.debug("Quantum parameters initialised as {}".format(self.quantum_weight.data))

    def forward(self, input_vector):
        return QNetFunction.apply(input_vector, self.quantum_weight, self.qnn, self.shots, self.save_statevectors)


# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" sub system """

from collections import defaultdict
import numpy as np
from scipy.linalg import sqrtm

from qiskit.quantum_info.states import partial_trace


def get_subsystem_density_matrix(statevector, trace_systems):
    """
    Compute the reduced density matrix of a quantum subsystem.

    Args:
        statevector (list|array): The state vector of the complete system
        trace_systems (list|range): The indices of the qubits to be traced out.

    Returns:
        numpy.ndarray: The reduced density matrix for the desired subsystem
    """
    rho = np.outer(statevector, np.conj(statevector))
    rho_sub = partial_trace(rho, trace_systems).data
    return rho_sub



def get_subsystem_fidelity(statevector, trace_systems, subsystem_state):
    """
    Compute the fidelity of the quantum subsystem.

    Args:
        statevector (list|array): The state vector of the complete system
        trace_systems (list|range): The indices of the qubits to be traced.
            to trace qubits 0 and 4 trace_systems = [0,4]
        subsystem_state (list|array): The ground-truth state vector of the subsystem

    Returns:
        numpy.ndarray: The subsystem fidelity
    """
    rho = np.outer(np.conj(statevector), statevector)
    rho_sub = partial_trace(rho, trace_systems).data
    rho_sub_in = np.outer(np.conj(subsystem_state), subsystem_state)
    fidelity = np.trace(
        sqrtm(
            np.dot(
                np.dot(sqrtm(rho_sub), rho_sub_in),
                sqrtm(rho_sub)
            )
        )
    ) ** 2
    return fidelity


def get_subsystems_counts(complete_system_counts, post_select_index=None, post_select_flag=None):
    """
    Extract all subsystems' counts from the single complete system count dictionary.

    If multiple classical registers are used to measure various parts of a quantum system,
    Each of the measurement dictionary's keys would contain spaces as delimiters to separate
    the various parts being measured. For example, you might have three keys
    '11 010', '01 011' and '11 011', among many other, in the count dictionary of the
    5-qubit complete system, and would like to get the two subsystems' counts
    (one 2-qubit, and the other 3-qubit) in order to get the counts for the 2-qubit
    partial measurement '11' or the 3-qubit partial measurement '011'.

    If the post_select_index and post_select_flag parameter are specified, the counts are
    returned subject to that specific post selection, that is, the counts for all subsystems where
    the subsystem at index post_select_index is equal to post_select_flag.


    Args:
        complete_system_counts (dict): The measurement count dictionary of a complete system
            that contains multiple classical registers for measurements s.t. the dictionary's
            keys have space delimiters.
        post_select_index (int): Optional, the index of the subsystem to apply the post selection
            to.
        post_select_flag (str): Optional, the post selection value to apply to the subsystem
            at index post_select_index.

    Returns:
        list: A list of measurement count dictionaries corresponding to
                each of the subsystems measured.
    """
    mixed_measurements = list(complete_system_counts)
    subsystems_counts = [defaultdict(int) for _ in mixed_measurements[0].split()]
    for mixed_measurement in mixed_measurements:
        count = complete_system_counts[mixed_measurement]
        subsystem_measurements = mixed_measurement.split()
        for k, d_l in zip(subsystem_measurements, subsystems_counts):
            if (post_select_index is None
                    or subsystem_measurements[post_select_index] == post_select_flag):
                d_l[k] += count
    return [dict(d) for d in subsystems_counts]