# Filename: JMH_VLS_Function_Database.py
# Author: Jessie M. Henderson

# A host of import statements for using:
# 1) All the basic circuit creation and visualization tools from Qiskit.
# 2) Noise imports and job monitoring for running on simulated hardware and actual quantum hardware.
# 3) Plotting tools from Matplotlib (includes Pylab import).
# 4) Numpy for matrix/vector manipulation, and Math for using sqrt.
# 6) Scipy.optimize and Qiskit.aqua to use various optimizers for minimize in the former 
# and to use SPSA in the latter.
from qiskit import *
from qiskit.providers.aer.noise import NoiseModel
from qiskit.visualization import plot_histogram
from qiskit.tools import job_monitor
from qiskit.transpiler import Layout
import matplotlib.pyplot as plt
import matplotlib
import pylab
import numpy as np
from scipy.optimize import minimize
from math import sqrt

# Description: The number of parameters in the circuit we are training depends
# upon the number of qubits and the number of "layers," where each layer is 
# a specified set of gates.
# The formula (encoded here) is
# num_parameters = number_of_qubits + 2*number_of_layers(number_of_qubits/2 + (number_of_qubits - 1)/2).
# Note that if the qubit count is odd, the position of Rys is slightly different
# on the last qubit.  In the round of Ry's prior to any layer, there is none on
# the last qubit.  Additionally, although there would not ordinarily be an Ry
# in the second round of Ry's per layer, for a circuit with odd qubits, the 
# _last_ layer should have an Ry on the last qubit in the second round of Rys.
# Thus, the number of Rys (and thus the number of parameters) does not change.
# Parameters: num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
# Return Value: An integer with the number of trainable parameters in the
# circuit.
def get_num_parameters(num_qubits, num_layers):
  return num_qubits + 2 * num_layers * ((num_qubits // 2) + ((num_qubits - 1) // 2))
  
# Description: The ansatz for our given circuit is composed of controlled-Z
# and Ry gates.  Each Ry gate has its own parameter, which we want to train.
# The ansatz has the following format: an initial "layer" of one Ry gate on
# each qubit, followed by uniform "layers" with a series of gates.  One layer
# is: 1) Controlled-Z gates on qubits 1/2, 3/4, ..., n-1/n, where n is the last
# qubit.  2) Ry gates on all qubits.  3) Controlled-Z gates on qubits
# 2/3, 4/5, ..., n-2/n-1, where n is the last qubit.  And 4) Ry gates
# on qubits 2 through n-1.  
# Parameters: params - An array with the parameters for each Ry gate.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
# Return Value: A Qiskit QuantumCircuit with the ansatz and a QuantumRegister
#               with the qubits.
def create_ansatz(params, num_qubits, num_layers):
  quantum_reg = QuantumRegister(num_qubits, name="q")
  classical_reg = ClassicalRegister(num_qubits, name="c")
  quantum_circuit = QuantumCircuit(quantum_reg, classical_reg)
  
  odd_qubit_count = False
  if (num_qubits % 2 != 0):
    odd_qubit_count = True

  # Every Ry gate needs to have a different parameter.
  cur_param_index = 0
  
  # There needs to be an Ry gate on every qubit to begin; this is an initialization that is only
  # performed once before all of the layers.
  # Note that the sole exception to this is if we have an odd number of qubits:
  # in that case, there is an Ry on every qubit except for the last one.
  last_qubit = 0
  if (odd_qubit_count == True):
    last_qubit = num_qubits-1
  else:
    last_qubit = num_qubits
  for index in range(last_qubit):
    quantum_circuit.ry(params[index], index)
    cur_param_index += 1
  
  # For each layer...
  for layer_index in range(num_layers):
    # For each qubit, add the following gates.
    # Add the controlled-Z gates, which connect qubits i and i+1, where i iterates over the qubits.
    for index in range(0, num_qubits, 2):
      if (index + 1 < num_qubits):
        quantum_circuit.cz(index, index+1)
      
    # Add the Ry gates, using our parameter index counter.
    for index in range(num_qubits):
      quantum_circuit.ry(params[cur_param_index], index)
      cur_param_index += 1
      
    # Add the second round of controlled Z-gates, which connect qubits i+1 and i+2, where i iterates
    # over the qubits.
    for index in range(1, num_qubits-1, 2):
        quantum_circuit.cz(index, index+1)
       
    # Add the second round of Ry gates; note that, this time, there should not be an Ry on the first
    # or last of the qubits.
    for index in range(1, num_qubits-1):
      quantum_circuit.ry(params[cur_param_index], index)
      cur_param_index += 1

    # If we have an odd number of qubits and we are on the last layer, add an
    # Ry on the last qubit even though the second round of Rys usually doesn't
    # have such a gate.
    if (odd_qubit_count == True and layer_index == num_layers-1):
      quantum_circuit.ry(params[cur_param_index], num_qubits-1)
      cur_param_index += 1
  return quantum_circuit, quantum_reg
  
# Description: See above for the detailed description of the ansatz. The utility
# in this function lies in making one "for display," meaning that it includes
# barriers to show the layers.
# Parameters: params - An array with the parameters for each Ry gate.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
# Return Value: A Qiskit QuantumCircuit with the ansatz and a QuantumRegister
#               with the qubits.
def create_ansatz_for_display(params, num_qubits, num_layers):
  quantum_reg = QuantumRegister(num_qubits, name="q")
  classical_reg = ClassicalRegister(num_qubits, name="c")
  quantum_circuit = QuantumCircuit(quantum_reg, classical_reg)
  
  odd_qubit_count = False
  if (num_qubits % 2 != 0):
    odd_qubit_count = True

  # Every Ry gate needs to have a different parameter.
  cur_param_index = 0
  
  # There needs to be an Ry gate on every qubit to begin; this is an initialization that is only
  # performed once before all of the layers.
  # Note that the sole exception to this is if we have an odd number of qubits:
  # in that case, there is an Ry on every qubit except for the last one.
  last_qubit = 0
  if (odd_qubit_count == True):
    last_qubit = num_qubits-1
  else:
    last_qubit = num_qubits
  for index in range(last_qubit):
    quantum_circuit.ry(params[index], index)
    cur_param_index += 1

  # Insert a barrier after this 'pre-layer.'
  quantum_circuit.barrier(quantum_reg)
  
  # For each layer...
  for layer_index in range(num_layers):
    # For each qubit, add the following gates.
    # Add the controlled-Z gates, which connect qubits i and i+1, where i iterates over the qubits.
    for index in range(0, num_qubits, 2):
      if (index + 1 < num_qubits):
        quantum_circuit.cz(index, index+1)
      
    # Add the Ry gates, using our parameter index counter.
    for index in range(num_qubits):
      quantum_circuit.ry(params[cur_param_index], index)
      cur_param_index += 1
      
    # Add the second round of controlled Z-gates, which connect qubits i+1 and i+2, where i iterates
    # over the qubits.
    for index in range(1, num_qubits-1, 2):
        quantum_circuit.cz(index, index+1)
       
    # Add the second round of Ry gates; note that, this time, there should not be an Ry on the first
    # or last of the qubits.
    for index in range(1, num_qubits-1):
      quantum_circuit.ry(params[cur_param_index], index)
      cur_param_index += 1
    
    # Insert a barrier after each layer.
    quantum_circuit.barrier(quantum_reg)

    # If we have an odd number of qubits and we are on the last layer, add an
    # Ry on the last qubit even though the second round of Rys usually doesn't
    # have such a gate.
    if (odd_qubit_count == True and layer_index == num_layers-1):
      quantum_circuit.ry(params[cur_param_index], num_qubits-1)
      cur_param_index += 1
  return quantum_circuit, quantum_reg
  
# Description: Build a numpy matrix representing the "A" portion of our problem,
#             Ax=b.  This builds A from a file, where each row of the matrix is 
#             printed without commas separating entries and with each row on its
#             own line.
# Parameters: num_qubits - The number of qubits in the circuit.
#             filename - A string with the name of the file holding the matrix.
# Return Value: The desired A, as a numpy matrix.
def create_A(num_qubits, filename):
  A = np.zeros((2**num_qubits, 2**num_qubits))
  row_index = 0
  
  file = open(filename, "r")
  nextline = file.readline()

  column_index = 0
  while nextline != '':
    space_index = nextline.find(" ")

    if (space_index == -1):
      space_index = len(nextline)
    
    next_element = nextline[0 : space_index]
    nextline = nextline[space_index+1 :]
    A[row_index][column_index] = float(next_element)
    column_index += 1

    if (nextline == ''):
      nextline = file.readline()
      column_index = 0
      row_index += 1
  
  """# To test if the matrix loaded correctly...
  for row in range(len(A)):
    for column in range(len(A[0])):
       print(A[row][column], " ", sep="", end="")
    print()
  """
  
  return A
  
# Description: Build a numpy vector representing the "b" portion of our problem,
#             Ax=b.  This builds b from a file, where entires are separated with
#             spaces.
# Parameters: num_qubits - The number of qubits in the circuit.
#             filename - A string with the name of the file holding the vector.
#             normalize - A flag indicating whether to normalize the vector.
# Return Value: The desired b, as a numpy vector.
def create_b(num_qubits, filename, normalize):
  b = np.zeros((2**num_qubits, 1))
    
  file = open(filename, "r")
  nextline = file.readline()

  column_index = 0
  while nextline != '':
    space_index = nextline.find(" ")

    if (space_index == -1):
      space_index = len(nextline)
    
    next_element = nextline[0 : space_index]
    nextline = nextline[space_index+1 :]
    b[column_index] = float(next_element)
    column_index += 1
  
  """# To test if the vector loaded correctly...
  for row in range(len(b)):
    print(b[row], " ", sep="", end="")
  print()
  """

  if (normalize):
    b = np.divide(b, sqrt(np.sum(b**2)))

  return b
  
# Description: Build a numpy vector representing the "x" portion of our problem,
#             Ax=b, where x has been classically-computed.
# Parameters: num_qubits - The number of qubits in the circuit.
#             filename - A string with the name of the file holding the vector.
#             normalize - A flag indicating whether we should normalize the vector.
# Return Value: The desired x, as a numpy vector.
def create_x(num_qubits, filename, normalize):
  x = np.zeros((2**num_qubits, 1))
    
  file = open(filename, "r")
  nextline = file.readline()

  column_index = 0
  while nextline != '':
    space_index = nextline.find(" ")

    if (space_index == -1):
      space_index = len(nextline)
    
    next_element = nextline[0 : space_index]
    nextline = nextline[space_index+1 :]
    x[column_index] = float(next_element)
    column_index += 1
  
  """# To test if the vector loaded correctly...
  for row in range(len(x)):
    print(x[row], " ", sep="", end="")
  print()
  """

  if (normalize):
    x = np.divide(x, sqrt(np.sum(x**2)))

  return x
  
# Description: Build the Hermitian for the pitchfork fracture problem.
# The formula is: H = A^dagger(I - |b><b|)A.  Recall that our problem is Ax=b, and we are seeking x.
# Parameters: num_qubits - An integer with the number of qubits in the circuit.
#             filename_A - A string with the filename for matrix A.
#             filename_b - A string with the filename for vector b.
#             normalize - A flag indicating whether to normalize b.
# Return Value: The desired Hamiltonian, stored as a numpy matrix.
def create_H(num_qubits, filename_A, filename_b, normalize):
  H = np.zeros((2**num_qubits, 2**num_qubits))
  
  A = create_A(num_qubits, filename_A)
  b = create_b(num_qubits, filename_b, normalize)

  A_dagger = A.conj().T

  b_dagger = b.conj().T

  b_outer_product =  b @ b_dagger

  I = np.identity(2**num_qubits)

  H = A_dagger @ (I - b_outer_product) @ A

  return H
  
# Description: Compute the fidelity of our answer by reading in a classically-
#              computed answer and comparing it to our answer.
# Parameters: num_qubits - An integer with the number of qubits in the circuit.
#             filename_classical_x - A string with the filename for vector x.
#             computed_x - The values of x we want to verify.
# Return Value: The fidelity, |<classical_x|computed_x>|^2.
def calculate_fidelity_from_file(num_qubits, filename_classical_x, computed_x):
  classical_x = create_x(num_qubits, filename_classical_x, True)
  
  # Normalize both vectors. (classical_x was created with normalization.)
  computed_x = np.divide(computed_x, sqrt(np.sum(computed_x**2)))
 
  classical_x_dagger = classical_x.conj().T

  # Compute the dot product.
  fidelity = 0
  for index in range(len(computed_x)):
    fidelity += classical_x_dagger[0][index]*computed_x[index]

  fidelity = (fidelity)**2

  return fidelity
  
# Description: Compute the fidelity of our answer by comparing a previously read
#              answer to our computed answer.
# Parameters: num_qubits - An integer with the number of qubits in the circuit.
#             classical_x - The values of the true x to which we are comparing.
#             computed_x - The values of x we want to verify.
# Return Value: The fidelity, |<classical_x|computed_x>|^2.
def calculate_fidelity(num_qubits, classical_x, computed_x):
  # Normalize both vectors.
  classical_x = np.divide(classical_x, sqrt(np.sum(classical_x**2)))
  computed_x = np.divide(computed_x, sqrt(np.sum(computed_x**2)))
 
  classical_x_dagger = classical_x.conj().T

  # Compute the dot product.
  fidelity = 0
  for index in range(len(computed_x)):
    try:
      fidelity += classical_x_dagger[0][index]*computed_x[index]
    except:
      fidelity += classical_x_dagger[index]*computed_x[index]

  fidelity = (fidelity)**2

  return fidelity
  
# Description: Diagonalize a given Hermitian matrix, so that it has the form
# H = WDW^dagger, where W is a unitary.
# Parameters: H - A numpy matrix with the Hermitian to diagonalize.
# Return Value: The desired W, D, and W^dagger matrices.
def diagonalize_H(H):
  D, W = np.linalg.eigh(H)
  W_dagger = W.conj().T
  return W, D, W_dagger
  
# Description: The following set of three functions provide a gradient-descent method of optimizing
#              without any black-box optimizer.
# 	       This function computes the first term in the parameter-shift rule (the +pi/2 term).
# Attributions: This code is a slightly-modified (to fit the current problem) 
#              version of Marco Cerezo's implementation of adaptive gradient-descent.
# Parameters: params - A list of the current parameters for the ansatz.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
#             H - A numpy matrix with the Hermitian encoding the linear systems problem.
#             index - An integer with the index of the parameter that we are currently optimizing.
# Return Value: A double with the cost of the parameters with +pi/2 adjustment.
def gradient_cost_plus_term(params, num_qubits, num_layers, H, index):
  adjusted_params = np.array([])
  num_parameters = get_num_parameters(num_qubits, num_layers)
  for i in range(0, num_parameters):
      if i == index:
          adjusted_params = np.append(adjusted_params, params[i] + np.pi / 2)
      else:
          adjusted_params = np.append(adjusted_params, params[i])
  return (cost_function_no_shot_noise(adjusted_params, num_qubits, num_layers, H))

# Description: The following two functions are part of the above three that provide a gradient-descent method of optimizing
#              without any black-box optimizer.
# 	       This function computes the second term in the parameter-shift rule (the -pi/2 term).
# Attributions: This code is a slightly-modified (to fit the current problem) 
#              version of Marco Cerezo's implementation of adaptive gradient-descent.
# Parameters: params - A list of the current parameters for the ansatz.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
#             H - A numpy matrix with the Hermitian encoding the linear systems problem.
#             index - An integer with the index of the parameter that we are currently optimizing.
# Return Value: A double with the cost of the parameters with -pi/2 adjustment.
def gradient_cost_minus_term(params, num_qubits, num_layers, H, index):
  adjusted_params = np.array([])
  num_parameters = get_num_parameters(num_qubits, num_layers)
  for i in range(0, num_parameters):
      if i == index:
          adjusted_params = np.append(adjusted_params, params[i] - np.pi / 2)
      else:
          adjusted_params = np.append(adjusted_params, params[i])
  return (cost_function_no_shot_noise(adjusted_params, num_qubits, num_layers, H))

# Description: This is the final function in the set of three that implements a gradient-descent
#              without any black-box optimizer.
# 	       This is an implementation of an analytical parameter-shift optimization.
# Attributions: This code is a slightly-modified (to fit the current problem) 
#              version of Marco Cerezo's implementation of adaptive gradient-descent.
# Parameters: params - A list of the current parameters for the ansatz.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
#             H - A numpy matrix with the Hermitian encoding the linear systems problem.
#             cost_func - A string with the name of the cost function implementation to use.
#             param_shift_plus_func - A string with the name of the +pi/2 component of the parameter-shift rule.
#             param_shift_minus_func - A string with the name of the -pi/2 component of the parameter-shift rule.
#             eta - A double with the 'learning rate,' which represents how far the function will seek to 'jump' 
#             based on whether the cost function slope is steep (large jumps) or shallow (small jumps).
#             eps_grad - A double representing how small the gradient should be before we start counting 'well jumps.'
#             This determines when we should stop the algorithm, if we are simply bouncing around in a local minimum.
#             eps_cost - A double with a desired cost to achieve. (Optimization will stop when this cost is achieved,
#             so making this arbitrarily small can give us arbitrary precision.)
#             iter_max - An integer with the maximum number of iterations to perform. 
#             well_iter_max - An integer with the maximum number of times we allow ourselves to hop around a local minimum
#             seeking a better result. (We start counting when a cost of eps_grad is achieved.)
# Return Value: A dictionary with the cost per iteration, as well as a list of lists with the parameters at each iteration.
def vanilla_gradient_optimization(params, num_qubits, num_layers, H, cost_func, param_shift_plus_func, param_shift_minus_func,
			eta, eps_grad, eps_cost, iter_max, well_iter_max):
  cur_iter_count = 0
  cur_well_count = 0
  num_parameters = get_num_parameters(num_qubits, num_layers)
  param1 = np.array([])
  costvalue = np.array([])
  all_params = []
  iter_cost_dict={}
   
  cost = cost_func(params, num_qubits, num_layers, H)
  costvalue = np.append(costvalue, cost)
    
  for i in range(0,num_parameters):
      param1 = np.append(param1, params[i])
        
  # While the iteration count is less than the maximum, and the well count is less than the maximum, and the achieved
  # cost is higher than the desired cost, continue the optimization.    
  while cur_iter_count < iter_max and cur_well_count < well_iter_max and cost > eps_cost:
    param2 = np.array([])
    param3 = np.array([])
    grad_vec = np.array([])
    
    # Evaluate the gradient for the current parameters.
    for i in range(0, num_parameters):
      grad_vec = np.append(grad_vec, 
      (param_shift_plus_func(param1, num_qubits, num_layers, H, i) - param_shift_minus_func(param1, num_qubits, num_layers, H, i)) / 2)
        
    # If we have achieved a gradient value indicating that we are in a local minimum, start counting the number
    # of steps that we will allow ourselves to explore this well before stopping.
    grad = np.linalg.norm(grad_vec)**2
    if grad <= eps_grad:
      cur_well_count += 1
    
    # Update the parameters using a standard, "vanilla" gradient-descent with adaptive learning rate.       
    param2 = np.real(param1-eta*grad_vec)
    for i in range(0,num_parameters):
      param3 = np.append(param3, param2[i])
    param3 = np.real(param2 - eta*grad_vec)
    
    # Put the "adaptive" in adaptive learning rate: based upon whether slope is steep or not, decide on how large our 
    # jump size should be.
    compare_cost = cost_func(param3, num_qubits, num_layers, H)
    if cost - compare_cost >= eta*grad:
      eta = 2*eta
      param1 = param3
    elif cost - compare_cost <(eta/2)*grad:
      eta = eta/2
      param1 = param2
    else:
      param1 = param2
    
    cost = cost_func(param1, num_qubits, num_layers, H)
    alpha_opt = param1
    
    # Save the data (itearation, cost and parameters), and 
    # update the overall number of iterations.
    iter_cost_dict.update({cur_iter_count:np.real(cost)})
    costvalue = np.append(costvalue, cost)
    all_params.append(param1)
    cur_iter_count += 1
    
    print("At iteration ", cur_iter_count, " the cost is ", cost, ".", sep="")
    
  # Return the dictionary of each iteration and cost.
  return iter_cost_dict, all_params
    
# Description: Estimate our cost function using finite sampling.
# Attributions: Thanks to Marianna Podzorova for the suggestion to 
#               normalize the distribution to avoid round-off error, as well as
#               for the suggestion to use Numpy's multinomial function, which
#               made the sampling process considerably simpler.
# Parameters: params - An array with the current values of the parameters.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
#             H - The classically-computed value of the Hermitian that we want
#             to obtain.
#             shot_count - The number of shots to use.
# Return Value: The value of the cost function: The real part of <psi|H|psi>, 
#               where psi is calculated using the statevector simulator.  Note
#               that the complex part for this Hermitian is always zero, but
#               we seek only the real part so that we have the right variable
#               type.
def cost_function_finite_sampling(params, num_qubits, num_layers, H, shot_count = 10000):
  # Create the ansatz circuit, and use the statevector simulator to run it.
  quantum_circuit, quantum_register = create_ansatz(params, num_qubits, num_layers)
  backend = Aer.get_backend('statevector_simulator')
  job = execute(quantum_circuit, backend)
  outputstate = job.result().get_statevector(quantum_circuit, decimals=11)

  # Diagnoalize H and use the result to obtain W^dagger(outputstate).
  # We will use that to estimate the cost using sampling.
  W, D, W_dagger = diagonalize_H(H)

  output_distribution = (W_dagger @ outputstate.real)**2
  # Normalize the distribution.  Although the sum should be 1, roundoff error
  # can make it slightly greater.
  output_distribution = np.divide(output_distribution, np.sum(output_distribution))

  # For each shot that we have, generate a random number using the weighted 
  # probabilities in output_distribution.  Then calculate our expected value
  # using the diagonal part of H: sum(D[i]*N_i)/N, where N_i is the number of
  # times D_i was chosen, and N is the number of shots.
  sample_values = np.random.multinomial(shot_count, output_distribution)
  cost = np.sum(sample_values*D) / shot_count
  return cost
    
# Description: Calculate the value of the cost function with no shot noise.
# Parameters: params - An array with the current values of the parameters.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
#             H - The classically-computed value of the Hermitian that we want
#             to obtain.
# Return Value: The value of the cost function, calculated using a noiseless 
#               Qiskit statevector.
def cost_function_no_shot_noise(params, num_qubits, num_layers, H):
  quantum_circuit, quantum_register = create_ansatz(params, num_qubits, num_layers)
  backend = Aer.get_backend('statevector_simulator')
  job = execute(quantum_circuit, backend)
  outputstate = job.result().get_statevector(quantum_circuit, decimals=11)
  
  return (outputstate @ H @ np.conj(outputstate)).real
  
# Description: This function implements a gradient for functions that satisfy
#              the parameter-shift rule.  Specifically, note the presence of a 
#              "plus" function and a "minus" function.  
# Parameters: params - An array with the current values of the parameters.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
#             H - The classically-computed value of the Hermitian that we want
#             to obtain.
#             shot_count - The number of shots to use.
# Return Value: The value of the gradient.
def parameter_shift_gradient_finite_sampling(params, num_qubits, num_layers, H, shot_count):
    shift_p = np.array(params)
    shift_n = np.array(params)
    grad = []
    for i in range(len(params)):
        shift_p[i] = params[i] + np.pi / 2
        shift_n[i] = params[i] - np.pi / 2

        cost_p = cost_function_finite_sampling(shift_p, num_qubits, num_layers, H, shot_count)
        cost_n = cost_function_finite_sampling(shift_n, num_qubits, num_layers, H, shot_count)

        grad.append((cost_p - cost_n) / 2)

        shift_p[i] = params[i]
        shift_n[i] = params[i]
    return np.array(grad)
 
 # Description: This function implements a gradient for functions that satisfy
#              the parameter-shift rule.  Specifically, note the presence of a 
#              "plus" function and a "minus" function.  Note that the sole
#              difference between this functiona and the previous one is that
#              this function calls the cost evaluation without shot noise.
# Parameters: params - An array with the current values of the parameters.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
#             H - The classically-computed value of the Hermitian that we want
#             to obtain.
# Return Value: The value of the gradient.
def parameter_shift_gradient_no_shot_noise(params, num_qubits, num_layers, H):
    shift_p = np.array(params)
    shift_n = np.array(params)
    grad = []
    for i in range(len(params)):
        shift_p[i] = params[i] + np.pi / 2
        shift_n[i] = params[i] - np.pi / 2

        cost_p = cost_function_no_shot_noise(shift_p, num_qubits, num_layers, H)
        cost_n = cost_function_no_shot_noise(shift_n, num_qubits, num_layers, H)

        grad.append((cost_p - cost_n) / 2)

        shift_p[i] = params[i]
        shift_n[i] = params[i]
    return np.array(grad)
    
# Description: This function plots two lines using user-defined constraints.
# Parameters: The parameters are very straightforward without further explanation.
# Return Value: None.
def plot_two_lines(x_values, y_values_1, y_values_2, x_axis_label, 
                                  y_axis_label, y_label_1, y_label_2, color_1, 
                                  color_2, title, font_size = 12, 
                                  filename = "JMHTempFile.pdf", log_scale = False,
                                  print_final_values = False,
                                  y_axis_ticks = [], x_axis_ticks = []):
  curPlot = plt.figure(figsize=(10, 5))
  curPlot = plt.gca()
  if (log_scale):
    curPlot.semilogy(x_values, y_values_1, '-o', label = y_label_1, color = color_1, markersize=1)
    curPlot.semilogy(x_values, y_values_2, '-o', label = y_label_2, color = color_2, markersize=1)
  else:
    curPlot.plot(x_values, y_values_1, '-o', label = y_label_1, color = color_1, markersize=1)
    curPlot.plot(x_values, y_values_2, '-o', label = y_label_2, color = color_2, markersize=1)

  plt.rcParams.update({'font.size': font_size})

  if (title != None):
    plt.title(title)
  plt.xlabel(x_axis_label)
  plt.ylabel(y_axis_label)
  plt.grid(True)
  plt.legend(loc='best')

  if (print_final_values):
    plt.text(60, 4.1, 'Final Value: ' + y_values_1[len(y_values_1)-1], fontsize=font_size,  color=color_1)
    plt.text(60, 4.1, 'Final Value: ' + y_values_2[len(y_values_2)-1], fontsize=font_size,  color=color_2)

  if y_axis_ticks != []:
    plt.yticks(np.arange(y_axis_ticks[0], y_axis_ticks[1]+y_axis_ticks[2], y_axis_ticks[2]))
    plt.ylim(y_axis_ticks[0], y_axis_ticks[1])
    plt.yticks()
  if x_axis_ticks != []:
    plt.xticks(np.arange(x_axis_ticks[0], x_axis_ticks[1]+x_axis_ticks[2], x_axis_ticks[2]))
    plt.xlim(x_axis_ticks[0], x_axis_ticks[1])
    plt.xticks()
  
  plt.savefig(filename)
  plt.show()
  
# Description: This function plots a user-specified number of lines using user-defined constraints.
# Code attribution: Thanks to https://stackoverflow.com/questions/3016283/create-a-color-generator-from-given-colormap-in-matplotlib
# for the color creation.
# Parameters: The parameters are very straightforward without further explanation.
# Return Value: None.
def plot_variable_num_lines(x_value_creation_array, y_values_array, x_axis_label, 
                                  y_axis_label, color_map_string, title, font_size = 20,
                                  show_legend = False, highlight_line_dict = {},
                                  filename = "JMHTempFile.pdf", log_scale = False,
                                  y_axis_ticks = [], x_axis_ticks = []):
  
  num_lines = len(y_values_array)
  num_colors = num_lines
  colors_array = []
  if (color_map_string[0] == "#"):
    for i in range(num_colors):
      colors_array.append(color_map_string)
  else:
    color_map = pylab.get_cmap(color_map_string)
    for i in range(num_colors):
      colors_array.append(color_map(1.*i/num_colors))
  
  # If the user desires to emphasize certain lines, adjust those colors in the 
  # color array.
  if (highlight_line_dict != None):
    for key, value in zip(highlight_line_dict.keys(), highlight_line_dict.values()):
      colors_array[key] = value

  y_labels_array = []
  string_base = "Circuit "
  for i in range(num_lines):
    y_labels_array.append(string_base + str(i+1))
  
  curPlot = plt.figure(figsize=(10, 5))
  curPlot = plt.gca()
  for i in range(num_lines):
    # If we are currently on a line to be highlighted, skip that line and plot it after the others
    # so that it is more prominent.
    x_values = np.arange(0, x_value_creation_array[i]+1)
    if (log_scale):
      curPlot.semilogy(x_values, y_values_array[i], '-o', label = y_labels_array[i], color = colors_array[i], markersize=1)
    else:
      curPlot.plot(x_values, y_values_array[i], '-o', label = y_labels_array[i], color = colors_array[i], markersize=1)
    
  # Plot all of the lines that need to be highlighted.
  for key in highlight_line_dict.keys():
    x_values = np.arange(0, x_value_creation_array[key]+1)
    if (log_scale):
      curPlot.semilogy(x_values, y_values_array[key], '-o', label = y_labels_array[key], color = colors_array[key], markersize=1)
    else:
      curPlot.plot(x_values, y_values_array[key], '-o', label = y_labels_array[key], color = colors_array[key], markersize=1)

  plt.rcParams.update({'font.size': font_size})

  if (title != None):
    plt.title(title)
  plt.xlabel(x_axis_label)
  plt.ylabel(y_axis_label)
  plt.grid(True)

  if (show_legend):
    plt.legend(loc='best')

  if y_axis_ticks != []:
   plt.yticks(np.arange(y_axis_ticks[0], y_axis_ticks[1]+y_axis_ticks[2], y_axis_ticks[2]))
   plt.ylim(y_axis_ticks[0], y_axis_ticks[1])
   plt.yticks()
  if x_axis_ticks != []:
   plt.xticks(np.arange(x_axis_ticks[0], x_axis_ticks[1]+x_axis_ticks[2], x_axis_ticks[2]))
   plt.xlim(x_axis_ticks[0], x_axis_ticks[1])
   plt.xticks()
  
  plt.savefig(filename)
  plt.show()
  
# Description: This function plots a single line using user-defined constraints.
# Parameters: The parameters are very straightforward without further explanation.
# Return Value: None.
def plot_single_line(x_values, y_values, x_axis_label, 
              y_axis_label, color, title, font_size = 12, 
              filename = "JMHTempFile.pdf", log_scale = False,
              y_axis_ticks = [], x_axis_ticks = []):
  
  curPlot = plt.figure(figsize=(10, 5))
  curPlot = plt.gca()
  if (log_scale):
    curPlot.semilogy(x_values, y_values, '-o', color = color, markersize=1)
  else:
    curPlot.plot(x_values, y_values, '-o', color = color, markersize=1)

  plt.rcParams.update({'font.size': font_size})

  if (title != None):
    plt.title(title)
  plt.xlabel(x_axis_label)
  plt.ylabel(y_axis_label)
  plt.grid(True)
  
  if y_axis_ticks != []:
    plt.yticks(np.arange(y_axis_ticks[0], y_axis_ticks[1]+y_axis_ticks[2], y_axis_ticks[2]))
    plt.ylim(y_axis_ticks[0], y_axis_ticks[1])
    plt.yticks()
  if x_axis_ticks != []:
    plt.xticks(np.arange(x_axis_ticks[0], x_axis_ticks[1]+x_axis_ticks[2], x_axis_ticks[2]))
    plt.xlim(x_axis_ticks[0], x_axis_ticks[1])
    plt.xticks()
  
  plt.savefig(filename)
  plt.show()
  
# Description: This class is a data structure for representing the data stored
#              during one run of this VQE.
#              The data stored in the file is, as below; note that **
#              indicates a user value.  All spacing is exactly as in the file:
#              Name: *UserNameOrInitials*
#              Date: *Month*-*Day*-*Year*
#              Ansatz: *Value* [Note: Value is currently one of IBM or VLS.]
#              Optimizer: *Value*
#              Maximum iterations: *Value*
#              Qubits: *Value*
#              Layers: *Value*
#              Kappa: *Value*
#              Shots: *Value*
#              Noise Setting: *Value* [Value is 0-6; see above.]
#              Number of optimizations: *Value*
#              Problem 0:
#              Initial Parameters: *Initial parameters, separated by spaces and with space after the last one*
#              Number of Iterations: *Value*
#              ParametersPerIteration: 
#              *First list of parameters, separated by spaces and with space after the last one*
#              *Second list of parameters, separated by spaces and with space after the last one*
#               ...
#              *Last list of parameters, separated by spaces and with space after the last one*
#              EndParametersPerIteration
#              Final Parameters: *Final parameters, separated by spaces and with space after the last one*
#            . . . 
#           Problem n:
#            . . . 
#           In addition to the information that can be read from the file,
#           this object stores information that can be calculated and subsequently
#           used including an array of costs for each iteration, an array of 
#           fidelities for each iteration, and an array of trace distances for
#           each iteration.
class JMH_VQE_train_data:
  def __init__(self):
    self.username = ''
    self.date = ''
    self.ansatz_type = ''
    self.optimizer_type = ''
    self.maximum_iterations = -1
    self.num_qubits = -1
    self.num_layers = -1
    self.shot_count = -1
    self.kappa = -1
    self.num_problems = -1
    self.all_initial_parameters_per_problem = []
    self.all_iterations_per_problem = []
    self.all_iterative_parameters_per_problem = []
    self.all_final_parameters_per_problem = []
    self.all_costs_per_iteration_per_problem = []
    self.all_fidelities_per_iteration_per_problem = []
    self.all_trace_distances_per_iteration_per_problem = []
    self.all_trace_distance_bounds_per_iteration_per_problem = []
    self.min_cost = -1
    self.min_cost_problem_index = -1
    self.min_cost_iteration_index = -1
    self.max_fidelity = -1
    self.max_fidelity_problem_index = -1
    self.max_fidelity_iteration_index = -1
    
# Description: This class is a data structure for representing the data stored
#              during one run of this VQE.
#              The data stored in the file is, as below; note that **
#              indicates a user value.  All spacing is exactly as in the file:
#              Name: *UserNameOrInitials*
#              Date: *Month*-*Day*-*Year*
#              Ansatz: *Value* [Note: Value is currently one of IBM or VLS.]
#              Machine: *Value*
#              Qubits to use: *List of values, separated by spaces*
#              Qubits: *Value*
#              Layers: *Value*
#              Shots: *Value*
#              Noise Setting: *Value* [Value is 0-6; see above.] 
#              Number of Runs: *Value*
#              Problem Index: *ValueOfIndexFromTrainSet*
#              {*DictionaryOfCountsFromQuantumHardware*}
#               . . . 
#              {*DictionaryOfCountsFromQuantumHardware*}
class JMH_VQE_run_data:
  def __init__(self):
    self.username = ''
    self.date = ''
    self.ansatz_type = ''
    self.machine = ''
    self.qubits_to_use = []
    self.num_qubits = -1
    self.num_layers = -1
    self.shot_count = -1
    self.num_problems = -1
    self.problem_index = -1
    self.all_count_results = []
 
 # Description: This function reads in the data that is the result of training
#              a circuit.
#              For details about the format of the file, please see the documentation
#              for the class JMH_VQE_train_data, above.
# Parameters: filename - A string with the name of the file from which to read.
# Return Value: A JMH_VQE_train_data object filled with the data from the file and 
#               with subsquent calculations.
def read_vqe_training_data_from_file(filename):
  data_collection = JMH_VQE_train_data()

  file = open(filename, 'r')
  cur_line = file.readline()

  in_problem = False
  cur_problem_index = 0
  while cur_line != '':
    # If we're not reading in data for a specific problem, read in the header
    # data or begin another problem.
    if (in_problem == False):
      space_index = cur_line.find(" ")
      if (cur_line[0:7] != "Problem"):
        if (cur_line[0:4] == "Name"):
          data_collection.username = cur_line[space_index+1 : ]
        elif (cur_line[0:4] == "Date"):
          data_collection.date = cur_line[space_index+1 : ]
        elif (cur_line[0:6] == "Ansatz"):
          data_collection.ansatz_type = cur_line[space_index+1 : ]
        elif (cur_line[0:9] == "Optimizer"):
          data_collection.optimizer_type = cur_line[space_index+1 : ]
        elif (cur_line[0:7] == "Maximum"):
          cur_line = cur_line[space_index+1 : ]
          space_index = cur_line.find(" ")
          data_collection.maximum_iterations = int(cur_line[space_index+1 : ])
        elif (cur_line[0:6] == "Qubits"):
          data_collection.num_qubits = int(cur_line[space_index+1 : ])
        elif (cur_line[0:6] == "Layers"):
          data_collection.num_layers = int(cur_line[space_index+1 : ])
        elif (cur_line[0:5] == "Shots"):
          data_collection.shot_count = int(cur_line[space_index+1 : ])
        elif (cur_line[0:5] == "Kappa"):
          data_collection.kappa = float(cur_line[space_index+1 : ])
        elif (cur_line[0:6] == "Number"):
          cur_line = cur_line[space_index+1 : ]
          space_index = cur_line.find(" ")
          cur_line = cur_line[space_index+1 : ]
          space_index = cur_line.find(" ")
          data_collection.num_problems = int(cur_line[space_index+1 : ])
      elif (cur_line[0:7] == "Problem"):
        in_problem = True
        cur_problem_index += 1
    elif (in_problem == True):
      if (cur_line[0:7] == "Initial"):
        # Get past the word "parameters".
        cur_line = cur_line[space_index+1 : ]
        space_index = cur_line.find(" ")
        cur_line = cur_line[space_index+1 : ]
        space_index = cur_line.find(" ")
        current_params = []
        while (space_index != -1):
          current_params.append(float(cur_line[0 : space_index]))
          cur_line = cur_line[space_index+1 : ]
          space_index = cur_line.find(" ")

        data_collection.all_initial_parameters_per_problem.append(current_params)

      elif (cur_line[0:6] == "Number"):
        # Get past the word "iterations".
        cur_line = cur_line[space_index+1 : ]
        space_index = cur_line.find(" ")
        cur_line = cur_line[space_index+1 : ]
        space_index = cur_line.find(" ")
        cur_line = cur_line[space_index+1 : ]
        space_index = cur_line.find(" ")
        data_collection.all_iterations_per_problem.append(int(cur_line[space_index+1 : ]))

      elif (cur_line[0:10] == "Parameters"):
        # Move to the first line of iterative parameters.
        cur_line = file.readline()

        current_iterative_params = []
        while (cur_line[0:13] != "EndParameters"):
          current_params = []
          space_index = cur_line.find(" ")
          while (space_index != -1):
            current_params.append(float(cur_line[0 : space_index]))
            cur_line = cur_line[space_index+1 : ]
            space_index = cur_line.find(" ")
          current_iterative_params.append(current_params)
          cur_line = file.readline()
          space_index = cur_line.find(" ")

        # Once we've found the EndParameters marker, add the array of arrays to our array,
        # and move on.
        data_collection.all_iterative_parameters_per_problem.append(current_iterative_params)
      
      elif (cur_line[0:5] == "Final"):
        # Get past the word "parameters".
        space_index = cur_line.find(" ")
        cur_line = cur_line[space_index+1 : ]
        space_index = cur_line.find(" ")
        cur_line = cur_line[space_index+1 : ]
        space_index = cur_line.find(" ")
        
        current_params = []
        while (space_index != -1):
          current_params.append(float(cur_line[0 : space_index]))
          cur_line = cur_line[space_index+1 : ]
          space_index = cur_line.find(" ")
          in_problem = False
        data_collection.all_final_parameters_per_problem.append(current_params)

    cur_line = file.readline()
  return data_collection
  
# Description: This function reads in the data from a run of a pre-trained circuit.
#              For details about the format of the file, please see the documentation
#              for the class JMH_VQE_run_data, above.
# Parameters: filename - A string with the name of the file from which to read.
# Return Value: A JMH_VQE_run_data object filled with the data from the file and 
#               with subsquent calculations.
def read_vqe_run_data_from_file(filename):
  data_collection = JMH_VQE_run_data()

  file = open(filename, 'r')
  cur_line = file.readline()

  cur_problem_index = 0
  while cur_line != '':
    space_index = cur_line.find(" ")
    if (cur_line[0:4] == "Name"):
      data_collection.username = cur_line[space_index+1 : ]
    elif (cur_line[0:4] == "Date"):
      data_collection.date = cur_line[space_index+1 : ]
    elif (cur_line[0:6] == "Ansatz"):
      data_collection.ansatz_type = cur_line[space_index+1 : ]
    elif (cur_line[0:7] == "Machine"):
      data_collection.machine = cur_line[space_index+1 : ]
    elif (cur_line[0:9] == "Qubits to"):
      # Get past the words "to" and "use."
      cur_line = cur_line[space_index+1 : ]
      space_index = cur_line.find(" ")
      cur_line = cur_line[space_index+1 : ]
      space_index = cur_line.find(" ")
      cur_line = cur_line[space_index+1 : ]
      space_index = cur_line.find(" ")
      # When there is only one character left, it is a space.
      while len(cur_line) > 1:
        space_index = cur_line.find(" ")
        data_collection.qubits_to_use.append(int(cur_line[0 : space_index+1]))
        cur_line = cur_line[space_index+1 : ]
    elif (cur_line[0:6] == "Qubits"):
      data_collection.num_qubits = int(cur_line[space_index+1 : ])
    elif (cur_line[0:6] == "Layers"):
      data_collection.num_layers = int(cur_line[space_index+1 : ])
    elif (cur_line[0:5] == "Shots"):
      data_collection.shot_count = int(cur_line[space_index+1 : ])
    elif (cur_line[0:6] == "Number"):
      cur_line = cur_line[space_index+1 : ]
      space_index = cur_line.find(" ")
      cur_line = cur_line[space_index+1 : ]
      space_index = cur_line.find(" ")
      data_collection.num_problems = int(cur_line[space_index+1 : ])
    elif (cur_line[0:7] == "Problem"):
      cur_line = cur_line[space_index+1 : ]
      space_index = cur_line.find(" ")
      cur_line = cur_line[space_index+1 : ]
      space_index = cur_line.find(" ")
      data_collection.problem_index = int(cur_line[space_index+1 : ])
    elif (cur_line[0:1] == "{"):
      cur_dict = {}
      cur_line = cur_line[1 : ]
      space_index = cur_line.find(" ")
      while (space_index != -1):
        cur_key = cur_line[0 : space_index-1]
        cur_line = cur_line[space_index+1: ]
        space_index = cur_line.find(" ")
        if (space_index == -1):
          space_index = cur_line.find("}")
          cur_value = cur_line[0 : space_index]
        else:
          cur_value = cur_line[0 : space_index-1]
        cur_line = cur_line[space_index+1 : ]
        space_index = cur_line.find(" ")
        cur_dict[cur_key] = int(cur_value)
      data_collection.all_count_results.append(cur_dict)

    cur_line = file.readline()
  return data_collection  
  
# Description: This function populates the costs/fidelities/trace distances per
#              iteration per problem for a given set of data.  All of this data
#              could be stored in the file but is not because it is easily 
#              recomputed from the stored parameters.  Storing more data than 
#              necessary in the file might make it too large to store and
#              work with.
# Parameters: data_object - The JMH_VQE_data object from which the parameters
#             will be accessed and to which the costs will be stored.
#             H - The matrix with the Hermitian for the problem.  This is the 
#             only information that is not stored in the data_object.  (H is 
#             not printed to the file because it is large and easily calculated
#             from the separately-stored A.txt and b.txt for the problem.)
#             kappa - A float with the condition number (defined as in Marco's)
#             VLS paper that can be used to compute the maximum on the trace
#             distance.
#             filename_x - A string with the filename for the true x vector that can be used to compute fidelity.
# Return Value: None; because the data object is updated via the parameter, 
#               there is no need to return anything.
def populate_iteration_per_problem_data_from_saved_params(data_object, H, kappa, filename_x):
  num_problems = data_object.num_problems
  num_qubits = data_object.num_qubits
  num_layers = data_object.num_layers
  true_x = create_x(num_qubits, filename_x, True)

  # For each problem...
  min_cost = -1
  min_cost_problem_index = -1
  min_cost_iteration_index = -1
  max_fidelity = -1
  max_fidelity_problem_index = -1
  max_fidelity_iteration_index = -1
  for problem in range(num_problems):
    current_param_collection = data_object.all_iterative_parameters_per_problem[problem]
    iteration_index = 0
    problem_costs = []
    problem_fidelities = []
    problem_trace_distances = []
    problem_trace_distance_bounds = []

    # For each set of parameters at each iteration of the current problem...
    for parameter_set in current_param_collection:
      cost = cost_function_no_shot_noise(parameter_set, num_qubits, num_layers, H)
      problem_costs.append(cost)

      # Compute the fidelities.
      quantum_circuit, quantum_register = create_ansatz(parameter_set, num_qubits, num_layers)
      backend = Aer.get_backend('statevector_simulator')
      job = execute(quantum_circuit, backend)
      outputstate = job.result().get_statevector(quantum_circuit, decimals=11)
      fidelity = calculate_fidelity(num_qubits, true_x, outputstate.data.real)
      problem_fidelities.append(fidelity)

      # Compute the trace distances (and the bounds on trace distance).
      #problem_trace_distances.append(0.5*sqrt(1-fidelity))
      #problem_trace_distance_bounds.append(sqrt(kappa**2*cost))

      # Consider minimum cost and maximum fidelity over both all problems
      # and all iterations within each problem.
      if (min_cost == -1 or cost < min_cost):
        min_cost = cost
        min_cost_problem_index = problem
        min_cost_iteration_index = iteration_index
      
      if (max_fidelity == -1 or fidelity > max_fidelity):
        max_fidelity = fidelity
        max_fidelity_problem_index = problem
        max_fidelity_iteration_index = iteration_index

      iteration_index += 1

    # After all iterations for this problem are complete, add the arrays of
    # costs, fidelities, trace distances, and trace distance bounds.
    data_object.all_costs_per_iteration_per_problem.append(problem_costs)
    data_object.all_fidelities_per_iteration_per_problem.append(problem_fidelities)
    
    # Because trace distance became less important in our investigations, comment out the code that computes trace distance for now.
    #data_object.all_trace_distances_per_iteration_per_problem.append(problem_trace_distances)
    #data_object.all_trace_distance_bounds_per_iteration_per_problem.append(problem_trace_distance_bounds)

  # After all problems and iterations are complete, update the maximum and
  # minimum values.
  data_object.min_cost = min_cost
  data_object.min_cost_problem_index = min_cost_problem_index
  data_object.min_cost_iteration_index = min_cost_iteration_index
  data_object.max_fidelity = max_fidelity
  data_object.max_fidelity_problem_index = max_fidelity_problem_index
  data_object.max_fidelity_iteration_index = max_fidelity_iteration_index
  
# Description: Generate a grid-value dictionary, in which we map each location in a 4x8 grid to 
#              a value to be plotted in that location.
# Parameters: vector_values - A vector with the values to be placed in the grid.
#             use_boundary_conditions - A flag that allows for adding boundary conditions to either side
#             of the grid.
#             boundary_conditions - An array with two values for the left and right hand sides of the plot, respectively.
#             unscale_normalization_constant - A constant that can be used to 'undo' normalization, if set.
# Return Value: A dictionary with keys of grid locations and values of the quantity to be put in that 
#               grid location.
def generate_grid_value_dictionary(vector_values, use_boundary_conditions = False, boundary_conditions = [], unscale_normalization_constant = -1):
  grid_value_dictionary = {}

  # If we are unscaling the normalization, multiply each element by the normalization constant.
  if (unscale_normalization_constant != -1):
    vector_values = np.multiply(vector_values, unscale_normalization_constant)

  # Convert vector values to a list, so that we don't have to deal with np arrays.
  try:
    # If the list is formatted such that we need to pull all of the values into a single row...
    if (isinstance(vector_values.T[0], float) == False):
      vector_values = vector_values.T[0].tolist()
    # If the list already has all values in a single row, simply convert it from a numpy array.
    else:
      vector_values = vector_values.tolist()
  # If we already have a list with all values in a single row, don't do anything.
  except:
    vector_values = vector_values

  if (use_boundary_conditions):
    for i in range(0, 8):
      vector_values.insert(i, boundary_conditions[0])
    for i in range(0, 8):
      vector_values.append(boundary_conditions[1])
  
  grid_row = 0
  grid_col = 7
  for element in vector_values:
    grid_value_dictionary[(grid_row, grid_col)] = element
    grid_col -= 1
    if (grid_col < 0):
      grid_row += 1
      grid_col = 7

  return grid_value_dictionary

# Description: Generate a plot with only the permeabilities (and no pressure solution).
# Parameters: permeability_color_array - Three colors for plotting low, medium, and high permeability,
#             respectively.
#             top_prong_varied - A boolean flag indicating whether or not the top-prong permeability 
#             should be different than that in the rest of the picthfork.
#             pitchfork_only - A flag indicating whether to print permeabilities surrounding the pitchfork.
# Return Value: The plot is created, and so nothing needs to be returned.
def plot_permeabilities_only(permeability_color_array = ['#800000', '#66ff66', '#ff33cc'], top_prong_varied = False, pitchfork_only = False):
  fig, ax = plt.subplots(figsize=(10, 5))

  # For consistency with the way that the colormap library plotter made the graph,
  # I want to have the y-axis go from 0 at top to 7 at bottom, and not the other way
  # around.
  plt.gca().invert_yaxis()
  
  row_elements = []
  col_elements = []
  for row in (0,6):
    for col in (0,8):
      row_elements.append(row)
      col_elements.append(col)

  largest_row = max(row_elements)
  largest_col = max(col_elements)

  # lr stands for left-to-right.
  # tb stands for top-to-bottom.
  lr_position = []
  tb_position = []
    
  # Starting from the left and moving to the right,
  # add the permeabilities that fall on the left-to-right edges of 
  # each colored face.
  cur_lr = -0.5
  while cur_lr <= 4.5:
    for cur_tb in range(0, 8):
      lr_position.append(cur_lr)
      tb_position.append(cur_tb)
    cur_lr += 1
    
  # Starting from the left and moving to the right, 
  # add the permeabilities that fall on the top-to-bottom edges of
  # each colored face.
  cur_lr = 0
  while cur_lr <= 5:
    cur_tb = 0.5
    while cur_tb <= 6.5:
      lr_position.append(cur_lr)
      tb_position.append(cur_tb)
      cur_tb += 1
    cur_lr += 1
      
  color_grid = []
  low_color = permeability_color_array[0]
  mid_color = permeability_color_array[1]
  high_color = permeability_color_array[2]
  for cur_lr, cur_tb in zip(lr_position, tb_position):
    # Either of the boundaries, which has no pitchfork.
    if (cur_lr == -0.5 or cur_lr >= 5):
      if (pitchfork_only):
        color_grid.append('#ffffff')
      else:
        color_grid.append(low_color)
    # Large middle prong of fracture...
    elif (cur_tb == 3):
      color_grid.append(mid_color)
    # 'Intersecting' prong of fracture...
    elif (cur_lr == 2):
      color_grid.append(mid_color)
    # Top prong...
    elif (cur_tb == 0 and cur_lr > 2):
      if (top_prong_varied):
        color_grid.append(high_color)
      else: 
        color_grid.append(mid_color)
    # Bottom prong...
    elif (cur_tb == 7 and cur_lr > 2):
      color_grid.append(mid_color)
    # Otherwise, we are not in the fracture...
    else:
      if (pitchfork_only):
        color_grid.append('#ffffff')
      else:
        color_grid.append(low_color)
  ax.scatter(lr_position, tb_position, c=color_grid)
    
  ax.xaxis.tick_bottom()

  fig.savefig("Permeabilities_Plot.pdf")

# Description: Create a colormap plot from the results in a specified solution.
#              This can be compared to results from a true solution for a visual 
#              comparison of how well the simulator/hardware did.
# Parameters: grid_value_dictionary - A dictionary with the values of the grid as keys (e.g., (0,0))
#             and the values to be placed in each grid location (e.g., probabilitiy from QC) as values.
#             color_scheme - A string with the name of one of Matplotlib's color schemes.
#             See https://matplotlib.org/stable/tutorials/colors/colormaps.html for options.
#             title - A string with the title of the plot. 
# Return Value: The plot is created, and so nothing needs to be returned.
def create_colormap_from_solution(grid_value_dictionary, color_scheme, title):
  fig, ax = plt.subplots(figsize=(10, 5))
  
  x_coordinates = []
  y_coordinates = []
  values = []

  # Fill the grid with colors.
  for grid, value in zip(grid_value_dictionary.keys(), grid_value_dictionary.values()):
    x_coordinates.append(grid[0])
    y_coordinates.append(grid[1])

    # Add the absolute value of the probability to convert the "negative" 
    # probabilities quantum can provide to postive ones.  Also, neglect the 
    # complex part as, for the problems we are interested in, that is always
    # zero.  (We cannot leave the zero value because matshow can't deal with it.)
    values.append(abs(value.real))
  grid_setup = np.zeros((max(x_coordinates)+1, max(y_coordinates)+1))
  for index in range(len(values)):
    grid_setup[x_coordinates[index]][y_coordinates[index]] = values[index]
  ax.matshow(grid_setup, cmap=plt.get_cmap(color_scheme), vmin=min(values), vmax=max(values))

  # Control plot aesthetics, including x-axis label position and title.
  ax.xaxis.tick_bottom()
  if (title != None):
    ax.set_title(title)
  
  # Add a color bar.
  norm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values))
  fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(color_scheme)), ax=ax)
  
  fig.savefig("Solution_Colormap.pdf")
  
# Description: Create a colormap plot from the results in a specified solution in such a way
#              that it could be used in a paper. Include the ability to overlay permeabilities
#              and to 'unscale' a normalization.
# Parameters: grid_value_dictionary - A dictionary with the values of the grid as keys (e.g., (0,0))
#             and the values to be placed in each grid location (e.g., probabilitiy from QC) as values.
#             color_scheme - A string with the name of one of Matplotlib's color schemes.
#             See https://matplotlib.org/stable/tutorials/colors/colormaps.html for options.
#             
#             use_permeability_overlay - A flag that allows for overlaying a permeability scatterplot.
#             permeability_color_array - Three colors for plotting low, medium, and high permeability,
#             respectively.
# Return Value: The plot is created, and so nothing needs to be returned.
def create_colormap_from_solution_for_aesthetic_display(grid_value_dictionary, color_scheme,
                                                        use_permeability_overlay = True, permeability_color_array = ['#800000', '#66ff66', '#0000ff']):
  fig, ax = plt.subplots(figsize=(10, 5))
  
  x_coordinates = []
  y_coordinates = []
  values = []

  # Fill the grid with colors.
  for grid, value in zip(grid_value_dictionary.keys(), grid_value_dictionary.values()):
    x_coordinates.append(grid[1])
    y_coordinates.append(grid[0])

    # Add the absolute value of the probability to convert the "negative" 
    # probabilities quantum can provide to postive ones.  Also, neglect the 
    # complex part as, for the problems we are interested in, that is always
    # zero.  (We cannot leave the zero value because matshow can't deal with it.)
    values.append(abs(value.real))

  grid_setup = np.zeros((max(x_coordinates)+1, max(y_coordinates)+1))
  for index in range(len(values)):
    grid_setup[x_coordinates[index]][y_coordinates[index]] = values[index]
  ax.matshow(grid_setup, cmap=plt.get_cmap(color_scheme), vmin=min(values), vmax=max(values))

  # If the user wants a permeability overlay, put the scatterplot of permeabilities
  # onto the colormap.  Note that this is hardcoded for now because I have not 
  # developed a generalized formula for positioning the permeabilities.
  if (use_permeability_overlay):
    # Establish the permeabilities that exist between the nodes, so that they can be plotted on top of the pressure solution.
    # First, find the largest value for both the number of rows and columns, so that can be used to determine how many
    # vertices there are.
    key_list = grid_value_dictionary.keys()
    row_elements = [a for (a,b) in key_list]
    col_elements = [b for (a,b) in key_list]
    largest_row = max(row_elements)
    largest_col = max(col_elements)

    # lr stands for left-to-right.
    # tb stands for top-to-bottom.
    lr_position = []
    tb_position = []
    
    # Starting from the left and moving to the right,
    # add the permeabilities that fall on the left-to-right edges of 
    # each colored face.
    cur_lr = -0.5
    while cur_lr <= 4.5:
      for cur_tb in range(0, 8):
        lr_position.append(cur_lr)
        tb_position.append(cur_tb)
      cur_lr += 1
    
    # Starting from the left and moving to the right, 
    # add the permeabilities that fall on the top-to-bottom edges of
    # each colored face.
    cur_lr = 0
    while cur_lr <= 5:
      cur_tb = 0.5
      while cur_tb <= 6.5:
        lr_position.append(cur_lr)
        tb_position.append(cur_tb)
        cur_tb += 1
      cur_lr += 1
      
    color_grid = []
    low_color = permeability_color_array[0]
    mid_color = permeability_color_array[1]
    high_color = permeability_color_array[2]
    for cur_lr, cur_tb in zip(lr_position, tb_position):
      # Either of the boundaries, which has no pitchfork.
      if (cur_lr == -0.5 or cur_lr >= 5):
        color_grid.append(low_color)
      # Large middle prong of fracture...
      elif (cur_tb == 3):
        color_grid.append(mid_color)
      # 'Intersecting' prong of fracture...
      elif (cur_lr == 2):
        color_grid.append(mid_color)
      # Top prong...
      elif (cur_tb == 0 and cur_lr > 2):
        color_grid.append(mid_color)
      # Bottom prong...
      elif (cur_tb == 7 and cur_lr > 2):
        color_grid.append(mid_color)
      # Otherwise, we are not in the fracture...
      else:
        color_grid.append(low_color)
    ax.scatter(lr_position, tb_position, c=color_grid)

  ax.xaxis.tick_bottom()
  
  # Add a color bar.
  norm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values))
  fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(color_scheme)), ax=ax)
  
  fig.savefig("Solution_Colormap_Better_Aesthetics.pdf")
  
# Description: Create a colormap plot from the results in a specified solution.
#              This can be compared to results from a true solution for a visual 
#              comparison of how well the simulator/hardware did.
#              Note that this version of the function takes an extra parameter 
#              because it prints the error for each value on each square of the 
#              plot.
# Parameters: grid_value_dictionary - A dictionary with the values of the grid as keys (e.g., (0,0))
#             and the values to be placed in each grid location (e.g., probabilitiy from QC) as values.
#             color_scheme - A string with the name of one of Matplotlib's color schemes.
#             See https://matplotlib.org/stable/tutorials/colors/colormaps.html for options.
#             title - A string with the title of the plot. 
#             true_values_dictionary - A dictionary with the values of the grid as keys
#             and the true value for each square as values.
#             text_color - A string with the color for printed text.
# Return Value: The plot is created, and so nothing needs to be returned.
def create_colormap_from_solution_with_text(grid_value_dictionary, color_scheme, title, true_values_dictionary, text_color):
  fig, ax = plt.subplots(figsize=(10, 5))
  
  x_coordinates = []
  y_coordinates = []
  values = []

  # Fill the grid with colors.
  for grid, value in zip(grid_value_dictionary.keys(), grid_value_dictionary.values()):
    x_coordinates.append(grid[0])
    y_coordinates.append(grid[1])

    # Add the absolute value of the probability to convert the "negative" 
    # probabilities quantum can provide to postive ones.  Also, neglect the 
    # complex part as, for the problems we are interested in, that is always
    # zero.  (We cannot leave the zero value because matshow can't deal with it.)
    values.append(abs(value.real))
  grid_setup = np.zeros((max(x_coordinates)+1, max(y_coordinates)+1))
  for index in range(len(values)):
    grid_setup[x_coordinates[index]][y_coordinates[index]] = values[index]
  ax.matshow(grid_setup, cmap=plt.get_cmap(color_scheme), vmin=min(values), vmax=max(values))

  for (x_val, y_val) in zip(x_coordinates, y_coordinates):
    cur_true_val = true_values_dictionary[(x_val,y_val)]
    to_print = abs(grid_setup[x_val][y_val] - cur_true_val)
    
    # If each value to print is an 'array' of one element, we need to print the single element
    # to avoid "[]" printing around the value.
    try:
      ax.text(y_val, x_val, round(to_print[0], 4), va='center', ha='center', color=text_color)
    # If it is already simply a float, we need only print the value.
    except:
      ax.text(y_val, x_val, round(to_print, 4), va='center', ha='center', color=text_color)

  # Control plot aesthetics, including x-axis label position and title.
  ax.xaxis.tick_bottom()
  if (title != None):
    ax.set_title(title)
  
  # Add a color bar.
  norm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values))
  fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(color_scheme)), ax=ax)
  
  fig.savefig("Solution_Colormap_With_Text.pdf")
  
 # Description: This function takes the optimal parameters found for a problem,
#              creates a circuit with those parameters, and measures all the qubits
#              in the circuit.  Use a specified shot count to obtain a probability
#              distribution of counts and to thus obtain our solution.  
# Parameters: optimal_params - A list of optimal parameters to be used in the circuit.
#             machine_choice - A string with the name of a machine whose noise we wish to simulate.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit ansatz.
#             shot_count - The number of shots to use.
#             qubits_to_use - An array with the indices of the physical qubits to use.
# Return Value: A dictionary with the counts (probabilities) for each bitstring result,
#               where each value in the bitstring result corresponds to a value in the
#               desired solution.
def run_circuit_with_simulated_hardware_noise(optimal_params, machine_choice, num_qubits, num_layers, shot_count, qubits_to_use):
  # Set up the machine that we will use to run 
  IBMQ.load_account()
  provider = IBMQ.get_provider(hub='ibm-q-lanl')
  device_backend = provider.get_backend(machine_choice)
  device_properties = device_backend.properties()
  device_coupling_map = device_backend.configuration().coupling_map
  device_noise_model = NoiseModel.from_backend(device_backend)
  basis_gates = device_noise_model.basis_gates
  
  # Create the circuit we will simulate.
  quantum_circuit, qr = create_ansatz(optimal_params, num_qubits, num_layers)
  for i in range(num_qubits):
  	quantum_circuit.measure(i, i)
  	
  # Execute the job on our simulated hardware.
  virtual_to_physical_qubits = Layout()
  for index in range(len(qubits_to_use)):
    virtual_to_physical_qubits.add(qr[index], qubits_to_use[index])
  job = execute(quantum_circuit, Aer.get_backend('qasm_simulator'), shots = shot_count, initial_layout=virtual_to_physical_qubits, coupling_map = device_coupling_map, noise_model = device_noise_model, basis_gates = basis_gates)
  simulation_results = job.result()
  simulation_counts = simulation_results.get_counts(quantum_circuit)
  return simulation_counts
  
# Description: This function runs the best parameters found, creates a circuit
#              with those parameters, and measures all the qubits in the circuit
#              on an IBMQ machine.
# Parameters: optimal_params - A list of optimal parameters to be used in the circuit.
#             machine_choice - A string with the name of a machine whose noise we wish to simulate.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit ansatz.
#             shot_count - The number of shots to use.
#             qubits_to_use - An array with the physical qubits to be used.
# Return Value: A dictionary with the counts (probabilities) for each bitstring result,
#               where each value in the bitstring result corresponds to a value in the
#               desired solution.  Additionally, return the number of shots that 
#               were used; this will not align with the parameter shots if more
#               than the maximum number of shots were requested.
def run_circuit_on_quantum_hardware(optimal_params, machine_choice, num_qubits, num_layers, shot_count, qubits_to_use):
  # Set up the machine that we will use to run 
  IBMQ.load_account()
  provider = IBMQ.get_provider(hub='ibm-q-lanl')
  device_backend = provider.get_backend(machine_choice)
  maximum_shots = device_backend.configuration().max_shots

  if (shot_count > maximum_shots):
    print("The maximum number of allowed shots on ", machine_choice, " is ", maximum_shots, ".", sep="")
    print("Currently, too many shots are requested, so changing shot count to ", maximum_shots, " shots.", sep="")
    shot_count = maximum_shots

  # Create the circuit we will simulate.
  quantum_circuit, qr = create_ansatz(optimal_params, num_qubits, num_layers)
  for i in range(num_qubits):
  	quantum_circuit.measure(i, i)
  	
  # Execute the job on our simulated hardware.
  virtual_to_physical_qubits = []
  for index in range(len(qubits_to_use)):
    virtual_to_physical_qubits.append(qubits_to_use[index])
  transpiled_circuit = transpile(quantum_circuit, device_backend, initial_layout=virtual_to_physical_qubits, optimization_level=3)
  job = device_backend.run(transpiled_circuit, shots=shot_count)
  job_monitor(job, interval=2)
  simulation_results = job.result()
  simulation_counts = simulation_results.get_counts(quantum_circuit)
  return simulation_counts, shot_count
 
# Description: This function takes the optimal parameters found for a problem,
#              and runs a circuit with those parameters.  It the statevector
#              simulator to obtain the final solution.
# Parameters: optimal_params - A list of optimal parameters to be used in the circuit.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit ansatz.
# Return Value: The real part of the statevector-computed solution.
def run_circuit_with_statevector_simulator(optimal_params, num_qubits, num_layers):
  quantum_circuit, quantum_register = create_ansatz(optimal_params, num_qubits, num_layers)
  backend = Aer.get_backend('statevector_simulator')
  job = execute(quantum_circuit, backend)
  outputstate = job.result().get_statevector(quantum_circuit, decimals=11)
  return outputstate.real

###################################################### Functions that perform a series of tasks using the functions above. ###############################

# Description: Create two heatmaps of a true, classically-computed solution.  One contains permeability overlay and the other does not.
# Parameters: num_qubits: An integer with the number of qubits in the specified circuit.
#             filename_x: A string with the filename containing a classically-computed, solution vector.
# Return Value: Because this function creates and saves plots, it does not return anything.   
def create_classically_computed_heat_map(num_qubits, filename_x):
    true_x = create_x(num_qubits, filename_x, True)
    true_x_non_normalized = create_x(num_qubits, filename_x, False).T[0]
    true_x_grid_val_dict = generate_grid_value_dictionary(true_x, True, [1.0, 0.0], sqrt(sum(true_x_non_normalized**2)))
    
    create_colormap_from_solution_for_aesthetic_display(true_x_grid_val_dict, 'plasma', True)
    create_colormap_from_solution_for_aesthetic_display(true_x_grid_val_dict, 'plasma', False)

# Description: Train a circuit without shot noise or hardware noise.
# Parameters: name - A string with the name/initials of the person collecting results.
#             date - A string with the date in month-day-year format.
#             ansatz - A string with the ansatz for the circuit. (For example, VLS or IBM.)
#             optimizer - A string with the name of the optimizer. (For example, CG or SPSA.)
#             maximum_iterations - An integer with the maximum number of iterations that the optimizer can perform.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
#             num_problems - An integer with the number of different circuits to train. (They will likely start with different initial parameters.)
#             kappa - A double with the condition number (as defined in the VLS paper) of the Ax=b system.  Note that this is set to -1, if not known/desired. 
#             (Kappa is important only when considering trace distance.)
#             loading_initial_parameters - An array with initial parameters to load, if applicable.
#             epsilon - A double (between 0 and 1) with a random 'kick' to apply to loaded parameters; ideally, this shouldn't be larger than 1, as that causes 
#             effectively random initial parameters to be used. If unwanted, use -1.
#             filename_A - A string with the file holding matrix A.
#             filename_b - A string with the file holding vector b.
#             optional_filename_tag - A string that can be added to a filename to distiguish it from previous data.  If None,
#             nothing will be added to the end of the filename.
# Return Value: Because the data is written to a file, nothing need be returned.
def train_no_noise(name, date, ansatz, optimizer, maximum_iterations, num_qubits, num_layers, num_problems, kappa, loading_initial_parameters, epsilon, filename_A, filename_b, optional_filename_tag = None):
    num_parameters = get_num_parameters(num_qubits, num_layers)
    H = create_H(num_qubits, filename_A, filename_b, True)
    filename_base = ""
    if (optional_filename_tag != None):
      filename_base = "Training_" + name + "_" + date + "_" + ansatz + "_" + optimizer + "_" + str(maximum_iterations) + "_" + str(num_layers) + "_" + str(-1) + "_" + str(num_problems) + "_" + optional_filename_tag
    else:
      filename_base = "Training_" + name + "_" + date + "_" + ansatz + "_" + optimizer + "_" + str(maximum_iterations) + "_" + str(num_layers) + "_" + str(-1) + "_" + str(num_problems)
    
    with open(filename_base + ".txt", 'w') as file:  
      file.write("Name: " + name + "\n")
      file.write("Date: " + date + "\n")
      file.write("Ansatz: " + ansatz + "\n")
      file.write("Optimizer: " + optimizer + "\n")
      file.write("Maximum iterations: " + str(maximum_iterations) + "\n")
      file.write("Qubits: " + str(num_qubits) + "\n")
      file.write("Layers: " + str(num_layers) + "\n")
      file.write("Shots: " + str(shot_count) + "\n")
      file.write("Kappa: " + str(kappa) + "\n")
      file.write("Noise Setting: 1\n")
      file.write("Number of problems: " + str(num_problems) + "\n")
      if (epsilon != -1):
        file.write("Epsilon (if training from previous params): " + str(epsilon) + "\n")

      initial_parameters = []
      if (len(loading_initial_parameters) != 0 and epsilon != -1):
        for problem in range(num_problems):
          # Obtain a vector of values between -1 and 1 that we will use to slightly change
          # the current parameters themselves, so that we reduce our chance of falling into
          # a local minimum.
          change_parameters = np.random.uniform(low=-1, high=1, size=(num_parameters,))
          # Normalize that vector.
          change_parameters = change_parameters/sqrt(sum(change_parameters**2))
          # Now, make the vector have norm less than or equal to some small epsilon, which we set above.
          change_parameters = change_parameters/(1/epsilon)
          
          # Add zero for any parameters that were not initialized in the previous run.
          loaded_parameters_modified_for_ansatz = current_standard_data_set.all_final_parameters_per_problem[current_standard_best_problem_index]
          loaded_params_length = len(current_standard_data_set.all_final_parameters_per_problem[current_standard_best_problem_index])
          for i in range(num_parameters-loaded_params_length):
            loaded_parameters_modified_for_ansatz.append(0)

          # Put the modified version of the loaded parameters into initial_parameters.
          initial_parameters.append(loaded_parameters_modified_for_ansatz + change_parameters)          
      elif (len(loading_initial_parameters) != 0):
        for problem in range(num_problems):
          loaded_parameters_modified_for_ansatz = loading_initial_parameters
          loaded_params_length = len(loaded_parameters_modified_for_ansatz)
          for i in range(num_parameters-loaded_params_length):
            loaded_parameters_modified_for_ansatz.append(0)
          initial_parameters.append(loaded_parameters_modified_for_ansatz)
      else:
        # Create however many sets of randomly-initialized parameters we want to use.
        for problem in range(num_problems):
          initial_parameters.append(np.random.rand(num_parameters) * 2 * np.pi)
      
      for problem in range(num_problems):
        print("\n\nSolving Problem ", problem, "...", sep="")
        file.write("Problem " + str(problem) + ":\n")
        file.write("Initial Parameters: ")
        for parameter in initial_parameters[problem]:
          file.write(str(parameter) + " ")
        file.write("\n")
        
        result = minimize(cost_function_no_shot_noise, initial_parameters[problem], jac=parameter_shift_gradient_no_shot_noise, method="CG", options={'maxiter':maximum_iterations, 'return_all':True}, args=(num_qubits, num_layers, H))
        
        file.write("Number of Iterations: " + str(result.nit) + "\n")
        
        file.write("ParametersPerIteration: " + "\n")
        for parameter_set in result.allvecs:
          for element in parameter_set:
            file.write(str(element) + " ")
          file.write("\n")
        file.write("EndParametersPerIteration" + "\n")
        
        file.write("Final Parameters: ")
        for element in result.x:
          file.write(str(element) + " ")
        file.write("\n")
      
    print("That's all! ", str(num_problems), " circuits have been trained with no noise.", sep="")
    
# Description: Load results from a circuit that was trained with no noise.
# Parameters: filename - A string with the filename of the data to read.
#             compute_additional_data - A boolean flag indicating whether or not to compute the additional data not stored in the file,
#             but relevant to some further data-processing tasks.  (For example, whether to compute cost/iteration, so that plot can be
#             created.)
#             filename_x - A string with the name of the file holding vector x.
#             filename_A - A string with the name of the file holding matrix A.
#             filename_b - A string with the name of the file holding vector b.
# Return Value: The JMH_VQE_train_data object with the read/computed data, and an integer with the index of the problem with best fidelity.
def train_no_noise_processing(filename, compute_additional_data, filename_x, filename_A, filename_b):
  data_from_file_no_noise = ''
  try:
    data_from_file_no_noise = read_vqe_training_data_from_file(filename)
  except IOError:
    print("I'm sorry, but the file you specified cannot be opened.  Quitting for now.")
    quit()
  
  name = data_from_file_no_noise.username
  date = data_from_file_no_noise.date
  ansatz = data_from_file_no_noise.ansatz_type
  optimizer = data_from_file_no_noise.optimizer_type
  maximum_iterations = data_from_file_no_noise.maximum_iterations
  shot_count = data_from_file_no_noise.shot_count
  num_qubits = data_from_file_no_noise.num_qubits
  num_layers = data_from_file_no_noise.num_layers
  num_problems = data_from_file_no_noise.num_problems
  kappa = data_from_file_no_noise.kappa
  num_parameters = get_num_parameters(num_qubits, num_layers)
  H = create_H(num_qubits, filename_A, filename_b, True)

  if (compute_additional_data):
    populate_iteration_per_problem_data_from_saved_params(data_from_file_no_noise, H, kappa, filename_x)
    # Print statistics for the entire set of circuits.  This can be used to find the
    # "best" circuit (where "best" is classified as that with highest fidelity) 
    # for future use.
    print("The maximum fidelity is ", data_from_file_no_noise.max_fidelity, ", which occurred at problem ", data_from_file_no_noise.max_fidelity_problem_index, " and iteration ", data_from_file_no_noise.max_fidelity_iteration_index, ".", sep="")
    print("The total number of iterations for problem ", data_from_file_no_noise.max_fidelity_problem_index, " was ", data_from_file_no_noise.all_iterations_per_problem[data_from_file_no_noise.max_fidelity_problem_index], ".", sep="")
    print("The minimum cost is ", data_from_file_no_noise.min_cost, ", which occurred at problem ", data_from_file_no_noise.min_cost_problem_index, " and iteration ", data_from_file_no_noise.min_cost_iteration_index, ".", sep="")
    print("The total number of iterations for problem ", data_from_file_no_noise.min_cost_problem_index, " was ", data_from_file_no_noise.all_iterations_per_problem[data_from_file_no_noise.min_cost_problem_index], ".", sep="")
    best_problem_index_no_noise = data_from_file_no_noise.max_fidelity_problem_index
    
  return data_from_file_no_noise, best_problem_index_no_noise
    
# Description: Train a circuit with shot noise, but without hardware noise.
# Parameters: name - A string with the name/initials of the person collecting results.
#             date - A string with the date in month-day-year format.
#             ansatz - A string with the ansatz for the circuit. (For example, VLS or IBM.)
#             optimizer - A string with the name of the optimizer. (For example, CG or SPSA.)
#             maximum_iterations - An integer with the maximum number of iterations that the optimizer can perform.
#             shot_count - An integer with the number of shots to use.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
#             num_problems - An integer with the number of different circuits to train. (They will likely start with different initial parameters.)
#             kappa - A double with the condition number (as defined in the VLS paper) of the Ax=b system.  Note that this is set to -1, if not known/desired. 
#             (Kappa is important only when considering trace distance.)
#             loading_initial_parameters - An array with initial parameters to load, if applicable.
#             epsilon - A double (between 0 and 1) with a random 'kick' to apply to loaded parameters; ideally, this shouldn't be larger than 1, as that causes 
#             effectively random initial parameters to be used. If unwanted, use -1.
#             filename_A - A string with the filename for matrix A.
#             filename_b - A string with the filename for vector b.
#             optional_filename_tag - A string that can be added to a filename to distiguish it from previous data.  If None,
#             nothing will be added to the end of the filename.
# Return Value: Because the data is written to a file, nothing need be returned.
def train_shot_noise(name, date, ansatz, optimizer, maximum_iterations, shot_count, num_qubits, num_layers, num_problems, kappa, loading_initial_parameters, epsilon, filename_A, filename_b, optional_filename_tag = None):
    num_parameters = get_num_parameters(num_qubits, num_layers)
    H = create_H(num_qubits, filename_A, filename_b, True)
    filename_base = ""
    if (optional_filename_tag != None):
      filename_base = "Training_" + name + "_" + date + "_" + ansatz + "_" + optimizer + "_" + str(maximum_iterations) + "_" + str(num_layers) + "_" + str(shot_count) + "_" + str(num_problems) + "_" + optional_filename_tag
    else:
      filename_base = "Training_" + name + "_" + date + "_" + ansatz + "_" + optimizer + "_" + str(maximum_iterations) + "_" + str(num_layers) + "_" + str(shot_count) + "_" + str(num_problems)
      
    with open(filename_base + ".txt", 'w') as file:  
      file.write("Name: " + name + "\n")
      file.write("Date: " + date + "\n")
      file.write("Ansatz: " + ansatz + "\n")
      file.write("Optimizer: " + optimizer + "\n")
      file.write("Maximum iterations: " + str(maximum_iterations) + "\n")
      file.write("Qubits: " + str(num_qubits) + "\n")
      file.write("Layers: " + str(num_layers) + "\n")
      file.write("Shots: " + str(shot_count) + "\n")
      file.write("Kappa: " + str(kappa) + "\n")
      file.write("Noise Setting: 1\n")
      file.write("Number of problems: " + str(num_problems) + "\n")
      if (epsilon):
        file.write("Epsilon (if training from previous params): " + str(epsilon) + "\n")

      initial_parameters = []
      if (len(loading_initial_parameters) != 0 and epsilon != -1):
        for problem in range(num_problems):
          # Obtain a vector of values between -1 and 1 that we will use to slightly change
          # the current parameters themselves, so that we reduce our chance of falling into
          # a local minimum.
          change_parameters = np.random.uniform(low=-1, high=1, size=(num_parameters,))
          # Normalize that vector.
          change_parameters = change_parameters/sqrt(sum(change_parameters**2))
          # Now, make the vector have norm less than or equal to some small epsilon, which we set above.
          change_parameters = change_parameters/(1/epsilon)
          
          # Add zero for any parameters that were not initialized in the previous run.
          loaded_parameters_modified_for_ansatz = current_standard_data_set.all_final_parameters_per_problem[current_standard_best_problem_index]
          loaded_params_length = len(current_standard_data_set.all_final_parameters_per_problem[current_standard_best_problem_index])
          for i in range(num_parameters-loaded_params_length):
            loaded_parameters_modified_for_ansatz.append(0)

          # Put the modified version of the loaded parameters into initial_parameters.
          initial_parameters.append(loaded_parameters_modified_for_ansatz + change_parameters)
      elif (len(loading_initial_parameters) != 0):
        for problem in range(num_problems):
          loaded_parameters_modified_for_ansatz = loading_initial_parameters
          loaded_params_length = len(loaded_parameters_modified_for_ansatz)
          for i in range(num_parameters-loaded_params_length):
            loaded_parameters_modified_for_ansatz.append(0)
          initial_parameters.append(loaded_parameters_modified_for_ansatz)
      else:
        # Create however many sets of randomly-initialized parameters we want to use.
        for problem in range(num_problems):
          initial_parameters.append(np.random.rand(num_parameters) * 2 * np.pi)

      for problem in range(num_problems):
        print("\n\nSolving Problem ", problem, "...", sep="")
        file.write("Problem " + str(problem) + ":\n")
        file.write("Initial Parameters: ")
        for parameter in initial_parameters[problem]:
          file.write(str(parameter) + " ")
        file.write("\n")
        
        result = minimize(cost_function_finite_sampling, initial_parameters[problem], jac=parameter_shift_gradient_finite_sampling, method="CG", options={'maxiter':maximum_iterations, 'return_all':True}, args=(num_qubits, num_layers, H, shot_count))
          
        file.write("Number of Iterations: " + str(result.nit) + "\n")

        file.write("ParametersPerIteration: " + "\n")
        for parameter_set in result.allvecs:
          for element in parameter_set:
            file.write(str(element) + " ")
          file.write("\n")
        file.write("EndParametersPerIteration" + "\n")
          
        file.write("Final Parameters: ")
        for element in result.x:
          file.write(str(element) + " ")
        file.write("\n")
          
    print("That's all! ", str(num_problems), " circuits have been trained with simulated (finite sampling) shot noise.", sep="")
    
# Description: Load results from a circuit that was trained with shot noise and without hardware noise.
# Parameters: filename - A string with the filename of the data to read.
#             compute_additional_data - A boolean flag indicating whether or not to compute the additional data not stored in the file,
#             but relevant to some further data-processing tasks.  (For example, whether to compute cost/iteration, so that plot can be
#             created.)
#             filename_x - A string with the filename for vector x.
#             filename_A - A string with the filename for matrix A.
#             filename_b - A string with the filename for vector b.
# Return Value: The JMH_VQE_train_data object with the read/computed data, and an integer with the index of the problem with best fidelity.
def train_shot_noise_processing(filename, compute_additional_data, filename_x, filename_A, filename_b):
  data_from_file_shot_noise = ''
  try:
    data_from_file_shot_noise = read_vqe_training_data_from_file(filename)
  except IOError:
    print("I'm sorry, but the file you specified cannot be opened.  Quitting for now.")
    quit()
    
  name = data_from_file_shot_noise.username
  date = data_from_file_shot_noise.date
  ansatz = data_from_file_shot_noise.ansatz_type
  optimizer = data_from_file_shot_noise.optimizer_type
  maximum_iterations = data_from_file_shot_noise.maximum_iterations
  shot_count = data_from_file_shot_noise.shot_count
  num_qubits = data_from_file_shot_noise.num_qubits
  num_layers = data_from_file_shot_noise.num_layers
  num_problems = data_from_file_shot_noise.num_problems
  kappa = data_from_file_shot_noise.kappa
  num_parameters = get_num_parameters(num_qubits, num_layers)
  H = create_H(num_qubits, filename_A, filename_b, True)

  best_problem_index_shot_noise = ''
  if (compute_additional_data):
    populate_iteration_per_problem_data_from_saved_params(data_from_file_shot_noise, H, kappa, filename_x)

    # Print statistics for the entire set of circuits.  This can be used to find the
    # "best" circuit (where "best" is classified as that with highest fidelity) 
    # for future use.
    print("The maximum fidelity is ", data_from_file_shot_noise.max_fidelity, ", which occurred at problem ", data_from_file_shot_noise.max_fidelity_problem_index, " and iteration ", data_from_file_shot_noise.max_fidelity_iteration_index, ".", sep="")
    print("The total number of iterations for problem ", data_from_file_shot_noise.max_fidelity_problem_index, " was ", data_from_file_shot_noise.all_iterations_per_problem[data_from_file_shot_noise.max_fidelity_problem_index], ".", sep="")
    print("The minimum cost is ", data_from_file_shot_noise.min_cost, ", which occurred at problem ", data_from_file_shot_noise.min_cost_problem_index, " and iteration ", data_from_file_shot_noise.min_cost_iteration_index, ".", sep="")
    print("The total number of iterations for problem ", data_from_file_shot_noise.min_cost_problem_index, " was ", data_from_file_shot_noise.all_iterations_per_problem[data_from_file_shot_noise.min_cost_problem_index], ".", sep="")
    best_problem_index_shot_noise = data_from_file_shot_noise.max_fidelity_problem_index
    
  return data_from_file_shot_noise, best_problem_index_shot_noise
    
# Description: Create a set of plots that are useful for display in a paper summarizing results:
#              First, a heatmap with the best solution frm the data set. And second, two line plots 
#              showing cost and fidelity per iteration.
# Parameters: data_object - A JMH_VQE_train_data object with data to be plotted.
#             best_solution_index - An integer with the index of the problem with best fidelity.  Currently, this is the
#             only heatmap we create.
#             filename_x - A string with the name of the file containing the true x.
# Return Value: Because plots are created, nothing need be returned.
def train_data_summary_plot_creation(data_object, best_solution_index, filename_x):
  # Create a heatmap of the best solution.
  cur_prob_state = run_circuit_with_statevector_simulator(data_object.all_final_parameters_per_problem[best_solution_index], data_object.num_qubits, data_object.num_layers)
  true_x_non_normalized = create_x(data_object.num_qubits, filename_x, False).T[0]
  grid_value_dict = generate_grid_value_dictionary(cur_prob_state, True, [1.0, 0.0], sqrt(sum(true_x_non_normalized**2)))
  create_colormap_from_solution_for_aesthetic_display(grid_value_dict, 'plasma', False)
   
  # Create line plots of cost and fidelity per iteration for all initial parameters used.
  plot_variable_num_lines(data_object.all_iterations_per_problem, data_object.all_costs_per_iteration_per_problem, "Iterations", 
                                    "Cost Function Value", "#6699ff", None, show_legend = False, highlight_line_dict = {best_solution_index: "#ff33cc"},
                          filename = "CostsPerIteration.pdf", log_scale = True)
  plot_variable_num_lines(data_object.all_iterations_per_problem, data_object.all_fidelities_per_iteration_per_problem, "Iterations", 
                                    "Fidelity", "#6699ff", None, show_legend = False, highlight_line_dict = {best_solution_index: "#ff33cc"},
                          filename = "FidelitiesPerIteration.pdf", log_scale = False)
                          
# Description: Create plots from loaded data in a "one plot per problem" (where problem means circuit with certain parameters)
#              approach.
# Parameters: data_object - A JMH_VQE_train_data object with data to be plotted.
# Return Value: Because plots are created, nothing need be returned.
def train_data_per_problem_plot_creation(data_object):
  for problem in range(data_object.num_problems):
    iterations = np.arange(0, data_object.all_iterations_per_problem[problem]+1)
    plot_single_line(iterations, data_object.all_costs_per_iteration_per_problem[problem], "Iterations", 
                                  "Cost Function Value", "#ff9966", "Problem " + str(problem) + ": Cost Function Value Versus Iteration", filename = "Cost_" + str(problem) + ".pdf", log_scale = True)
      
    plot_single_line(iterations, data_object.all_fidelities_per_iteration_per_problem[problem], "Iterations", 
                                    "Fidelity", "#ff9966", "Problem " + str(problem) + ": Fidelity Versus Iteration", filename = "Fidelity_" + str(problem) + ".pdf", log_scale = False)
    
    # An example of how to plot trace distance.  Because this became of less interest, I leave it only for reference.
    #plot_two_lines(iterations, data_object.all_trace_distances_per_iteration_per_problem[problem], data_object.all_trace_distance_bounds_per_iteration_per_problem[problem], "Iterations", "Trace Distance", "Trace Distance", "Upper Bound", "#ff99cc", "#3366ff", "Problem " + str(problem) + ": Trace Distance and Upper Bound", filename = "TraceDistance_" + str(problem) + ".pdf", log_scale = True)

    cur_prob_state = run_circuit_with_statevector_simulator(data_object.all_final_parameters_per_problem[problem], data_object.num_qubits, data_object.num_layers)
    grid_value_dict = generate_grid_value_dictionary(vector_values = cur_prob_state)
    create_colormap_from_solution(grid_value_dict, 'plasma', "Problem " + str(problem) + ": LANL Ansatz & No Noise")
                                 
# Description: Load results from a previous training, such that the best results can be used for subsequent training.
# Parameters: filename - A string with the filename of the data to read.
#             compute_additional_data - A boolean flag indicating whether or not to compute the additional data not stored in the file,
#             but relevant to some further data-processing tasks.  (For example, whether to compute cost/iteration, so that plot can be
#             created.)
#             filename_x - A string with the name of the file containing the true x solution.
#             filename_A - A string with the filename for matrix A.
#             filename_b - A string with the filename for vector b.
# Return Value: The JMH_VQE_train_data object with the read/computed data, and an integer with the index of the problem with best fidelity.
def load_parameter_starting_point(filename, compute_additional_data, filename_x, filename_A, filename_b):
  current_standard_data_set = ''
  try:
    current_standard_data_set = read_vqe_training_data_from_file(filename)
  except IOError:
    print("I'm sorry, but the file you specified cannot be opened.  Quitting for now.")
    quit()
  
  name = current_standard_data_set.username
  date = current_standard_data_set.date
  ansatz = current_standard_data_set.ansatz_type
  optimizer = current_standard_data_set.optimizer_type
  maximum_iterations = current_standard_data_set.maximum_iterations
  shot_count = current_standard_data_set.shot_count
  num_qubits = current_standard_data_set.num_qubits
  num_layers = current_standard_data_set.num_layers
  num_problems = current_standard_data_set.num_problems
  kappa = current_standard_data_set.kappa
  num_parameters = get_num_parameters(num_qubits, num_layers)
  H = create_H(num_qubits, filename_A, filename_b, True)

  if (compute_additional_data):
    populate_iteration_per_problem_data_from_saved_params(current_standard_data_set, H, kappa, filename_x)

    # Print statistics for the entire set of circuits.  This can be used to find the
    # "best" circuit (where "best" is classified as that with highest fidelity) 
    # for future use.
    print("The maximum fidelity is ", current_standard_data_set.max_fidelity, ", which occurred at problem ", current_standard_data_set.max_fidelity_problem_index, " and iteration ", current_standard_data_set.max_fidelity_iteration_index, ".", sep="")
    print("The total number of iterations for problem ", current_standard_data_set.max_fidelity_problem_index, " was ", current_standard_data_set.all_iterations_per_problem[current_standard_data_set.max_fidelity_problem_index], ".", sep="")
    print("The minimum cost is ", current_standard_data_set.min_cost, ", which occurred at problem ", current_standard_data_set.min_cost_problem_index, " and iteration ", current_standard_data_set.min_cost_iteration_index, ".", sep="")
    print("The total number of iterations for problem ", current_standard_data_set.min_cost_problem_index, " was ", current_standard_data_set.all_iterations_per_problem[current_standard_data_set.min_cost_problem_index], ".", sep="")
    current_standard_best_problem_index = current_standard_data_set.max_fidelity_problem_index
    
    # Create a heatmap of the best solution.
    cur_prob_state = run_circuit_with_statevector_simulator(current_standard_data_set.all_final_parameters_per_problem[current_standard_best_problem_index], num_qubits, num_layers)
    true_x_non_normalized = create_x(num_qubits, filename_x, False).T[0]
    grid_value_dict = generate_grid_value_dictionary(cur_prob_state, True, [1.0, 0.0], sqrt(sum(true_x_non_normalized**2)))
    create_colormap_from_solution_for_aesthetic_display(grid_value_dict, 'plasma', False) 
    
  return current_standard_data_set, current_standard_best_problem_index

# Description: Run a trained circuit using shot noise and simulated hardware noise for a device.
# Parameters: name - A string with the name/initials of the person collecting results.
#             date - A string with the date in month-day-year format.
#             ansatz - A string with the ansatz for the circuit. (For example, VLS or IBM.)
#             shot_count - An integer with the number of shots to use.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
#             machine - A string with the name of the IBMQ device to simulate.
#             number_of_problems - An integer with the number of times to simulate the circuit with the specified shot count.
#             qubits_to_use - An array of the integers with the qubits of the specified device to simulate.
#             data_object - A JMH_VQE_train_data object with the information from which we will run the circuit.
#             best_problem_index - An integer with the index of the problem in data_object that has the highest fidelity.
#             filename_A - A string with the filename for matrix A.
#             filename_b - A string with the filename for vector b.
#             optional_filename_tag - A string that can be added to a filename to distiguish it from previous data.  If None,
#             nothing will be added to the end of the filename.
# Return Value: Because the data is written to a file, there is nothing to return.
def run_shot_noise_hardware_noise(name, date, ansatz, shot_count, num_qubits, num_layers, machine, number_of_runs, qubits_to_use,
                                data_object, best_problem_index, filename_A, filename_b, optional_filename_tag = None):
    num_parameters = get_num_parameters(num_qubits, num_layers)
    H = create_H(num_qubits, filename_A, filename_b, True)
    filename_base = ""
    if (optional_filename_tag != None):
      filename_base = "Run_" + name + "_" + date + "_" + ansatz + "_" + "Simulated" + "_" + machine + "_" + str(num_layers) + "_" + str(shot_count) + "_" + str(number_of_runs) + "_" + optional_filename_tag
    else:
      filename_base = "Run_" + name + "_" + date + "_" + ansatz + "_" + "Simulated" + "_" + machine + "_" + str(num_layers) + "_" + str(shot_count) + "_" + str(number_of_runs)

    with open(filename_base + ".txt", 'w') as file:  
      file.write("Name: " + name + "\n")
      file.write("Date: " + date + "\n")
      file.write("Ansatz: " + ansatz + "\n")
      file.write("Machine: " + machine + "\n")
      file.write("Qubits to use: ")
      for qubit in qubits_to_use:
        file.write(str(qubit) + " ")
      file.write("\n")
      file.write("Qubits: " + str(num_qubits) + "\n")
      file.write("Layers: " + str(num_layers) + "\n")
      file.write("Shots: " + str(shot_count) + "\n")
      file.write("Noise Setting: 6\n")
      file.write("Number of Runs: " + str(number_of_runs) + "\n")
        
      print("\n\nSolving Problem ", best_problem_index, "...", sep="")
      file.write("Problem Index: " + str(best_problem_index) + "\n")
      
      for i in range(number_of_runs):
        result = run_circuit_with_simulated_hardware_noise(data_object.all_final_parameters_per_problem[best_problem_index], machine, num_qubits, num_layers, shot_count, qubits_to_use)
          
        file.write("{")
        counter = 0
        for element in result.keys():
          file.write(str(element) + ": " + str(result[element]))
          if (counter != len(result.keys())-1):
            file.write(", ")
          else:
            file.write("}\n")
            counter += 1

    print("That's all! Finished running the circuit with shot noise and simulated hardware noise.", sep="")
   
# Description: Run a trained circuit on quantum hardware.
# Parameters: name - A string with the name/initials of the person collecting results.
#             date - A string with the date in month-day-year format.
#             ansatz - A string with the ansatz for the circuit. (For example, VLS or IBM.)
#             shot_count - An integer with the number of shots to use.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
#             machine - A string with the name of the IBMQ device to simulate.
#             number_of_problems - An integer with the number of times to simulate the circuit with the specified shot count.
#             qubits_to_use - An array of the integers with the qubits of the specified device to simulate.
#             data_object - A JMH_VQE_train_data object with the information from which we will run the circuit.
#             best_problem_index - An integer with the index of the problem in data_object that has the highest fidelity.
#             filename_A - A string with the filename for matrix A.
#             filename_b - A string with the filename for vector b.
#             optional_filename_tag - A string that can be added to a filename to distiguish it from previous data.  If None,
#             nothing will be added to the end of the filename.
# Return Value: Because the data is written to a file, there is nothing to return.
def run_quantum_hardware(name, date, ansatz, shot_count, num_qubits, num_layers, machine, number_of_runs, qubits_to_use,
                        data_object, best_problem_index, filename_A, filename_b, optional_filename_tag = None):
    num_parameters = get_num_parameters(num_qubits, num_layers)
    H = create_H(num_qubits, filename_A, filename_b, True)
    filename_base = ""
    if (optional_filename_tag != None):
      filename_base = "Run_" + name + "_" + date + "_" + ansatz + "_" + machine + "_" + str(num_layers) + "_" + str(shot_count) + "_" + str(number_of_runs) + "_" + optional_filename_tag
    else:
      filename_base = "Run_" + name + "_" + date + "_" + ansatz + "_" + machine + "_" + str(num_layers) + "_" + str(shot_count) + "_" + str(number_of_runs)
     
    with open(filename_base + ".txt", 'w') as file:  
      file.write("Name: " + name + "\n")
      file.write("Date: " + date + "\n")
      file.write("Ansatz: " + ansatz + "\n")
      file.write("Machine: " + machine + "\n")
      file.write("Qubits to use: ")
      for qubit in qubits_to_use:
        file.write(str(qubit) + " ")
      file.write("\n")
      file.write("Qubits: " + str(num_qubits) + "\n")
      file.write("Layers: " + str(num_layers) + "\n")
      file.write("Noise Setting: 7\n")
      file.write("Number of Runs: " + str(number_of_runs) + "\n")
      shots_written = False
        
      print("\n\nSolving Problem ", best_problem_index, "...", sep="")
      file.write("Problem Index: " + str(best_problem_index) + "\n")

      for i in range(number_of_runs):
        result, adjusted_shots = run_circuit_on_quantum_hardware(data_object.all_final_parameters_per_problem[best_problem_index], machine, num_qubits, num_layers, shot_count, qubits_to_use)
     
        # We need to make sure to write the correct number of shots to a file.
        # For simulations, this is largely irrelevant, as the maximum number of 
        # shots is so large as to not be problematic.  However, for hardware,
        # the maximum number of shots varies much more often and might need
        # to be changed when the backend is initialized.  So, wait to record the 
        # number of shots until after we've completed the first run.
        if (shots_written == False):
          if (shot_count != ''):
            file.write("Shots: " + str(adjusted_shots) + "\n")
          else:
            file.write("Shots: -1")
          shots_written = True

        file.write("{")
        counter = 0
        for element in result.keys():
          file.write(str(element) + ": " + str(result[element]))
          if (counter != len(result.keys())-1):
            file.write(", ")
          else:
            file.write("}\n")
          counter += 1

    print("That's all! Finished running the circuit on quantum hardware.", sep="")
 
# Description: Load data from a previous run so that it can be plotted in a heatmap.
# Parameters: filename - The name of the file with the data to load/plot.
#             filename_x - A string with the name of the file containing the true x solution.
# Return Value: Because the plots are created in this function, there is no strict need to return anything.
# However, in keeping consistency with the rest of my "processing" functions, return the JMH_VQE_run_data
# object we create.
def run_data_processing(filename, filename_x):
  data_from_run = ''
  try:
    data_from_run = read_vqe_run_data_from_file(filename)
  except IOError:
    print("I'm sorry, but the file you specified cannot be opened.  Quitting for now.")
    quit()
  
  name = data_from_run.username
  date = data_from_run.date
  ansatz = data_from_run.ansatz_type
  machine = data_from_run.machine
  qubits_to_use = data_from_run.qubits_to_use
  shot_count = data_from_run.shot_count
  num_qubits = data_from_run.num_qubits
  num_layers = data_from_run.num_layers
  num_problems = data_from_run.num_problems

  all_results = [0*x for x in range(2**num_qubits)]
  
  for result_set in data_from_run.all_count_results:
    # Convert the dictionary keys from binary to decimal.
    dict_with_decimal_keys = {int(c, 2): v for c, v in result_set.items()}

    # For every element in the dictionary, add the value to the index matching
    # the given key in the vector.
    for key in dict_with_decimal_keys.keys():
      all_results[key] += dict_with_decimal_keys[key]

  # Convert all the elements in all_results to probabilities.
  all_results_converted_to_probs = []
  for element in all_results:
    all_results_converted_to_probs.append(sqrt(element / (shot_count*num_problems)))
        
  result_grid_val_dict = generate_grid_value_dictionary(vector_values = all_results_converted_to_probs)
  true_x = create_x(num_qubits, filename_x, True)
  true_grid_dict = generate_grid_value_dictionary(vector_values = true_x)
  create_colormap_from_solution(result_grid_val_dict, 'plasma', None)
  create_colormap_from_solution_with_text(result_grid_val_dict, 'plasma', None, true_grid_dict, "#00ffff")
  print("The fidelity is:", calculate_fidelity(num_qubits, true_x, np.array(all_results_converted_to_probs)))

  true_x_non_normalized = create_x(num_qubits, filename_x, False).T[0]
  result_grid_val_dict = generate_grid_value_dictionary(all_results_converted_to_probs, True, [1.0, 0.0], sqrt(sum(true_x_non_normalized**2)))
  create_colormap_from_solution_for_aesthetic_display(result_grid_val_dict, 'plasma', False)
  
  return data_from_run
  
# Description: Train a circuit without shot noise or hardware noise using a non-black-box implementation of
#              parameter-shift gradient-based descent.
# Parameters: name - A string with the name/initials of the person collecting results.
#             date - A string with the date in month-day-year format.
#             ansatz - A string with the ansatz for the circuit. (For example, VLS or IBM.)
#             optimizer - A string with the name of the optimizer. (For example, CG or SPSA.)
#             maximum_iterations - An integer with the maximum number of iterations that the optimizer can perform.
#             num_qubits - An integer with the number of qubits in the circuit.
#             num_layers - An integer with the number of layers in the circuit.
#             num_problems - An integer with the number of different circuits to train. (They will likely start with different initial parameters.)
#             kappa - A double with the condition number (as defined in the VLS paper) of the Ax=b system.  Note that this is set to -1, if not known/desired. 
#             (Kappa is important only when considering trace distance.)
#             loading_initial_parameters - An array with initial parameters to load, if applicable.
#             epsilon - A double (between 0 and 1) with a random 'kick' to apply to loaded parameters; ideally, this shouldn't be larger than 1, as that causes 
#             effectively random initial parameters to be used. If unwanted, use -1.
#             filename_A - A string with the file holding matrix A.
#             filename_b - A string with the file holding vector b.
#             eta - A double with the 'learning rate,' which represents how far the function will seek to 'jump' 
#             based on whether the cost function slope is steep (large jumps) or shallow (small jumps).
#             eps_grad - A double representing how small the gradient should be before we start counting 'well jumps.'
#             This determines when we should stop the algorithm, if we are simply bouncing around in a local minimum.
#             eps_cost - A double with a desired cost to achieve. (Optimization will stop when this cost is achieved,
#             so making this arbitrarily small can give us arbitrary precision.)
#             well_iter_max - An integer with the maximum number of times we allow ourselves to hop around a local minimum
#             seeking a better result. (We start counting when a cost of eps_grad is achieved.)
#             optional_filename_tag - A string that can be added to a filename to distiguish it from previous data.  If None,
#             nothing will be added to the end of the filename.
# Return Value: Because the data is written to a file, nothing need be returned.
def train_no_noise_custom_gradient_descent(name, date, ansatz, optimizer, maximum_iterations, num_qubits, num_layers, num_problems, kappa, loading_initial_parameters, epsilon, filename_A, filename_b, eta, eps_grad, eps_cost, well_iter_max, optional_filename_tag = None):
    num_parameters = get_num_parameters(num_qubits, num_layers)
    H = create_H(num_qubits, filename_A, filename_b, True)
    filename_base = ""
    if (optional_filename_tag != None):
      filename_base = "Training_" + name + "_" + date + "_" + ansatz + "_" + optimizer + "_" + str(maximum_iterations) + "_" + str(num_layers) + "_" + str(-1) + "_" + str(num_problems) + "_" + optional_filename_tag
    else:
      filename_base = "Training_" + name + "_" + date + "_" + ansatz + "_" + optimizer + "_" + str(maximum_iterations) + "_" + str(num_layers) + "_" + str(-1) + "_" + str(num_problems)
    
    with open(filename_base + ".txt", 'w') as file:  
      file.write("Name: " + name + "\n")
      file.write("Date: " + date + "\n")
      file.write("Ansatz: " + ansatz + "\n")
      file.write("Optimizer: " + optimizer + "\n")
      file.write("Maximum iterations: " + str(maximum_iterations) + "\n")
      file.write("Qubits: " + str(num_qubits) + "\n")
      file.write("Layers: " + str(num_layers) + "\n")
      file.write("Shots: -1\n")

      file.write("Kappa: " + str(kappa) + "\n")
      file.write("Noise Setting: 1\n")
      file.write("Number of problems: " + str(num_problems) + "\n")
      if (epsilon != -1):
        file.write("Epsilon (if training from previous params): " + str(epsilon) + "\n")
      file.write("Eta: " + str(eta) + "\n")
      file.write("Eps_grad: " + str(eps_grad) + "\n")
      file.write("Eps_cost: " + str(eps_cost) + "\n")
      file.write("Well_iter_max: " + str(well_iter_max) + "\n")

      initial_parameters = []
      if (len(loading_initial_parameters) != 0 and epsilon != -1):
        for problem in range(num_problems):
          # Obtain a vector of values between -1 and 1 that we will use to slightly change
          # the current parameters themselves, so that we reduce our chance of falling into
          # a local minimum.
          change_parameters = np.random.uniform(low=-1, high=1, size=(num_parameters,))
          # Normalize that vector.
          change_parameters = change_parameters/sqrt(sum(change_parameters**2))
          # Now, make the vector have norm less than or equal to some small epsilon, which we set above.
          change_parameters = change_parameters/(1/epsilon)
          
          # Add zero for any parameters that were not initialized in the previous run.
          loaded_parameters_modified_for_ansatz = current_standard_data_set.all_final_parameters_per_problem[current_standard_best_problem_index]
          loaded_params_length = len(current_standard_data_set.all_final_parameters_per_problem[current_standard_best_problem_index])
          for i in range(num_parameters-loaded_params_length):
            loaded_parameters_modified_for_ansatz.append(0)

          # Put the modified version of the loaded parameters into initial_parameters.
          initial_parameters.append(loaded_parameters_modified_for_ansatz + change_parameters)          
      elif (len(loading_initial_parameters) != 0):
        for problem in range(num_problems):
          loaded_parameters_modified_for_ansatz = loading_initial_parameters
          loaded_params_length = len(loaded_parameters_modified_for_ansatz)
          for i in range(num_parameters-loaded_params_length):
            loaded_parameters_modified_for_ansatz.append(0)
          initial_parameters.append(loaded_parameters_modified_for_ansatz)
      else:
        # Create however many sets of randomly-initialized parameters we want to use.
        for problem in range(num_problems):
          initial_parameters.append(np.random.rand(num_parameters) * 2 * np.pi)
      
      for problem in range(num_problems):
        print("\n\nSolving Problem ", problem, "...", sep="")
        file.write("Problem " + str(problem) + ":\n")
        file.write("Initial Parameters: ")
        for parameter in initial_parameters[problem]:
          file.write(str(parameter) + " ")
        file.write("\n")
       
        result_dict, result_params = vanilla_gradient_optimization(initial_parameters[problem], num_qubits, num_layers, H,                   
                                     cost_function_no_shot_noise, gradient_cost_plus_term, gradient_cost_minus_term, eta, eps_grad, eps_cost, 
                                     maximum_iterations, well_iter_max) 
        
        file.write("Number of Iterations: " + str(len(result_dict.keys())) + "\n")
        
        file.write("ParametersPerIteration: " + "\n")
        for parameter_set in result_params:
          for element in parameter_set:
            file.write(str(element) + " ")
          file.write("\n")
        file.write("EndParametersPerIteration" + "\n")
        
        file.write("Final Parameters: ")
        for element in result_params[len(result_params)-1]:
          file.write(str(element) + " ")
        file.write("\n")
        
    print("That's all! ", str(num_problems), " circuits have been trained with no noise.", sep="")
