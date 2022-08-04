# Filename: JMH_VLS_Pitchfork_Fracture_Execution.py
# Author: Jessie M. Henderson
# Date: July 19, 2021 [Adapted from code started on June 28, 2021]
# Description: This code solves the pitchfork fracture problem using information from the 
# user to determine the problem and the specifications under which the problem should be solved. 
# There are then a number of possible combinations:
# 0) Train with no noise.
# 1) Train with shot noise.
# 2) Train with shot noise and hardware noise.
# 3) Train with quantum hardware.
# 4) Run with no noise.
# 5) Run with shot noise.
# 6) Run with shot noise and hardware noise. (i.e., Use simulated hardware.)
# 7) Run with quantum hardware.
# To specify what you would like to do, use the variables in the next block.
# A few additional notes:
# - Some of the above options are not implemented because there hasn't been a need for them.
# - The words "problem" and "circuit" are used somewhat interchangably in this
# code.  Although not ideal, both terms refer to results from a training set, where a specified number of randomly
# initialized parameters were trained in order to find the best parameters that 
# we can for a single Ax=b problem.  (Training with multiple randomly-generated
# parameters reduces the risk of becoming stuck in a local minimum.)

# Import everything we need to solve pitchfork fracture problems using VLS.
import JMH_VLS_Function_Database as jmh
from qiskit import IBMQ

###### Variables for the user to set.  Please look through all of the settings below until you see the 'closing bar' of pound signs. ####################

# These dictate what functionality you want to run. 
# Note that variables that are currently commented out are those that represent functionality that would be easily implemented based upon that which
# already exists, but which has not been necessary yet.
load_parameter_starting_point = False
train_no_noise = False
train_no_noise_without_black_box_optimizer = True
train_no_noise_processing = False
train_no_noise_plot_creation = False
train_shot_noise = False
train_shot_noise_processing = False
train_shot_noise_plot_creation = False
# train_shot_noise_hardware_noise = False
# train_shot_noise_hardware_noise_processing = False
# train_shot_noise_hardware_noise_plot_creation = False
# train_quantum_hardware = False
# train_quantum_hardware_processing = False
# train_quantum_hardware_plot_creation = False
# run_no_noise = False
# run_shot_noise = False
run_shot_noise_hardware_noise = False
run_shot_noise_hardware_noise_processing = False
run_quantum_hardware = False
run_quantum_hardware_processing = False

# These are the settings to use for training.
name_train = "JMH"
date_train = "8-3-21"
ansatz_train = "VLS"
optimizer_train = "VanillaGradient-Descent"
maximum_iterations_train = 2000
# Set shot_count to -1, if using noiseless training.
shot_count_train = -1
num_qubits_train = 5
num_layers_train = 100
num_problems_train = 1
# Set kappa to -1, if you do not care about computing trace distance upper bounds.
kappa_train = -1
# Set epsilon to -1, if you do not want to randomly-adjust pre-loaded parameters or if you are not pre-loading parameters at all.
epsilon_train = -1
filename_A_train = "A.txt"
filename_b_train = "b.txt"
filename_x_train = "x.txt"
opt_filename_train_tag = None

""" These are additional settings for training, if you are not using a black-box optimizer, and are
instead using our "vanilla" gradient-descent with adpative learning rate.
A description of each variable:
eta_train - A double with the 'learning rate,' which represents how far the function will seek to 'jump' 
based on whether the cost function slope is steep (large jumps) or shallow (small jumps).
eps_grad_train - A double representing how small the gradient should be before we start counting 'well jumps.'
This determines when we should stop the algorithm, if we are simply bouncing around in a local minimum.
eps_cost_train - A double with a desired cost to achieve. (Optimization will stop when this cost is achieved,
so making this arbitrarily small can give us arbitrary precision.)
well_iter_max_train - An integer with the maximum number of times we allow ourselves to hop around a local minimum
seeking a better result. (We start counting when a cost of eps_grad is achieved.)
If you have no idea what to set these values to, you can use these suggested values:
eta_train = 1.0
eps_grad_train = 10**-4
eps_cost_train = 10**-10
iter_max_train = 2000
well_iter_max_train = 50
"""
eta_train = 1.0
eps_grad_train = 10**-4
eps_cost_train = 10**-10
well_iter_max_train = 50

# These are the settings to use for running a previously-trained circuit.
name_run = "JMH"
date_run = "7-20-21"
ansatz_run = "VLS"
shot_count_run = 10**5
num_qubits_run = 5
num_layers_run = 5
machine_run = "ibmq_mumbai"
number_of_runs_run = 1
qubits_to_use_run = [0, 1, 4, 7, 10]
# Fill in data_set_to_run with "current_loaded" for any data set loaded using the load_parameter_starting_point option;
# "train_no_noise" for a data set loaded using the train_no_noise_processing option; or 
# "train_shot_noise" for a data set loaded using the train_shot_noise_processing option.
data_set_to_run = "train_shot_noise"
filename_A_run = "A_train.txt"
filename_b_run = "b_train.txt"
filename_x_run = "x_train.txt"
opt_filename_run_tag = "10000xMorePermeable"

# These are the settings to use for loading data of any kind.
load_parameter_filename = ""
load_parameter_compute_additional_data = False
filename_A_load = "A_load.txt"
filename_b_load = "b_load.txt"
filename_x_load = "x_load.txt"
train_no_noise_filename = ""
train_no_noise_compute_additional_data = False
train_shot_noise_filename = ""
train_shot_noise_compute_additional_data = False
run_shot_noise_hardware_noise_filename = ""
run_quantum_hardware_filename = ""
########################################################################################################################################################

#################################### Variables the user should leave alone. Please don't touch! :) #####################################################
current_standard_data_set = None
current_standard_best_problem_index = None
train_no_noise_data_set = None
train_no_noise_best_problem_index = None
train_shot_noise_data_set = None
train_shot_noise_best_problem_index = None
run_shot_noise_hardware_noise_data = None
run_quantum_hardware_data = None
loaded_IBMQ_account = False
# Other code developers, use the below variable if you want to test new options without the restrictions above.
jmh_code_in_development = False
#######################################################################################################################################################

print("Welcome to the Pitchfork Fracture VLS Solver!")
print("This code solves a given pitchfork fracture problem that has been converted into a linear system of equations.")
print("Specifically, this script offers a number of options that--once set--allow the code to autonomously run to completion.")
print("Ease of use (and laziness of my developer??) dictate that information is not entered in real-time, but is instead set")
print("in a specified location at the top of this script.")

print("Have you specified all of your desired settings?  Enter 'y' to continue, or anything else to stop this program and check.")
user_continue_choice = input()

if (user_continue_choice != 'y' and user_continue_choice != 'Y'):
  print("Okay, I will quit so you can go check on your settings.  TTFN!")
  quit()

print("Additionally, the correct A, x, and b need to be specified in properly-formatted files in the current directory.")
print("For full formatting instructions, please see the documentation for creating A, x, and b in JMH_VLS_Function_Database.")
print("Are you confident that your A, x, and b are ready to go?  Enter 'y' to continue, or anything else to stop this program and check.")
user_continue_choice = input()

if (user_continue_choice != 'y' and user_continue_choice != 'Y'):
  print("Okay, I will quit so you can go check on your settings.  TTFN!")
  quit()

print("Alrighty, final check: any previously trained or run data files that will be loaded also need to be in the current directory OR you need to supply a full file path.")
print("Do you have such data ready if you plan to use it?  Enter 'y' to continue, or anything else to stop this program and check.")
user_continue_choice = input()

if (user_continue_choice != 'y' and user_continue_choice != 'Y'):
  print("Okay, I will quit so you can go check on your settings.  TTFN!")
  quit()
  
print("Since everything is ready, let's get started!")

# Take action, depending upon what the user specified.
if (load_parameter_starting_point):
  print("Loading previously-stored training data...")
  current_standard_data_set, current_standard_best_problem_index = jmh.load_parameter_starting_point(load_parameter_filename, load_parameter_compute_additional_data, filename_x_load, filename_A_load, filename_b_load)

if (train_no_noise):
  print("Training with no noise...")
  if (load_parameter_starting_point):
    jmh.train_no_noise(name_train, date_train, ansatz_train, optimizer_train, maximum_iterations_train, 
    num_qubits_train, num_layers_train, num_problems_train, kappa_train, current_standard_data_set.all_final_parameters_per_problem[current_standard_best_problem_index], epsilon_train, filename_A_train, filename_b_train, opt_filename_train_tag)
    
  else:
    jmh.train_no_noise(name_train, date_train, ansatz_train, optimizer_train, maximum_iterations_train, 
    num_qubits_train, num_layers_train, num_problems_train, kappa_train, [], epsilon_train, filename_A_train, filename_b_train, opt_filename_train_tag)

if (train_no_noise_without_black_box_optimizer):
  print("Training with no noise and a non-black-box optimizer (custom gradient descent)...")
  if (load_parameter_starting_point):
    jmh.train_no_noise_custom_gradient_descent(name_train, date_train, ansatz_train, optimizer_train, maximum_iterations_train,
    					num_qubits_train, num_layers_train, num_problems_train, kappa_train,              
    					current_standard_data_set.all_final_parameters_per_problem[current_standard_best_problem_index], 
    					epsilon_train, filename_A_train, filename_b_train, eta_train, eps_grad_train, eps_cost_train,
    					well_iter_max_train, opt_filename_train_tag)

  else:
    jmh.train_no_noise_custom_gradient_descent(name_train, date_train, ansatz_train, optimizer_train, maximum_iterations_train,
    					num_qubits_train, num_layers_train, num_problems_train, kappa_train, [], epsilon_train, filename_A_train, 						filename_b_train, eta_train, eps_grad_train, eps_cost_train,
    					well_iter_max_train, opt_filename_train_tag) 
    					
if (train_no_noise_processing):
  print("Processing data from a no-noise training run...")
  train_no_noise_data_set, train_no_noise_best_problem_index = jmh.train_no_noise_processing(train_no_noise_filename, train_no_noise_compute_additional_data, filename_x_train, filename_A_train, filename_b_train)  
  
if (train_no_noise_plot_creation):
  print("Creating plots from no-noise training data...")
  jmh.train_data_summary_plot_creation(train_no_noise_data_set, train_no_noise_best_problem_index, filename_x_train)

if (train_shot_noise):
  print("Training with shot noise...")
  if (load_parameter_starting_point):
    jmh.train_shot_noise(name_train, date_train, ansatz_train, optimizer_train, maximum_iterations_train, shot_count_train, num_qubits_train, num_layers_train, num_problems_train, kappa_train, current_standard_data_set.all_final_parameters_per_problem[current_standard_best_problem_index], epsilon_train, filename_A_train, filename_b_train, opt_filename_train_tag)
  else:
    jmh.train_shot_noise(name_train, date_train, ansatz_train, optimizer_train, maximum_iterations_train, shot_count_train, num_qubits_train, num_layers_train, num_problems_train, kappa_train, [], epsilon_train, filename_A_train, filename_b_train, opt_filename_train_tag)
    
if (train_shot_noise_processing):
  print("Processing data from a shot-noise training run...")
  train_shot_noise_data_set, train_shot_noise_best_problem_index = jmh.train_shot_noise_processing(train_shot_noise_filename, train_shot_noise_compute_additional_data, filename_x_train, filename_A_train, filename_b_train)
  
if (train_shot_noise_plot_creation):
  print("Creating plots from shot-noise training data...")
  jmh.train_data_summary_plot_creation(train_shot_noise_data_set, train_shot_noise_best_problem_index, filename_x_train)
  
if (run_shot_noise_hardware_noise):
  print("Running a loaded circuit with shot noise and simulated hardware noise...")
  
  if (loaded_IBMQ_account == False):
    try:
      IBMQ.load_account()
    except:
      print("Enter your IBMQ account token.")
      token = input()
      IBMQ.save_account(token)
    loaded_IBMQ_account = True
  
  if (data_set_to_run == "current_loaded"):
    jmh.run_shot_noise_hardware_noise(name_run, date_run, ansatz_run, shot_count_run, num_layers_run, num_layers_run, machine_run,
  number_of_runs_run, qubits_to_use_run, current_standard_data_set, current_standard_best_problem_index, filename_A_load, filename_b_load, opt_filename_run_tag)
  
  elif (data_set_to_run == "train_no_noise"):
    jmh.run_shot_noise_hardware_noise(name_run, date_run, ansatz_run, shot_count_run, num_layers_run, num_layers_run, machine_run,
  number_of_runs_run, qubits_to_use_run, train_no_noise_data_set, train_no_noise_best_problem_index, filename_A_run, filename_b_run, opt_filename_run_tag)
  
  elif (data_set_to_run == "train_shot_noise"):
    jmh.run_shot_noise_hardware_noise(name_run, date_run, ansatz_run, shot_count_run, num_layers_run, num_layers_run, machine_run,
  number_of_runs_run, qubits_to_use_run, train_shot_noise_data_set, train_shot_noise_best_problem_index, filename_A_run, filename_b_run, opt_filename_run_tag)
  
  else:
    print("Error! I don't know which data set to run.  Please review the data_set_to_run variable at the beginning of this script.")
    quit()
  
if (run_shot_noise_hardware_noise_processing):
  print("Loading data from a previous shot noise/simulated hardware noise run.")
  run_shot_noise_hardware_noise_data = jmh.run_data_processing(run_shot_noise_hardware_noise_filename, filename_x_run)
  
if (run_quantum_hardware):
  print("Running a loaded circuit on quantum hardware...")
  
  if (loaded_IBMQ_account == False):
    try:
      IBMQ.load_account()
    except:
      print("Enter your IBMQ account token.")
      token = input()
      IBMQ.save_account(token)
    loaded_IBMQ_account = True
    
  if (data_set_to_run == "current_loaded"):
    jmh.run_quantum_hardware(name_run, date_run, ansatz_run, shot_count_run, num_layers_run, num_layers_run, machine_run,
  number_of_runs_run, qubits_to_use_run, current_standard_data_set, current_standard_best_problem_index, filename_A_load, filename_b_load, opt_filename_run_tag)
  
  elif (data_set_to_run == "train_no_noise"):
    jmh.run_quantum_hardware(name_run, date_run, ansatz_run, shot_count_run, num_layers_run, num_layers_run, machine_run,
  number_of_runs_run, qubits_to_use_run, train_no_noise_data_set, train_no_noise_best_problem_index, filename_A_run, filename_b_run, opt_filename_run_tag)
  
  elif (data_set_to_run == "train_shot_noise"):
    jmh.run_quantum_hardware(name_run, date_run, ansatz_run, shot_count_run, num_layers_run, num_layers_run, machine_run,
  number_of_runs_run, qubits_to_use_run, train_shot_noise_data_set, train_shot_noise_best_problem_index, filename_A_run, filename_b_run, opt_filename_run_tag)
  
  else:
    print("Error! I don't know which data set to run.  Please review the data_set_to_run variable at the beginning of this script.")
    quit()

if (run_quantum_hardware_processing):
  print("Loading data from a previous run on quantum hardware...")
  run_quantum_hardware_data = jmh.run_data_processing(run_quantum_hardware_filename, filename_x_run)
    					
print("That's all!")
