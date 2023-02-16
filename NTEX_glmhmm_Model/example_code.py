# %%
import pickle
from sklearn.utils import shuffle
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import ssm
import autograd.numpy as np
from NTEX_glmhmm_Model.LapseModel_ntex import lapse_model
from NTEX_glmhmm_Model.lapse_utils_ntex import get_parmin, get_parmax, get_parstart, fit_lapse_multiple_init, \
    calculate_predictive_acc_lapse_model
from NTEX_glmhmm_Model.glm_utils_ntex import fit_glm, calculate_predictive_acc_glm, \
    plot_input_vectors, append_zeros, fit_glm_multiple_init
from NTEX_glmhmm_Model.glm_hmm_utils_ntex import calculate_predictive_acc_glmhmm_ntex_parted, \
    fit_glmhmm_multiple_init, get_posterior_states_labels, get_posterior_states_labels_parted
from NTEX_glmhmm_Model.plotting_utils_ntex import plot_glmhmm_weights

# %%
xxx
# %% load example data the merged version or the parted version
container = np.load('../example_data/'
                    'preprocessed_Input_Output_example_merged.npz',
                    allow_pickle=True)
data = [container[key] for key in container]
input_example_merged = data[0]
y_example_merged = data[1]

container = np.load('../example_data/'
                    'preprocessed_Input_Output_example_parted.npz',
                    allow_pickle=True)
data = [container[key] for key in container]
input_example_parted = data[0]
y_example_parted = data[1]
results_dir = '../results/'

# For following part, I will use the parted version of data as example:
#Shuffle parted data
input_shuffled, y_shuffled = shuffle(input_example_parted, y_example_parted,
                                   random_state=66)

#%% Here gives an example of 5-fold model training (iterations of using each 1/5 sessions of data as test set
# and the other 4/5 sessions as training set), the following part set up all necessary hyperparameters
data_size = len(input_shuffled)
Acc_GLM_5fold = []
Param_GLM_5fold = []
Acc_HMM_5fold = []
Param_HMM_5fold = []

fold = 5
M_GLM = 4  # Number of inputs for each trial for the model
# Since GLM model already has the bias colum, so remove this column before feeding to it, which is column 4
input_index_glm = [0, 1, 2, 3]
M_HMM = 5  # Number of inputs for each trial for the model
input_index_hmm = [0, 1, 2, 3, 4]
C = 2  # number of output types/categories here with only two: response or not
n_init = 5  # number of times for independent runs in which the one with the best predictive acc would be selected out

D = 1  # dimension of data (observations)
N_em_iters = 500  # number of max EM iterations to prevent non-converging situation
global_fit = True  # If global_fit true, use GLM parameter as initial params for glmhmm
# If global_fit false, pretrained glmhmm params are needed
transition_alpha = 1  # Hyperparameter
prior_sigma = 100  # Hyperparameter
K_states = 3  # Number of states you wish to observe

# get time now and result storing path to saving data such as the trained glmhmm parameters
results_dir = '../results/'
#%% Now is the example of starting the 5 fold training
for j in range(fold):
    # this part is too separate training and test set, the code is not optimal
    # get hmm training and test set, for hmm training, the data should be parted into sessions
    input_this_test_hmm = [
        x[:, input_index_hmm]
        for c, x in enumerate(input_shuffled)
        if j*len(input_shuffled)/fold <= c <= (j+1)*len(input_shuffled)/fold]
    y_this_test_hmm = [
        x[:, :]
        for c, x in enumerate(y_shuffled)
        if j*len(input_shuffled)/fold <= c <= (j+1)*len(input_shuffled)/fold]
    input_this_train_hmm = [
        x[:, input_index_hmm]
        for c, x in enumerate(input_shuffled)
        if c < j*len(input_shuffled)/fold or c > (j+1)*len(input_shuffled)/fold]
    y_this_train_hmm = [
        x[:, :]
        for c, x in enumerate(y_shuffled)
        if c < j*len(input_shuffled)/fold or c > (j+1)*len(input_shuffled)/fold]

    # get glm training and test set, for glm training, the data should not be parted, that concatenaed
    input_this_test_glm = []
    input_this_train_glm = []
    y_this_test_glm = []
    y_this_train_glm = []
    for c, x in enumerate(input_shuffled):
        if j*len(input_shuffled)/ fold <= c <= (j + 1)*len(input_shuffled)/ fold:
            if len(input_this_test_glm) == 0:
                input_this_test_glm = x[:, input_index_glm]
                y_this_test_glm = y_shuffled[c]
            else:
                input_this_test_glm = np.concatenate((
                    input_this_test_glm, x[:, input_index_glm]))
                y_this_test_glm = np.concatenate((
                    y_this_test_glm, y_shuffled[c]))
        else:
            if len(input_this_train_glm) == 0:
                input_this_train_glm = x[:, input_index_glm]
                y_this_train_glm = y_shuffled[c]
            else:
                input_this_train_glm = np.concatenate((
                    input_this_train_glm, x[:, input_index_glm]))
                y_this_train_glm = np.concatenate((
                    y_this_train_glm, y_shuffled[c]))

    # run GLM model training
    best_param_glm, best_acc_glm = fit_glm_multiple_init(
        input_this_train_glm, y_this_train_glm,
        input_this_test_glm, y_this_test_glm,
        M_GLM, C, n_init)
    Param_GLM_5fold.append(best_param_glm)
    Param_GLM_5fold.append(best_acc_glm)

    # start GLM HMM training
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H%M%S")
    weights_glmhmm_example, acc_glmhmm_example = fit_glmhmm_multiple_init(
        input_this_train_hmm, y_this_train_hmm,
        input_this_test_hmm, y_this_test_hmm,
        [np.ones([len(x), 1]) for x in y_this_train_hmm],
        # this parameter provides a function to exclude some trials, if you want to use all trials,
        # create a matrix of ones in the same shape as the y dataset
        K_states, D, M_HMM, C, N_em_iters, transition_alpha,
        prior_sigma, global_fit, best_param_glm,
        'training_cache/glmhmm_' + time_str,
        n_init, partition=True)
    Param_HMM_5fold.append(weights_glmhmm_example)
    Acc_HMM_5fold.append(acc_glmhmm_example)
print(Acc_HMM_5fold)
#%% save best trained model parameters and accuracy for each fold
results_dir = '../results/'
with open(results_dir+'Accs_5fold_example_' + time_str, 'wb') as f:
    # indent=2 is not needed but makes the file human-readable
    # if the data is nested
    pickle.dump(Acc_HMM_5fold, f)
with open(results_dir+'Params_5fold_example_' + time_str, 'wb') as f:
    # indent=2 is not needed but makes the file human-readable
    # if the data is nested
    pickle.dump(Param_HMM_5fold, f)
# %% load accs and params trained
results_dir = '../results/'
with open(results_dir+"Accs_5fold_example_2023-02-16-163152", 'rb') as f:
    Acc_HMM_load = pickle.load(f)
with open(results_dir+"Params_5fold_example_2023-02-16-163152", 'rb') as f:
    Param_HMM_load = pickle.load(f)

# %% Load the best trained parameter and the example data before shuffling to calculate
# predicted labels and states of each trial in correct order, which could be saved and
# provide information for further nalysis
Best_param = Param_HMM_load[Acc_HMM_load.index(max(Acc_HMM_load))]
K_states = 3
container = np.load('../example_data/'
                    'preprocessed_Input_Output_example_parted.npz',
                    allow_pickle=True)
data = [container[key] for key in container]
input_example_parted = data[0]
y_example_parted = data[1]
# states_probs: for each trial what is its probability in state1, state2 and so on
# predicted_states: basically find the max one among the states_probs for each trial
# predicted_response_prob: the predicted probability of response
# predicted_label: If predicted_response_prob > 50, labeled as response trial, otherwise no-response trial
states_probs, predicted_states, predicted_label, predicted_response_prob = \
    get_posterior_states_labels_parted(
        input_example_parted, y_example_parted,
        Best_param, K_states, range(K_states))

results_dir = '../results/'
now = datetime.now()
time_str = now.strftime("%Y-%m-%d-%H%M%S")
np.savez(results_dir+'predicted_states_and_labels_' + time_str + '.npz',
         states_probs,
         predicted_states,
         predicted_response_prob,
         predicted_label
         )