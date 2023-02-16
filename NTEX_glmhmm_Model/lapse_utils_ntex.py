# Fit lapse model to IBL data
import autograd.numpy as np
import autograd.numpy.random as npr
from NTEX_glmhmm_Model.LapseModel_ntex import lapse_model

def calculate_predictive_acc_lapse_model(lapse_glm_weights, lapse_p,
                                         num_lapse_params, inpt, y):
    M = inpt.shape[1]
    from LapseModel_ntex import lapse_model
    new_lapse_model = lapse_model(M, num_lapse_params)
    if num_lapse_params == 1:
        new_lapse_model.params = [lapse_glm_weights, np.array([lapse_p])]
    else:
        new_lapse_model.params = [lapse_glm_weights, lapse_p]
    prob_right = np.exp(new_lapse_model.calculate_logits(inpt))
    prob_right = prob_right[:, 1]
    # Get the predicted label for each time step:
    predicted_label = np.around(prob_right, decimals=0).astype('int')
    # Examine at appropriate idx
    predictive_acc = np.sum(
        y[:,
          0] == predicted_label[:]) / len(predicted_label)
    return predictive_acc

def fit_lapse_multiple_init(
        input_train, y_train, input_test,
        y_test, m, n_lapse, n):
    param_list_temp = []
    acc_list = []
    for k in range(n):
        print(2)
        parmin_grid = np.array(
            [get_parmin(ii, m) for ii in range(m + 1 + n_lapse)])
        parmax_grid = np.array(
            [get_parmax(ii, m) for ii in range(m + 1 + n_lapse)])
        parstart = parmin_grid + np.random.rand(
            parmin_grid.size) * (parmax_grid - parmin_grid)
        # Instantiate new model
        new_model = lapse_model(m, n_lapse)
        # Initialize parameters as parstart
        new_model.params = [parstart[range(m + 1)], parstart[(m + 1):]]
        # Fit model, and obtain loglikelihood and parameters
        new_model.fit_lapse_model(datas=[y_train],
                                  inputs=[input_train],
                                  masks=None,
                                  tags=None)
        params_lapse = new_model.params
        if n_lapse == 1:
            predictive_acc = calculate_predictive_acc_lapse_model(
                params_lapse[0], params_lapse[1][0],
                n_lapse, input_test, y_test)
        else:
            predictive_acc = calculate_predictive_acc_lapse_model(
                params_lapse[0], params_lapse[1],
                n_lapse, input_test, y_test)
        param_list_temp.append(params_lapse)
        acc_list.append(predictive_acc)
    best_init = acc_list.index(max(acc_list))
    best_acc = acc_list[best_init]
    best_param = param_list_temp[best_init]
    return best_param, best_acc

def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    return inpt, y, session

def load_session_fold_lookup(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table

def load_animal_list(list_file):
    container = np.load(list_file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list

# Parameter grid to search over when using multiple initializations
def get_parmax(i, M):
    if i <= M:
        return 10
    else:
        return 1


def get_parmin(i, M):
    if i <= M:
        return -10
    else:
        return 0


def get_parstart(i, M):
    if i <= M:
        return 2 * npr.randn(1)
    else:
        gamma = np.maximum(0.05 + 0.03 * npr.rand(1), 0)
        gamma = np.minimum(gamma, 1)
        return gamma


# Reshape hessian and calculate its inverse
def calculate_std(hessian):
    # Calculate inverse of Hessian (this is what we will actually use to
    # calculate variance)
    inv_hessian = np.linalg.inv(hessian)
    # Take diagonal elements and calculate square root
    std_dev = np.sqrt(np.diag(inv_hessian))
    return std_dev
