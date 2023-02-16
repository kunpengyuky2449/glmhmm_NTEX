import autograd.numpy as np
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from NTEX_glmhmm_Model.GLM_ntex import glm

npr.seed(65)

def fit_glm_multiple_init(
        input_train, y_train, input_test,
        y_test, m, c, nn_init):
    param_list_temp = []
    acc_list = []
    for k in range(nn_init):
        #print('1')
        _, weights = fit_glm(
            [input_train],
            [y_train], m, c)
        param_list_temp.append(weights)
        acc_list.append(calculate_predictive_acc_glm(
            weights, input_test, y_test))
    best_init = acc_list.index(max(acc_list))
    best_acc = acc_list[best_init]
    best_param = param_list_temp[best_init]
    return best_param, best_acc

def calculate_predictive_acc_glm(glm_weights, inpt, y):
    M = inpt.shape[1]
    C = 2
    # Calculate test loglikelihood
    from GLM_ntex import glm
    new_glm = glm(M, C)
    # Set parameters to fit parameters:
    new_glm.params = glm_weights
    # time dependent logits:
    prob_right = np.exp(new_glm.calculate_logits(inpt))
    prob_right = prob_right[:, 0, 1]
    # Get the predicted label for each time step:
    predicted_label = np.around(prob_right, decimals=0).astype('int')
    # Examine at appropriate idx
    predictive_acc = np.sum(
        y[:, 0] == predicted_label[:]) / len(y[:, 0])
    return predictive_acc

def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    return inpt, y, session


def fit_glm(inputs, datas, M, C):
    new_glm = glm(M, C)
    new_glm.fit_glm(datas, inputs, masks=None, tags=None)
    # Get loglikelihood of training data:
    loglikelihood_train = new_glm.log_marginal(datas, inputs, None, None)
    recovered_weights = new_glm.Wk
    return loglikelihood_train, recovered_weights


# Append column of zeros to weights matrix in appropriate location
def append_zeros(weights):
    weights_tranpose = np.transpose(weights, (1, 0, 2))
    weights = np.transpose(
        np.vstack([
            weights_tranpose,
            np.zeros((1, weights_tranpose.shape[1], weights_tranpose.shape[2]))
        ]), (1, 0, 2))
    return weights


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


def plot_input_vectors(Ws,
                       figure_directory,
                       title='true',
                       save_title="true",
                       labels_for_plot=[]):
    K = Ws.shape[0]
    K_prime = Ws.shape[1]
    M = Ws.shape[2] - 1
    fig = plt.figure(figsize=(7, 9), dpi=80, facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.15,
                        bottom=0.27,
                        right=0.95,
                        top=0.95,
                        wspace=0.3,
                        hspace=0.3)

    for j in range(K):
        for k in range(K_prime - 1):
            # plt.subplot(K, K_prime, 1+j*K_prime+k)
            plt.plot(range(M + 1), -Ws[j][k], marker='o')
            plt.plot(range(-1, M + 2), np.repeat(0, M + 3), 'k', alpha=0.2)
            plt.axhline(y=0, color="k", alpha=0.5, ls="--")
            if len(labels_for_plot) > 0:
                plt.xticks(list(range(0, len(labels_for_plot))),
                           labels_for_plot,
                           rotation='90',
                           fontsize=12)
            else:
                plt.xticks(list(range(0, 3)),
                           ['Stimulus', 'Past Choice', 'Bias'],
                           rotation='90',
                           fontsize=12)
            plt.ylim((-3, 6))

    fig.text(0.04,
             0.5,
             "Weight",
             ha="center",
             va="center",
             rotation=90,
             fontsize=15)
    fig.suptitle("GLM Weights: " + title, y=0.99, fontsize=14)
    fig.savefig(figure_directory + 'glm_weights_' + save_title + '.png')
