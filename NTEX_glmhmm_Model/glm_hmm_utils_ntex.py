# Functions to assist with GLM-HMM model fitting
import sys
import ssm
import autograd.numpy as np
import autograd.numpy.random as npr


def get_posterior_states_labels(input, output, hmm_params, K, permutation):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M = input.shape[1]
    D = output.shape[1]
    this_hmm = ssm.HMM(K,
                       D,
                       M,
                       observations="input_driven_obs",
                       observation_kwargs=dict(C=2),
                       transitions="standard")
    this_hmm.params = hmm_params
    # Get expected states:
    expectations = this_hmm.expected_states(data=output,
                                            input=input,
                                            mask=np.ones(output.shape))
    posterior_probs = expectations[0]
    posterior_probs = posterior_probs[:, permutation]
    states_max_posterior = np.argmax(posterior_probs, axis=1)
    prob_right = np.exp(this_hmm.observations.calculate_logits(input=input))
    prob_right2 = prob_right[:, :, 1]
    final_prob_response = np.sum(np.multiply(posterior_probs, prob_right2), axis=1)
    # Get the predicted label for each time step:
    predicted_label = np.around(final_prob_response, decimals=0).astype('int')
    return posterior_probs, states_max_posterior, predicted_label, final_prob_response


def get_posterior_states_labels_parted(inputs, datas, hmm_params, K, permutation):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M = inputs[0].shape[1]
    D = datas[0].shape[1]
    this_hmm = ssm.HMM(K,
                       D,
                       M,
                       observations="input_driven_obs",
                       observation_kwargs=dict(C=2, prior_sigma=2),
                       transitions="sticky",
                       transition_kwargs=dict(alpha=2, kappa=0))
    this_hmm.params = hmm_params
    # Get expected states:
    expectations = [
        this_hmm.expected_states(data=data,
                                 input=input,
                                 mask=np.ones(data.shape))[0]
        for data, input in zip(datas, inputs)
    ]
    # Convert this now to one array:
    posterior_probs = np.concatenate(expectations, axis=0)
    posterior_probs = posterior_probs[:, permutation]
    states_max_posterior = np.argmax(posterior_probs, axis=1)
    prob_right = [
        np.exp(this_hmm.observations.calculate_logits(input=input))
        for data, input in zip(datas, inputs)
    ]
    prob_right = np.concatenate(prob_right, axis=0)
    # Now multiply posterior probs and prob_right:
    prob_right = prob_right[:, :, 1]
    # Now multiply posterior probs and prob_right and sum over latent axis:
    final_prob_right = np.sum(np.multiply(posterior_probs, prob_right), axis=1)
    # Get the predicted label for each time step:
    predicted_label = np.around(final_prob_right, decimals=0).astype('int')
    return posterior_probs, states_max_posterior, predicted_label, final_prob_right


def fit_glm_hmm_ntex(datas, inputs, masks, K, D, M, C, N_em_iters,
                   transition_alpha, prior_sigma, global_fit,
                   params_for_initialization, save_title):
    '''
    Instantiate and fit GLM-HMM model
    :param datas:
    :param inputs:
    :param masks:
    :param K:
    :param D:
    :param M:
    :param C:
    :param N_em_iters:
    :param global_fit:
    :param save_title:
    :return:
    '''
    if global_fit == True:
        # Prior variables
        # Choice of prior
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs",
                           observation_kwargs=dict(C=C,
                                                   prior_sigma=prior_sigma),
                           transitions="sticky",
                           transition_kwargs=dict(alpha=transition_alpha,
                                                  kappa=0))
        # Initialize observation weights as GLM weights with some noise:
        glm_vectors_repeated = np.tile(params_for_initialization, (K, 1, 1))
        glm_vectors_with_noise = glm_vectors_repeated + np.random.normal(
            0, 0.2, glm_vectors_repeated.shape)
        this_hmm.observations.params = glm_vectors_with_noise
    else:
        # Choice of prior
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs",
                           observation_kwargs=dict(C=C,
                                                   prior_sigma=prior_sigma),
                           transitions="sticky",
                           transition_kwargs=dict(alpha=transition_alpha,
                                                  kappa=0))
        # Initialize HMM-GLM with global parameters:
        this_hmm.params = params_for_initialization
        # Get log_prior of transitions:
    print("=== fitting GLM-HMM ========")
    sys.stdout.flush()
    # Fit this HMM and calculate marginal likelihood
    lls = this_hmm.fit(datas,
                       inputs=inputs,
                       masks=masks,
                       method="em",
                       num_iters=N_em_iters,
                       initialize=False,
                       tolerance=10 ** -4)
    # Save raw parameters of HMM, as well as loglikelihood during training
    np.savez(save_title, this_hmm.params, lls)
    return this_hmm.params, lls


def fit_glmhmm_multiple_init(
        input_train, y_train, input_test, y_test, mask,
        k, d, m, c, n_iters, trans_alpha, pr_sigma, glb_fit,
        weights_glm, save_path, nn_init, partition):
    param_list = []
    acc_list = []
    for ii in range(nn_init):
        weights_temp, _ = fit_glm_hmm_ntex(
            y_train, input_train, mask,
            k, d, m, c, n_iters,
            trans_alpha, pr_sigma, glb_fit,
            weights_glm, save_path)
        if not partition:
            acc_temp = calculate_predictive_acc_glmhmm_ntex(
                input_test, y_test, weights_temp, k,
                range(k), trans_alpha, pr_sigma)
        else:
            acc_temp = calculate_predictive_acc_glmhmm_ntex_parted(
                input_test, y_test, weights_temp, k,
                range(k), trans_alpha, pr_sigma)
        param_list.append(weights_temp)
        acc_list.append(acc_temp)
    best_init = acc_list.index(max(acc_list))
    best_acc = acc_list[best_init]
    best_param = param_list[best_init]
    return best_param, best_acc

def calculate_predictive_acc_glmhmm_ntex(input, output, hmm_params, K,
                                     permutation, transition_alpha, prior_sigma):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M = input.shape[1]
    D = output.shape[1]
    this_hmm = ssm.HMM(K,
                       D,
                       M,
                       observations="input_driven_obs",
                       observation_kwargs=dict(C=2, prior_sigma=prior_sigma),
                       transitions="sticky",
                       transition_kwargs=dict(alpha=transition_alpha, kappa=0))
    this_hmm.params = hmm_params
    # Get expected states:
    expectations = this_hmm.expected_states(data=output,
                                            input=input,
                                            mask=np.ones(output.shape))
    posterior_probs = expectations[0]
    posterior_probs = posterior_probs[:, permutation]
    prob_right = np.exp(this_hmm.observations.calculate_logits(input=input))
    prob_right2 = prob_right[:, :, 1]
    final_prob_right = np.sum(np.multiply(posterior_probs, prob_right2), axis=1)
    # Get the predicted label for each time step:
    predicted_label = np.around(final_prob_right, decimals=0).astype('int')
    # Examine at appropriate idx
    predictive_acc = np.sum(output == predicted_label.reshape(output.shape)) / len(output)
    return predictive_acc


def calculate_predictive_acc_glmhmm_ntex_parted(inputs, datas, hmm_params, K,
                                      permutation, transition_alpha, prior_sigma):
    # Run forward algorithm on hmm with these parameters and collect gammas:
    M = inputs[0].shape[1]
    D = datas[0].shape[1]
    this_hmm = ssm.HMM(K,
                       D,
                       M,
                       observations="input_driven_obs",
                       observation_kwargs=dict(C=2, prior_sigma=prior_sigma),
                       transitions="sticky",
                       transition_kwargs=dict(alpha=transition_alpha, kappa=0))
    this_hmm.params = hmm_params
    # Get expected states:
    expectations = [
        this_hmm.expected_states(data=data,
                                 input=input,
                                 mask=np.ones(data.shape))[0]
        for data, input in zip(datas, inputs)
    ]
    # Convert this now to one array:
    posterior_probs = np.concatenate(expectations, axis=0)
    posterior_probs = posterior_probs[:, permutation]
    prob_right = [
        np.exp(this_hmm.observations.calculate_logits(input=input))
        for data, input in zip(datas, inputs)
    ]
    prob_right = np.concatenate(prob_right, axis=0)
    # Now multiply posterior probs and prob_right:
    prob_right = prob_right[:, :, 1]
    # Now multiply posterior probs and prob_right and sum over latent axis:
    final_prob_right = np.sum(np.multiply(posterior_probs, prob_right), axis=1)
    # Get the predicted label for each time step:
    predicted_label = np.around(final_prob_right, decimals=0).astype('int')
    # Examine at appropriate idx
    for c, x in enumerate(datas):
        if c == 0:
            y = datas[c]
        else:
            y = np.concatenate(
                (y, datas[c]
                 ), axis=0)
    predictive_acc = np.sum(
        y == predicted_label.reshape(y.shape))/len(y)
    return predictive_acc


def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    return inpt, y, session


def load_cluster_arr(cluster_arr_file):
    container = np.load(cluster_arr_file, allow_pickle=True)
    data = [container[key] for key in container]
    cluster_arr = data[0]
    return cluster_arr


def load_glm_vectors(glm_vectors_file):
    container = np.load(glm_vectors_file)
    data = [container[key] for key in container]
    loglikelihood_train = data[0]
    recovered_weights = data[1]
    return loglikelihood_train, recovered_weights


def load_global_params(global_params_file):
    container = np.load(global_params_file, allow_pickle=True)
    data = [container[key] for key in container]
    global_params = data[0]
    return global_params


def partition_data_by_session(inpt, y, mask, session):
    '''
    Partition inpt, y, mask by session
    :param inpt: arr of size TxM
    :param y:  arr of size T x D
    :param mask: Boolean arr of size T indicating if element is violation or
    not
    :param session: list of size T containing session ids
    :return: list of inpt arrays, data arrays and mask arrays, where the
    number of elements in list = number of sessions and each array size is
    number of trials in session
    '''
    inputs = []
    datas = []
    indexes = np.unique(session, return_index=True)[1]
    unique_sessions = [session[index] for index in sorted(indexes)]
    counter = 0
    masks = []
    for sess in unique_sessions:
        idx = np.where(session == sess)[0]
        counter += len(idx)
        inputs.append(inpt[idx, :])
        datas.append(y[idx, :])
        masks.append(mask[idx, :])
    assert counter == inpt.shape[0], "not all trials assigned to session!"
    return inputs, datas, masks


def load_session_fold_lookup(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table


def load_animal_list(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list


def launch_glm_hmm_job(inpt, y, session, mask, session_fold_lookup_table, K, D,
                       C, N_em_iters, transition_alpha, prior_sigma, fold,
                       iter, global_fit, init_param_file, save_directory):
    print("Starting inference with K = " + str(K) + "; Fold = " + str(fold) +
          "; Iter = " + str(iter))
    sys.stdout.flush()
    sessions_to_keep = session_fold_lookup_table[np.where(
        session_fold_lookup_table[:, 1] != fold), 0]
    idx_this_fold = [str(sess) in sessions_to_keep for sess in session]
    this_inpt, this_y, this_session, this_mask = inpt[idx_this_fold, :], \
                                                 y[idx_this_fold, :], \
                                                 session[idx_this_fold], \
                                                 mask[idx_this_fold]
    # Only do this so that errors are avoided - these y values will not
    # actually be used for anything (due to violation mask)
    this_y[np.where(this_y == -1), :] = 1
    inputs, datas, masks = partition_data_by_session(
        this_inpt, this_y, this_mask, this_session)
    # Read in GLM fit if global_fit = True:
    if global_fit == True:
        _, params_for_initialization = load_glm_vectors(init_param_file)
    else:
        params_for_initialization = load_global_params(init_param_file)
    M = this_inpt.shape[1]
    npr.seed(iter)
    fit_glm_hmm(datas,
                inputs,
                masks,
                K,
                D,
                M,
                C,
                N_em_iters,
                transition_alpha,
                prior_sigma,
                global_fit,
                params_for_initialization,
                save_title=save_directory + 'glm_hmm_raw_parameters_itr_' +
                           str(iter) + '.npz')


def fit_glm_hmm(datas, inputs, masks, K, D, M, C, N_em_iters,
                transition_alpha, prior_sigma, global_fit,
                params_for_initialization, save_title):
    '''
    Instantiate and fit GLM-HMM model
    :param datas:
    :param inputs:
    :param masks:
    :param K:
    :param D:
    :param M:
    :param C:
    :param N_em_iters:
    :param global_fit:
    :param glm_vectors:
    :param save_title:
    :return:
    '''
    if global_fit == True:
        # Prior variables
        # Choice of prior
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs",
                           observation_kwargs=dict(C=C,
                                                   prior_sigma=prior_sigma),
                           transitions="sticky",
                           transition_kwargs=dict(alpha=transition_alpha,
                                                  kappa=0))
        # Initialize observation weights as GLM weights with some noise:
        glm_vectors_repeated = np.tile(params_for_initialization, (K, 1, 1))
        glm_vectors_with_noise = glm_vectors_repeated + np.random.normal(
            0, 0.2, glm_vectors_repeated.shape)
        this_hmm.observations.params = glm_vectors_with_noise
    else:
        # Choice of prior
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs",
                           observation_kwargs=dict(C=C,
                                                   prior_sigma=prior_sigma),
                           transitions="sticky",
                           transition_kwargs=dict(alpha=transition_alpha,
                                                  kappa=0))
        # Initialize HMM-GLM with global parameters:
        this_hmm.params = params_for_initialization
        # Get log_prior of transitions:
    print("=== fitting GLM-HMM ========")
    sys.stdout.flush()
    # Fit this HMM and calculate marginal likelihood
    lls = this_hmm.fit(datas,
                       inputs=inputs,
                       masks=masks,
                       method="em",
                       num_iters=N_em_iters,
                       initialize=False,
                       tolerance=10 ** -4)
    # Save raw parameters of HMM, as well as loglikelihood during training
    np.savez(save_title, this_hmm.params, lls)
    return None


def create_violation_mask(violation_idx, T):
    """
    Return indices of nonviolations and also a Boolean mask for inclusion (1
    = nonviolation; 0 = violation)
    :param test_idx:
    :param T:
    :return:
    """
    mask = np.array([i not in violation_idx for i in range(T)])
    nonviolation_idx = np.arange(T)[mask]
    mask = mask + 0
    assert len(nonviolation_idx) + len(
        violation_idx
    ) == T, "violation and non-violation idx do not include all dta!"
    return nonviolation_idx, np.expand_dims(mask, axis=1)
