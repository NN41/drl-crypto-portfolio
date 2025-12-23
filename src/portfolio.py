import torch
import torch.nn.functional as F
import numpy as np

def next_mu(mu, c_s, c_p, w_prime, w):
    '''
    Calculates the next element in the sequence that converges to the transaction remainder factor (mu).
    Arguments: sell commission rate (c_s), purchase commission rate (c_p), weights before reallocation at time t (w_prime),
    weights after reallocation at time t (w). A commission rate of 0.0001 means 0.01%.
    '''
    sum_of_relus = torch.sum(F.relu(w_prime[:, 1:] - mu * w[:, 1:]), dim=1, keepdim=True)
    main_part = (1 - c_p * w_prime[:, :1] - (c_s + c_p - c_s * c_p) * sum_of_relus)
    final_multiplier = 1 / (1 - c_p * w[:, :1])
    return final_multiplier * main_part

def approximate_mu(w_prev, y, w, commission_rate, train_mode=False, return_w_prime=False):
    '''
    Approximates the transaction remainder factor mu.
    All input tensors must include cash component and must contain a batch dimension.
    '''
    steps_train_mode = 10 # 10 is MORE than enough to get the consecutive change below 1e-7
    max_steps_test_mode = 50
    threshold = 1e-9

    w_prime = (w_prev * y) / torch.sum((w_prev * y), dim=1, keepdim=True)

    assert w_prev.dim() == w.dim() == y.dim() == 2, "All inputs must have batch dimension"
    assert np.mean(np.abs(np.sum(w.detach().cpu().numpy(), axis=1) - 1)) < 1e-6
    assert np.mean(np.abs(np.sum(w_prime.detach().cpu().numpy(), axis=1) - 1)) < 1e-6

    mu_0 = 1 - commission_rate * torch.sum(torch.abs(w_prime[:, 1:] - w[:, 1:]), dim=1, keepdim=True)

    all_mu = [mu_0]
    step = 0
    while True:
        mu_prev = all_mu[-1]
        mu_next = next_mu(mu_prev, commission_rate, commission_rate, w_prime, w)
        mu_next = torch.clamp(mu_next, min=0.0, max=1.0) # necessary to deal with the edge case where all money is in the cash component
        all_mu.append(mu_next)

        step += 1
        if train_mode:
            # policy network is in train mode, so we iterate until we have a nested gradient graph of X layers deep
            if step >= steps_train_mode:
                break
        else:
            # policy network is in eval mode, so we dynamically approximate until the errors between two consecutive iterations are all small enough,
            # or until we have reached the max number of steps.
            consecutive_errors = np.abs((mu_prev - mu_next).detach().cpu().numpy())
            if np.all(consecutive_errors < threshold) or (step >= max_steps_test_mode):
                break

    final_mu = all_mu[-1]
    if return_w_prime:
        return final_mu, w_prime
    else:
        return final_mu