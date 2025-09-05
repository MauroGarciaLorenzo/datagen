import os

import numpy as np
try:
    from pycompss.api.task import task
    from pycompss.api.api import compss_wait_on
    from pycompss.api.constraint import constraint
except ImportError:
    from datagen.dummies.task import task
    from datagen.dummies.api import compss_wait_on
    from datagen.dummies.constraint import constraint


@constraint(computing_units=os.environ.get('COMPUTING_UNITS', '1'))
@task(returns=2, on_failure='FAIL')
def eval_stability(case, f, func_params, **kwargs):
    """Call objective function and return its result.

    :param case: Involved cases
    :param f: Objective function
    :param kwargs: Additional keyword arguments
    :return: Result of the evaluation
    """
    return f(case=case, func_params=func_params, **kwargs)


def calculate_entropy(freqs):
    """Obtain cell entropy from stability and non-stability frequencies.

    :param freqs: two-element list with the frequency (1-based) of stable and
    non-stable cases, respectively
    :return: Entropy
    """
    cell_entropy = 0
    for i in range(len(freqs)):
        if freqs[i] != 0:
            cell_entropy = cell_entropy - freqs[i] * np.log(freqs[i])
    return cell_entropy


# def eval_entropy(stabilities, entropy_parent):
#     """Calculate entropy of the cell using its list of stabilities.

#     :param stabilities: List of stabilities (result of the evaluation of every
#     case)
#     :param entropy_parent: Parent entropy based on concrete cases (those which
#     correspond to the cell)
#     :return: Entropy and delta entropy
#     """
#     stabilities = [x for x in stabilities if x >= 0]
#     freqs = []
#     counter = 0
#     for stability in stabilities:
#         if stability == 1:
#             counter += 1
#     freqs.append(counter / len(stabilities))
#     freqs.append((len(stabilities) - counter) / len(stabilities))
#     entropy = calculate_entropy(freqs)
#     if entropy_parent is None:
#         delta_entropy = 1
#     else:
#         delta_entropy = entropy - entropy_parent
#     return entropy, delta_entropy

def eval_entropy(stabilities, entropy_parent):
    """Calculate entropy of the cell using its list of stabilities.

    :param stabilities: List of stabilities (result of the evaluation of every
    case)
    :param entropy_parent: Parent entropy based on concrete cases (those which
    correspond to the cell)
    :return: Entropy and delta entropy
    """
    freqs = []
    counter = 0
    for stability in stabilities:
        if stability == 1:
            counter += 1
    freqs.append(counter / len(stabilities))
    freqs.append((len(stabilities) - counter) / len(stabilities))
    entropy = calculate_entropy(freqs)
    if entropy_parent is None:
        delta_entropy = 1
    else:
        delta_entropy = entropy - entropy_parent
    return entropy, delta_entropy