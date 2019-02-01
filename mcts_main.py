# author : Arin Minasian

from node_selection import Node


def run_mcts(n_a, n, items, objective, objective_reqs, itr_limit, c_p, policy=None):
    """
    Run Monte-Carlo tree-search for selecting n items out of n_a without replacement

    :param n_a: number of possible initial choices (size of pool)
    :param n: number of items to be selected
    :param items: [n_a x 1] list of objects, the selection pool
    :param objective: callable, to be used in default policy to evaluate reward for selection
    :param objective_reqs: dictionary of required parameters for calling objective
    :param itr_limit: max number of simulations (play-outs) in each epoch of MCTS
    :param c_p: parameter of UCT child selection
    :param policy: sampling policy
    :return: selected_items : [n, 1] selected items
    :return v_t : final reward of this run
    """
    # starting with an empty root
    root = Node(n_a, policy)
    root.n_v = 1

    for epoch in range(n):
        # one decision making epoch
        for itr in range(itr_limit):

            # tree policy expands the decision making tree
            current = tree_policy_ucb(root, n, c_p)
            current.set_dist(policy)
            # default policy estimates reward from newly added node
            reward = default_policy(current, n, items, objective, objective_reqs)
            # backup updates statistics of all nodes in the path to newly added one
            backup(current, reward)

        root = root.get_best_child(0)

    return root.get_selected(items), objective(root.get_selected(items), objective_reqs)


def tree_policy_ucb(root, n, c_p):
    """
    Expand tree starting at current_node according to UCT with parameter c_p and return newly added node.

    :param root:  root of current search tree
    :param n: number of RRHs to be selected, used for leaf detection
    :param c_p: c parameter in UCB
    :return: expanded node
    """

    current = root
    while not current.is_leaf(n):

        # if current node is not fully explore than add a child to it
        if not current.is_explored():
            return current.expand()
        else:
            current = current.get_best_child(c_p)

    return current


def default_policy(current, n, items, objective, objective_reqs):
    """
    Run random simulation (play-out) starting from root and return corresponding reward

    :param current: Node, node to begin simulation from
    :param n: number of items to be selected (max height of tree)
    :param items: collection of items to be selected from (pool)
    :param objective: callable, reward function
    :param objective_reqs: dictionary with required parameters for objective
    :return: objective of reward
    """
    while not current.is_leaf(n):

        candid = current.get_next_state(current.get_random_child())
        candid.parent = current
        current = candid

    # return objective
    return objective(current.get_selected(items), objective_reqs)


def backup(current, reward):
    """
    Apply backup operation to update rewards in all nodes leading to current (parents)

    :param current: current node to backup from
    :param reward: reward that needs to be added to previous values of rewards in parent nodes
    :return: None
    """

    # have to double check this code!
    while current.parent:

        current.r += reward
        current.n_v += 1
        current = current.parent

    # updating root
    current.reward += reward
    current.n_v += 1


if __name__ == "__main__":

    pass
