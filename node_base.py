import numpy as np
from abc import abstractmethod


class NodeBase:
    """
    Generic class for nodes in a search tree
    attributes:

    n_v : number of times this node has been visited
    n_c : number of children already expanded and added to tree
    n_a_ : max number of actions available at each state
    r : reward accumulated in this tree
    parent : incoming action for this state
    state : a data structure represnting state, must be implemented in subclasses
    children : [n_a_ x 1] array of children, if a child is not selected then that entry is None
    policy_ : [n_a x 1] decision making policy for this node
    incoming_action : action that lead to this node

    Note: at this stage all attributes were considered public
    """

    def __init__(self, n_a, dist=None):

        self.n_a_ = n_a
        self.dist_ = dist[:] if dist is not None else np.ones(n_a) / n_a  # if dist not provided use uniform
        self.n_v = 0
        self.n_c = 0
        self.r = 0
        self.state = None
        self.parent = None
        self.children = [None for _ in range(n_a)]
        self.incoming_action = -1

    def __str__(self):
        """
        Return string represntation of node
        :return:
        """
        return "reward = {}, n_v = {}, n_c = {}".format(self.r, self.n_v, self.n_c)

    def normalize_dist(self):
        """
        Normalize self.dist_ making sure its a probability vector
        :return:
        """
        self.dist_ = self.dist_[:] / np.sum(self.dist_)

    def copy(self):
        """
        Create a copy of self and return it

        Note : a better way is to override __deepcopy__

        :param cls: class of copied object, this is to enable extension of this method with subclasses
        :return:
        """
        cls = self.__class__
        other = cls(self.n_a_, self.dist_)
        other.n_v = self.n_v
        other.n_c = self.n_c
        other.reward = self.r
        other.parent = self.parent
        other.children = self.children[:]
        other.incoming_action = self.incoming_action

        return other

    @abstractmethod
    def get_next_state(self, action):
        """
        Apply action and return subsequent state

        To be implemented in subclasses
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def get_allowed_actions(self):
        """
        Return set of allowed actions from this state
        :return: list of allowed actions
        """
        raise NotImplementedError()

    def get_random_child(self):
        """
        Get a random child according to self.dist_
        :return:
        """
        return np.random.choice(self.n_a_, self.dist_)

    def add_child(self, action):
        """
        Add a child to list of expanded children if not selected already, return new child or None if failed
        :param action:
        :return:
        """
        # updating self
        self.n_c += 1
        # creating new_child and filling it with relevant info
        new_child = self.get_next_state(action)
        self.children[action] = new_child
        return new_child

    def expand(self):
        """
        Expand current node by adding a previously un-explored child to it and return the newly added chile

        :param self: this node
        :return: Node, expanded child
        """
        candid = self.get_random_child()
        # making sure selected child is not already in tree
        while self.children[candid] is not None:
            candid = self.get_random_child()
        return self.add_child(candid)

    @abstractmethod
    def is_leaf(self, height):
        """
        Return True iff self is a leaf node
        :param height: height of the tree
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def is_explored(self):
        """
        Return True iff self is fully explored
        :return:
        """
        raise NotImplementedError


    def set_dist(self, new_dist):
        """
        Change prior distribution vector for selecting children in this node.

        :param new_dist:
        :return: None
        """
        self.dist_ = new_dist[:] / np.sum(new_dist)

    def get_best_child(self, c_p=1):
        """
        Return the child with largest UCT value (see MCTS literature for details)

        Method can be ovridden in subclasses if other tree policy is used

        :param c_p: exploration/exploitation trade-off
        :return: child node with largest uct value
        """
        max_reward, best_child = 0, None
        for child_idx, child in enumerate(self.children):
            n_v = 0 if child is None else child.n_v
            reward = 0 if child is None else child.r
            ucb = reward / n_v + c_p * self.dist_[child_idx] * np.sqrt(2 * np.log(self.n_v) / n_v)

            if ucb > max_reward:
                max_reward = ucb
                best_child = child

        return best_child


if __name__ == "__main__":

    n = NodeBase(6)
    n_copy = n.copy()