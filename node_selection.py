import numpy as np
import scipy.io
from node_base import NodeBase


class Node(NodeBase):
    """
    Nodes for a combinatorial problem involving selection of n out of n_a possibilities, without replacement.

    This is used to build a tree for Monte-Carlo tree search
    inhereted attributes:

    n_v : number of times this node has been visited
    n_c : number of children already expanded and added to tree
    n_a_ : max number of actions available at each state
    r : reward accumulated in this tree
    parent : incoming action for this state
    children : [n_a_ x 1] array of children, if a child is not selected then that entry is None
    policy_ : [n_a x 1] decision making policy for this node
    incoming_action : action that lead to this node

    new attributes :

    n_s : number of selected possible choices in this state
    state : [n_a_ x 1] binary array indicating selected items
    selected : [n_s x 1] array of indexes of selected items
    """

    def __init__(self, n_a, dist=None, selected=()):
        """

        :param n_a_: size of pool of candidates
        :param selected: [n_s x 1] list of items already selected in this stat
        :param dist: decision making policy for this state
        """
        NodeBase.__init__(self, n_a, dist)
        self.n_s = len(selected)
        self.selected = list(selected)
        self.state = [0 if i not in selected else 1 for i in range(n_a)]

    def __str__(self):
        """
        Return a string representation of self
        :return:
        """
        return NodeBase.__str__(self) +  "selected : {}".format(
                self.reward, self.n_v, self.n_c, self.n_s, self.selected)

    def copy(self):
        """
        Create a copy of self and return it
        :return:
        """
        other = NodeBase.copy(self)
        other.selected = self.selected[:]
        other.state = self.state[:]
        other.n_s = self.n_s

        return other

    def is_explored(self):
        """
        Return True iff self is fully explored

        In context of selection a node is explored iff all its possible children have been visited
        :return:
        """
        return self.n_c == self.n_a_ - self.n_s

    def get_next_state(self, action):
        """
        Apply action and return subsequent state

        To be implemented in subclasses
        :return:
        """
        return Node(self.n_a_, self.selected + [action], self.dist_)

    def get_allowed_actions(self):
        """
        Return set of allowed actions from this state
        :return: [(n_a - n_s) x 1] list of allowed actions
        """
        return [action for action in range(self.n_a_) if not self.state[action]]

    def is_leaf(self, height):
        """
        Return True iff self is a leaf node
        :param height: height of the tree
        :return:
        """
        return self.n_s == height

    def get_selected(self, items):
        """
        Return a list of selected items from items

        :param items: [n_a x 1] array of objects to be selected froom (the pool)
        :return: [n_s x 1] array of objects selected in this node
        """
        return np.array([items[i] for i in self.selected])


if __name__ == "__main__":

    pass
