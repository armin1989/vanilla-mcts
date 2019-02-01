# vanilla-mcts
This repo contains objects and functions for running Monte-Carlo Tree search.

Monte-Carlo Tree search (MCTS) is a novel tree search algorithm that has yielded outstanding performance in the field of artificial intelligence for playing games. AlphaGoZero, AlphaGoLee and other state-of-the-art intelligent agents use MCTS as part of their decision making process.

In this repo I have provided the classes and functions necessary for building a basic version of MCTS. A desciription of the files is as follows:

- node_base.py: Contains a based node class for nodes in the MCTS tree. Subclassses based on this class can be written to tailor the node to any specific "decision making" problem

- node_selection.py : Contains a node class suitable for solving combinatorial "selection" problems. This is derived from base node class in node_base.py

- mcts_main.py : Includes the MCTS algorithm functions
