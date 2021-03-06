# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
import copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def goalTest(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getActions(self, state):
        """
        Given a state, returns available actions.
        Returns a list of actions
        """        
        util.raiseNotDefined()

    def getResult(self, state, action):
        """
        Given a state and an action, returns resulting state.
        """
        util.raiseNotDefined()

    def getCost(self, state, action):
        """
        Given a state and an action, returns step cost, which is the incremental cost 
        of moving to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class Node:
    """
    Search node object for your convenience.

    This object uses the state of the node to compare equality and for its hash function,
    so you can use it in things like sets and priority queues if you want those structures
    to use the state for comparison.

    Example usage:
    >>> S = Node("Start", None, None, 0)
    >>> A1 = Node("A", S, "Up", 4)
    >>> B1 = Node("B", S, "Down", 3)
    >>> B2 = Node("B", A1, "Left", 6)
    >>> B1 == B2
    True
    >>> A1 == B2
    False
    >>> node_list1 = [B1, B2]
    >>> B1 in node_list1
    True
    >>> A1 in node_list1
    False
    """
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.

    You are not required to implement this, but you may find it useful for Q5.
    """
    "*** YOUR CODE HERE ***"
    frontier = util.Queue()
    explored = []
    frontier_states = []
    start_state = problem.getStartState()
    #([path],[actions])
    frontier.push(([start_state],[]))
    while not (frontier.isEmpty()):
        cur_info = frontier.pop()
        depth = len(cur_info[0])-1
        cur_state = cur_info[0][depth]
        if (problem.goalTest(cur_state)):
            #if the cur_state is the goal, return the actions leading to it
            return cur_info[1]
        explored.append(cur_state)
        next_actions = problem.getActions(cur_state)
        for action in next_actions:
            next_state = problem.getResult(cur_state,action)
            if next_state not in explored and next_state not in frontier_states:
                new_info = copy.deepcopy(cur_info)
                new_info[0].append(next_state)
                new_info[1].append(action)
                frontier.push(new_info)
                frontier_states.append(next_state)
    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
def iterativeDeepeningSearch(problem):
    """
    Perform DFS with increasingly larger depth. Begin with a depth of 1 and increment depth by 1 at every step.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.goalTest(problem.getStartState()))
    print("Actions from start state:", problem.getActions(problem.getStartState()))

    Then try to print the resulting state for one of those actions
    by calling problem.getResult(problem.getStartState(), one_of_the_actions)
    or the resulting cost for one of these actions
    by calling problem.getCost(problem.getStartState(), one_of_the_actions)

    Note: In the autograder, "nodes expanded" is equivalent to the nodes on which getActions 
    was called. To make the autograder happy, do the depth check after the goal test but before calling getActions.

    """
    "*** YOUR CODE HERE ***"
    def dfsHelper(problem, limit):
        frontier = []
        explored = []
        frontier_states = []
        start_state = problem.getStartState()
        #([path],[actions])
        frontier.append(([start_state],[]))
        while not (len(frontier) == 0):
            cur_info = frontier.pop()
            depth = len(cur_info[0])-1
            cur_state = cur_info[0][depth]
            if (problem.goalTest(cur_state)):
                #if the cur_state is the goal, return the actions leading to it
                return cur_info[1]
            explored.append(cur_state)
            if (depth < limit):
                next_actions = problem.getActions(cur_state)
                for action in next_actions:
                    next_state = problem.getResult(cur_state,action)
                    if next_state not in explored and next_state not in frontier_states:
                        new_info = copy.deepcopy(cur_info)
                        new_info[0].append(next_state)
                        new_info[1].append(action)
                        frontier.append(new_info)
                        frontier_states.append(next_state)
        return None
    limit = 0
    while True:
        #will always find a solution
        solution = dfsHelper(problem,limit)
        if (solution != None):
            return solution
        limit += 1
    util.raiseNotDefined()
    
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    explored = []
    frontier_states = []
    startNode = Node(problem.getStartState(), None, None, 0)
    #the start state has parent and action both set to None
    frontier.push(startNode,heuristic(startNode.state, problem))

    def backtrackActions(node):
        #return the series of actions leading to that state
        actions = []
        if node is None: return []
        if node.action is not None:
            actions.append(node.action)
        parent = node.parent
        while (parent is not None):
            if (parent.action is not None):
                actions = [parent.action] + actions
            node = parent
            parent = node.parent
        return actions

    while (not frontier.isEmpty()):
        cur_node = frontier.pop()
        if problem.goalTest(cur_node.state):
            #print("called backtrack")
            return backtrackActions(cur_node)
        explored.append(cur_node.state)
        next_actions = problem.getActions(cur_node.state)
        for action in next_actions:
            new_state = problem.getResult(cur_node.state, action)
            step_cost = problem.getCost(cur_node.state, action)
            new_node = Node(new_state, cur_node, action, step_cost)
            #f(n) = g(n) + h(n) where n is the new_state
            priority = problem.getCostOfActions(backtrackActions(new_node)) + heuristic(new_state, problem) 
            if new_state not in frontier_states and new_state not in explored:
                frontier.push(new_node, priority)
                frontier_states.append(new_state)
            else:
               if new_state in frontier_states and new_state not in explored:
                    #handle possible replace
                    frontier.update(new_node, priority)
    return None 

# Abbreviations
bfs = breadthFirstSearch
astar = aStarSearch
ids = iterativeDeepeningSearch
