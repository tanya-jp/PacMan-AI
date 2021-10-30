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

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def findPath(node, parent):
    """
    Fnd path recursively
    :param node: goal node
    :param parent: parents dictionary to find parent of each node
    :return: path from start to goal
    """
    action = []
    curr = node
    while curr[1] is not None:
        action.insert(0, curr[1])
        curr = parent[curr]
    return action


def searchWithoutCosts(fringe, problem):
    """
    Returns path to the goal for DFS and BFS
    """

    # parent[node1] = node2 shows that the parent of node1 is node2
    parent = dict()
    # States that have already been visited
    visited = set()

    start_state = problem.getStartState()
    # Insert starting node with action=None and cost=0 to fringe
    fringe.push((start_state, None, 0))

    while True:

        if fringe.isEmpty():
            return

        node = fringe.pop()
        state = node[0]

        # Find path recursively
        if problem.isGoalState(state):
            return findPath(node, parent)

        # Make graph
        if state not in visited:
            visited.add(state)

            for s in problem.getSuccessors(state):
                successor_state = s[0]

                if successor_state not in visited:
                    fringe.push(s)
                    parent[s] = node

    return []


def searchWithCosts(fringe, problem, heuristic):
    """
    Returns path to the goal for A* and UCS
    """

    # Dict to track the cost of each node
    cost = dict()
    # parent[node1] = node2 shows that the parent of node1 is node2
    parent = dict()
    # States that have already been visited
    visited = set()

    # Insert starting node with action=None and cost=0 to fringe
    start_state = problem.getStartState()
    start_state_h = heuristic(start_state, problem)
    fringe.push(item=(start_state, None), priority=0)
    # The cost is the priority of this item
    cost[start_state] = 0

    while True:

        if fringe.isEmpty():
            return

        node = fringe.pop()
        state = node[0]

        # Find path recursively
        if problem.isGoalState(state):
            return findPath(node, parent)

        # Make graph and calculate costs
        if state not in visited:
            visited.add(state)

            for s in problem.getSuccessors(state):
                successor_state = s[0]
                successor_cost = s[2]

                if successor_state not in visited:
                    # g from start state to this successor state
                    g = cost[state] + successor_cost
                    # h based on given heuristic
                    h = heuristic(successor_state, problem)
                    # If this path is new or better(cost is lower) fringe will be updated
                    totalCost = g+h
                    fringe.update(s[:2], totalCost)
                    # Update cost of this path
                    cost[successor_state] = g
                    # Define parent of this node base on state and successor
                    parent[s[:2]] = node

    return []


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    return searchWithoutCosts(util.Stack(), problem)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    return searchWithoutCosts(util.Queue(), problem)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    return searchWithCosts(util.PriorityQueue(), problem, nullHeuristic)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # return searchWithCosts(util.PriorityQueue(), problem, heuristic)

    # Set of visited positions in order to avoid revisiting them again
    # Initialize the explored set to be empty
    visited = set()

    fringe = util.PriorityQueue()
    # Insert starting node with action=None and cost=0 to fringe
    start_state = problem.getStartState()
    start_h = heuristic(start_state, problem)
    fringe.push(item=(start_state, []), priority=start_h)

    while True:

        if fringe.isEmpty():
            return False

        node = fringe.pop()
        state = node[0]
        path = node[1]

        visited.add(state)

        if problem.isGoalState(state):
            return path

        for x in problem.getSuccessors(state):
            successor_state = x[0]
            successor_action = x[1]
            successor_cost = x[2]
            sPath = path + [successor_action]
            sCost = problem.getCostOfActions(sPath) + heuristic(successor_state, problem)
            if (successor_state not in visited) and (successor_state not in (y[2][0] for y in fringe.heap)):
                fringe.push((successor_state, sPath), sCost)

            elif successor_state in (y[2][0] for y in fringe.heap):
                for s in fringe.heap:
                    if successor_state == s[2][0]:
                        if s[0] > sCost:
                            fringe.update((successor_state, sPath), sCost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
