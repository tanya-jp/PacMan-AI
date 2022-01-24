# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*
        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def calculateBestQValue(self, state, bestQValue):
        """
        Finds best value of Q based on possible actions of each state
        """
        for action in self.mdp.getPossibleActions(state):
            QValue = self.computeQValueFromValues(state, action)
            bestQValue = max(bestQValue, QValue)
        return bestQValue


    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        # For each k, which means each iteration,
        # this loop computes all of the values for each state
        for each in range(self.iterations):

            updatedValues = util.Counter()

            for s in self.mdp.getStates():
                if not self.mdp.isTerminal(s):
                    updatedValues[s] = -999
                    updatedValues[s] = self.calculateBestQValue(s, updatedValues[s])
            self.values = updatedValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Qvalues = util.Counter()
        # nextState = next possible state by doing this action
        # T = probability of reaching nextState
        for nextState, T in self.mdp.getTransitionStatesAndProbs(state, action):
            R = self.mdp.getReward(state, action, nextState)
            # self.discount = gama
            # self.value is being computed in runValueIteration
            Qvalues[nextState] = T * (R + self.discount*self.values[nextState])
        return Qvalues.totalCount()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Computing action(Vk+1(s)) by using Q values
        actionValues = util.Counter()
        for a in self.mdp.getPossibleActions(state):
            actionValues[a] = self.computeQValueFromValues(state, a)
        return actionValues.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # The only difference between this part and runValueIteration in ValueIterationAgent
        # is that in this part value should be compute for only one state,
        # so iteration should be on states(each time one state)
        s = self.mdp.getStates()
        for each in range(self.iterations):
            # state_num defines the value should be updated for which state
            state_num = each % len(s)
            state = s[state_num]
            if not self.mdp.isTerminal(state):
                updatedQValue = -999
                self.values[state] = self.calculateBestQValue(state, updatedQValue)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # predecessors is a dictionary where each key is a state and the value is a set
        # containing all the predecessors of that state
        # Duplication is controlled here
        predecessors = collections.defaultdict(set)

        minHeap = util.PriorityQueue()

        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
                    # Only the more than zero probability shows that
                    # by this node reaching next considered node is reachable
                    # So only nonzero probability of being reached should be added
                    if probability > 0:
                        predecessors[nextState].add(state)

        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                diff = abs(self.calculateBestQValue(s, -9999) - self.values[s])
                # Because the priority queue is a min-heap, so diff should be negative
                minHeap.push(s, -diff)
        for each in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if minHeap.isEmpty():
                break
            else:
                state = minHeap.pop()
                self.values[state] = self.calculateBestQValue(state, -9999)
                for predecessor in predecessors[state]:
                    diff = abs(self.calculateBestQValue(predecessor, -9999) - self.values[predecessor])
                    if diff > self.theta:
                        minHeap.update(predecessor, -diff)

