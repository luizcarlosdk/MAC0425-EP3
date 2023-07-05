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
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
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

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"

#              Some useful mdp methods you will use:
#              mdp.getStates()
#              mdp.getPossibleActions(state)
#              mdp.getTransitionStatesAndProbs(state, action)
#              mdp.getReward(state, action, nextState)
#              mdp.isTerminal(state)

       
        #vk+1() = max(do estado) somatoria T(s,a,s')[R(s,a,s') + disconto * estado anterior(atual na primeira iteração)]
        iterações = self.iterations
        mdp = self.mdp
        disconto = self.discount


        for iteração in range(iterações):
            valores = util.Counter()
            estados = mdp.getStates()

            for estado in estados:
                if mdp.isTerminal(estado):
                    valores[estado] = 0
                else:
                    maxValue = float("-inf")
                    CurrentStateActions = mdp.getPossibleActions(estado)
                    for ação in CurrentStateActions:
                        
                        currentValue = self.computeQValueFromValues(estado, ação)

                        if(currentValue > maxValue):
                            maxValue = currentValue
                    valores[estado] = maxValue

            self.values = valores

        return self.values
                        


            # for cada estado do getStates atual
            #   transição* [Recompensa do estado atual até o proximo + disconto* estado anterior)]
            
            # if currentState > vk+1  (aq pega o max)
            #   vk+1 = currentState 

        

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
        result = 0
        mdp = self.mdp
        disconto = self.discount
        for next, prob in self.mdp.getTransitionStatesAndProbs(state,action):
            result += prob * (mdp.getReward(state, action, next) + (disconto * self.values[next]))

        return result

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.mdp.getPossibleActions(state)
        
        value = float("-inf")
        result = None
        for action in actions:
            currentValue = self.computeQValueFromValues(state, action)
            if currentValue > value:
                value = currentValue
                result = action

        return result

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
