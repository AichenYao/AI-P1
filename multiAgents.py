# multiAgents.py
# --------------
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


# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        # If we find a food close to us, then base up
        # If closer to ghost, then base down
        # If newScaredTimes is bigger, then base up
        
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newX, newY = newPos
        oldFood = currentGameState.getFood()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newGhosts = [ghostState.getPosition() for ghostState in newGhostStates]
        newFoodCount = newFood.count(True)
        oldFoodCount = oldFood.count(True)
        base = 0
        walls = successorGameState.getWalls()
        if (walls[newX][newY] == True):
            # don't hit a wall
            base -= 1

        for ghost in newGhosts:  
            # use the exponential function so that the closer we are to a food,
            # the more we subtract from base
            curr_d = manhattanDistance(ghost, newPos)
            base -= math.exp(-curr_d+2)

        if (newFoodCount == oldFoodCount):  
            # if we did not eat a food, then we need to subtract the distance
            # from the closest food 
            dis = -1
            for food in newFood.asList():
                curr_d = manhattanDistance(food, newPos)
                if (dis == -1 or curr_d < dis):
                    dis = curr_d
        else:
            dis = 0
        base -= dis
        return base


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 7)
    """

    def getAction(self, gameState):
        "*** YOUR CODE HERE ***"
        num_ghosts = gameState.getNumAgents()-1
        def max_value(state, curr_depth):
            #only Pacman will take the max, so no need to pass in agentIndex here
            #it will always be 0
            actions = gameState.getLegalActions(0)
            # if there are no more legal actions or if we have reached the
            # designated depth, then this state is a leaf
            if (curr_depth == self.depth or len(actions) == 0):
                print("actions:")
                print((len(actions) == 0))
                return self.evaluationFunction(state)
            best_value = -10000000
            for action in actions:
                next_state = gameState.generateSuccessor(0, action)
                next_value = min_value(next_state, curr_depth, 1)
                if next_value > best_value:
                    best_value = next_value
            return best_value
        
        def min_value(state, curr_depth, agentIndex):
            actions = gameState.getLegalActions(agentIndex)
            if (curr_depth == self.depth or len(actions) == 0):
                print("actions:")
                print((len(actions) == 0))
                return self.evaluationFunction(state)
            if (agentIndex == num_ghosts):
                # if all ghosts have taken a step for this turn (layer), 
                # then Pacman shall move
                best_min_value = 10000000
                for action in actions:
                    next_state = gameState.generateSuccessor(agentIndex, action)
                    next_value = max_value(next_state, curr_depth+1)
                    if next_value < best_min_value:
                        best_min_value = next_value
                return best_min_value
            else:
                best_min_value = 10000000
                for action in actions:
                    # we want to next ghost to move on this current depth
                    next_state = gameState.generateSuccessor(agentIndex, action)
                    next_value = min_value(next_state, agentIndex+1, curr_depth)
                    if next_value < best_min_value:
                        best_min_value = next_value
                return best_min_value
            
        dic = {}
        #(action: evalation score)
        actions = gameState.getLegalActions(0)
        for action in actions:
            next_state = gameState.generateSuccessor(0, action)
            next_value = min_value(next_state,0,1)
            dic[action] = next_value
        max_res = max(dic.values())
        for key in dic:
            if (dic[key] == max_res):
                return key
        return actions[0]
        #cite https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        #util.raiseNotDefined()
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 8)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # print("total depth:")
        # print(self.depth)
        # print("total ghosts:")
        # print(num_ghosts)
        def max_value(state, curr_depth):
            actions = state.getLegalActions(0)
            if (curr_depth == self.depth-1 or len(actions) == 0):
                return self.evaluationFunction(state)
            best_value = -math.inf
            for action in actions:
                next_state = state.generateSuccessor(0,action)
                best_value = max(best_value,exp_value(next_state,curr_depth+1,1))
            return best_value

        def exp_value(state, curr_depth, agentIndex):
            actions = state.getLegalActions(agentIndex)
            if (len(actions) == 0):
                return self.evaluationFunction(state)
            value_total = 0
            count = len(actions)
            for action in actions:
                next_state = state.generateSuccessor(agentIndex, action)
                if (agentIndex != state.getNumAgents()-1):
                    cur_value = exp_value(next_state,curr_depth, agentIndex+1)
                else:
                    cur_value = max_value(next_state, curr_depth)
                value_total += cur_value
            if (count != 0):
                return float(value_total)/float(count)
            return 0
        
        actions = gameState.getLegalActions(0)
        best_score = -math.inf
        best_action = None
        for action in actions:
            next_state = gameState.generateSuccessor(0,action)
            cur_score = exp_value(next_state,0,1)
            if (cur_score > best_score):
                best_score = cur_score
                best_action = action
        return best_action
        #util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 9).
    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newX, newY = newPos
    oldFood = currentGameState.getFood()
    #newFood = successorGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newGhosts = [ghostState.getPosition() for ghostState in newGhostStates]
    #newFoodCount = newFood.count(True)
    oldFoodCount = oldFood.count(True)
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    base = 0
    base -= oldFoodCount
    base += sum(newScaredTimes)
    base += currentGameState.getScore()
    
    walls = currentGameState.getWalls()
    if (walls[newX][newY] == True):
        base -= 1

    for ghost in newGhosts: 
        curr_d = manhattanDistance(ghost, newPos)
        base -= 2**(-curr_d+2)
 
    dis = 0
    for food in oldFood.asList():
        curr_d = manhattanDistance(food , newPos)
        if (dis == 0 or curr_d < dis):
            dis = curr_d
    base -= dis

    return base


# Abbreviation
better = betterEvaluationFunction
