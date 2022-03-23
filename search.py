"""
In search.py, you will implement generic search algorithms
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()




def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

	print("Start:", problem.get_start_state().state)
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """
    "*** YOUR CODE HERE ***"
    s = util.Stack()
    d = {}
    visited = set()

    start = problem.get_start_state()
    s.push(start)
    d[start] = None

    while not s.isEmpty():
        u = s.pop()
        # if not u in visited:
        visited.add(u)
        for tup in problem.get_successors(u):
            if tup[0] in visited:
                continue
            a = tup[0]
            d[a] = (u, tup[1])
            s.push(a)

            if problem.is_goal_state(a):
                curr_state = a
                l = []
                while d[curr_state] != None:
                    l.append(d[curr_state][1])
                    curr_state = d[curr_state][0]
                return l[::-1]
    # util.raiseNotDefined()


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"

    parent = dict()
    visited = set()
    queue = []
    queue.append(problem.get_start_state())
    board = queue[0]
    parent[board] = "end"

    while queue:
        curr = queue.pop(0)
        visited.add(curr)
        if problem.is_goal_state(curr):
            break
        for succ in problem.get_successors(curr):
            board = succ[0]
            move = succ[1]
            if board not in visited:
                parent[board] = (curr, move)
                queue.append(board)
    final = []
    while parent[board] != "end":
        final = [parent[board][1]] + final
        board = parent[board][0]

    return final