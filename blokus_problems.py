from board import Board
from search import SearchProblem, ucs, astar
import util
import math
import numpy as np


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)


class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.targets = [(0, 0), (0, board_w - 1), (board_h - 1, 0), (board_h - 1, board_w - 1)]
        # removing the starting point from the targets
        if starting_point in self.targets:
            self.targets.remove(starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        # checking if all the targets are filled
        value = all([state.state[loc[0], loc[1]] != -1 for loc in self.targets])
        return value

    def get_successors(self, state):
        """
        we used the same function as yours
        :param state: the curr state
        :return: all the successors
        """
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        the cost of all the actions in a list of actions
        :param actions: the list of actions
        :return: the cost
        """
        sum = 0
        for move in actions:
            sum += move.piece.get_num_tiles()
        return sum


def new_min_dists(state, locations, max_int):
    """
    finding the min straight-line-distance to each location from the closest filled location
    :param state: the current state
    :param locations: the locations to reach
    :param max_int: the maimum distance possible
    :return: a list of min distances
    """
    points = np.asarray(np.where(state.state != -1))
    dists = [max_int] * len(locations)
    for i in range(len(locations)):
        dist_2 = np.sum([(points[0] - locations[i][0]) ** 2, (points[1] - locations[i][1]) ** 2], axis=0)
        dists[i] = math.sqrt(np.min(dist_2))

    # if one of the locations that is adjacent to the target
    # is filled there is no solution, so we make the heuristic distance
    # high because the actual distance is infinity
    w, h = state.board_w, state.board_h
    for i in range(len(locations)):
        i, j = locations[i]
        if state.state[i,j] == 0:
            continue
        if i-1 >= 0 and state.state[i-1,j] == 0:
            return [max_int*10]
        if i+1 < w and state.state[i+1,j] == 0:
            return [max_int*10]
        if j-1 >= 0 and state.state[i,j-1] == 0:
            return [max_int*10]
        if j+1 < h and state.state[i,j+1] == 0:
            return [max_int*10]
    return dists


def blokus_corners_heuristic(state, problem):
    """
    the corners problem heuristic.
    :param state: the current state
    :param problem: the problem
    :return: the heuristic guess
    """
    max_int = state.board_w * state.board_h  # a number that can be used as max int to the distances in the board
    locations = problem.targets
    dists = new_min_dists(state, locations, max_int)

    """
    the heuristic is to take the max between the sum of minimum
    distances to the corners and dividing it by two and the max distance. 
    It is admissible because we can write a proof
    that at most half of the moves directly to a corner 
    can get us closer to another 2 corners. so 1/2 * 1/4 + 1/2 *3/4 = 1/2.
    and the max distance is obviously admissible because
    we must go at least the minimum distance to the corner.
    
    We must admit that the heuristic is not consistent, however we are not
    required to supply a consistent heuristic AND we are in a tree search
    so the optimality is still correct because it requires only admissible functions
    """
    a = max(sum(dists) / 2, max(dists))

    return a


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.targets = targets.copy()
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        we have reached the goal if all the targets are filled
        :param state: the current state
        :return: whether it is a goal state
        """
        for tup in self.targets:
            i, j = tup
            if state.state[i, j] == -1:
                return False

        return True

    def get_successors(self, state):
        """
        we used the same function as yours
        :param state: the curr state
        :return: all the successors
        """
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        the cost of all the actions in a list of actions
        :param actions: the list of actions
        :return: the cost
        """
        sum = 0
        for move in actions:
            sum += move.piece.get_num_tiles()

        return sum


def blokus_cover_heuristic(state, problem):
    """
    the cover problem heuristic.
    :param state: the current state
    :param problem: the problem
    :return: the heuristic guess
    """
    max_int = state.board_w * state.board_h  # a number that can be used as max int to the distances in the board
    locations = problem.targets
    dists = new_min_dists(state, locations, max_int)
    """
    the heuristic is to take the max of minimum distances to the corners.
    It is admissible because we can't get to a corner in less than the min distance to it
    and it is obviously consistent because a real step can get us closer at most the 
    straight-line-distance
    """
    a = max(dists)

    """
    next we added the basic heuristic for targets that are not filled,
    and in the final heuristic we will take the max between this and the 
    other on (max(dists))
    """
    not_filled = 0
    for i,j in problem.targets:
        if state.state[i,j] == -1:
            not_filled += 1

    return max(a, not_filled)


def distance(t1, t2):
    """
    a distance from point to point
    :param t1: the first point
    :param t2: the second point
    :return: the distance
    """
    return math.sqrt((t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2)


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0
        self.starting_point = starting_point
        self.targets = targets.copy()
        self.curr_target = None
        self.curr_state_finished_last = self.board
        self.start = (0,0)
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.curr_state_finished_last

    def get_cost_of_actions(self, actions):
        """
        the cost of all the actions in a list of actions
        :param actions: the list of actions
        :return: the cost
        """
        sum = 0
        for move in actions:
            sum += move.piece.get_num_tiles()
        return sum

    def is_goal_state(self, state):
        i, j = self.curr_target
        if state.state[i, j] == -1:
            return False
        self.curr_state_finished_last = state
        return True

    def get_dist_from_targets(self, state, targets):
        """mini = state.board_w * state.board_h
        for targ in targets:
            dist = get_dist_from_targets(state, targ)
            if dist < mini:
                mini = dist
        return mini"""
        max_int = state.board_w * state.board_h  # a number that can be used as max int to the distances in the board
        points = np.asarray(np.where(state.state != -1))
        targets = np.array(targets)
        if points.shape[1] == 0:
            points = np.array([[self.starting_point[0]], [self.starting_point[1]]])
        dists = [max_int] * len(targets)
        for i in range(len(targets)):
            dist_2 = np.sum([(points[0] - targets[i][0]) ** 2, (points[1] - targets[i][1]) ** 2], axis=0)
            dists[i] = math.sqrt(np.min(dist_2))
        idx = np.argmin(dists)

        return dists, targets[idx]

    def get_successors(self, state):
        """
        we used the same function as yours
        :param state: the curr state
        :return: all the successors
        """
        self.expanded = self.expanded + 1
        succs = [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

        # we added a little upgrade to the algorithm and we return the successors
        # sorted by which one is closest to the target. It improves the amount
        # of expanded nodes and even gives us better solutions.
        d = {}
        for succ in succs:
            d[succ] = self.get_dist_from_targets(succ[0], [self.curr_target])[0][0]
        d = sorted(d, key=d.get)
        return d
        # return sorted([(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)])

    def solve(self):
        back = []
        curr_state = self.get_start_state()
        while self.targets:
            print(self.start)
            _, targ = self.get_dist_from_targets(curr_state, self.targets)
            self.curr_target = targ
            moves = ucs(self)
            curr_state = self.curr_state_finished_last
            back += moves
            self.targets.remove((targ[0], targ[1]))
        self.curr_state_finished_last = self.board
        return back


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0
        self.starting_point = starting_point
        self.targets = targets.copy()

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def get_successors(self, state):
        """
        we used the same function as yours
        :param state: the curr state
        :return: all the successors
        """
        self.expanded = self.expanded + 1
        succs = [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]
        return succs

    def get_cost_of_actions(self, actions):
        """
        the cost of all the actions in a list of actions
        :param actions: the list of actions
        :return: the cost
        """
        sum = 0
        for move in actions:
            sum += move.piece.get_num_tiles()
        return sum

    def is_goal_state(self, state):
        for i, j in self.targets:
            if state.state[i, j] == -1:
                return False
        return True

    def solve(self):
        "*** YOUR CODE HERE ***"
        return astar(problem=self,heuristic=blokus_cover_heuristic)
