from board import Board
from search import SearchProblem, ucs
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
    the heuristic is to take the sum of minimum distances to the corners
    and dividing it by two. It is admissible because we can write a proof
    that at most half of the moves directly to a corner 
    can get us closer to another 2 corners. so 1/2 * 1/4 + 1/2 *3/4 = 1/2
    """
    a = sum(dists) / 2

    return a


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.targets = targets.copy()
        self.expanded = 0
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        "*** YOUR CODE HERE ***"
        for tup in self.targets:
            i, j = tup
            if state.state[i, j] == -1:
                return False

        return True

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
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        "*** YOUR CODE HERE ***"
        sum = 0
        for move in actions:
            sum += move.piece.get_num_tiles()

        return sum


def blokus_cover_heuristic(state, problem):
    "*** YOUR CODE HERE ***"
    max_int = state.board_w * state.board_h  # a number that can be used as max int to the distances in the board
    locations = problem.targets
    dists = new_min_dists(state, locations, max_int)

    a = max(dists)

    return a


def distance(t1, t2):
    return math.sqrt((t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2)


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        for tup in self.targets:
            i, j = tup
            if state.state[i, j] == -1:
                return False

        return True

    def reached_a_target(self, state, targets):
        t = []
        for tup in targets:
            i, j = tup
            if state.state[i, j] != -1:
                t.append(tup)
        return t

    def get_dist_from_targets(self, state, targets):
        """mini = state.board_w * state.board_h
        for targ in targets:
            dist = get_dist_from_targets(state, targ)
            if dist < mini:
                mini = dist
        return mini"""
        max_int = state.board_w * state.board_h  # a number that can be used as max int to the distances in the board
        locations = targets
        dists = [max_int] * len(locations)

        # if the cover is filled:
        for i in range(len(locations)):
            if state.state[locations[i][0], locations[i][1]] != -1:
                dists[i] = 0

        valid_places_list = []
        for i in range(0, state.board_w):
            for j in range(0, state.board_h):
                if state.state[i, j] != -1:
                    valid_places_list.append((j, i))

        for place in valid_places_list:
            for i in range(len(locations)):
                if distance(place, locations[i]) < dists[i]:
                    dists[i] = distance(place, locations[i])

        mini = max_int
        idx = -1
        for i in range(len(locations)):
            if dists[i] == 0:
                continue
            elif dists[i] < mini:
                mini = dists[i]
                idx = i

        return mini, locations[idx]

    def get_dist_from_target(self, state, target):
        """mini = state.board_w * state.board_h
        for targ in targets:
            dist = get_dist_from_targets(state, targ)
            if dist < mini:
                mini = dist
        return mini"""
        max_int = state.board_w * state.board_h
        dists = max_int

        # if the cover is filled:
        if state.state[target[0], target[1]] != -1:
            return 0

        valid_places_list = []
        for i in range(0, state.board_w):
            for j in range(0, state.board_h):
                if state.state[i, j] != -1:
                    valid_places_list.append((j, i))

        for place in valid_places_list:
            if distance(place, target) < dists:
                dists = distance(place, target)
        return dists

    def get_successors(self, state, targets):
        self.expanded = self.expanded + 1
        succs = [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]
        d = {}
        for succ in succs:
            d[succ] = self.get_dist_from_targets(succ[0], targets)[0]
        d = sorted(d, key=d.get)
        return d

    def get_successors_one_target(self, state, target):
        self.expanded = self.expanded + 1
        succs = [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]
        d = {}
        for succ in succs:
            d[succ] = self.get_dist_from_target(succ[0], target)
        d = sorted(d, key=d.get)
        return d

    def solve(self):
        remaining_targets = self.targets.copy()

        s = util.Stack()
        d = {}
        visited = set()
        start = self.get_start_state()
        s.push(start)
        d[start] = None
        while not s.isEmpty():
            u = s.pop()
            # if not u in visited:
            visited.add(u)
            for tup in self.get_successors(u, remaining_targets):
                if tup[0] in visited:
                    continue
                a = tup[0]
                d[a] = (u, tup[1])

                reached = self.reached_a_target(a, remaining_targets)
                s.push(a)

                if reached:
                    temp_reached = [targ for targ in remaining_targets if targ not in reached]
                    if not temp_reached:
                        curr_state = a
                        l = []
                        while d[curr_state] != None:
                            l.append(d[curr_state][1])
                            curr_state = d[curr_state][0]
                        return l[::-1]

    def solved(self):
        remaining_targets = self.targets.copy()

        s = util.Stack()
        d = {}
        visited = set()

        start = self.get_start_state()
        s.push((start, []))
        d[start] = None
        _, target = self.get_dist_from_targets(start, remaining_targets)
        while not s.isEmpty():
            u, reached = s.pop()
            if self.get_dist_from_target(u, target) == 0:
                remaining_targets.remove(target)
                _, target = self.get_dist_from_targets(start, remaining_targets)
            print(remaining_targets)

            # if not u in visited:
            visited.add(u)
            for tup in self.get_successors_one_target(u, target):
                if tup[0] in visited:
                    continue
                a = tup[0]
                d[a] = (u, tup[1])

                reached = self.reached_a_target(a, remaining_targets)
                s.push((a, reached))

                if reached:
                    temp_reached = [targ for targ in remaining_targets if targ not in reached]
                    if not temp_reached:
                        curr_state = a
                        l = []
                        while d[curr_state] != None:
                            l.append(d[curr_state][1])
                            curr_state = d[curr_state][0]
                        return l[::-1]


class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
