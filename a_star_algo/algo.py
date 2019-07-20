# https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0  # G is the distance between the current node and the start node.
        self.h = 0. # H is the heuristic - estimated distance from the current node to the end node.
                    # using the Eucleandian distance squared : a2 + b2 = c2
        self.f = 0. # F is the total cost of the node.

    def __eq__(self, other):
        return self.position == other.position


def get_node_with_smallest_f(open_list):
    assert open_list != []
    current_node = open_list[0]
    current_index = 0
    for index, item in enumerate(open_list):
        if item.f < current_node.f:
            current_node = item
            current_index = index
    return current_node, current_index

def get_path_to_node(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1] # Return reversed path

def get_children_nodes(current_node, maze):
    # get all the children / neighbors
    children = []
    for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
    # Adjacent squares
        node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
        if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
            continue
        if maze[node_position[0]][node_position[1]] != 0:
            continue
        new_node = Node(current_node, node_position)
        children.append(new_node)
    return children

def AStar(maze, start, end):
    # Returns a list of tuples as a path from the
    # given start to the given end in the given maze

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []       # like a frontier
    closed_list = []     # like where we camre from

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node, current_index = get_node_with_smallest_f(open_list)

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            # print("DONE")
            return get_path_to_node(current_node)

        children = get_children_nodes(current_node, maze)
        children = [c for c in children if c not in closed_list]

        for child in children:
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            open_list.append(child)