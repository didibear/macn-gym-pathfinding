import gym
import gym_pathfinding

from gym_pathfinding.games.gridworld import generate_grid, MOUVEMENT
from astar import astar
from tqdm import tqdm
import numpy as np
import operator
import itertools

def generate_dataset(size, shape, *, grid_type="free", verbose=False):
    """
    Arguments
    ---------
    size : number of training set generated
    shape : the grid shape
    grid_type : the type of grid ("free", "obstacle", "maze")

    Return
    ------
    return images, S1s, S2s, labels

    image : (m, n, 2) grid with state and goal on the 3rd axis
        state = (m, n) grid with 1 and 0 ;
        goal = (m, n) grid with 10 at goal position

    S1 : vertical position of the player
    S2 : horizontal position of the player
    label : the action made
    """
    if verbose: progress_bar = tqdm(total=size)

    images = []
    S1s = []
    S2s = []
    labels = []

    n = 0

    while True:

        grid, start, goal = generate_grid(shape, grid_type=grid_type)
        path, action_planning = compute_action_planning(grid, start, goal)

        goal_grid = create_goal_grid(grid.shape, goal)
        image = np.stack([grid, goal_grid], axis=2)

        for action, position in zip(action_planning, path):
            images.append(image)
            S1s.append(position[0])
            S2s.append(position[1])
            labels.append(action)

            if verbose : progress_bar.update(1)

            n += 1 
            if n >= size:
                if verbose : progress_bar.close()
                return images, S1s, S2s, labels

# reversed MOUVEMENT dict
ACTION = {mouvement: action for action, mouvement in dict(enumerate(MOUVEMENT)).items()}

def compute_action_planning(grid, start, goal):
    path = astar(grid, start, goal)

    action_planning = []
    for i in range(len(path) - 1):
        pos = path[i]
        next_pos = path[i+1]
        
        # mouvement = (-1, 0), (1, 0), (0, -1), (0, 1)
        mouvement = tuple(map(operator.sub, next_pos, pos))

        action_planning.append(ACTION[mouvement])
        
    return path, action_planning


def create_goal_grid(shape, goal):
    goal_grid = np.zeros(shape, dtype=np.int8)
    goal_grid[goal] = 10
    return goal_grid



def main():
    import joblib
    import argparse

    parser = argparse.ArgumentParser(description='Generate data (images, S1s, S2s, labels)')
    parser.add_argument('--out', '-o', type=str, default='./data/dataset.pkl', help='Path to save the dataset')
    parser.add_argument('--size', '-s', type=int, default=100000, help='Number of example')
    parser.add_argument('--shape', type=int, default=[9, 9], nargs=2, help='Shape of the grid (e.g. --shape 9 9)')
    parser.add_argument('--grid_type', type=str, default='obstacle', help='Type of grid : "free", "obstacle" or "maze"')
    args = parser.parse_args()

    dataset = generate_dataset(args.size, args.shape, 
        grid_type=args.grid_type, verbose=True
    )


    print("saving data into {}".format(args.out))

    # np.save(args.out, dataset)
    joblib.dump(dataset, args.out)

    print("done")

if __name__ == "__main__":
    main()
