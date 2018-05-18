#!/usr/bin/env python

"""
Highly simplified version of the MDP agents described in

Vogel, Adam and Jurafsky, Daniel. 2010. Learning to follow navigational
directions. In Proceedings of the 48th Annual Meeting of the
Association for Computational Linguistics, 806–814. ACL.

The goal is to highlight what agents like this learn and how they
learn it.  For a simple example:

python navagent.py

---Chris Potts
"""

import random
from collections import defaultdict
from math import exp, sqrt
from operator import itemgetter


# Used for informative ASCII map printouts:
PATH_END_SYMBOL = "*"
PATH_WEST_SYMBOL = "<"
PATH_EAST_SYMBOL = ">"
PATH_NORTH_SYMBOL = "^"
PATH_SOUTH_SYMBOL = "v"


class NavigationalAgent:
    def __init__(self,
                 xlim=5,
                 ylim=10,
                 phi=None,
                 actions=('north', 'south', 'east', 'west'),
                 obstacles=[]):
        # These determine the size of the map:
        self.xlim = xlim
        self.ylim = ylim
        # These must be string names of movement methods of the class:
        self.actions = actions
        # Feature function on states; defaults to one that tracks associations
        # between words in utterance and corresponding intended actions:
        if not phi:
            self.phi = self.phi_unigrams
        else:
            self.phi = phi
        # Places in the map where the agent cannot travel:
        self.obstacles = obstacles
        # Initialized at all 0s; to be learned:
        self.weights = defaultdict(float)
        # Parameter for the scoring function:
        self.temperature = 0.25

    ##################################################################
    ## Usage (presumably post-training)
    
    def follow_instructions(self, instructions=['RIGHT', 'RIGHT', 'DOWN'], location=(0,0), display=True, gold=[]):
        """Follow instructions using the learned weights. instructions is a
        list of strings, and location is the (x,y) starting coordinate."""
        path = []
        for utt in instructions:
            action = self.predict(utt)
            next_loc = self.move(location, action)
            direction = self.get_direction(location, next_loc)
            path.append((location, direction))
            location = next_loc
        path.append((next_loc, PATH_END_SYMBOL))
        if display:
            self.display_construal(instructions, path, gold)
        return path

    def get_direction(self, current_location, next_location):
        """ASCII symbols for the current trajectory; used only for map displays."""
        cx, cy = current_location
        nx, ny = next_location
        if cy < ny:   return PATH_EAST_SYMBOL
        elif cy > ny: return PATH_WEST_SYMBOL
        elif cx < nx: return PATH_SOUTH_SYMBOL
        else:         return PATH_NORTH_SYMBOL

    ##################################################################
    ## Training

    def train(self, D=[], T=10):
        """SARSA online learning algorithm. D should be a list of lists of
        (utterance, location) tuples, T is the number of iterations."""
        # Learning rate, defined in terms of T to keep the interface simple:
        eta = sqrt(1.0/T)
        # Begin T iterations:         
        for t in range(T):
            random.shuffle(D)
            for directed_path in D:
                # Initial location and utterance:                
                u, loc = directed_path[0]
                # Select an action based on the random weights and u (unlikely to be smart!):
                a = self.predict(u)
                # Iterate through the nonfinal states:
                for state_index in range(len(directed_path)-2):
                    # Execute the action:
                    next_loc = self.move(loc, a)
                    # Use the actual next utterace to pick the next action:
                    next_u = directed_path[state_index+1][0]
                    next_a = self.predict(next_u)
                    # Gradient: R(s,a) with respect to the gold next state, plus the temporal difference:
                    delta = self.reward(loc, a, directed_path[state_index+1][1]) + self.score(next_u, next_a) - self.score(u, a)
                    # Weight update:
                    for f, val in self.phi(u, a).items():
                        self.weights[f] += eta * val * delta

    ##################################################################
    ## Scoring for best action selection
                        
    def predict(self, u):
        """Pick the best action given the utterance u and the learned weights.
        Ties are resolved by random selection."""
        # List of (score, action) pairs.
        scores = [(self.score(u, a), a) for a in self.actions]
        # Find the highest score at the bottom of the list:
        max_score = sorted(scores)[-1][0]
        # Return a random selection from the highest scoring actions:
        max_actions = [a for v, a in scores if v == max_score]
        return random.choice(max_actions)

    def score(self, u, a):
        """Score an action a given an given an utterance u as the exponentiated
        inner product of the feature representation and the weights, modualted by the
        temperature parameter."""
        return exp(self.temperature * sum(self.weights[f]*val for f, val in self.phi(u, a).items()))

    def phi_unigrams(self, u, a):
        """Simple bag-of-words association between utterances and actions."""
        d = defaultdict(float)
        for word in u.split():
            d[(word, a)] += 1.0
        return d

    def reward(self, loc, a, next_loc):
        """R(s, a) = 1.0 if a(s) puts the agent in the location of s+1, else 0.0"""
        if self.move(loc, a) == next_loc:
            return 1.0
        return 0.0

    ##################################################################
    ## Moving

    def move(self, loc, a):
        """Move by executing action a from the current location."""
        return getattr(self, a)(*loc)
    
    def north(self, x, y):
        if x > 0 and (x-1, y) not in self.obstacles:
            return (x-1, y)
        return (x,y)
    
    def south(self, x, y):
        if x < self.xlim-1 and (x+1, y) not in self.obstacles:
            return (x+1, y)
        return (x,y)
    
    def east(self, x, y):
        if y < self.ylim-1 and (x, y+1) not in self.obstacles:
            return (x, y+1)
        return (x, y)
    
    def west(self, x, y):
        if y > 0 and (x, y-1) not in self.obstacles:
            return (x, y-1)
        return (x, y)

    ##################################################################
    ## Visualization for study

    def print_weights(self, hide_zeros=True):
        tups = sorted(self.weights.items(), key=itemgetter(1), reverse=True)
        if hide_zeros:
            tups = [(x,y) for x,y in tups if y > 0.0]
        for key, val in tups:
            print key, val

    def display_construal(self, instructions, path, gold):
        print self.viz(path=path)
        print 'Instructions:', instructions
        print 'Path:', [x[0] for x in path]
        if gold:
            print 'Gold:', gold
                            
    def viz(self, path=[]):
        initial_position = None
        if path:
            initial_position = path[0][0]
        path_locs = [x[0] for x in path]
        path_dirs = [x[1] for x in path]
        s = ""
        # Top wall:
        s += "+---" * self.ylim + "+\n"
        for i in range(self.xlim):
            s += "|"
            # Rooms:
            for j in range(self.ylim):
                if (i,j) == initial_position:
                    s += " @  "                                               
                elif (i,j) in path_locs:
                    index = path_locs.index((i,j))                  
                    s += " %s  " % path_dirs[index]
                elif (i,j) in self.obstacles:
                    s += "###|"
                elif (i, j+1) in self.obstacles or j == self.ylim-1:
                    s += "   |"
                else:
                    s += "    "
            s += "\n"
            # Walls:
            for j in range(self.ylim):                    
                if (i,j) in self.obstacles or i == self.xlim-1 or (i+1, j) in self.obstacles:
                    s += "+---"
                else:
                    s += "+   "            
            s += "+\n"
        return s

######################################################################
            
if __name__ == '__main__':
                
    training_set = [
        [('RIGHT', (2,2)), ('RIGHT', (2,3)), ('', (2,4))],
        [('LEFT', (2,2)),  ('LEFT', (2,1)),  ('', (2,0))],
        [('DOWN', (2,2)),  ('DOWN', (3,2)),  ('', (4,2))],
        [('UP', (2,2)),    ('UP', (1,2)),    ('', (0,2))]        
    ]
        
    agent = NavigationalAgent()
    agent.train(D=training_set)
    
    print "======================================================================"
    print 'Learned weights'
    agent.print_weights(hide_zeros=False)

    print "======================================================================"
    print "Test runs"
    
    test_set = [
        ['RIGHT', 'RIGHT', 'DOWN', 'DOWN'],
        ['MOVE RIGHT', 'GO RIGHT', 'RIGHT AGAIN', 'DOWN MAN!', 'LEFT', 'LEFT AGAIN SPORT!'],
        ['DOWN', 'DOWN', 'RIGHT', 'RIGHT', 'RIGHT', 'UP', 'UP', 'RIGHT'],
        ['RIGHT', 'RIGHT', 'DOWN', 'RIGHT', 'DOWN', 'RIGHT', 'RIGHT']  
    ]
    
    for instructions in test_set:
        print '--------------------------------------------------'
        agent.follow_instructions(instructions=instructions, location=(0,0), display=True)
   
