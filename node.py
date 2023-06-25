import numpy as np
import config

class Node:
    def __init__(self,state,prior,parent=None,is_negating=None):
        self.N = 0 # number of visits
        self.Val = 0 # sum of value
        self.state = state # game state
        self.prior = prior
        self.children = []
        self.parent=parent
        self.is_negating = False if is_negating is None else is_negating

    # return ucb score of node
    def UCB(self,time):
          q = -self.Q() if self.is_negating else self.Q()
          return q + (2 * self.prior * ((np.log(time)/self.N) ** 0.5))

    # return expected value of node
    def Q(self):
          return self.Val/self.N
    
    # expand
    def expand(self,policy_network):
        next_states = config.get_actions(self.state)
        policy = config.get_policy(policy_network,self.state)

        for i,prior in enumerate(policy):
             self.children.append(Node(next_states[i],prior,parent=self,is_negating=not self.is_negating))

    def is_expanded(self):
        return bool(len(self.children))
    
    def is_leaf_node(self):
        return (not self.is_expanded())

    def is_root(self):
        return self.parent is None
    
    # dont call if node is terminal
    def get_next_node(self,policy_network,time):
        # ignore terminal nodes
        if config.is_terminal_state(self.state):
            return self
        # if not expanded expand  
        if not self.is_expanded():
            self.expand(policy_network)

        # calculate ucb scores
        ucbs = [self.UCB(child,time) for child in self.children]
        mX_ucb = -np.inf
        next_node = None

        # get most promising child
        for p,ucb in enumerate(ucbs):
            if ucb > mX_ucb:
                  mX_ucb = ucb
                  next_node = self.children[p]

        return next_node
    
    def backup_rewards(self,value_network):
      # get reward
      reward = config.evaluate_state(value_network,self.state)
      
      # back propagate rewards
      node = self
      while True:
            node.Val += reward
            node.N += 1

            if node.is_root():
                break
            
            node = node.parent
      
      return node
    

    def copy(self):
      node = Node(self.state,self.prior,self.parent)
      node.children = self.children
      node.N = self.N
      node.Val = self.Val

      return node
      
      
            



          


        


