from node import Node
import networks
from tqdm import tqdm
import config

class AlphaZero:
      def __init__(self,policynet,valuenet,num_steps,num_simulations,root_state):
            self.policy_net = policynet
            self.value_net = valuenet
            self.num_steps = num_steps
            self.num_simulations = num_simulations
            self.root_state = root_state
      
      def start(self):
            for step in range(1,self.num_steps+1):
                  preformance, visited = self.mcts()

                  state_val_pairs, state_dist_pairs = self.get_training_data(visited)
                  self.train(state_val_pairs, state_dist_pairs)

                  print ('step:',step,'preformace:',preformance)

      def mcts(self):
            root_node = Node(state=self.root_state,prior=None) # define root node
            visited = []
            for simulation in tqdm(range(1,self.num_simulations+1), desc='simulating '):
                  # play simulation from root
                  node = root_node            
                  # progress until leaf node
                  while not node.is_leaf_node():
                        if node not in visited:
                              visited.append(node)
                        node = node.get_next_node(self.policy_net, simulation)
                  # expand leaf node
                  node = node.get_next_node(self.policy_net, simulation)
                  # backpropagate rewards
                  root_node = node.backup_rewards(self.value_net)

            return config.evaluate_preformance(self.policy_net, self.value_net), visited
      
      def get_training_data(self,visited):
            state_val_pairs = [[visited[i].state,visited[i].Q] for i in range(len(visited))]
            state_dist_pairs = []
            for node in visited:
              visit_counts = [child.N for child in node.children]
              state_dist_pairs.append([node.state,networks.softmax(visit_counts)])
            
            return state_val_pairs, state_dist_pairs
      
      def train(self,state_val_pairs,state_dist_pairs):
        x_val,y_val = self.split(state_val_pairs)
        x_dist,y_dist = self.split(state_dist_pairs)

        self.value_net.train(x_val,y_val)
        self.state_dist_pairs.train(x_dist,y_dist)
                        

