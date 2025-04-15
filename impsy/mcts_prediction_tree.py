import numpy as np
import time
from typing import List, Tuple, Optional, Callable
import math

class MCTSNode:
    def __init__(self, output: np.ndarray, parent=None, lstm_states=None):
        self.output = output
        self.parent = parent
        self.lstm_states = lstm_states
        self.gmm = None
        self.children = []
        
        # MCTS specific attributes
        self.visits = 0
        self.value = 0.0
        self.untried_actions = []  # Will be populated with samples from GMM
        
    def add_child(self, output: np.ndarray, lstm_states) -> 'MCTSNode':
        """Add a child node with the given output and return it"""
        child = MCTSNode(output, parent=self, lstm_states=lstm_states)
        self.children.append(child)
        return child
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried"""
        return len(self.untried_actions) == 0
    
    def best_child(self, exploration_weight: float) -> 'MCTSNode':
        """
        Select the best child according to UCT formula:
        UCT = value/visits + exploration_weight * sqrt(2 * ln(parent_visits) / visits)
        """
        if not self.children:
            return None
        
        uct_values = []
        for child in self.children:
            # Exploitation term
            exploitation = child.value / child.visits if child.visits > 0 else 0
            
            # Exploration term
            exploration = exploration_weight * math.sqrt(2 * math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
            
            uct_values.append(exploitation + exploration)
        
        return self.children[np.argmax(uct_values)]


class MCTSPredictionTree:
    def __init__(self, 
                 max_simulation_depth: int = 2, 
                 branching_factor: int = 3, 
                 exploration_weight: float = 1.0,
                 initial_lstm_states=None):
        self.root = None
        self.initial_lstm_states = initial_lstm_states
        self.max_simulation_depth = max_simulation_depth
        self.branching_factor = branching_factor
        self.exploration_weight = exploration_weight
        self.best_branch = (None, None, float('-inf'))
        self.nodes_searched = 0
    
    def search(self, 
               memory: List[np.ndarray], 
               predict_function: Callable, 
               sample_function: Callable,
               heuristic_function: Callable,
               time_limit_ms: int = 1000) -> Tuple[np.ndarray, List, float]:
        """
        Run MCTS for the given time limit and return the best branch found
        
        Args:
            memory: Initial sequence of notes
            predict_function: Function to predict next note probabilities
            sample_function: Function to sample from GMM
            heuristic_function: Function to evaluate a branch
            time_limit_ms: Time limit in milliseconds
            
        Returns:
            Tuple of (best branch, lstm states, score)
        """
        # Reset statistics
        self.nodes_searched = 0
        self.best_branch = (None, None, float('-inf'))
        
        # Create nodes for the entire memory
        nodes = [MCTSNode(output) for output in memory]
        for i in range(1, len(nodes)):
            nodes[i].parent = nodes[i-1]
            nodes[i-1].children.append(nodes[i])

        # Print the values of all memory nodes
        for node in nodes:
            print(f"Node output: {node.output}, visits: {node.visits}")
        
        self.root = nodes[0]
        
        # Set up the initial state
        current_node = nodes[-1]
        
        # Get initial GMM for root - only call predict_function once
        current_node.gmm, current_node.lstm_states = predict_function(memory[-1], lstm_states=self.initial_lstm_states)
        
        # Initialize untried actions with samples from GMM
        current_node.untried_actions = [sample_function(current_node.gmm) for _ in range(self.branching_factor)]
        
        # Set up time limit
        end_time = time.time() + (time_limit_ms / 1000.0)
        
        # MCTS main loop
        while time.time() < end_time:
            # Selection and Expansion
            selected_node, new_memory = self._select_and_expand(current_node, memory[1:], predict_function, sample_function)
            
            # Simulation
            simulation_result = self._simulate(selected_node, new_memory, predict_function, sample_function, heuristic_function)
            
            # Backpropagation
            self._backpropagate(selected_node, simulation_result)
        
        # Return best branch after time is up
        best_child = self._get_best_child(current_node)
        if best_child:
            branch, lstm_branch = self._extract_branch(best_child)
            branch_score = heuristic_function(branch)
            self.best_branch = (branch, lstm_branch, branch_score)
        
        return self.best_branch

    def _select_and_expand(self, node: MCTSNode, memory: List[np.ndarray], 
                           predict_function: Callable, sample_function: Callable) -> Tuple[MCTSNode, List[np.ndarray]]:
        """
        Select a node to expand using UCT and expand it
        Returns the newly expanded node and the current memory state
        """
        self.nodes_searched += 1
        current = node
        current_memory = memory.copy()
        
        # Selection phase - navigate down the tree until we reach a node that isn't fully expanded
        while current.is_fully_expanded() and current.children:
            current = current.best_child(self.exploration_weight)
            current_memory = current_memory[1:] + [current.output]
            self.nodes_searched += 1
        
        # Expansion phase - if we can expand, pick an untried action and create a new node
        if not current.is_fully_expanded():
            action = current.untried_actions.pop(0)
            
            # Create new child with parent's lstm_states (will be updated during simulation)
            new_node = current.add_child(action, current.lstm_states)
            
            # Update memory with the new action
            current_memory = current_memory[1:] + [action]
            
            # During the simulation step, we'll update the GMM and lstm_states for this node
            return new_node, current_memory
        
        return current, current_memory
    
    def _simulate(self, node: MCTSNode, memory: List[np.ndarray], 
                 predict_function: Callable, sample_function: Callable, 
                 heuristic_function: Callable) -> float:
        """
        Run a simulation from the given node to estimate its value
        """
        current = node
        current_memory = memory.copy()
        depth = 0
        
        # Important: Get the GMM and lstm_states for the current node if not already computed
        # This avoids the duplicate predict_function call in the expansion phase
        if current.gmm is None:
            current.gmm, current.lstm_states = predict_function(current_memory[-1], lstm_states=current.parent.lstm_states)
            # Pre-populate untried actions for future expansions
            if not current.untried_actions:
                current.untried_actions = [sample_function(current.gmm) for _ in range(self.branching_factor)]
        
        lstm_states = current.lstm_states
        
        # Continue simulation until we reach max depth
        simulation_outputs = []
        while depth < self.max_simulation_depth:
            # Predict next node - single call to get both GMM and next LSTM state
            gmm, lstm_states = predict_function(current_memory[-1], lstm_states=lstm_states)
            action = sample_function(gmm)
            
            # Update memory
            current_memory = current_memory[1:] + [action]
            simulation_outputs.append(action)
            depth += 1
        
        # Evaluate the final state using heuristic function
        full_branch = self._extract_branch(node)[0]  # Get the full path
        if simulation_outputs:
            full_branch = np.concatenate([full_branch, np.array(simulation_outputs)])  # Add simulation path
        
        return heuristic_function(full_branch)
    
    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        """Update node statistics going up the tree"""
        current = node
        while current:
            current.visits += 1
            current.value += result
            current = current.parent
    
    def _extract_branch(self, node: MCTSNode) -> Tuple[np.ndarray, List]:
        """Extract the complete branch from root to the given node"""
        output_branch = []
        lstm_states_branch = []
        current = node
        
        while current:
            output_branch.append(current.output)
            lstm_states_branch.append(current.lstm_states)
            current = current.parent
        
        return np.array(output_branch[::-1]), lstm_states_branch[::-1]
    
    def _get_best_child(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Get the child with the highest visit count (most robust choice)"""
        if not node.children:
            return None
        
        visit_counts = [child.visits for child in node.children]
        return node.children[np.argmax(visit_counts)]
    
    def get_num_nodes(self) -> int:
        """Return the total number of nodes searched during the MCTS process"""
        return self.nodes_searched
    
    def get_highest_heuristic_branch(self) -> np.ndarray:
        """
        Return the branch with the highest heuristic score found during MCTS
        """
        # Start at root node
        current = self.root
        # Recrusively find the best child node until finding a child with no children
        while current.children:
            best_child = max(current.children, key=lambda child: child.visits)
            current = best_child

        # Extract the branch from the best child node
        branch, _ = self._extract_branch(current)
        return branch
    
    def graph_tree(self, ax=None, title="Monte Carlo Tree Search Visualization"):
        """
        Visualize the MCTS tree in 3D.
        
        Args:
            ax: Optional matplotlib 3D axis. If None, a new figure and axis will be created.
            title: Title for the plot.
            
        Returns:
            The matplotlib figure and axis.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        # Create figure and 3D axis if not provided
        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure
        
        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Pitch')
        ax.set_zlabel('Tree Depth')
        ax.set_title(title)
        
        # Extract winning branch for highlighting
        winning_branch = self.get_highest_heuristic_branch()
        print(f"Winning branch: {winning_branch}")
        
        # Recursively plot nodes and edges
        def plot_node(node, depth=0, parent_pos=None):
            if node is None:
                return
            
            # If this node has less than 5 visits and is not at the root, skip it
            if node.visits < 5 and depth > 0:
                return
            
            # Skip root nodes that don't have music output data
            if depth < 2 and (node.output is None or len(node.output) < 2):
                for child in node.children:
                    plot_node(child, depth + 1)
                return
                    
            # Extract x (time) and y (pitch) from node output
            # Ensure we have at least 2 elements
            if node.output is None or len(node.output) < 2:
                x, y = 0, 0  # Default values if output doesn't have enough elements
            else:
                x, y = node.output[0], node.output[1]
            
            # Position in 3D space
            pos = (x, y, depth)
            print(f"Node position: {pos}, Visits: {node.visits}")
            
            # Check if this node is part of the winning branch
            is_winning = node.output in winning_branch if node.output is not None else False
            
            # Plot node
            if node.visits > 0:  # Only plot nodes that have been visited
                # Normalize size (min size 20, max size 200)
                node_size = 20 + 20 * math.pow(node.visits, 0.4)
                
                # Color based on whether it's part of the winning branch
                node_color = 'yellow' if is_winning else 'skyblue'
                
                # Plot the node
                ax.scatter(x, y, depth, s=node_size, c=node_color, edgecolor='black', alpha=0.7)
                
                # Add text annotation with visits count
                ax.text(x, y, depth, f"{node.visits}", fontsize=8)
                
                # Draw edge to parent
                if parent_pos is not None:
                    # Normalize edge width
                    edge_width = 1 + math.pow(node.visits, 0.3)
                    
                    # Line coordinates
                    xs = [parent_pos[0], x]
                    ys = [parent_pos[1], y]
                    zs = [parent_pos[2], depth]
                    
                    # Edge color
                    edge_color = 'gold' if is_winning else 'gray'
                    
                    # Plot edge
                    ax.plot(xs, ys, zs, linewidth=edge_width, color=edge_color, alpha=0.6)
            
            # Recursively plot children
            for child in node.children:
                plot_node(child, depth + 1, pos)
        
        # Start plotting from root
        plot_node(self.root)
        
        # Add legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle
        
        legend_elements = [
            Circle((0, 0), 0.1, facecolor='skyblue', edgecolor='black', label='Regular Node'),
            Circle((0, 0), 0.1, facecolor='yellow', edgecolor='black', label='Winning Branch'),
            Line2D([0], [0], color='gray', lw=2, label='Regular Edge'),
            Line2D([0], [0], color='gold', lw=2, label='Winning Path')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Enable grid
        ax.grid(True)
        
        # Auto-adjust view
        ax.view_init(elev=30, azim=30)

        # Adjust Z axis ticks and direction
        z_min, z_max = ax.get_zlim()
        ax.set_zticks(range(int(z_min), int(z_max) + 1))
        ax.invert_zaxis()  # Makes depth 0 at the top
        
        plt.tight_layout()
        return fig, ax