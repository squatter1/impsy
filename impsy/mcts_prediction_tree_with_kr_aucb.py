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
        self.failed_progressive_widening = 0  # Count of failed progressive widening attempts, too many leads to assumption all valid unique children have been added
        
    def add_child(self, output: np.ndarray) -> 'MCTSNode':
        """Add a child node with the given output and return it"""
        child = MCTSNode(output, parent=self)
        self.children.append(child)
        return child
    
    def best_child(self, exploration_weight: float, verbose: bool = False) -> Optional['MCTSNode']:
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
            
            if verbose:
                print(f"Child output: {child.output}, Exploitation: {exploitation}, Exploration: {exploration}")
            uct_values.append(exploitation + exploration)
        
        if verbose:
            print("FINAL UCT SELECTION: ", self.children[np.argmax(uct_values)].output)
        return self.children[np.argmax(uct_values)]
    
    def most_visited_child(self) -> Optional['MCTSNode']:
        """Return the child with the most visits"""
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.visits)  

    def get_all_actions(self) -> List[np.ndarray]:
        """Get all actions tried from this node (for kernel regression)"""
        return [child.output for child in self.children]
    
    def get_all_values(self) -> List[float]:
        """Get all values for actions tried from this node (for kernel regression)"""
        return [child.value / child.visits if child.visits > 0 else 0 for child in self.children]
    
    def get_all_visits(self) -> List[int]:
        """Get all visit counts for actions tried from this node (for kernel density)"""
        return [child.visits for child in self.children] 


class MCTSPredictionTree:
    def __init__(self, 
                 initial_lstm_states: np.ndarray,
                 simulation_depth: int = 2, 
                 exploration_weight: float = 1.0,
                 progressive_widening_k: float = 1.0,
                 progressive_widening_alpha: float = 0.5,
                 starting_originality_distances: np.ndarray = np.array([0.12, 0.01]),
                 min_originality_distances: np.ndarray = np.array([0.03, 0.005]),
                 snap_to_semitones: bool = True,
                 max_samples_for_originality: int = 10,
                 # Parameters for KR-AUCB
                 kr_lambda_start: float = 0.0,
                 kr_lambda_target: float = 0.8,
                 kr_lambda_schedule_iters: int = 1000):
        self.root = None
        self.initial_lstm_states = initial_lstm_states
        self.simulation_depth = simulation_depth
        self.exploration_weight = exploration_weight
        self.progressive_widening_k = progressive_widening_k
        self.progressive_widening_alpha = progressive_widening_alpha
        self.starting_originality_distances = starting_originality_distances
        self.min_originality_distances = min_originality_distances
        self.snap_to_semitones = snap_to_semitones
        self.max_samples_for_originality = max_samples_for_originality

        # KR-AUCB parameters
        self.kr_lambda_start = kr_lambda_start
        self.kr_lambda_target = kr_lambda_target
        self.kr_lambda_schedule_iters = kr_lambda_schedule_iters
        self.kr_lambda = kr_lambda_start  # Current lambda value (will increase over time)

        self.verbose = True  # Set to True for verbose output
        
        self.heuristic_function = None  # TODO delete
    
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
        self.heuristic_function = heuristic_function # TODO delete
        # Reset statistics
        self.nodes_searched = 0
        self.branches_simulated = 0
        self.kr_lambda = self.kr_lambda_start
        
        # Create nodes for the entire memory
        nodes = [MCTSNode(output) for output in memory]
        for i in range(1, len(nodes)):
            nodes[i].parent = nodes[i-1]
            nodes[i-1].children.append(nodes[i])
        
        self.root = nodes[0]
        
        # Set up the initial state
        initial_node = nodes[-1]
        
        # Get the GMM for the root
        initial_node.gmm, initial_node.lstm_states = predict_function(initial_node.output, init_lstm_states=self.initial_lstm_states)
        
        # Set up time limit
        end_time = time.time() + (time_limit_ms / 1000.0)
        
        # MCTS main loop
        while time.time() < end_time:
            # Selection and Expansion
            if self.verbose:
                print("Selecting and Expanding")
            selected_node = self._select_and_expand(initial_node, sample_function)
            if self.verbose:
                print(f"Selected node: {selected_node.output}")
            
            # Simulation
            if self.verbose:
                print("Simulating")
            simulation_result = self._simulate(selected_node, predict_function, sample_function, heuristic_function)
            if self.verbose:
                print(f"Simulation result: {simulation_result}")
            
            # Backpropagation
            if self.verbose:
                print("Backpropagating")
            self._backpropagate(selected_node, simulation_result)

            # Update lambda for KR-AUCB asymptotic policy
            self.branches_simulated += 1
            if self.branches_simulated <= self.kr_lambda_schedule_iters:
                progress = self.branches_simulated / self.kr_lambda_schedule_iters
                self.kr_lambda = self.kr_lambda_start + progress * (self.kr_lambda_target - self.kr_lambda_start)           
        
        # Return best child after time is up, argmax over visits
        # best child = argmax of initial_node.children visits
        best_child = initial_node.most_visited_child()
        if self.verbose:
            print("Initial node:", initial_node.output)
        if self.verbose:
            print("Best child:", best_child.output)
        return (best_child.output, initial_node.lstm_states)

    def _check_progressive_widening(self, node: MCTSNode, sample_function: Callable) -> None:
        """
        Check if we should add more actions according to progressive widening
        formula: k * n^alpha where n is the visit count
        
        Uses per-dimension originality constraints - action is original if at least one dimension
        meets the required distance.
        """
        if node.gmm is None or node.failed_progressive_widening >= 2:
            return
            
        max_actions = max(1, int(self.progressive_widening_k * (node.visits ** self.progressive_widening_alpha)))
        
        # If we already have enough children, don't add more
        current_actions = len(node.children)
        if current_actions >= max_actions:
            return
        
        if self.verbose:
            print("Progressive widening:")
        if self.verbose:
            print(f"Current actions: {current_actions}, Max actions: {max_actions}")
            
        # We need to add more actions
        # Get all existing actions to check for originality
        existing_actions = [child.output for child in node.children]
        
        # Add a new action with originality constraint
        num_samples = 0
        while current_actions < max_actions:
            # Sample a new action
            new_action = sample_function(node.gmm)
            
            # Calculate dynamic epsilon based on number of samples for each dimension
            # Dynamic epsilon: scale from starting to min distance based on number of samples
            if node.failed_progressive_widening == 0:
                scaling_factor = np.minimum(1.0, num_samples / self.max_samples_for_originality)
            else:
                scaling_factor = 1.0
            epsilon = self.starting_originality_distances - scaling_factor * (
                self.starting_originality_distances - self.min_originality_distances)
            
            # Check if the action is original enough in at least one dimension
            is_original = True
            
            # Check distance to all existing actions
            for action in existing_actions:
                # Calculate per-dimension distances
                distances = np.abs(new_action - action)

                if self.snap_to_semitones:
                    # Calculate the distance of the second dimension differently, first round to the nearest 0.01
                    # This is for semitone snapping pitch instruments
                    distances[1] = np.abs(np.round(new_action[1], 2) - np.round(action[1], 2))
                
                # Action is not original if ALL dimensions are too close
                if np.all(distances < epsilon):
                    is_original = False
                    break
            
            # If original, add it; otherwise, try again with a relaxed constraint
            if is_original:
                node.add_child(new_action)
                existing_actions.append(new_action)
                current_actions += 1
                num_samples = 0  # Reset counter for next action
                if self.verbose:
                    print(f"Added action: {new_action}, Current actions: {current_actions}")
            else:
                num_samples += 1
                # If we've tried too many times, break and don't add any action, as it may be too crowded
                if num_samples >= self.max_samples_for_originality:
                    node.failed_progressive_widening += 1
                    if self.verbose:
                        print(f"Failed progressive widening: {node.failed_progressive_widening}")
                    break

    def _kernel_function(self, a: np.ndarray, b: np.ndarray, sigma: float = 0.1) -> float:
        """
        Gaussian kernel function: K(â,a) from Eq. (15)
        
        K(â,a) = exp(-0.5·(â-a)ᵀ·Σ⁻¹·(â-a)) / √((2π)^n|Σ|)
        
        For simplicity, we use a diagonal covariance matrix Σ = σ²I
        """
        # Calculate squared distance
        d = np.sum(((a - b) / sigma) ** 2)
        n = len(a)
        
        # Calculate kernel value
        return np.exp(-0.5 * d) / np.sqrt((2 * np.pi) ** n * sigma ** (2 * n))
    
    def _kr_aucb_selection(self, node: MCTSNode) -> Optional[MCTSNode]:
        """
        Select child using KR-AUCB formula (Eq. 17)
        
        arg max[E[v̄_â|â] + c·P_asym(â)·(√(∑W(a))/W(â))]
        """
        
        # Get all actions, values, and visit counts for kernel regression
        actions = node.get_all_actions()
        values = node.get_all_values()
        visits = node.get_all_visits()
        
        # Convert to numpy arrays
        actions = np.array(actions)
        values = np.array(values)
        visits = np.array(visits)
        
        kr_aucb_values = []
        
        # Calculate KR-AUCB for each child
        for i, child in enumerate(node.children):
            a_hat = child.output
            
            # Calculate kernel weights for all actions relative to this action
            kernel_weights = np.array([self._kernel_function(a_hat, a) for a in actions])
            
            # Calculate E[v̄_â|â] using kernel regression (Eq. 13)
            # E[v̄_â|â] = (∑_a K(â,a)·v̄_a·n_a) / (∑_a K(â,a)·n_a)
            weighted_sum_values = np.sum(kernel_weights * values * visits)
            sum_weights = np.sum(kernel_weights * visits)
            expected_value = weighted_sum_values / sum_weights if sum_weights > 0 else 0
            
            # Calculate W(â) using kernel density (Eq. 14)
            # W(â) = ∑_a K(â,a)·n_a
            effective_visits = np.sum(kernel_weights * visits)
            
            # Calculate P_asym - asymptotic policy (Eq. 16)
            # P_asym = λ·P_prior + (1-λ)·P_uniform
            if hasattr(child, 'prior_prob') and child.prior_prob is not None:
                # Use stored prior probability if available
                p_prior = child.prior_prob
            else:
                # If prior not available, use equal weights (will be adjusted by GMM later)
                p_prior = 1.0 / len(node.children)
            
            p_uniform = 1.0 / len(node.children)
            p_asym = self.kr_lambda * p_prior + (1.0 - self.kr_lambda) * p_uniform
            
            # Calculate exploration term
            total_effective_visits = np.sum([np.sum([self._kernel_function(a, a_other) for a_other in actions] * visits) for a in actions])
            exploration_term = self.exploration_weight * p_asym * np.sqrt(total_effective_visits) / effective_visits if effective_visits > 0 else float('inf')
            
            # Calculate KR-AUCB value (Eq. 17)
            if self.verbose:
                print(f"Child output: {child.output}, Exploitation: {expected_value}, Exploration: {exploration_term}")
            kr_aucb_value = expected_value + exploration_term
            kr_aucb_values.append(kr_aucb_value)
        
        # Return child with highest KR-AUCB value
        print("FINAL KR-AUCB SELECTION: ", node.children[np.argmax(kr_aucb_values)].output)
        return node.children[np.argmax(kr_aucb_values)]
    
    def _select_and_expand(self, node: MCTSNode, sample_function: Callable) -> Tuple[MCTSNode, List[np.ndarray]]:
        """
        Select a node to expand using UCT and expand it
        Apply progressive widening to dynamically increase available actions
        Returns the newly expanded node and the current memory state
        """
        self.nodes_searched += 1
        current = node
        
        # Selection phase - navigate down the tree with progressive widening at each node
        if self.verbose:
            print("Selection phase")
        while current.children:  # As long as we have children to explore
            # Check progressive widening condition
            self._check_progressive_widening(current, sample_function)
                
            # Otherwise, select best child according to KR-AUCB
            if self.verbose:
                print("Selecting best child with KR-AUCB")
            current = self._kr_aucb_selection(current)
            if self.verbose:
                print(f"Selected child: {current.output}")
            self.nodes_searched += 1
            
        # Expansion phase, this is simply another progressive widening check to add a child since the current node will have no children
        if self.verbose:
            print("Expansion phase")
        self._check_progressive_widening(current, sample_function)
        if current.children:
            # If child added by progressive widening, select one
            current = current.children[0]
        
        return current
    
    def _simulate(self, selected_node: MCTSNode, predict_function: Callable, sample_function: Callable, heuristic_function: Callable) -> float:
        """
        Run a simulation from the given node to estimate its value
        """
        depth = 0

        if self.verbose:
            print("Simulation from branch: ", self._extract_branch(selected_node))
        
        # Get the GMM and lstm_states for the current node if not already computed
        if selected_node.gmm is None:  
            selected_node.gmm, selected_node.lstm_states = predict_function(selected_node.output, init_lstm_states=selected_node.parent.lstm_states)
        
        # Continue simulation until we reach max depth
        current_gmm = selected_node.gmm
        current_lstm_states = selected_node.lstm_states
        simulation_outputs = []
        while depth < self.simulation_depth:
            # Sample next node
            action = sample_function(current_gmm)
            simulation_outputs.append(action)
            depth += 1

            # If max depth not reached, predict next GMM and lstm states
            if depth < self.simulation_depth:
                current_gmm, current_lstm_states = predict_function(action, init_lstm_states=current_lstm_states)
        
        # Concatenate the simulated outputs with the original branch
        simulated_branch = np.concatenate([self._extract_branch(selected_node), np.array(simulation_outputs)])
        if self.verbose:
            print("Simulated branch: ", simulated_branch)

        # Evaluate the simulated branch using the heuristic function
        return heuristic_function(simulated_branch)
    
    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        """Update node statistics going up the tree"""
        current = node
        while current:
            current.visits += 1
            current.value += result
            if self.verbose:
                print(f"Backpropagating: {current.output}, Visits: {current.visits}, Value: {current.value}, Average: {current.value / current.visits}")
            current = current.parent
    
    def _extract_branch(self, node: MCTSNode) -> Tuple[np.ndarray, List]:
        """Extract the complete branch from root to the given node"""
        output_branch = []
        current = node
        while current:
            output_branch.append(current.output)
            current = current.parent
        return np.array(output_branch[::-1])
    
    def get_num_nodes(self) -> int:
        """Return the total number of nodes searched during the MCTS process"""
        return self.nodes_searched
    
    def get_num_branches(self) -> int:
        """Return the total number of branches simulated during the MCTS process"""
        return self.branches_simulated
    
    def get_best_branch(self) -> np.ndarray:
        """
        Return the entire branch with the most visits during the search.
        """
        # Start at root node
        current = self.root
        # Recursively find the child with the highest visit count
        while current.children:
            current = current.most_visited_child()

        # Extract the branch from the best child node
        branch = self._extract_branch(current)
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
        ax.set_xlabel('Time Interval (ms)')
        ax.set_ylabel('Pitch (Hz)')
        ax.set_zlabel('Tree Depth')
        ax.set_title(title)
        
        # Extract winning branch for highlighting
        winning_branch = self.get_best_branch()
        print(f"Winning branch: {winning_branch}")
        # Print the heuristic value of the winning branch
        heuristic_value = self.heuristic_function(winning_branch)
        print(f"Heuristic value of winning branch: {heuristic_value}")
        
        # Recursively plot nodes and edges
        def plot_node(node, depth=0, parent_pos=None):
            if node is None:
                return
            
            # If this node has less than 5 visits and is not at the root, skip it
            if node.visits < 3 and depth > 0:
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
                # Convert output[1] (note) to frequency in Hz
                # Assuming output[1] is in semitones, convert to Hz using A4 = 440 Hz, A4 = 54, 10 notes = 1 octave, eg A5 = 64 TODO: change for 12 note model
                hz = 440 * math.pow(2, (node.output[1]*100 - 54) / 10)
                x, y = node.output[0] * 1000, hz
            
            # Position in 3D space
            pos = (x, y, depth)
            
            # Check if this node is part of the winning branch
            is_winning = node.output in winning_branch if node.output is not None else False
            
            # Plot node
            if node.visits > 0:  # Only plot nodes that have been visited
                # Normalize size (min size 20, max size 200)
                node_size = -80 + 65 * math.pow(node.visits + 2, 0.25)
                
                # Color based on whether it's part of the winning branch
                node_color = 'yellow' if is_winning else 'skyblue'
                
                # Plot the node
                if is_winning:
                    # Highlight winning nodes by putting them in front of all others
                    ax.scatter(x, y, depth, s=node_size, c=node_color, edgecolor='black', alpha=1)
                else:
                    # Regular node
                    ax.scatter(x, y, depth, s=node_size, c=node_color, edgecolor='black', alpha=0.35)
                
                # Add text annotation with visits count
                #ax.text(x, y, depth, f"{node.visits}", fontsize=8)
                
                # Draw edge to parent
                if parent_pos is not None:
                    # Normalize edge width
                    edge_width = -8 + 6.5 * math.pow(node.visits + 2, 0.25)
                    
                    # Line coordinates
                    xs = [parent_pos[0], x]
                    ys = [parent_pos[1], y]
                    zs = [parent_pos[2], depth]
                    
                    # Edge color
                    edge_color = 'gold' if is_winning else 'gray'
                    
                    # Plot edge
                    if is_winning:
                        # Highlight winning path by putting it in front of all others
                        ax.plot(xs, ys, zs, linewidth=edge_width, color=edge_color, alpha=1)
                    else:
                        # Regular edge
                        ax.plot(xs, ys, zs, linewidth=edge_width, color=edge_color, alpha=0.15)
            
            # Recursively plot children
            for child in node.children:
                plot_node(child, depth + 1, pos)
        
        # Start plotting from root
        plot_node(self.root)
        
        # Add legend
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markeredgecolor='black', label='Chosen Node', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markeredgecolor='black', label='Explored Node', markersize=10),
            Line2D([0], [0], color='gold', lw=2, label='Chosen Path'),
            Line2D([0], [0], color='gray', lw=2, label='Explored Path')
        ]
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Enable grid
        ax.grid(True)
        
        # Auto-adjust view
        ax.view_init(elev=8, azim=78)

        # Adjust Z axis ticks and direction
        z_min, z_max = ax.get_zlim()
        ax.set_zticks(range(int(z_min), int(z_max) + 1))
        ax.invert_zaxis()  # Makes depth 0 at the top
        
        plt.tight_layout()
        return fig, ax