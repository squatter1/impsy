import numpy as np
import time
from typing import List, Tuple, Optional, Callable
import math

class MCTSNode:
    def __init__(self, output: np.ndarray, parent=None, lstm_states=None, snap_dp: [Optional[int]] = [None, 2]):
        self.output = output
        # For each dimension, check if snapping is enabled and adjust distance accordingly
        for i in range(len(output)):
            if snap_dp[i] is not None:
                # Round to the snapped number of decimal places
                # This is for semitone snapping pitch instruments and other non-continuous cases
                # If i is 1, aka pitch, convert to semitones first
                if i == 1:
                    self.output[i] *= 1.27
                self.output[i] = np.round(self.output[i], snap_dp[i])
                # If i is 1, aka pitch, convert back
                if i == 1:
                    self.output[i] /= 1.27
        self.parent = parent
        self.lstm_states = lstm_states
        self.gmm = None
        self.children = []
        
        # MCTS specific attributes
        self.visits = 0
        self.value = 0.0
        self.best_value = 0.0  # Best found heuristic value of any path from this node
        self.failed_progressive_widening = 0  # Count of failed progressive widening attempts, too many leads to assumption all valid unique children have been added

    def delete_children(self, exception: Optional['MCTSNode'] = None) -> None:
        for child in self.children:
            if child != exception:
                child.delete_children()
        self.children.clear()  # Clear the list of children, this will trigger python's garbage collection
        
    def add_child(self, output: np.ndarray, snap_dp: [Optional[int]]) -> 'MCTSNode':
        """Add a child node with the given output and return it"""
        child = MCTSNode(output, parent=self, snap_dp=snap_dp)
        self.children.append(child)
        return child
    
    def uct_child(self, greedy_weight: float, exploration_weight: float, verbose: bool = False) -> Optional['MCTSNode']:
        """
        Select the best child according to UCT formula:
        UCT = greedy_weight*best_value + (1-greedy_weight)*value/visits + exploration_weight * sqrt(2 * ln(parent_visits) / visits)
        """
        uct_values = []
        for child in self.children:
            # Exploitation term
            exploitation = greedy_weight * child.best_value + (1 - greedy_weight) * child.value / child.visits if child.visits > 0 else 0
            
            # Exploration term
            exploration = exploration_weight * math.sqrt(math.log(self.visits) / child.visits) if child.visits > 0 else float('inf')
            
            if verbose:
                print(f"Child output: {child.output}, Exploitation: {exploitation}, Exploration: {exploration}")
            uct_values.append(exploitation + exploration)
        
        if verbose:
            print("FINAL UCT SELECTION: ", self.children[np.argmax(uct_values)].output)
        return self.children[np.argmax(uct_values)]
    
    def most_visited_child(self) -> Optional['MCTSNode']:
        """Return the child with the most visits"""
        return max(self.children, key=lambda child: child.visits)  

class MCTSPredictionTree:
    def __init__(self, 
                 root_output: np.ndarray,
                 initial_lstm_states: np.ndarray,
                 predict_function: Callable, 
                 sample_function: Callable,
                 simulation_depth: int = 2, 
                 greedy_weight: float = 0.5,
                 exploration_weight: float = 1.0,
                 progressive_widening_k: float = 2.5,
                 progressive_widening_alpha: float = 0.25,
                 min_originality_distances: np.ndarray = np.array([0.08, None]),
                 expansion_samples: int = 10,
                 snap_dp: [Optional[int]] = [None, 2],
                 verbose: bool = False):
        self.root = MCTSNode(root_output, snap_dp=snap_dp)
        self.root.gmm, self.root.lstm_states = predict_function(self.root.output, init_lstm_states=initial_lstm_states) # Get the GMM for the root
        self.initial_lstm_states = initial_lstm_states
        self.predict_function = predict_function
        self.sample_function = sample_function
        self.simulation_depth = simulation_depth
        self.greedy_weight = greedy_weight
        self.exploration_weight = exploration_weight
        self.progressive_widening_k = progressive_widening_k
        self.progressive_widening_alpha = progressive_widening_alpha
        self.min_originality_distances = min_originality_distances
        self.expansion_samples = expansion_samples
        self.snap_dp = snap_dp

        self.verbose = verbose

    def set_root(self, new_root_output: np.ndarray) -> None:
        """Set the new root to the child of the current root with the given output"""
        # Find the child with the given output
        for child in self.root.children:
            if np.array_equal(child.output, new_root_output):
                self.root.delete_children(exception=child)  # Delete all other children of the current root
                # Set the new root to this child
                self.root = child
                break
        else:
            raise ValueError(f"Child with output {new_root_output} not found in root children.")
    
    def search(self, 
               memory: List[np.ndarray], 
               heuristic_functions: List[Tuple[Callable, Callable, float]],
               time_limit_ms: float = None,
               max_iterations: int = None
              ) -> Tuple[np.ndarray, List, float]:
        """
        Run MCTS for the given time limit and return the best branch found
        
        Args:
            memory: Initial sequence of notes prior to root node
            heuristic_functions: Functions to evaluate a branch, in tuples for memory, branch, and importance multiplier
            time_limit_ms: Search time limit in milliseconds
            iterations: Maximum number of iterations to run MCTS
            
        Returns:
            Tuple of (best_child, lstm states)
        """
        # Check if at least one of time_limit_ms or max_iterations is provided
        if time_limit_ms is None and max_iterations is None:
            raise ValueError("Either time_limit_ms or max_iterations must be provided.")
        # Convert memory to numpy array if not already
        if isinstance(memory, list):
            memory = np.array(memory)
        # Apply snapping to the memory if enabled
        for i in range(len(memory)):
            memory[i] = np.array(memory[i])
            for j in range(len(memory[i])):
                if self.snap_dp[j] is not None:
                    # If j is 1, aka pitch, convert to semitones first
                    if j == 1:
                        memory[i][j] *= 1.27
                    memory[i][j] = np.round(memory[i][j], self.snap_dp[j])
                    # If j is 1, aka pitch, convert back
                    if j == 1:
                        memory[i][j] /= 1.27
        if self.verbose:
            print("Starting MCTS search with memory: ", memory)
        
        # Get heuristic values for the memory
        heuristic_memories = []
        for heuristic_function in heuristic_functions:
            # Get the heuristic value for this function
            heuristic_memories.append(heuristic_function[0](memory))

        # Reset statistics
        self.nodes_searched = 0
        self.branches_simulated = 0
        
        # Set up time limit
        if time_limit_ms is None:
            time_limit_ms = 3600000 # Default to 1 hour
        end_time = time.time() + (time_limit_ms / 1000.0)
        # Set up max iterations
        if max_iterations is None:
            max_iterations = 1000000 # Default to 1 million iterations
        
        # MCTS main loop
        while time.time() < end_time and self.branches_simulated < max_iterations:
            # Selection and Expansion
            if self.verbose:
                print("Selecting and Expanding")
            selected_node = self._select_and_expand(self.root)
            if self.verbose:
                print(f"Selected node: {selected_node.output}")
            
            # Simulation
            if self.verbose:
                print("Simulating")
            simulation_result = self._simulate(selected_node, memory, heuristic_functions, heuristic_memories)
            if self.verbose:
                print(f"Simulation result: {simulation_result}")
            
            # Backpropagation
            if self.verbose:
                print("Backpropagating")
            self._backpropagate(selected_node, simulation_result)

            self.branches_simulated += 1        
        
        # Return best child after time is up, argmax over visits
        best_child = self.root.most_visited_child()
        if self.verbose:
            print("Initial node:", self.root.output)
        if self.verbose:
            print("Best child:", best_child.output)
        return (best_child.output, self.root.lstm_states)

    def _check_progressive_widening(self, node: MCTSNode) -> None:
        """
        Check if we should add more actions according to progressive widening
        formula: k * n^alpha where n is the visit count
        
        Uses per-dimension originality constraints - action is original if at least one dimension
        meets the required distance.
        """
        if node.gmm is None or node.failed_progressive_widening >= 5:
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
        if self.verbose:
            print(f"Existing actions: {existing_actions}")

        if len(existing_actions) == 0:
            new_action = self.sample_function(node.gmm)
            node.add_child(new_action, self.snap_dp)
            return
        
        # Add a new action with originality constraints
        while current_actions < max_actions:
            best_new_action = None
            best_originality_score = 0
            for _ in range(self.expansion_samples):
                # Sample a new action
                new_action = self.sample_function(node.gmm)
                # Perform snapping if enabled
                for i in range(len(new_action)):
                    if self.snap_dp[i] is not None:
                        # If i is 1, aka pitch, convert to semitones first
                        if i == 1:
                            new_action[i] *= 1.27
                        new_action[i] = np.round(new_action[i], self.snap_dp[i])
                        # If i is 1, aka pitch, convert back
                        if i == 1:
                            new_action[i] /= 1.27
                
                # Check if the action is original enough in at least one dimension
                is_original = True
                originality_scores = []
                for action in existing_actions:
                    # Calculate per-dimension multiplicative distances (ratio between values)
                    # We use max/min to ensure ratio is always >= 1.0
                    ratios = np.maximum(new_action, action) / np.maximum(np.minimum(new_action, action), 1e-10)
                    # Subtract 1 to get the actual multiplicative distance (e.g., 1.1x becomes 0.1)
                    mult_distances = ratios - 1.0
                    
                    # Create a mask for dimensions where min_originality_distances is not None
                    valid_dimensions = np.array([d is not None for d in self.min_originality_distances])
                    
                    # For None dimensions, we just check they're not exactly equal (distance > 0)
                    none_dimensions = ~valid_dimensions
                    if none_dimensions.any():
                        none_dim_values_equal = np.isclose(mult_distances[none_dimensions], 0.0, atol=1e-10)
                    
                    # Check originality condition:
                    # 1. For dimensions with min distance requirement: distance must be >= min_distance
                    # 2. For dimensions with None: values must not be equal (distance > 0)
                    if valid_dimensions.any():
                        valid_dims_too_close = np.all(
                            mult_distances[valid_dimensions] < np.array(self.min_originality_distances)[valid_dimensions]
                        )
                    else:
                        valid_dims_too_close = False
                        
                    if none_dimensions.any():
                        none_dims_all_equal = np.all(none_dim_values_equal)
                    else:
                        none_dims_all_equal = False

                    # Action is not original if both conditions fail
                    if (valid_dimensions.any() and valid_dims_too_close) and (not none_dimensions.any() or none_dims_all_equal):
                        is_original = False
                        break
                
                    # Calculate originality score using only dimensions with actual minimum requirements
                    if valid_dimensions.any():
                        score = np.sum(
                            mult_distances[valid_dimensions] / 
                            np.array(self.min_originality_distances)[valid_dimensions]
                        )
                        originality_scores.append(score)
                if is_original and originality_scores:
                    originality_score = np.min(originality_scores)
                    if originality_score > best_originality_score:
                        best_originality_score = originality_score
                        best_new_action = new_action
            
            # If original action found, add it
            if best_new_action is not None:
                node.add_child(best_new_action, self.snap_dp)
                existing_actions.append(best_new_action)
                current_actions += 1
                node.failed_progressive_widening = 0  # Reset failed progressive widening count
                if self.verbose:
                    print(f"Added action: {best_new_action}, Current actions: {current_actions}")
            # Otherwise don't add anything
            else:
                node.failed_progressive_widening += 1
                if self.verbose:
                    print(f"Failed progressive widening: {node.failed_progressive_widening}")
                break
    
    def _select_and_expand(self, node: MCTSNode) -> Tuple[MCTSNode, List[np.ndarray]]:
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
            self._check_progressive_widening(current)
                
            # Otherwise, select best child according to uct
            if self.verbose:
                print("Selecting best child with UCT")
            current = current.uct_child(self.greedy_weight, self.exploration_weight, verbose=self.verbose)
            if self.verbose:
                print(f"Selected child: {current.output}")
            self.nodes_searched += 1
            
        # Expansion phase, this is simply another progressive widening check to add a child since the current node will have no children
        if self.verbose:
            print("Expansion phase")
        self._check_progressive_widening(current)
        if current.children:
            # If child added by progressive widening, select one
            current = current.children[0]
        
        return current
    
    def _simulate(self, selected_node: MCTSNode, memory: List[np.ndarray], heuristic_functions: List[Tuple[Callable, Callable, float]], heuristic_memories: List[Tuple]) -> float:
        """
        Run a simulation from the given node to estimate its value
        """
        depth = 0

        if self.verbose:
            print("Simulation from branch: ", self._extract_branch(selected_node))
        
        # Get the GMM and lstm_states for the current node if not already computed
        if selected_node.gmm is None:  
            selected_node.gmm, selected_node.lstm_states = self.predict_function(selected_node.output, init_lstm_states=selected_node.parent.lstm_states)
        
        # Continue simulation until we reach max depth
        current_gmm = selected_node.gmm
        current_lstm_states = selected_node.lstm_states
        simulation_outputs = []
        while depth < self.simulation_depth:
            # Sample next node
            action = self.sample_function(current_gmm)
            simulation_outputs.append(action)
            depth += 1

            # If max depth not reached, predict next GMM and lstm states
            if depth < self.simulation_depth:
                current_gmm, current_lstm_states = self.predict_function(action, init_lstm_states=current_lstm_states)
        
        # Concatenate the simulated outputs with the original branch
        simulated_branch = np.concatenate([self._extract_branch(selected_node), np.array(simulation_outputs)])
        if self.verbose:
            print("Simulated branch: ", simulated_branch)

        # Calculate heuristic value for the simulated branch
        heuristic_value = 0.0
        for i in range(len(heuristic_functions)):
            heuristic_function = heuristic_functions[i]
            heuristic_memory = heuristic_memories[i]
            # Get the heuristic value for this function
            function_value = heuristic_function[1](heuristic_memory, simulated_branch, heuristic_function[2])
            if self.verbose:
                print(f"Calculating heuristic value for function: {heuristic_function[1].__name__}")
                print(f"Result: {function_value}")
            heuristic_value += function_value
        
        if self.verbose:
            print(f"Total heuristic value: {heuristic_value}")
        return -heuristic_value
    
    def _backpropagate(self, node: MCTSNode, result: float) -> None:
        """Update node statistics going up the tree"""
        current = node
        while current:
            current.visits += 1
            current.value += result
            if result > current.best_value:
                current.best_value = result
            if self.verbose:
                print(f"Backpropagating: {current.output}, Visits: {current.visits}, Value: {current.value}, Average: {current.value / current.visits}, Best: {current.best_value}")
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
    
    
    def graph_tree(self, ax=None, title="Monte Carlo Tree Search Visualization", has_winning_branch=True, zoomed=False, min_visits=3) -> Tuple:
        """
        Visualize the MCTS tree in 3D.
        
        Args:
            ax: Optional matplotlib 3D axis. If None, a new figure and axis will be created.
            title: Title for the plot. Ignored if zoomed=True.
            has_winning_branch: Whether to highlight the winning branch.
            zoomed: When True, optimizes the visualization for a zoomed view (no title, 
                   legend inside graph, maximized graph space, larger text, and larger nodes).
            
        Returns:
            The matplotlib figure and axis.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Create figure and 3D axis if not provided
        if ax is None:
            fig = plt.figure(figsize=(99.13386, 55.31496))

            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect(aspect = (2,1,1))
            #ax = fig.add_axes([0.05, 0.05, 0.95, 0.95], projection='3d')
        
        # Set labels and title (with larger text if zoomed)
        fontsize = 27.5 if zoomed else 10
        labelpad = 20 if zoomed else 0
        ax.set_xlabel('Time Interval (ms)', fontsize=fontsize, labelpad=(labelpad+5))
        ax.set_ylabel('Pitch (Hz)', fontsize=fontsize, labelpad=labelpad)
        ax.set_zlabel('Tree Depth', fontsize=fontsize, labelpad=(labelpad-5))
        
        # Only set title if not zoomed
        if not zoomed:
            ax.set_title(title)
        
        # Extract winning branch for highlighting
        if has_winning_branch:
            winning_branch = self.get_best_branch()
            print(f"Winning branch: {winning_branch}")
            ## Print the heuristic value of the winning branch
            #import impsy.heuristics as heuristics
            #heuristic_value = heuristics.rhythmic_consistency_to_value(winning_branch, value=winning_branch[0][0], verbose=True)
            #print(f"Heuristic value of winning branch: {heuristic_value}")
        else:
            winning_branch = None
        
        # Recursively plot nodes and edges
        def plot_node(node, depth=0, parent_pos=None, min_visits=3):
            if node is None:
                return
            
            # Show nodes with at least 3 visits
            min_visits = 3
            
            # If this node has less than min_visits and is not at the root, skip it
            if node.visits < min_visits and depth > 0:
                return
            
            ## Skip root nodes that don't have music output data
            #if depth < 2 and (node.output is None or len(node.output) < 2):
            #    for child in node.children:
            #        plot_node(child, depth + 1, min_visits=min_visits)
            #    return
                    
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
            if has_winning_branch:
                is_winning = node.output[0] in winning_branch[:, 0]
            else:
                is_winning = False
            
            # Plot node
            if node.visits > 0:  # Only plot nodes that have been visited
                # Normalize size (min size 20, max size 200)
                # Double the node size if zoomed
                size_multiplier = 2 if zoomed else 1
                node_size = size_multiplier * (-80 + 65 * math.pow(node.visits + 2, 0.25))
                
                # Color based on whether it's part of the winning branch
                node_color = 'yellow' if is_winning else 'skyblue'
                
                # Plot the node
                if is_winning:
                    # Highlight winning nodes by putting them in front of all others
                    ax.scatter(x, y, depth, s=node_size, c=node_color, edgecolor='black', alpha=1)
                else:
                    # Regular node
                    if has_winning_branch:
                        ax.scatter(x, y, depth, s=node_size, c=node_color, edgecolor='black', alpha=0.35)
                    else:
                        ax.scatter(x, y, depth, s=node_size, c=node_color, edgecolor='black', alpha=0.8)
                
                # Add text annotation with visits count
                #ax.text(x, y, depth, f"{node.visits}", fontsize=8)
                
                # Draw edge to parent
                if parent_pos is not None:
                    # Normalize edge width
                    edge_width = -8 + 6.5 * math.pow(node.visits + 2, 0.25)
                    # Make edges thicker if zoomed
                    if zoomed:
                        edge_width *= 1.5
                    
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
                        if has_winning_branch:
                            ax.plot(xs, ys, zs, linewidth=edge_width, color=edge_color, alpha=0.15)
                        else:
                            ax.plot(xs, ys, zs, linewidth=edge_width, color=edge_color, alpha=0.6)
            
            # Recursively plot children
            for child in node.children:
                plot_node(child, depth + 1, pos, min_visits=min_visits)
        
        # Start plotting from root
        plot_node(self.root, min_visits=min_visits)
        
        # Add legend
        from matplotlib.lines import Line2D
        
        markersize = 30 if zoomed else 10
        lw = 8 if zoomed else 2
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markeredgecolor='black', label='Chosen Node', markersize=markersize),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markeredgecolor='black', label='Explored Node', markersize=markersize),
            Line2D([0], [0], color='gold', lw=lw, label='Chosen Path'),
            Line2D([0], [0], color='gray', lw=lw, label='Explored Path')
        ]
        
        # Set legend
        if not zoomed:
            legend_loc = 'upper right'
            ax.legend(handles=legend_elements, loc=legend_loc, fontsize=fontsize)
        
        # Enable grid
        ax.grid(True)
        
        # Auto-adjust view
        ax.view_init(elev=8, azim=78)

        # Adjust Z axis ticks and direction
        z_min, z_max = ax.get_zlim()
        ax.set_zticks(range(0, int(z_max) + 1))
        ax.invert_zaxis()  # Makes depth 0 at the top

        # Adjust x axis ticks intervals
        if zoomed:
            # Set x axis ticks to be in intervals of 300ms from 0 to 1200
            x_ticks = np.arange(0, 1201, 300)
        else:
            # Set x axis ticks to be in intervals of 200ms from 0 to 1200
            x_ticks = np.arange(0, 1201, 200)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis='x', labelsize=fontsize)

        # Adjust y axis ticks intervals
        if zoomed:
            # Set y axis ticks to be in intervals of 300Hz from 200 to 800
            y_ticks = np.arange(200, 801, 300)
        else:
            # Set y axis ticks to be in intervals of 100Hz from 200 to 800
            y_ticks = np.arange(200, 801, 100)
        ax.set_yticks(y_ticks)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.tick_params(axis='z', labelsize=fontsize)
        
        # Make the graph take up the entire figure space if zoomed
        plt.tight_layout()
        #if zoomed:
        #    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #    ax.set_box_aspect([1, 1, 1])
            
        return