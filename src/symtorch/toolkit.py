#Interpretability toolkit 

import warnings
warnings.filterwarnings("ignore", message="torch was imported before juliacall")

import torch
import torch.nn as nn
import math
import time
import numpy as np
import os
import pickle
from typing import Optional, Dict, Any, List, Callable
from pysr import PySRRegressor
from .SymbolicMLP import SymbolicMLP

class PruningMLP(SymbolicMLP):
    """
    A PyTorch module wrapper that adds dynamic pruning and symbolic regression capabilities to MLPs.

    This class extends SymbolicMLP to provide progressive dimensionality reduction through pruning
    while maintaining interpretability features. It dynamically removes less important output
    dimensions during training based on activation variance, then applies symbolic regression
    to the remaining active dimensions.

    The wrapper maintains full compatibility with PyTorch's training pipeline and inherits
    all SymbolicMLP functionality for symbolic regression on pruned dimensions.

    Attributes:
        InterpretSR_MLP (nn.Module): The wrapped PyTorch MLP model (inherited from SymbolicMLP)
        mlp_name (str): Human-readable name for the MLP instance (inherited from SymbolicMLP)
        pysr_regressor (dict): Dictionary mapping active dimensions to fitted symbolic regression models (inherited from SymbolicMLP)
        initial_dim (int): Initial output dimensionality before pruning
        current_dim (int): Current number of active dimensions
        target_dim (int): Final target dimensionality after pruning
        pruning_schedule (dict): Mapping from epoch to target dimensions
        pruning_mask (torch.Tensor): Boolean mask indicating active dimensions
        
    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from symtorch import PruningMLP
        >>>
        >>> # Create a composite model
        >>> class SimpleModel(nn.Module):
        ...     def __init__(self, input_dim, output_dim, output_dim_f=32, hidden_dim=128):
        ...         super(SimpleModel, self).__init__()
        ...         self.f_net = nn.Sequential(
        ...             nn.Linear(input_dim, hidden_dim),
        ...             nn.ReLU(),
        ...             nn.Linear(hidden_dim, output_dim_f)
        ...         )
        ...         self.g_net = nn.Linear(output_dim_f, output_dim)
        ...
        ...     def forward(self, x):
        ...         x = self.f_net(x)
        ...         x = self.g_net(x)
        ...         return x
        >>>
        >>> # Create model and wrap f_net with pruning
        >>> model = SimpleModel(input_dim=5, output_dim=1, output_dim_f=32)
        >>> model.f_net = PruningMLP(model.f_net,
        ...                          initial_dim=32,
        ...                          target_dim=2,
        ...                          mlp_name="f_net")
        >>> 
        >>> # Set up pruning schedule
        >>> epochs = 100
        >>> model.f_net.set_schedule(total_epochs=epochs, end_epoch_frac=0.7)
        >>> 
        >>> # During training loop
        >>> for epoch in range(epochs):
        ...     # ... training code ...
        ...     model.f_net.prune(epoch, validation_data)  # Prune based on importance
        >>> 
        >>> # Apply symbolic regression to active dimensions only
        >>> regressor = model.f_net.distill(train_inputs)
        >>> 
        >>> # Switch to using symbolic equations for active dimensions
        >>> model.f_net.switch_to_equation()
        >>> # Switch back to using the MLP
        >>> model.f_net.switch_to_mlp()
    """
    
    def __init__(self, mlp: nn.Module, initial_dim: int, target_dim: int, mlp_name: str = None):
        """
        Initialise the PruningMLP wrapper.

        Args:
            mlp (nn.Module): The PyTorch MLP model to wrap
            initial_dim (int): Initial output dimensionality before pruning
            target_dim (int): Target output dimensionality after pruning
            mlp_name (str, optional): Human-readable name for this MLP instance.
                                    If None, generates a unique name based on object ID.
        """
        # Initialize SymbolicMLP with the MLP
        super().__init__(mlp, mlp_name or f"pruned_mlp_{id(self)}")
        
        # Add pruning-specific attributes
        self.initial_dim = initial_dim
        self.current_dim = initial_dim 
        self.target_dim = target_dim
        self.pruning_schedule = None
        # Register pruning_mask as a buffer so it moves with the model
        self.register_buffer('pruning_mask', torch.ones(self.current_dim, dtype=torch.bool))
    

    def set_schedule(self, total_epochs: int, decay_rate: str = 'cosine', end_epoch_frac: float = 0.5):
        """
        Set up the pruning schedule for progressive dimensionality reduction.
        
        Creates a schedule that progressively reduces dimensions from initial_dim to target_dim
        over the specified fraction of training epochs using the chosen decay strategy.
        
        Args:
            total_epochs (int): Total number of training epochs
            decay_rate (str, optional): Pruning schedule type. Options:
                                      - 'cosine': Cosine annealing schedule (default)
                                      - 'linear': Linear reduction schedule
                                      - 'exp': Exponential decay schedule
            end_epoch_frac (float, optional): Fraction of total epochs to complete pruning by.
                                             Defaults to 0.5 (pruning ends halfway through training)
                                             
        Example:
            >>> pruned_mlp.set_schedule(total_epochs=100, decay_rate='cosine', end_epoch_frac=0.7)
        """
        
        prune_end_epoch = int(end_epoch_frac * total_epochs)
        prune_epochs = prune_end_epoch

        dims_to_prune = self.initial_dim - self.target_dim
        schedule_dict = {}

        #different pruning schedules
        #exponential decay
        if decay_rate == 'exp':
            decay_rate = 3.0
            max_decay = 1 - math.exp(-decay_rate)

            for epoch in range(prune_end_epoch):
                progress = epoch / prune_epochs
                raw_decay = 1 - math.exp(-decay_rate * progress)
                decay_factor = raw_decay / max_decay

                dims_pruned = math.ceil(dims_to_prune * decay_factor)
                target_dims = max(self.initial_dim - dims_pruned, self.target_dim)
                schedule_dict[epoch] = target_dims

        #linear decay
        elif decay_rate == 'linear':
            for epoch in range(prune_end_epoch):
                progress = epoch / prune_epochs
                dims_pruned = math.ceil(dims_to_prune * progress)
                target_dims = max(self.initial_dim - dims_pruned, self.target_dim)
                schedule_dict[epoch] = target_dims

        #cosine decay
        elif decay_rate == 'cosine':
            for epoch in range(prune_end_epoch):
                progress = epoch / prune_epochs
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                dims_pruned = math.ceil(dims_to_prune * (1 - cosine_decay))
                target_dims = max(self.initial_dim - dims_pruned, self.target_dim)
                schedule_dict[epoch] = target_dims

        #keep target_dim for the last part of training
        for epoch in range(prune_end_epoch, total_epochs):
            schedule_dict[epoch] = self.target_dim

        self.pruning_schedule = schedule_dict

    def prune(self, epoch: int, sample_data: torch.Tensor, parent_model=None):
        """
        Perform pruning for the current epoch based on the pruning schedule.
        
        Evaluates the importance of each output dimension by computing the standard deviation
        of activations across the sample data. Retains the most important dimensions according
        to the current epoch's target dimensionality.
        
        Args:
            epoch (int): Current training epoch
            sample_data (torch.Tensor): Sample input data to evaluate dimension importance.
                                       Typically a subset of validation data.
            parent_model (nn.Module, optional): The parent model containing this PruningMLP instance.
                                              If provided, will trace intermediate activations to get
                                              the actual outputs at this layer level for importance evaluation.
                                       
        Note:
            This method should be called during each training epoch. If the current epoch
            is not in the pruning schedule, no pruning is performed.
        """

        if epoch not in self.pruning_schedule:
            return
        if self.pruning_schedule is None:
            assert 'Pruning schedule is not set.'
            
        target_dims = self.pruning_schedule[epoch]
        
        with torch.no_grad():
            # Extract outputs at this layer level for importance evaluation
            if parent_model is not None:
                with self._capture_layer_output(parent_model, sample_data) as (_, layer_outputs):
                    pass
                
                # Use captured intermediate data
                if layer_outputs:
                    output_array = layer_outputs[0]
                else:
                    raise RuntimeError("Failed to capture intermediate activations. Ensure parent_model contains this SymbolicMLP instance.")
            else:
                # Original behavior - use MLP directly
                output_array = self.InterpretSR_MLP(sample_data)

            output_importance = output_array.std(dim=0)
            most_important = torch.argsort(output_importance)[-target_dims:]
            
            new_mask = torch.zeros_like(self.pruning_mask)
            new_mask[most_important] = True
            # Update the registered buffer (this maintains device consistency)
            self.pruning_mask.data = new_mask.data
            self.current_dim = target_dims

    def get_active_dimensions(self):
        """
        Get indices of currently active (non-masked) dimensions.
        
        Returns:
            list: List of integer indices for dimensions that are currently active
                 (not pruned/masked)
                 
        Example:
            >>> active_dims = pruned_mlp.get_active_dimensions()
            >>> print(f"Active dimensions: {active_dims}")
            Active dimensions: [5, 12, 18]
        """
        return torch.where(self.pruning_mask)[0].tolist()

    def distill(self, inputs, output_dim: int = None, parent_model=None,
                 variable_transforms: Optional[List[Callable]] = None,
                 save_path: str = None,
                 sr_params: Optional[Dict[str, Any]] = None,
                 fit_params: Optional[Dict[str, Any]] = None):
        """
        Discover symbolic expressions for active (non-pruned) dimensions only.

        Overrides SymbolicMLP's distill method to focus symbolic regression on dimensions
        that survived the pruning process, ignoring inactive/masked dimensions.

        Args are the same as parent class distill method.
                
        Returns:
            dict: Dictionary mapping active dimension indices to fitted PySRRegressor objects
        """
        active_dims = self.get_active_dimensions()
        if not active_dims:
            print("No active dimensions to interpret!")
            return {}
        
        # Filter active dimensions based on output_dim parameter
        if output_dim is not None:
            if output_dim not in active_dims:
                print(f"❗Requested output dimension {output_dim} is not active. Active dimensions: {active_dims}")
                return {}
            target_dims = [output_dim]
        else:
            target_dims = active_dims
        
        # Extract inputs and outputs at this layer level
        if parent_model is not None:
            with self._capture_layer_output(parent_model, inputs) as (layer_inputs, layer_outputs):
                pass
            
            # Use captured intermediate data
            if layer_inputs and layer_outputs:
                actual_inputs = layer_inputs[0]
                full_output = layer_outputs[0]
                # Filter to active dimensions only
                active_output = full_output[:, self.pruning_mask]
            else:
                raise RuntimeError("Failed to capture intermediate activations. Ensure parent_model contains this SymbolicMLP instance.")
        else:
            # Original behavior - use MLP directly
            actual_inputs = inputs
            self.InterpretSR_MLP.eval()
            with torch.no_grad():
                full_output = self.InterpretSR_MLP(inputs)
                # Filter to active dimensions only  
                active_output = full_output[:, self.pruning_mask]

        # Extract fit parameters
        if fit_params is None:
            fit_params = {}
        
        variable_names = fit_params.get('variable_names', None)
        
        # Apply variable transformations if provided
        if variable_transforms is not None:
            # Validate inputs
            if variable_names is not None and len(variable_names) != len(variable_transforms):
                raise ValueError(f"Length of variable_names ({len(variable_names)}) must match length of variable_transforms ({len(variable_transforms)})")
            
            # Apply transformations
            transformed_inputs = []
            for i, transform_func in enumerate(variable_transforms):
                try:
                    transformed_var = transform_func(actual_inputs)
                    # Ensure the result is 1D (batch_size,)
                    if transformed_var.dim() > 1:
                        transformed_var = transformed_var.flatten()
                    transformed_inputs.append(transformed_var.detach().cpu().numpy())
                except Exception as e:
                    raise ValueError(f"Error applying transformation {i}: {e}")
            
            # Stack transformed variables into input matrix
            actual_inputs_numpy = np.column_stack(transformed_inputs)
            
            # Store transformation info for later use in switch_to_equation
            self._variable_transforms = variable_transforms
            self._variable_names = variable_names
            
            print(f"🔄 Applied {len(variable_transforms)} variable transformations")
            if variable_names:
                print(f"   Variable names: {variable_names}")
        else:
            # Use original inputs
            actual_inputs_numpy = actual_inputs.detach().cpu().numpy()
            self._variable_transforms = None
            # Still store variable names even without transforms for switch_to_equation
            self._variable_names = variable_names

        timestamp = int(time.time())
        
        # Extract sr_params with defaults
        if sr_params is None:
            sr_params = {}
        
        # Set output_dims for compatibility
        self.output_dims = self.initial_dim
        
        # Run SR for active dimensions only
        results = {}
        
        for i, dim_idx in enumerate(target_dims):
            print(f"🛠️ Running SR on active dimension {dim_idx} ({i+1}/{len(target_dims)})")
            
            run_id = f"dim{dim_idx}_{timestamp}"
            final_sr_params = self._create_sr_params(save_path, run_id, sr_params)
            
            regressor = PySRRegressor(**final_sr_params)
            
            # Find the index of this dimension in the active output
            active_dim_index = active_dims.index(dim_idx)
            
            # Prepare fit arguments
            fit_args = [actual_inputs_numpy, active_output[:, active_dim_index].detach().cpu().numpy()]
            final_fit_params = dict(fit_params)  # Copy to avoid modifying original
            
            regressor.fit(*fit_args, **final_fit_params)
            
            results[dim_idx] = regressor
            
            best_eq = regressor.get_best()['equation']
            print(f"💡Best equation for active dimension {dim_idx}: {best_eq}")
        
        # Store in the format expected by SymbolicMLP (merge with existing dict)
        self.pysr_regressor = self.pysr_regressor | results
        
        print(f"❤️ SR on {self.mlp_name} active dimensions complete.")
        
        # For backward compatibility
        if output_dim is not None:
            return results.get(output_dim)
        else:
            return results

    def switch_to_equation(self, complexity: list = None):
        """
        Switch forward pass to use symbolic equations for active dimensions only.

        Overrides SymbolicMLP's switch_to_equation to handle pruned architectures correctly.
        Active dimensions use their discovered symbolic expressions, while inactive
        dimensions output zeros as enforced by the pruning mask.
        
        Args:
            complexity (list or int, optional): Specific complexity levels to use.
                                               If list, maps to active dimensions in order.
                                               If int, uses same complexity for all active dimensions.
                                               If None, uses best overall equation for each active dimension.
        """
        active_dims = self.get_active_dimensions()
        if not active_dims:
            print("❗No active dimensions to switch to equations.")
            return
        
        # Check if we have equations for all active dimensions
        if not hasattr(self, 'pysr_regressor') or not self.pysr_regressor:
            print("❗No equations found. You need to first run .distill.")
            return
        
        missing_dims = []
        for dim in active_dims:
            if dim not in self.pysr_regressor:
                missing_dims.append(dim)
        
        if missing_dims:
            print(f"❗Missing equations for active dimensions {missing_dims}. You need to run .distill on all active dimensions first.")
            return
        
        # Store original MLP for potential restoration
        if not hasattr(self, '_original_mlp'):
            self._original_mlp = self.InterpretSR_MLP
        
        # Set output_dims to initial_dim for compatibility
        self.output_dims = self.initial_dim
        
        # Get equations for active dimensions only
        equation_funcs = {}
        equation_vars = {}
        equation_strs = {}
        
        for i, dim_idx in enumerate(active_dims):
            # Get complexity for this specific dimension
            dim_complexity = None
            if complexity is not None:
                if isinstance(complexity, list):
                    if i < len(complexity):
                        dim_complexity = complexity[i]
                else:
                    dim_complexity = complexity
            
            result = self._get_equation(dim_idx, dim_complexity)
            if result is None:
                print(f"⚠️ Failed to get equation for dimension {dim_idx}")
                return
                
            f, vars_sorted = result
            
            # Map variables to indices using helper method
            try:
                var_indices = self._map_variables_to_indices(vars_sorted, dim_idx)
            except ValueError as e:
                print(f"⚠️ Warning: {e}")
                return
            
            equation_funcs[dim_idx] = f
            equation_vars[dim_idx] = var_indices
            
            # Get equation string for display
            regressor = self.pysr_regressor[dim_idx]
            if dim_complexity is None:
                equation_strs[dim_idx] = regressor.get_best()["equation"]
            else:
                matching_rows = regressor.equations_[regressor.equations_["complexity"] == dim_complexity]
                equation_strs[dim_idx] = matching_rows["equation"].values[0]
        
        # Store the equation information
        self._equation_funcs = equation_funcs
        self._equation_vars = equation_vars
        self._using_equation = True
        
        # Print success messages
        print(f"✅ Successfully switched {self.mlp_name} to symbolic equations for {len(active_dims)} active dimensions:")
        for dim_idx in active_dims:
            print(f"   Dimension {dim_idx}: {equation_strs[dim_idx]}")
            
            # Display variable names properly
            var_names_display = []
            if hasattr(self, '_variable_names') and self._variable_names is not None:
                # Use custom variable names
                for idx in equation_vars[dim_idx]:
                    if idx < len(self._variable_names):
                        var_names_display.append(self._variable_names[idx])
                    else:
                        var_names_display.append(f"transform_{idx}")
            else:
                # Use default x0, x1, etc. format
                var_names_display = [f'x{i}' for i in equation_vars[dim_idx]]
            
            print(f"   Variables: {var_names_display}")
        
        print(f"🎯 Active dimensions {active_dims} now using symbolic equations.")
        print(f"🔒 Inactive dimensions will output zeros.")

    def forward(self, x):
        """
        Forward pass through the model with pruning mask applied.

        Uses parent's forward method and applies pruning mask to enforce
        zero outputs for inactive dimensions.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Output tensor with inactive dimensions masked to zero
        """
        if hasattr(self, '_using_equation') and self._using_equation:
            # For equation mode, we need to handle the PruningMLP case specially
            batch_size = x.shape[0]
            # Initialize output tensor with zeros for all dimensions
            output = torch.zeros(batch_size, self.initial_dim, dtype=x.dtype, device=x.device)
            
            # Fill in active dimensions with symbolic equations
            active_dims = self.get_active_dimensions()
            for dim_idx in active_dims:
                if dim_idx in self._equation_funcs:
                    equation_func = self._equation_funcs[dim_idx]
                    var_indices = self._equation_vars[dim_idx]
                    
                    # Extract variables needed for this dimension
                    try:
                        selected_inputs = self._extract_variables_for_equation(x, var_indices, dim_idx)
                        
                        # Convert to numpy for the equation function
                        numpy_inputs = [inp.detach().cpu().numpy() for inp in selected_inputs]
                        
                        # Evaluate the equation for this dimension
                        result = equation_func(*numpy_inputs)
                        
                        # Convert back to torch tensor with same device/dtype as input
                        result_tensor = torch.tensor(result, dtype=x.dtype, device=x.device)
                        
                        # Ensure result is 1D (batch_size,)
                        if result_tensor.dim() == 0:
                            result_tensor = result_tensor.expand(batch_size)
                        elif result_tensor.dim() > 1:
                            result_tensor = result_tensor.flatten()
                        
                        output[:, dim_idx] = result_tensor
                    except Exception as e:
                        print(f"⚠️ Error evaluating equation for dimension {dim_idx}: {e}")
            
            # Apply pruning mask to ensure inactive dimensions are zero
            return output * self.pruning_mask
        else:
            # Use parent's forward method and apply pruning mask
            output = super().forward(x)
            return output * self.pruning_mask

    def get_importance(self, sample_data: torch.Tensor, parent_model=None):
        """
        Get ordered list of output dimensions from most to least important.

        For PruningMLP, this considers only currently active (non-pruned) dimensions.
        Evaluates importance by computing standard deviation across sample data,
        with higher standard deviation indicating higher importance.

        Args:
            sample_data (torch.Tensor): Sample input data to evaluate dimension importance.
                                       Typically a subset of validation data.
            parent_model (nn.Module, optional): The parent model containing this PruningMLP instance.
                                              If provided, will trace intermediate activations to get
                                              the actual outputs at this layer level for importance evaluation.
                                              
        Returns:
            dict: Dictionary with keys:
                - 'importance': List of dimension indices ordered from most important to least important.
                               Only includes currently active (non-pruned) dimensions.
                - 'std': List of standard deviation values corresponding to the ordered active dimensions
            
        Example:
            >>> result = model.f_net.get_importance(validation_data)
            >>> print(f"Most important active dimension: {result['importance'][0]} (std: {result['std'][0]})")
            >>> print(f"Least important active dimension: {result['importance'][-1]} (std: {result['std'][-1]})")
        """
        # Get full importance from parent class
        full_result = super().get_importance(sample_data, parent_model)
        
        # Filter to only active dimensions
        active_dims = self.get_active_dimensions()
        if not active_dims:
            return {'importance': [], 'std': []}
        
        # Filter parent results to active dimensions only
        active_importance = []
        active_std = []
        
        for i, dim_idx in enumerate(full_result['importance']):
            if dim_idx in active_dims:
                active_importance.append(dim_idx)
                active_std.append(full_result['std'][i])
        
        return {
            'importance': active_importance,
            'std': active_std
        }

    def save_model(self, save_path: str, save_pytorch: bool = True, save_regressors: bool = True):
        """
        Save the PruningMLP model including PyTorch weights, PySR regressors, and pruning state.

        Extends SymbolicMLP's save functionality to include pruning-specific information:
        - PyTorch model state dict (if save_pytorch=True)
        - All fitted PySR regressors for active dimensions (if save_regressors=True)
        - Model metadata and configuration
        - Pruning mask and dimension information
        - Pruning schedule and current state
        
        Args:
            save_path (str): Base path for saving (without extension)
            save_pytorch (bool, optional): Whether to save PyTorch model state. Defaults to True.
            save_regressors (bool, optional): Whether to save PySR regressors. Defaults to True.
            
        Example:
            >>> model.f_net = PruningMLP(model.f_net, initial_dim=32, target_dim=2, mlp_name="f_net")
            >>> # ... train, prune, and run distill ...
            >>> model.f_net.save_model("./saved_models/pruned_model")
            
        Note:
            This creates multiple files with pruning-specific metadata:
            - {save_path}_pytorch.pth: PyTorch model state
            - {save_path}_metadata.pkl: Model configuration, pruning state, and metadata
            - {save_path}_regressor_dim{i}.pkl: PySR regressors (only for active dimensions)
        """
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        saved_files = []
        
        # Save PyTorch model state
        if save_pytorch:
            pytorch_path = f"{save_path}_pytorch.pth"
            torch.save(self.InterpretSR_MLP.state_dict(), pytorch_path)
            saved_files.append(pytorch_path)
            print(f"✅ Saved PyTorch model state to {pytorch_path}")
        
        # Save enhanced metadata for PruningMLP
        metadata = {
            'mlp_name': self.mlp_name,
            'output_dims': getattr(self, 'output_dims', None),
            'variable_transforms_available': hasattr(self, '_variable_transforms') and self._variable_transforms is not None,
            'variable_names': getattr(self, '_variable_names', None),
            'using_equation': getattr(self, '_using_equation', False),
            'class_name': self.__class__.__name__,
            'equation_vars': getattr(self, '_equation_vars', {}),
            'regressor_dimensions': list(self.pysr_regressor.keys()) if hasattr(self, 'pysr_regressor') else [],
            # Pruning-specific metadata
            'initial_dim': self.initial_dim,
            'current_dim': self.current_dim,
            'target_dim': self.target_dim,
            'pruning_mask': self.pruning_mask.tolist() if hasattr(self, 'pruning_mask') else None,
            'pruning_schedule': self.pruning_schedule,
            'active_dimensions': self.get_active_dimensions()
        }
        
        metadata_path = f"{save_path}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        saved_files.append(metadata_path)
        print(f"✅ Saved pruning model metadata to {metadata_path}")
        
        # Save PySR regressors (only for active dimensions)
        if save_regressors and hasattr(self, 'pysr_regressor') and self.pysr_regressor:
            active_dims = self.get_active_dimensions()
            regressor_files = []
            
            for dim, regressor in self.pysr_regressor.items():
                if dim in active_dims:  # Only save regressors for active dimensions
                    regressor_path = f"{save_path}_regressor_dim{dim}.pkl"
                    try:
                        # Use PySR's built-in pickling support
                        with open(regressor_path, 'wb') as f:
                            pickle.dump(regressor, f)
                        regressor_files.append(regressor_path)
                        saved_files.append(regressor_path)
                        print(f"✅ Saved regressor for active dimension {dim} to {regressor_path}")
                    except Exception as e:
                        print(f"⚠️ Warning: Could not save regressor for dimension {dim}: {e}")
                else:
                    print(f"ℹ️ Skipping inactive dimension {dim} regressor")
            
            if regressor_files:
                print(f"✅ Saved {len(regressor_files)} PySR regressors for active dimensions")
        elif save_regressors:
            print("ℹ️ No PySR regressors found to save")
        
        print(f"🎯 Pruning model save complete. Created {len(saved_files)} files with base name: {save_path}")
        print(f"📊 Saved state: {self.current_dim}/{self.initial_dim} active dimensions")
        return saved_files

    @classmethod
    def load_model(cls, save_path: str, mlp_architecture: nn.Module = None, device: str = 'cpu'):
        """
        Load a previously saved PruningMLP model with all components.

        Reconstructs the complete PruningMLP instance including:
        - PyTorch model weights (requires architecture)
        - All fitted PySR regressors for active dimensions
        - Model metadata and configuration
        - Pruning mask and dimension state
        - Pruning schedule
        
        Args:
            save_path (str): Base path used during saving (without extension)
            mlp_architecture (nn.Module, optional): PyTorch model architecture to load weights into.
                                                   If None, only metadata and regressors are loaded.
            device (str, optional): Device to load tensors to ('cpu', 'cuda', etc.). Defaults to 'cpu'.
            
        Returns:
            PruningMLP: Reconstructed PruningMLP instance with loaded components

        Example:
            >>> # Create same architecture as original
            >>> mlp = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 32))
            >>> loaded_model = PruningMLP.load_model("./saved_models/pruned_model", mlp)
            >>> # Model ready to use with pruning state and equations
            >>> loaded_model.switch_to_equation()
            
        Note:
            The mlp_architecture must match the original architecture exactly for weight loading.
            Pruning state and active dimensions are automatically restored.
        """
        # Load metadata first
        metadata_path = f"{save_path}_metadata.pkl"
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Verify this is a PruningMLP save
        if metadata['class_name'] != 'PruningMLP':
            raise ValueError(f"Expected PruningMLP save file, found {metadata['class_name']}")

        print(f"📂 Loading PruningMLP model: {metadata['mlp_name']}")
        
        # Extract pruning dimensions
        initial_dim = metadata['initial_dim']
        current_dim = metadata['current_dim']
        target_dim = metadata['target_dim']
        
        if mlp_architecture is None:
            raise ValueError("mlp_architecture is required when loading PruningMLP")

        # Create PruningMLP instance
        instance = cls(mlp_architecture,
                      initial_dim=initial_dim,
                      target_dim=target_dim,
                      mlp_name=metadata['mlp_name'])
        
        # Load PyTorch weights if available
        pytorch_path = f"{save_path}_pytorch.pth"
        if os.path.exists(pytorch_path):
            state_dict = torch.load(pytorch_path, map_location=device, weights_only=True)
            instance.InterpretSR_MLP.load_state_dict(state_dict)
            instance.InterpretSR_MLP.eval()  # Ensure model is in eval mode after loading
            print(f"✅ Loaded PyTorch weights from {pytorch_path}")
        else:
            print(f"⚠️ PyTorch weights file not found: {pytorch_path}")
        
        # Restore pruning state
        instance.current_dim = current_dim
        instance.pruning_schedule = metadata.get('pruning_schedule')
        
        # Restore pruning mask
        if metadata['pruning_mask'] is not None:
            pruning_mask = torch.tensor(metadata['pruning_mask'], dtype=torch.bool)
            instance.pruning_mask.data = pruning_mask.data
        
        # Restore general metadata
        instance.output_dims = metadata.get('output_dims')
        instance._variable_names = metadata.get('variable_names')
        instance._using_equation = metadata.get('using_equation', False)
        instance._equation_vars = metadata.get('equation_vars', {})
        
        # Load PySR regressors
        regressor_dims = metadata.get('regressor_dimensions', [])
        instance.pysr_regressor = {}
        equation_funcs = {}
        
        loaded_regressors = 0
        for dim in regressor_dims:
            regressor_path = f"{save_path}_regressor_dim{dim}.pkl"
            if os.path.exists(regressor_path):
                try:
                    with open(regressor_path, 'rb') as f:
                        regressor = pickle.load(f)
                    instance.pysr_regressor[dim] = regressor
                    
                    # Rebuild equation function if model was using equations
                    if instance._using_equation and dim in instance._equation_vars:
                        result = instance._get_equation(dim)
                        if result is not None:
                            equation_funcs[dim] = result[0]
                    
                    loaded_regressors += 1
                    print(f"✅ Loaded regressor for dimension {dim}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not load regressor for dimension {dim}: {e}")
            else:
                print(f"⚠️ Warning: Regressor file not found for dimension {dim}: {regressor_path}")
        
        if loaded_regressors > 0:
            print(f"✅ Loaded {loaded_regressors} PySR regressors")
            
            # Restore equation functions if model was using equations
            if instance._using_equation and equation_funcs:
                instance._equation_funcs = equation_funcs
                print(f"✅ Restored symbolic equation functions for {len(equation_funcs)} dimensions")
        else:
            print("ℹ️ No PySR regressors found to load")
        
        # Display loaded state
        active_dims = instance.get_active_dimensions()
        print(f"📊 Loaded pruning state: {len(active_dims)}/{initial_dim} active dimensions: {active_dims}")
        print(f"🎯 Pruning model loading complete: {metadata['mlp_name']}")
        
        return instance