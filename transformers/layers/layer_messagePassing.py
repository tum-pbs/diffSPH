import warnings
from numpy import integer
import torch
from torch import Tensor
import torch.nn as nn

from .windows import getWindowFunction

try:
    import torch_geometric
    from torch_geometric.utils import scatter, segment
    from torch_geometric.utils.num_nodes import maybe_num_nodes
except ImportError:
    torch_geometric = None
from typing import Optional, Union, Tuple
 

from .activation import getActivationLayer
from .basisFunctions import basisEncoderLayer
from .layer_positionEncoder import BasisEncoder, computeBasisEncoderOutputShape
from .networkUtil import verboseBannerPrint
from .networkUtil import verbosePrint
from .sparse import buildSparseTensor
from .softmax import softmax
from .mlp import buildMLPwDict, getDefaultMLPDict


from typing import List, Optional

def checkTensorShape(tensor: Tensor, expected_shape: List[str], shape_dict: dict, verbose: bool = False, logName: Optional[str] = None):
    if tensor is None:
        return
    # if verbose:
    #     name = f' for {logName}' if logName is not None else ''
    #     print(f'Checking tensor{name} shape: {tensor.shape} against expected: {expected_shape}')
    shape = tensor.shape
    if len(shape) != len(expected_shape):
        raise ValueError(f'Expected tensor to have {len(expected_shape)} dimensions, got {len(shape)} dimensions with shape {shape}')
    for i, dim in enumerate(expected_shape):
        if isinstance(dim, int):
            if shape[i] != dim:
                raise ValueError(f'Expected dimension {i} of tensor to have size {dim}, got {shape[i]}')
        elif '*' in dim or '//' in dim:
            LHS, RHS = dim.split('//') if '//' in dim else dim.split('*')
            if LHS.isdigit() and RHS.isdigit():
                lhs = int(LHS)
                rhs = int(RHS)
                if shape[i] % rhs != 0 or shape[i] // rhs != lhs:
                    raise ValueError(f'Expected dimension {i} of tensor to have size {lhs}*{rhs}, got {shape[i]}')  
            elif LHS.isdigit() and RHS in shape_dict:
                lhs = int(LHS)
                rhs = shape_dict[RHS]
                if rhs is not None and (shape[i] % rhs != 0 or shape[i] // rhs != lhs):
                    raise ValueError(f'Expected dimension {i} of tensor to have size {lhs}*{rhs} ({RHS}), got {shape[i]}')  
            elif LHS in shape_dict and RHS.isdigit():
                lhs = shape_dict[LHS]
                rhs = int(RHS)
                if lhs is not None and (shape[i] % rhs != 0 or shape[i] // rhs != lhs):
                    raise ValueError(f'Expected dimension {i} of tensor to have size {lhs} ({LHS})*{rhs}, got {shape[i]}')
            elif LHS in shape_dict and RHS in shape_dict:
                lhs = shape_dict[LHS]
                rhs = shape_dict[RHS]
                if lhs is not None and rhs is not None and (shape[i] % rhs != 0 or shape[i] // rhs != lhs):
                    raise ValueError(f'Expected dimension {i} of tensor to have size {lhs} ({LHS})*{rhs} ({RHS}), got {shape[i]}')
            else:
                raise ValueError(f'Unknown dimension specifier: {dim}')
        else:
            if dim.isdigit():
                expected_dim = int(dim)
                if shape[i] != expected_dim:
                    raise ValueError(f'Expected dimension {i} of tensor to have size {expected_dim}, got {shape[i]}')
            elif dim in shape_dict:
                expected_dim = shape_dict[dim]
                if expected_dim is not None and shape[i] != expected_dim:
                    raise ValueError(f'Expected dimension {i} of tensor to have size {expected_dim} ({dim}), got {shape[i]}')
            elif dim == '*':
                continue
            else:
                raise ValueError(f'Unknown dimension specifier: {dim}')
    if verbose:
        name = f' for {logName}' if logName is not None else ''
        print(f'Tensor{name} has expected shape: {shape}')

"""
This is a message passing layer that is part of the transformer architecture, however, it is expanded in functionality
to match what is normally expected from a graph neural network layer.

The normal message passing inputs are:
- queryTokens: the tokens for which we want to compute the new representation (shape: [batch_size, numQueryTokens, latentSpaceSize]) [i]
- keyTokens: the tokens that provide the context (shape: [batch_size, num_nodes_neighbor, latentSpaceSize]) [j]
- edge_index: the indices of the edges in the sparse neighborhood (shape: [2, num_edges]) where the first row are the indices for queryTokens and the second row for keyTokens
- edge_attr: the features associated with each edge (shape: [num_edges, edgeFeatureSize]) (spatial information mostly)
- edge_vector: the length of each edge (shape: [num_edges, spatial_dim]) (optional, can be computed from edge_attr if needed)

By adding an attention Mechanism, we add an aditional input:
- attention_values: the values to modulate the attention scores (shape: [num_edges, num_attention_heads]) (optional)

We also add a shepard like scaling value to support spatial normalization
- S_k: the shepard values to scale the attention scores (shape: [num_edges]) (optional)

The output is:
- outputTokens: the new representation of the query tokens (shape: [batch_size, numQueryTokens, latentSpaceSize])
The output tokens have the same shape as the input query tokens, but their values have been updated based on the message passing mechanism.

This leaves the central steps, which we can handle via branching and also add edge gating steps to all networks:
1. Project to  [b,n,h,t]
2. Gather to q_i, q_j [ne,h,t]
3. Apply message generation logic:
    - Transformer:
        - Nothing to do
    - GNN:
        - gather the relevant features as inputs, e.g., concatenate [q_i, q_j, edge_attr, edge_attention]
        - feed the concatenated features into an MLP: [ne,C]->[ne,h*t]
        - Reshape to match transformer style [ne,h,t]
    - CConv:
        - Construct a weight matrix that maps L to L features, i.e., [h*t,h*t], conditioned on [b,b]: [ne,b,b]->[ne,h*t,h*t]
        - Apply the weight matrix on the incoming features: [ne,h*t,h*t].[ne,h*t] -> [ne,h*t]
        - Reshape to match transformer style [ne,h,t]
4. Apply edge_attention if given (expand the attention values to match) [ne,h,t]x[ne,h,1] -> [ne,h,t]
5. Apply edge_weighting using s_k [ne,h,t]x[ne] -> [ne,h,t]
6. Compute the window function if required [ne,d] -> [ne] and apply it [ne,h,t]x[ne] -> [ne,h,t]
7. Apply edge gating (with optional basis encodes): [ne,d]->[ne,b*b], apply linear projection [ne,b*b]->[ne,h] and apply to messages [ne,h,t]x[ne,h]->[ne,h,t]
8. scatter sum up the result [ne,h,t]->[b,n,h,t]
9. project the resulting values linearly using W_o after flattening: [b,n,h*t] -> [b,n,L]

For generalization we treat edge attributes, i.e., features of edges, and spatial relations, i.e., relative positions, distances, etc., separately. This means that the layer can be used in non-spatial contexts, spatial-contexts and mixed contexts. Consequently, we have the following parameters:

- node_feature_dim: the size of the input tokens [I], required
- transformer_dim:  the size of each attention head [T], required
- edge_feature_dim: the size of the edge features [E], Optional, if not given, it is set to 0
- spatial_dim:      the size of the spatial dimension [D], Optional, if not given, it is set to 0
- attention_heads:  the number of attention heads [H], default: 1
- latent_dim:       the latent feature size of the layer [L], optional, if not given, it is set H * T
- output_dim:       the size of the output tokens [O], optional, if not given, it is set to L

- split_across_heads: whether to split the latent space across attention heads (True) or have each head operate on the full latent space (False), default: True, if True, then either L must be a multiple of H or an input projection is used to map the input tokens to a latent space that is divisble by H. If false, the gathered features are repeated across heads.
- use_input_proj:  whether to use an input projection layer to map input tokens to latent space, default: True
- use_output_proj: whether to use an output projection layer to map latent space to output tokens, default: True
- message_mode: 'transformer', 'gnn', 'cconv', default: 'gnn':
    - transformer: standard transformer attention mechanism
    - gnn:         graph neural network style message passing using an MLP to compute messages
    - cconv:       continuous convolution style message passing using a weight matrix conditioned on edge attributes and/or spatial relations

- input_proj_linear: whether the input projection is linear (True) or an MLP (False), default: True
- input_proj_mlp_dict: dictionary defining the MLP for the input projection, default: None
- output_proj_linear: whether the output projection is linear (True) or an MLP (False), default: True
- output_proj_mlp_dict: dictionary defining the MLP for the output projection, default: None

- relative_position_bias: whether to use relative position bias, default: False, if True, spatial edge features are included in the GNN input for message generation, only in GNN mode
- rpb_base_encoding: whether to use a basis function encoding for the relative position bias, default: True
- rpb_base_terms: number of basis functions to use for the relative position bias, default: 8
- rpb_base_mode: mode for the basis function encoding, one of 'cat', 'sum', 'prod', 'outer', 'i', 'j', 'k', default: 'cat'
- rpb_proj: whether to use a linear projection after the basis function encoding for the relative position bias, default: True
- rpb_split: whether to split the relative position bias across attention heads, default: True
- rpb_dim: dimensionality of the relative position bias after projection, default: None, if not given, it is set to T in split mode and H*T otherwise
- rpb_proj_linear: whether the projection for the relative position bias is linear (True) or an MLP (False), default: True
- rpb_proj_mlp_dict: dictionary defining the MLP for the relative position bias projection, default: None

- window_function: whether to use a window function based on spatial relations, default: False
- window_function_type: type of window function
- gnn_window_function: whether to use the window function as an additional input to the message generation (True) or to scale the messages (False), default: False

- gnn_node_i_features: whether to include the features of the target node (i) in the message generation, default: False
- gnn_node_j_features: whether to include the features of the source node (j) in the message generation, default: True
- gnn_node_sum_features: whether to include the sum of the features of the target and source nodes (i+j) in the message generation, default: False
- gnn_node_diff_features: whether to include the difference of the features of the target and source nodes (i-j) in the message generation, default: False
- gnn_edge_features: whether to include edge features in the message generation, default: True
- gnn_attention_features: whether to include attention values in the message generation, default: True, if False, attention values are only used to scale the messages
- gnn_spatial_features: whether to include spatial edge features in the message generation, default, True
- gnn_spatial_distance: whether to include the distance of the edge in the message generation, default: False
- gnn_mlp_dict: dictionary defining the MLP for the message generation in GNN mode, default: None
"""


class MessagePassingLayer(torch.nn.Module):
    def __init__(self, 
                node_feature_dim: int,
                transformer_features: int,
                edgeFeatureSize: int = 0,
                spatial_dim: int = 0,
                multi_heads: int = 1,
                latent_dim: Optional[int] = None,
                output_dim: Optional[int] = None,

                split_across_heads: bool = True,
                use_input_proj: bool = True,
                use_output_proj: bool = True,
                message_mode: str = 'gnn', # 'transformer', 'gnn', 'cconv'

                input_proj_linear: bool = True,
                input_proj_mlp_dict: Optional[dict] = None,
                output_proj_linear: bool = True,
                output_proj_mlp_dict: Optional[dict] = None,
                skipConnections: bool = True,

                relative_position_bias: bool = False,
                rpb_base_encoding: bool = True,
                rpb_base_terms: int = 8,
                rpb_base_basis: str = 'ffourier', # 'gaussian', 'fourier', 'ffourier'
                rpb_base_mode: str = 'cat', # 'cat', 'sum', 'prod', 'outer', 'i', 'j', 'k'
                rpb_proj: bool = True,
                rpb_split: bool = True,
                rpb_dim: Optional[int] = None,
                rpb_proj_linear: bool = True,
                rpb_proj_mlp_dict: Optional[dict] = None,

                window_function: bool = False,
                window_function_type: str = 'cubicSpline', # 'gaussian', 'cosine', 'tanh'
                window_function_normalize: bool = False,
                window_function_as_gate: bool = True,

                gnn_window_function: bool = False,
                gnn_node_i_features: bool = False,
                gnn_node_j_features: bool = True,
                gnn_node_sum_features: bool = False,
                gnn_node_diff_features: bool = False,
                gnn_edge_features: bool = True,
                gnn_attention_features: bool = True,
                gnn_spatial_features: bool = True,
                gnn_spatial_distance: bool = False,
                gnn_mlp_dict: Optional[dict] = None,     

                # Each edge gate is realized by using a linear projection to compute a gating value with an optional activation
                edge_gating: bool = False,
                edge_gating_repeat: bool = False, # whether to repeat the gating values across heads (True) or have a separate gating value for each head (False)
                edge_gating_mode: str = 'multiply', # 'multiply', 'add'
                edge_gating_edge_vectors: bool = False, 
                edge_gating_edge_features: bool = False, 
                edge_gating_rpb: bool = False, 
                edge_gating_activation: Optional[str] = 'sigmoid', # 'sigmoid', 'tanh', 'celu', None

                message_activation: Optional[str] = None,

                multiHeadAggregation: str = 'concat', # 'mean', 'concat'
                verbose: bool = False
                 ):
        super(MessagePassingLayer, self).__init__()
        verboseBannerPrint('Initializing MessagePassingLayer', True)

        self.node_feature_dim = node_feature_dim
        self.transformer_features = transformer_features
        self.edgeFeatureSize = edgeFeatureSize
        self.spatial_dim = spatial_dim
        self.multi_heads = multi_heads
        self.latent_dim = latent_dim if latent_dim is not None else multi_heads * transformer_features
        self.output_dim = output_dim if output_dim is not None else self.node_feature_dim
        verbosePrint(f'Dimension Features:\n\tnode_feature_dim: {node_feature_dim}, transformer_features: {transformer_features}, edgeFeatureSize: {edgeFeatureSize}, spatial_dim: {spatial_dim}', verbose)
        verbosePrint(f'\tmulti_heads: {multi_heads}, latent_dim: {self.latent_dim}, output_dim: {self.output_dim}', verbose)

        self.split_across_heads = split_across_heads
        self.multiHeadAggregation = multiHeadAggregation
        self.use_input_proj = use_input_proj
        self.use_output_proj = use_output_proj
        self.message_mode = message_mode
        self.input_proj_linear = input_proj_linear
        self.input_proj_mlp_dict = input_proj_mlp_dict if input_proj_mlp_dict is not None else getDefaultMLPDict()
        self.output_proj_linear = output_proj_linear
        self.output_proj_mlp_dict = output_proj_mlp_dict if output_proj_mlp_dict is not None else getDefaultMLPDict()
        self.skipConnections = skipConnections

        if message_activation is None:
            self.activation = nn.Identity()
            self.activationName = 'identity'
        else:
            self.activationName = message_activation.split('(')[0] if '(' in message_activation else message_activation
            activationArguments = () if '(' not in message_activation else message_activation[message_activation.index('(')+1:message_activation.index(')')].split(',')
            activationArguments = tuple([float(arg) for arg in activationArguments])
            self.activation = getActivationLayer(self.activationName, *activationArguments)


        verbosePrint(f'Architecture:\n\tsplit_across_heads: {split_across_heads}, use_input_proj: {use_input_proj}, use_output_proj: {use_output_proj}, message_mode: {message_mode}', verbose, separator=True)
        verbosePrint(f'\tinput_proj_linear: {input_proj_linear}, input_proj_mlp_dict: {self.input_proj_mlp_dict}', verbose)
        verbosePrint(f'\toutput_proj_linear: {output_proj_linear}, output_proj_mlp_dict: {self.output_proj_mlp_dict} multiHeadAggregation: {self.multiHeadAggregation}, skipConnections: {self.skipConnections}', verbose)
        verbosePrint(f'\tmessage_activation: {self.activationName}', verbose)

        self.relative_position_bias = relative_position_bias
        self.rpb_base_encoding = rpb_base_encoding
        self.rpb_base_terms = rpb_base_terms
        self.rpb_base_basis = rpb_base_basis
        self.rpb_base_mode = rpb_base_mode
        self.rpb_proj = rpb_proj
        self.rpb_split = rpb_split
        self.rpb_dim = rpb_dim if rpb_dim is not None else (multi_heads * transformer_features if rpb_split else transformer_features)
        self.rpb_proj_linear = rpb_proj_linear
        self.rpb_proj_mlp_dict = rpb_proj_mlp_dict if rpb_proj_mlp_dict is not None else getDefaultMLPDict()
        verbosePrint(f'Relative Position Bias:\n\trelative_position_bias: {relative_position_bias}, rpb_base_encoding: {rpb_base_encoding}, rpb_base_terms: {rpb_base_terms}, rpb_base_mode: {rpb_base_mode} rpb_base_basis: {rpb_base_basis}', verbose, separator=True)
        verbosePrint(f'\trpb_proj: {rpb_proj}, rpb_split: {rpb_split}, rpb_dim: {self.rpb_dim}', verbose)
        verbosePrint(f'\trpb_proj_linear: {rpb_proj_linear}, rpb_proj_mlp_dict: {self.rpb_proj_mlp_dict}', verbose)

        self.window_function = window_function
        self.window_function_type = window_function_type
        self.window_function_normalize = window_function_normalize
        self.gnn_window_function = gnn_window_function
        self.window_function_as_gate = window_function_as_gate
        verbosePrint(f'Window Function:\n\twindow_function: {window_function}, window_function_type: {window_function_type}, gnn_window_function: {gnn_window_function}, window_function_as_gate: {window_function_as_gate}', verbose, separator=True)


        self.gnn_node_i_features = gnn_node_i_features
        self.gnn_node_j_features = gnn_node_j_features
        self.gnn_node_sum_features = gnn_node_sum_features
        self.gnn_node_diff_features = gnn_node_diff_features
        self.gnn_edge_features = gnn_edge_features
        self.gnn_attention_features = gnn_attention_features
        self.gnn_spatial_features = gnn_spatial_features
        self.gnn_spatial_distance = gnn_spatial_distance
        self.gnn_mlp_dict = gnn_mlp_dict if gnn_mlp_dict is not None else getDefaultMLPDict()
        verbosePrint(f'GNN Message Generation:\n\tgnn_node_i_features: {gnn_node_i_features}, gnn_node_j_features: {gnn_node_j_features}, gnn_node_sum_features: {gnn_node_sum_features}, gnn_node_diff_features: {gnn_node_diff_features}', verbose, separator=True)
        verbosePrint(f'\tgnn_edge_features: {gnn_edge_features}, gnn_attention_features: {gnn_attention_features}, gnn_spatial_features: {gnn_spatial_features}, gnn_spatial_distance: {gnn_spatial_distance}', verbose)
        verbosePrint(f'\tgnn_mlp_dict: {self.gnn_mlp_dict}', verbose)

        self.edge_gating = edge_gating
        self.edge_gating_repeat = edge_gating_repeat

        if edge_gating_activation is None:
            self.gatingActivation = nn.Identity()
            self.gatingActivationName = 'identity'
        else:
            self.gatingActivationName = edge_gating_activation.split('(')[0] if '(' in edge_gating_activation else edge_gating_activation
            activationArguments = () if '(' not in edge_gating_activation else edge_gating_activation[edge_gating_activation.index('(')+1:edge_gating_activation.index(')')].split(',')
            activationArguments = tuple([float(arg) for arg in activationArguments])
            self.gatingActivation = getActivationLayer(self.gatingActivationName, *activationArguments)
        
        self.edge_gating_edge_vectors = edge_gating_edge_vectors
        self.edge_gating_edge_features = edge_gating_edge_features
        self.edge_gating_rpb = edge_gating_rpb
        self.edge_gating_mode = edge_gating_mode

        verbosePrint(f'Edge Gating:\n\tedge_gating: {self.edge_gating}, edge_gating_repeat: {self.edge_gating_repeat}, edge_gating_edge_vectors: {self.edge_gating_edge_vectors}, edge_gating_edge_features: {self.edge_gating_edge_features}, edge_gating_rpb: {self.edge_gating_rpb}\n\tedge_gating_activation: {self.gatingActivationName}', verbose, separator=True)

        self.verbose = verbose

        shape_dict = {
            'D': spatial_dim,
            'E': edgeFeatureSize,
            'H': self.multi_heads,
            'L': self.latent_dim,
            'T': self.transformer_features,
            'I': self.node_feature_dim,
            'O': self.output_dim
        }


        verboseBannerPrint('Building MessagePassingLayer', verbose)

        verboseBannerPrint('Building Input Projection', verbose)
        if self.use_input_proj:
            verbosePrint(f'Input projection enabled', verbose)
            verbosePrint(f'\tInput feature size: {self.node_feature_dim}', verbose)
            verbosePrint(f'\tLatent feature size: {self.latent_dim}', verbose)

            if self.latent_dim % self.multi_heads != 0 and self.split_across_heads:
                raise ValueError(f'latent_dim must be a multiple of multi_heads if split_across_heads is True, got latent_dim={self.latent_dim}, multi_heads={self.multi_heads}')

            if self.input_proj_linear:
                verbosePrint(f'Using linear input projection layer', verbose)
                self.inputProjection = nn.Linear(self.node_feature_dim, self.latent_dim)
                verbosePrint(f'\tShape: {self.node_feature_dim} -> {self.latent_dim}', verbose)
            else:
                verbosePrint(f'Using MLP input projection layer', verbose)
                self.inputProjection = buildMLPwDict(self.input_proj_mlp_dict, verbose, inputDim=self.node_feature_dim, outputDim=self.latent_dim, verbosePrefix='\t')
        else:
            verbosePrint(f'Input projection disabled, using identity', verbose)
            if self.node_feature_dim != self.latent_dim:
                raise ValueError(f'node_feature_dim must be equal to latent_dim if use_input_proj is False, got node_feature_dim={self.node_feature_dim}, latent_dim={self.latent_dim}')
            self.inputProjection = nn.Identity()

            if self.latent_dim % self.multi_heads != 0 and self.split_across_heads:
                raise ValueError(f'latent_dim must be a multiple of multi_heads if split_across_heads is True, got latent_dim={self.latent_dim}, multi_heads={self.multi_heads}')
            
        verboseBannerPrint('Building Relative Position Bias', verbose)
        if self.relative_position_bias or self.message_mode == 'cconv':
            verbosePrint(f'Relative position bias enabled', verbose)

            verbosePrint(f'\tSpatial dimension: {self.spatial_dim}', verbose)
            verbosePrint(f'\tSplit across heads: {self.rpb_split}', verbose)
            verbosePrint(f'\tRelative position bias dimension: {self.rpb_dim}', verbose)

            ################################################################################
            # Start by collecting all properties for the RPB
            ################################################################################

            spatial_dim = self.spatial_dim

            basis_terms = self.rpb_base_terms
            basis_function = self.rpb_base_basis
            mode = self.rpb_base_mode

            skip_basis = not self.rpb_base_encoding
            split_across_heads = self.rpb_split

            project_out = self.rpb_proj
            project_linear = self.rpb_proj_linear
            project_mlp_dict = self.rpb_proj_mlp_dict
            out_dim = self.rpb_dim

            basisEncoderOutputShape = computeBasisEncoderOutputShape(
                spatial_dim=spatial_dim,
                basis_terms=basis_terms,
                basis_function=basis_function,
                skip_basis=skip_basis,
                mode=mode,
                project_out=project_out,
                out_dim=out_dim,
                verbose=False
            )
            verbosePrint(f'\t\trpb basis function encoding output shape: {basisEncoderOutputShape}', self.verbose)
            out_dim = basisEncoderOutputShape
            
            self.rpbEncoder = BasisEncoder(
                spatial_dim=spatial_dim,
                basis_terms=basis_terms,
                basis_function=basis_function,
                skip_basis=skip_basis,
                mode=mode,
                
                out_dim=out_dim,
                
                project_mlp_properties=project_mlp_dict,
                project_linear=project_linear,
                project_out= project_out,

                verbose=verbose, verbosePrefix='\t\t',
            )
            self.rpbDim = self.rpbEncoder.outputShape
            verbosePrint(f'\trpb encoder output shape: {self.rpbDim}', verbose)
        else:
            self.rpbDim = 0
            self.rpbEncoder = None
            verbosePrint(f'Relative position bias disabled', verbose)
            

        verboseBannerPrint('Building Window Function', verbose)
        if self.window_function:
            verbosePrint(f'Window function enabled', verbose)
            if self.spatial_dim <= 0:
                raise ValueError(f'spatial_dim must be > 0 if window_function is True, got spatial_dim={self.spatial_dim}')
            verbosePrint(f'\tWindow function type: {self.window_function_type}', verbose)
            if self.gnn_window_function:
                verbosePrint(f'\tUsing window function as input to message generation', verbose)
                self.windowDim = 1
            else:
                verbosePrint(f'\tUsing window function to scale messages', verbose)
                self.windowDim = 0
        else:
            verbosePrint(f'Window function disabled', verbose)
            self.windowDim = 0

        verboseBannerPrint('Building Message Generation', verbose)
        if self.message_mode == 'transformer':
            # In transformer mode, we do not need to do anything special here as the message is simply the result of multiplying the attention mechanism with the value vectors with no learnable parameters
            verbosePrint(f'Transformer message mode enabled', verbose)
            pass
        elif self.message_mode == 'gnn':
            verbosePrint(f'GNN message mode enabled', verbose)
            raise NotImplementedError('Only transformer message mode is implemented yet')
        elif self.message_mode == 'cconv':
            """ 
Continuous convolution mode, i.e., we construct a weight matrix that maps the incoming features to the outgoing features conditioned on edge spatial relations

In the basic cconv paper (and follow up work) this is achieved by evaluating some basis functions $b(x)$ for each incoming spatial dimension which gives a basis tensor of shape
[ num_edges, num_basis_functions ^ spatial_dim ], e.g., [n_e, 4,4], in 2D.

Then given an incoming feature vector of shape [num_edges, in_features], e.g., [n_e, 16], the approach of CConv is computing a 4 dimensional weight tensor of shape [num_basis_functions ^ spatial_dim, in_features, out_features], e.g., [4,4,16,16]. This tensor can be used via
conv = torch.einsum('nu, nv, nw, uvwio,ni -> no',u,v,w,weight, x_j[batch])

where u,v,w are the basis function evaluations for each spatial dimension, e.g., [n_e,4], and weight is the weight tensor, e.g., [4,4,16,16], and x_j are the incoming features, e.g., [n_e,16], to produce the outgoing features no, e.g., [n_e,16].

Pulling apart the einsum, we can see that this is equivalent to first computing a weight matrix for each edge via
W_e = torch.einsum('nu, nv, uvio -> ncio',u,v,weight) which gives a weight matrix of shape [n_e, 16, 16]. Then we can apply this weight matrix to the incoming features via
out_e = torch.einsum('nio, ni -> no', W_e, x_j) to get the outgoing features [n_e, 16].

Similarly we could combined u,v,w as before into a single  tensor with the same result.

 """
            verbosePrint(f'Continuous Convolution message mode enabled', verbose)
            raise NotImplementedError('Only transformer message mode is implemented yet')
        else:
            raise ValueError(f'message_mode must be one of "transformer", "gnn", or "cconv", got {self.message_mode}')


        verboseBannerPrint('Building Output Projection', verbose)
        if self.multiHeadAggregation not in ['concat', 'mean']:
            raise ValueError(f'multiHeadAggregation must be one of "concat" or "mean", got {self.multiHeadAggregation}')
        
        output_proj_input_dim = self.transformer_features if self.multiHeadAggregation == 'mean' else self.transformer_features * self.multi_heads


        if self.use_output_proj:
            verbosePrint(f'Output projection enabled', verbose)
            odim = output_proj_input_dim

            verbosePrint(f'\tLatent feature size: {odim}', verbose)
            verbosePrint(f'\tOutput feature size: {self.output_dim}', verbose)

            if self.output_proj_linear:
                verbosePrint(f'Using linear output projection layer', verbose)
                self.outputProjection = nn.Linear(odim, self.output_dim)
                verbosePrint(f'\tShape: {odim} -> {self.output_dim}', verbose)
            else:
                verbosePrint(f'Using MLP output projection layer', verbose)
                self.outputProjection = buildMLPwDict(self.output_proj_mlp_dict, verbose, inputDim=odim, outputDim=self.output_dim, verbosePrefix='\t')
        else:
            verbosePrint(f'Output projection disabled, using identity', verbose)
            if self.node_feature_dim != self.output_dim:
                raise ValueError(f'node_feature_dim must be equal to output_dim if use_output_proj is False, got node_feature_dim={self.node_feature_dim}, output_dim={self.output_dim}')
            self.outputProjection = nn.Identity()

        verboseBannerPrint('Building Edge Gating', verbose)
        if self.edge_gating:
            verbosePrint(f'Edge gating enabled', verbose)
            
            if self.edge_gating_repeat:
                verbosePrint(f'\tRepeating gating values across heads', verbose)
                gating_output_dim = self.transformer_features
                gating_heads = 1
            else:
                verbosePrint(f'\tUsing separate gating values for each head', verbose)
                gating_output_dim = self.transformer_features * self.multi_heads
                gating_heads = self.multi_heads

            if self.edge_gating_edge_features and self.edgeFeatureSize <= 0:
                raise ValueError(f'edgeFeatureSize must be > 0 if edge_gating_edge_features is not "none", got edgeFeatureSize={self.edgeFeatureSize}')
            if self.edge_gating_edge_vectors  and self.spatial_dim <= 0:
                raise ValueError(f'spatial_dim must be > 0 if edge_gating_edge_vectors is not "none", got spatial_dim={self.spatial_dim}')
            if self.edge_gating_rpb  and self.rpbEncoder is None:
                raise ValueError(f'rpb must be enabled if edge_gating_rpb is not "none", got rpbDim={self.rpbDim}')

            if self.edge_gating_edge_features:
                verbosePrint(f'\tUsing edge features for gating with method: {self.edge_gating_edge_features}', verbose)
                self.edge_gating_W_edge_features = nn.Linear(self.edgeFeatureSize, gating_output_dim)
                verbosePrint(f'\t\tShape: {self.edgeFeatureSize} -> {gating_output_dim}', verbose)
            else:
                self.edge_gating_W_edge_features = None
            if self.edge_gating_edge_vectors:
                verbosePrint(f'\tUsing edge vectors for gating with method: {self.edge_gating_edge_vectors}', verbose)
                self.edge_gating_W_edge_vectors = nn.Linear(self.spatial_dim, gating_output_dim)
                verbosePrint(f'\t\tShape: {self.spatial_dim} -> {gating_output_dim}', verbose)
            else:
                self.edge_gating_W_edge_vectors = None
            if self.edge_gating_rpb:
                verbosePrint(f'\tUsing rpb for gating with method: {self.edge_gating_rpb}', verbose)
                self.edge_gating_W_rpb = nn.Linear(self.rpbDim, gating_output_dim)
                verbosePrint(f'\t\tShape: {self.rpbDim} -> {gating_output_dim}', verbose)
            else:
                self.edge_gating_W_rpb = None
            verbosePrint(f'\tGating activation: {self.gatingActivationName}', verbose)
        else:
            verbosePrint(f'Edge gating disabled', verbose)
            self.edge_gating_W_edge_features = None
            self.edge_gating_W_edge_vectors = None
            self.edge_gating_W_rpb = None

        verboseBannerPrint('MessagePassingLayer Built', verbose)


    def forward(self, 
                queryTokens: Tensor,                        # Shape [B, nQ, I] or [nQ, I]
                keyTokens: Tensor,                          # Shape [B, nK, I] or [nK, I]

                edge_index: Tensor,                         # Shape [2, num_edges]
                edge_attr: Optional[Tensor] = None,         # Shape [nE, EF]
                edge_vector: Optional[Tensor] = None,       # Shape [nE, D]

                attention_values: Optional[Tensor] = None,  # Shape [H, nE] or [nE]
                S_k: Optional[Tensor] = None                # Shape [nE]
                ) -> Tensor:  # Output shape [B, nQ, O] or [nQ, O]
        ################################################################################
        #                             Validate input shapes                            #
        ################################################################################
        numQueryTokens = queryTokens.shape[-2]
        numKeyTokens   = keyTokens.shape[-2]
        numEdges = edge_index.shape[-1]
        batch_size = queryTokens.shape[0] if len(queryTokens.shape) > 2 else 1
        unsqueezed_batch = False
        if len(queryTokens.shape) == 2:
            unsqueezed_batch = True
            queryTokens = queryTokens.unsqueeze(0)
        if len(keyTokens.shape) == 2:
            keyTokens = keyTokens.unsqueeze(0)

        spatial_dim = 0 if edge_vector is None else edge_vector.shape[-1]
        edgeFeatureSize = 0 if edge_attr is None else edge_attr.shape[-1]
        if len(attention_values.shape) == 1:
            attention_values = attention_values.unsqueeze(0)
        attention_heads = 0 if attention_values is None else attention_values.shape[0]

        if spatial_dim != self.spatial_dim:
            raise ValueError(f'spatial_dim of edge_vector must be equal to spatial_dim of layer, got {spatial_dim} and {self.spatial_dim}')
        if edgeFeatureSize != self.edgeFeatureSize:
            raise ValueError(f'edgeFeatureSize of edge_attr must be equal to edgeFeatureSize of layer, got {edgeFeatureSize} and {self.edgeFeatureSize}')
        if attention_heads != self.multi_heads and attention_values is not None:
            raise ValueError(f'attention_heads of attention_values must be equal to multi_heads of layer, got {attention_heads} and {self.multi_heads}')
        if edge_index.shape[0] != 2:
            raise ValueError(f'edge_index must have shape [2, num_edges], got {edge_index.shape}')
        if edge_index.shape[1] != numEdges:
            raise ValueError(f'edge_index second dimension must be equal to number of edges, got {edge_index.shape[1]} and {numEdges}')
        if attention_values is not None and attention_values.shape[1] != numEdges:
            raise ValueError(f'attention_values second dimension must be equal to number of edges, got {attention_values.shape[0]} and {numEdges}')

        rows = edge_index[0]  # Indices for query tokens
        cols = edge_index[1]  # Indices for key tokens
        if len(queryTokens) == 1:
            queryTokens = queryTokens.unsqueeze(0)
        if len(keyTokens) == 1:
            keyTokens = keyTokens.unsqueeze(0)

        verboseBannerPrint('MessagePassingLayer Forward', self.verbose)
        shape_dict = {
            'B': batch_size,
            'nQ': numQueryTokens,
            'nK': numKeyTokens,
            'nE': numEdges,
            'D': spatial_dim,
            'E': edgeFeatureSize,
            'H': attention_heads,
            'L': self.latent_dim,
            'T': self.transformer_features,
            'I': self.node_feature_dim,
            'O': self.output_dim,
            'RPB': self.rpbDim,
            'G': self.transformer_features if self.edge_gating_repeat else self.transformer_features * self.multi_heads
        }

        verbosePrint(f'Input Shapes:', self.verbose)
        for key, value in shape_dict.items():
            verbosePrint(f'\t{key}: {value}', self.verbose)

        verbosePrint(f'\tedge_index: {edge_index.shape}', self.verbose)
        verbosePrint(f'\tedge_attr: {edge_attr.shape if edge_attr is not None else None}', self.verbose)
        verbosePrint(f'\tedge_vector: {edge_vector.shape if edge_vector is not None else None}', self.verbose)
        verbosePrint(f'\tattention_values: {attention_values.shape if attention_values is not None else None}', self.verbose)
        verbosePrint(f'\tS_k: {S_k.shape if S_k is not None else None}', self.verbose)

        checkShapes = False
        
        ################################################################################
        #                             Begin Forward Pass                             #
        ################################################################################


        ##############################################################################
        #                            Step 1: Input Projection                         #
        ##############################################################################
        verboseBannerPrint('Projection Step', self.verbose)
        checkTensorShape(queryTokens, ['B', 'nQ', 'I'], shape_dict, checkShapes, 'queryTokens')
        checkTensorShape(keyTokens, ['B', 'nK', 'I'], shape_dict, checkShapes, 'keyTokens')

        queryLatent = self.inputProjection(queryTokens)
        keyLatent = self.inputProjection(keyTokens)

        verbosePrint(f'Query Tokens: {queryTokens.shape} [B, nQ, I] -> {queryLatent.shape} [B, nQ, L]', self.verbose)
        verbosePrint(f'Key Tokens: {keyTokens.shape} [B, nK, I] -> {keyLatent.shape} [B, nK, L]', self.verbose)

        checkTensorShape(queryLatent, ['B', 'nQ', 'L'], shape_dict, checkShapes, 'queryLatent')
        checkTensorShape(keyLatent, ['B', 'nK', 'L'], shape_dict, checkShapes, 'keyLatent')

        if self.split_across_heads:
            queryLatent = queryLatent.view(batch_size, numQueryTokens, self.multi_heads, self.latent_dim // self.multi_heads)
            keyLatent = keyLatent.view(batch_size, numKeyTokens, self.multi_heads, self.latent_dim // self.multi_heads)
            verbosePrint(f'Reshaped Query Tokens for multi-head: {queryLatent.shape} [B, nQ, H, T]', self.verbose)
            verbosePrint(f'Reshaped Key Tokens for multi-head: {keyLatent.shape} [B, nK, H, T]', self.verbose)

            checkTensorShape(queryLatent, ['B', 'nQ', 'H', 'T'], shape_dict, checkShapes, 'queryLatent multi-head')
            checkTensorShape(keyLatent, ['B', 'nK', 'H', 'T'], shape_dict, checkShapes, 'keyLatent multi-head')
        else:
            queryLatent = queryLatent.unsqueeze(2).repeat(1, 1, self.multi_heads, 1)
            keyLatent = keyLatent.unsqueeze(2).repeat(1, 1, self.multi_heads, 1)
            verbosePrint(f'Repeated Query Tokens for multi-head: {queryLatent.shape} [B, nQ, H, T]', self.verbose)
            verbosePrint(f'Repeated Key Tokens for multi-head: {keyLatent.shape} [B, nK, H, T]', self.verbose)

            checkTensorShape(queryLatent, ['B', 'nQ', 'H', 'T'], shape_dict, checkShapes, 'queryLatent multi-head')
            checkTensorShape(keyLatent, ['B', 'nK', 'H', 'T'], shape_dict, checkShapes, 'keyLatent multi-head')

        queryLatent = queryLatent.permute(2, 0, 1, 3).reshape(self.multi_heads, -1, self.latent_dim // self.multi_heads) # [h, b*n, t]
        keyLatent = keyLatent.permute(2, 0, 1, 3).reshape(self.multi_heads, -1, self.latent_dim // self.multi_heads)     # [h, b*m, t]

        verbosePrint(f'Final Query Tokens for multi-head: {queryLatent.shape} [H, B*N, T]', self.verbose)
        verbosePrint(f'Final Key Tokens for multi-head: {keyLatent.shape} [H, B*M, T]', self.verbose)
        checkTensorShape(queryLatent, ['H', 'B*nQ', 'T'], shape_dict, checkShapes, 'queryLatent multi-head final')
        checkTensorShape(keyLatent, ['H', 'B*nK', 'T'], shape_dict, checkShapes, 'keyLatent multi-head final')

        ##############################################################################
        #                        Step 2: Gather Node Features                         #
        ##############################################################################
        verboseBannerPrint('Gather Step', self.verbose)

        f_i = queryLatent[:, rows, :]  # [h, ne, t]
        V_j = keyLatent[:, cols, :]    # [h, ne, t]

        verbosePrint(f'Gathered Node Features f_i: {f_i.shape} [H, NE, T]', self.verbose) # Not used for transformer logic
        verbosePrint(f'Gathered Value Tokens V_j: {V_j.shape} [H, NE, T]', self.verbose)
        checkTensorShape(f_i, ['H', 'nE', 'T'], shape_dict, checkShapes, 'f_i')
        checkTensorShape(V_j, ['H', 'nE', 'T'], shape_dict, checkShapes, 'V_j')

        ##############################################################################
        #                    Step 3: Edge Feature and Vector Processing                #
        ##############################################################################
        verboseBannerPrint('Edge Feature Processing Step', self.verbose)
        if self.rpbEncoder is not None:
            verbosePrint('Encoding edge vectors with RPB encoder', self.verbose)
            encodedEdges = self.rpbEncoder(edge_vector)
            verbosePrint(f'Encoded edge vectors: {encodedEdges.shape} [nE, RPB]', self.verbose)
            checkTensorShape(encodedEdges, ['nE', 'RPB'], shape_dict, checkShapes, 'encodedEdges')
        else:
            encodedEdges = None
            verbosePrint('No RPB encoder, skipping edge vector encoding', self.verbose)

        if self.window_function:            
            verboseBannerPrint(f'Applying Window Function...', self.verbose)
            edgeLengths = torch.linalg.norm(edge_vector, dim=-1)
            verbosePrint(f'Edge lengths min: {edgeLengths.min().item():.4f}, max: {edgeLengths.max().item():.4f}, mean: {edgeLengths.mean().item():.4f}, std: {edgeLengths.std().item():.4f}', self.verbose)
            windowScaling = getWindowFunction(self.window_function_type, norm= None)(torch.linalg.norm(edge_vector, dim=-1)) 
            verbosePrint(f'\tWindow function shape: {windowScaling.shape} [E]', self.verbose)      
            if self.window_function_normalize:
                verbosePrint(f'\tNormalizing window function by number of neighbors', self.verbose)
                numNeighbors = scatter(torch.ones_like(rows), rows, dim=0, dim_size=batch_size*numQueryTokens, reduce='sum')  # Shape: [num_nodes_current]
                verbosePrint(f'\tNumber of neighbors per node min: {numNeighbors.min().item():.4f}, max: {numNeighbors.max().item():.4f}, median: {numNeighbors.median().item():.4f}', self.verbose)

                windowScaling_sum = scatter(windowScaling, rows, dim=0, dim_size=batch_size*numQueryTokens, reduce='sum')  # Shape: [num_nodes_current]
                verbosePrint(f'\tWindow function sum per node min: {windowScaling_sum.min().item():.4f}, max: {windowScaling_sum.max().item():.4f}, mean: {windowScaling_sum.mean().item():.4f}, std: {windowScaling_sum.std().item():.4f}', self.verbose)

                windowScaling_sum = windowScaling_sum[rows]  # Shape: [num_edges]
                # print(windowScaling_sum)
                windowScaling = numNeighbors[rows] * windowScaling / (windowScaling_sum + 1e-16)


                windowScaling_sum = scatter(windowScaling, rows, dim=0, dim_size=batch_size*numQueryTokens, reduce='sum')  # Shape: [num_nodes_current]
                verbosePrint(f'\tWindow function sum per node after norm min: {windowScaling_sum.min().item():.4f}, max: {windowScaling_sum.max().item():.4f}, mean: {windowScaling_sum.mean().item():.4f}, std: {windowScaling_sum.std().item():.4f}', self.verbose)

            verbosePrint(f'\tWindow function stats - min: {windowScaling.min().item():.4f}, max: {windowScaling.max().item():.4f}, mean: {windowScaling.mean().item():.4f}, std: {windowScaling.std().item():.4f}', self.verbose)
            verbosePrint(f'\tWindow function after normalization shape: {windowScaling.shape} [nE]', self.verbose)
            checkTensorShape(windowScaling, ['nE'], shape_dict, checkShapes, 'windowScaling')

        ##############################################################################
        #                        Step 4: Message Generation                           #
        ##############################################################################
        verboseBannerPrint('Message Generation Step', self.verbose)

        if self.message_mode == 'transformer':
            verbosePrint(f'Using Transformer style message generation', self.verbose)
            if attention_values is None: 
                raise ValueError('attention_values must be provided for transformer message mode')
            # We already computed the attention score with all scaling in the transformer layer
            # So the attention_values input is already computed. We just need to apply it to V_j
            messages = V_j
            checkTensorShape(messages, ['H', 'nE', 'T'], shape_dict, checkShapes, 'messages')

            verbosePrint(f'\tGenerated Messages: {messages.shape} [H, NE, T]', self.verbose)
            verbosePrint(f'\tAttention Values: {attention_values.shape if attention_values is not None else None}[H, NE]', self.verbose)

            attentionValues = attention_values.unsqueeze(-1)  # [H, NE, 1]
            verbosePrint(f'\tReshaped Attention Values: {attentionValues.shape} [H, NE, 1]', self.verbose)

            final_messages = messages * attentionValues  # [H, NE, T] * [H, NE, 1] -> [H, NE, T]
            verbosePrint(f'\tApplied Attention Values to Messages: {messages.shape} [H, NE, T]', self.verbose)

            checkTensorShape(final_messages, ['H', 'nE', 'T'], shape_dict, checkShapes, 'messages')
            checkTensorShape(attentionValues, ['H', 'nE', 1], shape_dict, checkShapes, 'attentionValues')
        else:
            raise NotImplementedError('Only transformer message mode is implemented yet')

        ##############################################################################
        #                        Step 5: Edge Gating                                   #
        ##############################################################################

        verboseBannerPrint('Edge Gating Step', self.verbose)
        if self.edge_gating:
            verbosePrint(f'Applying edge gating to messages', self.verbose)

            gating_inputs = []
            if self.edge_gating_edge_features != 'none' and edge_attr is not None:
                edge_features_contribution = self.edge_gating_W_edge_features(edge_attr)  # [nE, G]
                verbosePrint(f'\tEdge features contribution: {edge_features_contribution.shape} [nE, G]', self.verbose)
                checkTensorShape(edge_features_contribution, ['nE', 'G'], shape_dict, checkShapes, 'edge_features_contribution')
                gating_inputs.append(edge_features_contribution)
            if self.edge_gating_edge_vectors != 'none' and edge_vector is not None:
                edge_vectors_contribution = self.edge_gating_W_edge_vectors(edge_vector)  # [nE, G]
                verbosePrint(f'\tEdge vectors contribution: {edge_vectors_contribution.shape} [nE, G]', self.verbose)
                checkTensorShape(edge_vectors_contribution, ['nE', 'G'], shape_dict, checkShapes, 'edge_vectors_contribution')
                gating_inputs.append(edge_vectors_contribution)
            if self.edge_gating_rpb != 'none' and encodedEdges is not None:
                rpb_contribution = self.edge_gating_W_rpb(encodedEdges)  # [nE, G]
                verbosePrint(f'\tRPB contribution: {rpb_contribution.shape} [nE, G]', self.verbose)
                checkTensorShape(rpb_contribution, ['nE', 'G'], shape_dict, checkShapes, 'rpb_contribution')
                gating_inputs.append(rpb_contribution)

            if len(gating_inputs) == 0:
                raise ValueError('No inputs for edge gating, please check edge_gating settings and provided inputs')
            gating_input = sum(gating_inputs)  # [nE, G]
            verbosePrint(f'\tCombined gating input: {gating_input.shape} [nE, G]', self.verbose)
            checkTensorShape(gating_input, ['nE', 'G'], shape_dict, checkShapes, 'gating_input')

            if self.gatingActivation is not None:
                verbosePrint(f'\tApplying gating activation: {self.gatingActivationName}', self.verbose)
                gating_values = self.gatingActivation(gating_input)  # [nE, G]
            else:
                gating_values = gating_input
            verbosePrint(f'\tGating values after activation: {gating_values.shape} [nE, G]', self.verbose)
            checkTensorShape(gating_values, ['nE', 'G'], shape_dict, checkShapes, 'gating_values')

            if self.edge_gating_repeat:
                gating_values = gating_values.unsqueeze(0).repeat(self.multi_heads, 1, 1)  # [H, nE, T]
                verbosePrint(f'\tRepeated gating values across heads: {gating_values.shape} [H, nE, T]', self.verbose)
                checkTensorShape(gating_values, ['H', 'nE', 'T'], shape_dict, checkShapes, 'gating_values repeated')
            else:
                gating_values = gating_values.view(-1, self.multi_heads, self.transformer_features).permute(1,0,2)  # [H, nE, T]
                verbosePrint(f'\tReshaped gating values for heads: {gating_values.shape} [H, nE, T]', self.verbose)
                checkTensorShape(gating_values, ['H', 'nE', 'T'], shape_dict, checkShapes, 'gating_values reshaped')

            if self.edge_gating_mode == 'multiply':
                verbosePrint(f'\tApplying multiplicative gating', self.verbose)
                final_messages = final_messages * gating_values  # [H, nE, T]
            elif self.edge_gating_mode == 'add':
                verbosePrint(f'\tApplying additive gating', self.verbose)
                final_messages = final_messages + gating_values  # [H, nE, T]
            else:
                raise ValueError(f'Invalid edge_gating_mode: {self.edge_gating_mode}, must be "multiply" or "add"')

        ###############################################################################
        #                      Step 6: Window Function Application                    #
        ###############################################################################
        verboseBannerPrint('Window Function Step', self.verbose)
        if self.window_function and self.window_function_as_gate:        # Normalize to sum to 1 for each query node
            # print(f'Window Scaling min: {windowScaling.min().item():.4f}, max: {windowScaling.max().item():.4f}, mean: {windowScaling.mean().item():.4f}, std: {windowScaling.std().item():.4f}')

            verbosePrint(f'\tApplying Window Function Scaling to messages', self.verbose)

            verbosePrint(f'\tMessages before window function shape: {final_messages.shape} [H x E x F]', self.verbose)
            verbosePrint(f'\tWindow function shape: {windowScaling.shape} [E]', self.verbose)
          
            # Window function scaling shape: [E] -> [1, H, E]
            # First expand then repeat to match attention weights shape
            windowScaling_expanded = windowScaling.view(1, -1, 1)
            checkTensorShape(windowScaling_expanded, [1, 'nE', 1], shape_dict, checkShapes, 'windowScaling_expanded')

            final_messages = final_messages * windowScaling_expanded
            verbosePrint(f'\tMessages after window function shape: {final_messages.shape} [H x nE x T]', self.verbose)
            checkTensorShape(final_messages, ['H', 'nE', 'T'], shape_dict, checkShapes, 'final_messages after window function')


        verboseBannerPrint('Aggregation Step', self.verbose)
        message_values = final_messages.reshape(-1, self.transformer_features)  # [H*NE, T]
        batch_size_edges = 1

        if torch_geometric is not None:
            verbosePrint(f'Using PyTorch Geometric for message aggregation', self.verbose)
            messages_transposed = message_values.view(self.multi_heads, numEdges, self.transformer_features).permute(1, 0, 2)  # Shape: [E, H, F]
            aggregated_messages_sparse_geometric = torch_geometric.utils.scatter(
                messages_transposed, rows, dim=0, dim_size=batch_size * numQueryTokens, reduce='sum'
            )
            verbosePrint(f'Aggregated Messages Sparse Geometric: {aggregated_messages_sparse_geometric.shape} [B*nQ x H  x T]', self.verbose)
            checkTensorShape(aggregated_messages_sparse_geometric, ['B*nQ', 'H', 'T'], shape_dict, checkShapes, 'aggregated_messages_sparse_geometric')

            aggregated_messages_sparse = aggregated_messages_sparse_geometric.transpose(0, 1).reshape(self.multi_heads, numQueryTokens * batch_size, self.transformer_features).transpose(0, 1)
        else:
            raise NotImplementedError('PyTorch Geometric is required for message aggregation in this implementation')

        dense_output = aggregated_messages_sparse.to_dense()


        verbosePrint(f'Dense output shape: {dense_output.shape} [B*nQ x H x T ]', self.verbose)
        attentionOutputSparse = dense_output.reshape(batch_size, numQueryTokens, -1)#.transpose(0, 1)
        checkTensorShape(attentionOutputSparse, ['B', 'nQ', 'H*T'], shape_dict, checkShapes, 'attentionOutputSparse')
        verbosePrint(f'Attention output sparse shape: {attentionOutputSparse.shape} [B x nQ x H*T]', self.verbose)
        verbosePrint(f'Projecting attention output back to latent space', self.verbose, separator=True)
        # Project back to latent space
        
        if self.multiHeadAggregation == 'mean':
            attentionOutputSparse = attentionOutputSparse.view(batch_size, numQueryTokens, self.multi_heads, self.transformer_features)
            attentionOutput = attentionOutputSparse
            verbosePrint(f'Attention output shape before mean aggregation: {attentionOutput.shape} [B x nQ x H x T]', self.verbose)
            outputTokens = attentionOutput.mean(dim=2)
            verbosePrint(f'Attention output shape after mean aggregation: {outputTokens.shape} [B x nQ x T]', self.verbose)
            checkTensorShape(attentionOutputSparse, ['B', 'nQ', 'H', 'T'], shape_dict, checkShapes, 'attentionOutputSparse for mean')
            checkTensorShape(outputTokens, ['B', 'nQ', 'T'], shape_dict, checkShapes, 'outputTokens')
            aggregatedTokens = outputTokens
        else:
            verbosePrint(f'Using concatenation for multi-head aggregation', self.verbose)
            aggregatedTokens = attentionOutputSparse.view(batch_size, numQueryTokens, self.multi_heads * self.transformer_features)
            checkTensorShape(aggregatedTokens, ['B', 'nQ', 'H*T'], shape_dict, checkShapes, 'aggregatedTokens for concat')


        verbosePrint(f'Applying output projection to attention output {aggregatedTokens.shape}', self.verbose)
        outputTokens = self.outputProjection(aggregatedTokens)
        verbosePrint(f'Attention output shape after projection: {outputTokens.shape} [B x nQ x O]', self.verbose)
        # Residual Connection and Layer Norm (Post-Attention)
        verbosePrint(f'Applying residual connection: {queryTokens.shape} [B x nQ x I ]', self.verbose)

        if self.skipConnections:
            outputTokens = outputTokens + queryTokens
        verbosePrint(f'Output Tokens shape after residual connection: {outputTokens.shape} [B x nQ x O]', self.verbose)

        if unsqueezed_batch:
            outputTokens = outputTokens.squeeze(0)
            verbosePrint(f'Removed batch dimension, final output shape: {outputTokens.shape} [nQ x O]', self.verbose)

        return outputTokens