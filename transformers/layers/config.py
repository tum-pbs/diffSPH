from layers.layer_inputEncoder import InputEncodeLayer
from layers.layer_attentionMechanism import AttentionMechanismLayer
from layers.layer_messagePassing import MessagePassingLayer
from layers.layer_ffn import FeedForwardNetworkLayer

inputEncoderConfig = {
    # 'input_dim': inputFeatures.shape[-1],    # Required
    # 'output_dim': latentSpaceSize,           # Required
    # 'spatial_dim': edge_attr.shape[-1],      # Required
    'linear': True,
    'mlpDict': None,

    'absolutePositionBias': False,
    'apbDict': {
        'scalePositions': False,
        'multiplicative': False,
        'baseEncoding': True,
        'baseFunction': 'ffourier',
        'baseTerms': 8,
        'baseMode': 'cat',

        'linear': True,
        'mlpDict': None,

    },
    'verbose': True
}

attentionMechanismConfig = {
    # 'latent_dim': latentSpaceSize,
    # 'edge_dim': dim,
    # 'transformerFeatures': transformerFeatures, # Optional, per head
    # 'multi_heads': multiHeads,

    # Attention specifics
    'attention':{
        'mechanism': 'dot',
        'dropout': 0.0,
        'scaling': True,
        'clipping': False,
        'clipThreshold': 1.0,
        'activation': 'leaky_relu(0.2)',
        'softmax': True,
        'mlpDict': None,
    },
    # Query/Key specifics
    'encoder':{ 
        'active': True,
        'linear': True,
        'shareQueryKey': False, # True for GAT
        'mlpDict': None,
        'cConvMode': False,
    },
    'relativePositionBias': {
        'active': True,

        'postAttention': True, # If True, add after attention, else before attention
        'multiplicative': False,

        'baseEncoding': True,
        'baseFunction': 'ffourier',
        'baseTerms': 8,
        'baseMode': 'cat', # 'cat', 'add', 'outer'

        'encoder': None, # If None, use linear encoding after basis function computation
        'linear': True,
        'mlpDict': None,
        'rpbTerms': None, # used for encoder, optional,
        'split': False, # If True, split the RPB across the heads, otherwise repeat
    },
    # Window function specifics
    'windowFunction': {
        'active': False,
        'type': 'cubicSpline',
        'postSoftmax': False, # If True, apply after softmax, else before softmax
    },
    # Miscellaneous
    'verbose': True,
}

messagePassingConfig ={
    # Shape Parameters
    # 'node_features': latentSpaceSize,
    # 'transformerFeatures': transformerFeatures,
    # 'edgeFeatureSize': 0,
    # 'spatial_dim': dim,
    # 'multi_heads': multiHeads,

    # General Hyperparameters    
    'mode': 'gnn', # 'gnn', 'cconv', 'transformer'
    'skipConnections': True,
    'verbose': True,

    # Message Passing Options
    'activation': None, # Activation applied before message passing
    'message_content': {
        # Per node information, node_j represents the Value tokens in a transformer architecture
        'use_node_i': False,
        'use_node_j': True,
        'use_node_ij_sum': False,
        'use_node_ij_diff': False,

        # Edge information
        'use_edge_features': False,
        'use_spatial_features': True,
        'use_spatial_distances': False,
        'use_window_function': False,

        # Attention information
        'use_attention': True,
    },

    # Computed Shapes but can be overriden
    'latent_dim': None, # If None, use transformerFeatures * multi_heads
    'output_dim': None, # If None, use node_features

    # Input/Output Options
    'input_projection': True,
    'output_projection': True,
    'input_projection_mlpDict': None,
    'output_projection_mlpDict': None,

    # Attention Mechanism Options
    'split_across_heads': True, # If True, split latent_dim across heads, otherwise use latent_dim for each head
    'aggregation': 'concat', # 'concat', 'mean'

    # GNN Options
    'gnn':{
        'per_head': True, # If True, use separate GNN for each head, otherwise share GNN across heads
        'linear': True,
        'mlpDict': None,
    },

    # CConv Options
    'cconv':{
        'linear': True,
        'mlpDict': None,
        'latent_projection': True,
        'latent_dim': 4
    },

    # Gating Options
    'gating': {
        'active': True,
        'per_head': False,
        'mode': 'multiply', # 'multiply', 'add'

        'activation': 'sigmoid',

        'use_spatial_features': False, # edge_vectors
        'use_edge_features': False,
        'use_rpb': True, # relative position bias
    },

    # Relative Position Bias Options
    'relativePositionBias': {
        'active': True, 
        'multiplicative': False,

        'baseEncoding': True,
        'baseFunction': 'ffourier',
        'baseTerms': 8,
        'baseMode': 'cat', # 'cat', 'add', 'outer'

        'encoder': None, # If None, use linear encoding after basis function computation
        'linear': True,
        'mlpDict': None,
        'rpbTerms': None, # used for encoder, optional,
        'split': False, # If True, split the RPB across the heads, otherwise repeat
    },

    # Window Function
    'windowFunction': {
        'active': False,
        'type': 'cubicSpline',
        'normalize': False, 
        'as_gate': False, # If True, use as gate, otherwise only compute as feature
    },
}

ffnConfig = {
    # 'input_dim': latentSpaceSize,
    # 'output_dim': 1,

    'pre_norm': False,
    'post_norm': True,

    'skip_connection': False,

    'linear': False,
    'mlpDict': None,

    'verbose': True,
}



#######################################################

import copy

def getDefaultConfiguration(
        input_token_dim: int,
        output_token_dim: int,

        spatial_dim: int,

        transformer_features: int = 32,
        multi_heads: int = 1,

        latent_dim: int = None, # default to input_token_dim
        edge_feature_dim: int = 0, # default to 0
        inputEncoder: bool = True, # perform an initial encoding into a latent space, only needed on the first transformer layer
        ffn: bool = True, # use a feed-forward network after the message-passing layer    

        verbose: bool = False,
):  
    inputEncoderConfig_ = copy.deepcopy(inputEncoderConfig)
    attentionMechanismConfig_ = copy.deepcopy(attentionMechanismConfig)
    messagePassingConfig_ = copy.deepcopy(messagePassingConfig)
    ffnConfig_ = copy.deepcopy(ffnConfig)

    config = {
        'input_dim': input_token_dim,
        'output_dim': output_token_dim,
        'spatial_dim': spatial_dim,
        'edge_feature_dim': edge_feature_dim,

        'transformer_features': transformer_features,
        'multi_heads': multi_heads,

        'latent_dim': latent_dim if latent_dim is not None else input_token_dim,

        'inputEncoder': inputEncoderConfig_ if inputEncoder else None,
        'attentionMechanism': attentionMechanismConfig_,
        'messagePassing': messagePassingConfig_,
        'ffn': ffnConfig_ if ffn else None,
    }

    config['inputEncoder']['input_dim'] = input_token_dim
    config['inputEncoder']['output_dim'] = config['latent_dim']
    config['inputEncoder']['spatial_dim'] = spatial_dim

    config['attentionMechanism']['latent_dim'] = config['latent_dim']
    config['attentionMechanism']['edge_dim'] = spatial_dim
    config['attentionMechanism']['transformerFeatures'] = transformer_features
    config['attentionMechanism']['multi_heads'] = multi_heads

    config['messagePassing']['node_features'] = config['latent_dim']
    config['messagePassing']['transformerFeatures'] = transformer_features
    config['messagePassing']['edgeFeatureSize'] = edge_feature_dim
    config['messagePassing']['spatial_dim'] = spatial_dim
    config['messagePassing']['multi_heads'] = multi_heads

    config['inputEncoder']['verbose'] = verbose
    config['attentionMechanism']['verbose'] = verbose
    config['messagePassing']['verbose'] = verbose

    if ffn:
        config['ffn']['input_dim'] = config['latent_dim']
        config['ffn']['output_dim'] = output_token_dim
        config['ffn']['spatial_dim'] = spatial_dim
        config['ffn']['verbose'] = verbose

    return config
    

# The configuration should be a dictionary that matches the parameters of InputEncodeLayer
# However, for readability we will use a more structured approach where the APB is a dictionary itself
def buildEncoderFromDict(configDict):
    # Extract APB dictionary if it exists
    apbDict = configDict.pop('apbDict', {})
    
    # Create the InputEncodeLayer instance
    encoder = InputEncodeLayer(
        input_dim = configDict.get('input_dim'),
        output_dim = configDict.get('output_dim'),
        spatial_dim = configDict.get('spatial_dim'),

        linearEncode = configDict.get('linear', True),
        encoderMLPDict = configDict.get('mlpDict', None),

        absolutePositionBias = configDict.get('absolutePositionBias', False),
        
        absolutePositionBiasScaledPositions = apbDict.get('scalePositions', False),
        absolutePositionBiasMultiplicative = apbDict.get('multiplicative', False),

        absolutePositionBiasBaseEncoding = apbDict.get('baseEncoding', True),
        absolutePositionBiasBaseFunction = apbDict.get('baseFunction', 'ffourier'),
        absolutePositionBiasBaseTerms = apbDict.get('baseTerms', 16),
        absolutePositionBiasBaseMode = apbDict.get('baseMode', 'cat'),

        absolutePositionBiasLinear = apbDict.get('linear', True),
        absolutePositionBiasMLPDict = apbDict.get('mlpDict', None),

        verbose = configDict.get('verbose', False)
    )
    
    return encoder

# inputEncoder = buildEncoderFromDict(config['inputEncoder']).to(device)

def buildAttentionMechanismFromDict(configDict):
    attentionConfig = configDict.get('attention', {})
    encoderConfig = configDict.get('encoder', {})
    rpbConfig = configDict.get('relativePositionBias', {})
    windowFunctionConfig = configDict.get('windowFunction', {})

    attentionMechanism = AttentionMechanismLayer(
        latent_dim = configDict.get('latent_dim'),
        edge_dim = configDict.get('edge_dim'),
        transformer_features = configDict.get('transformerFeatures', None),
        num_heads = configDict.get('multi_heads', 1),

        # Attention specifics
        attentionMechanism = attentionConfig.get('mechanism', 'dot'),
        attentionDropout = attentionConfig.get('dropout', 0.0),
        attentionScaling = attentionConfig.get('scaling', True),
        attentionClipping = attentionConfig.get('clipping', False),
        attentionClippingValue = attentionConfig.get('clipThreshold', 1.0),
        attentionActivation = attentionConfig.get('activation', 'leaky_relu(0.2)'),
        skipSoftmax = attentionConfig.get('softmax', False),
        attentionScoreMLPDict = attentionConfig.get('mlpDict', None),

        # Query/Key specifics
        encodeTokens = encoderConfig.get('active', True),
        encodeTokensLinear = encoderConfig.get('linear', True),
        encodeTokensShared = encoderConfig.get('shareQueryKey', False), # True for GAT
        encodeTokensMLPDict = encoderConfig.get('mlpDict', None),
        cConvMode = encoderConfig.get('cConvMode', False), # 'add', 'subtract', 'concat'

        # Relative Position Bias specifics
        relativePositionBias = rpbConfig.get('active', True),

        relativePositionBiasAfterAttention = rpbConfig.get('postAttention', True), # If True, add after attention, else before attention
        relativePositionBiasMultiplicative = rpbConfig.get('multiplicative', False),

        relativePositionBiasBaseEncoding = rpbConfig.get('baseEncoding', True),
        relativePositionBiasBaseFunction = rpbConfig.get('baseFunction', 'ffourier'),
        relativePositionBiasBaseTerms = rpbConfig.get('baseTerms', 8),
        relativePositionBiasBaseMode = rpbConfig.get('baseMode', 'cat'), # 'cat', 'add', 'outer'

        relativePositionBiasEncoder = rpbConfig.get('encoder', None), # If None, use linear encoding after basis function computation
        relativePositionBiasLinear = rpbConfig.get('linear', True),
        relativePositionBiasMLPDict = rpbConfig.get('mlpDict', None),
        relativePositionBiasDim = rpbConfig.get('rpbTerms', None), # used for encoder, optional,
        relativePositionBiasSplit = rpbConfig.get('split', False), # If True, split the RPB across the heads, otherwise repeat
        # Window function specifics
        windowFunction = windowFunctionConfig.get('active', False),
        windowFunctionType = windowFunctionConfig.get('type', 'cubicSpline'),
        windowFunctionBeforeSoftmax = not windowFunctionConfig.get('postSoftmax', False), # If True, apply after softmax, else before softmax
        # Miscellaneous
        verbose = configDict.get('verbose', False),
    )
    return attentionMechanism

# attentionMechanism = buildAttentionMechanismFromDict(config['attentionMechanism']).to(device)

def buildMessagePassingFromDict(configDict):
    messagePassingConfig = configDict
    gnnConfig = messagePassingConfig.get('gnn', {})
    cconvConfig = messagePassingConfig.get('cconv', {})
    gatingConfig = messagePassingConfig.get('gating', {})
    windowFunctionConfig = messagePassingConfig.get('windowFunction', {})
    rpbConfig = messagePassingConfig.get('relativePositionBias', {})
    messageContentConfig = messagePassingConfig.get('message_content', {})

    messagePassing = MessagePassingLayer(
        node_feature_dim=messagePassingConfig.get('node_features'),
        transformer_features=messagePassingConfig.get('transformerFeatures'),
        edgeFeatureSize=messagePassingConfig.get('edge_feature_dim', 0),
        spatial_dim=messagePassingConfig.get('spatial_dim'),
        multi_heads=messagePassingConfig.get('multi_heads', 1),

        latent_dim = messagePassingConfig.get('latent_dim', None), # If None, use transformerFeatures * multi_heads
        output_dim = messagePassingConfig.get('output_dim', None), # If None, use node_features

        message_mode = messagePassingConfig.get('mode', 'transformer'), # 'gnn', 'cconv', 'transformer'
        message_activation=messagePassingConfig.get('activation', None), # Activation applied before message passing
        split_across_heads=messagePassingConfig.get('split_across_heads', True), # If True, split latent_dim across heads, otherwise use latent_dim for each head

        use_input_proj=messagePassingConfig.get('input_projection', True),
        use_output_proj=messagePassingConfig.get('output_projection', True),
        input_proj_linear=messagePassingConfig.get('input_projection_linear', True),
        output_proj_linear=messagePassingConfig.get('output_projection_linear', True),
        input_proj_mlp_dict=messagePassingConfig.get('input_projection_mlpDict', None),
        output_proj_mlp_dict=messagePassingConfig.get('output_projection_mlpDict', None),

        skipConnections=messagePassingConfig.get('skipConnections', True),

        relative_position_bias= rpbConfig.get('active', True),

        rpb_base_encoding=rpbConfig.get('baseEncoding', True),
        rpb_base_basis=rpbConfig.get('baseFunction', 'ffourier'),
        rpb_base_terms=rpbConfig.get('baseTerms', 8),
        rpb_base_mode=rpbConfig.get('baseMode', 'cat'), # 'cat', 'add', 'outer'

        rpb_proj=rpbConfig.get('encoder', None), # If None, use linear encoding after basis function computation
        rpb_proj_linear=rpbConfig.get('linear', True),
        rpb_proj_mlp_dict=rpbConfig.get('mlpDict', None),

        rpb_dim = rpbConfig.get('rpbTerms', None), # used for encoder, optional,
        rpb_split = rpbConfig.get('split', False), # If True, split the RPB across the heads, otherwise repeat

        window_function=windowFunctionConfig.get('active', False),
        window_function_type=windowFunctionConfig.get('type', 'cubicSpline'),
        window_function_normalize=windowFunctionConfig.get('normalize', False),
        window_function_as_gate=windowFunctionConfig.get('as_gate', False), # If True, use as gate, otherwise only compute as feature        

        gnn_linear = gnnConfig.get('linear', True),
        gnn_mlp_dict = gnnConfig.get('mlpDict', None),
        gnn_per_head = gnnConfig.get('per_head', True), #

        gnn_window_function=messageContentConfig.get('use_window_function', False),
        gnn_node_i_features=messageContentConfig.get('use_node_i', False),
        gnn_node_j_features=messageContentConfig.get('use_node_j', True),
        gnn_node_sum_features=messageContentConfig.get('use_node_ij_sum', False),
        gnn_node_diff_features=messageContentConfig.get('use_node_ij_diff', False),
        gnn_edge_features=messageContentConfig.get('use_edge_features', False),
        gnn_spatial_features=messageContentConfig.get('use_spatial_features', True),
        gnn_spatial_distance=messageContentConfig.get('use_spatial_distances', False),
        gnn_attention_features=messageContentConfig.get('use_attention', True),

        cconv_use_latent_proj=cconvConfig.get('latent_projection', True),
        cconv_latent_dim=cconvConfig.get('latent_dim', None),
        cconv_use_linear=cconvConfig.get('linear', True),
        cconv_mlp_dict=cconvConfig.get('mlpDict', None),

        edge_gating=gatingConfig.get('active', True),
        edge_gating_repeat=gatingConfig.get('per_head', False),
        edge_gating_mode=gatingConfig.get('mode', 'multiply'), # 'multiply',
        edge_gating_activation=gatingConfig.get('activation', 'sigmoid'),

        edge_gating_edge_vectors=gatingConfig.get('use_spatial_features', False), # edge_vectors
        edge_gating_edge_features=gatingConfig.get('use_edge_features', False),
        edge_gating_rpb=gatingConfig.get('use_rpb', True), # relative position bias

        multiHeadAggregation=messagePassingConfig.get('aggregation', 'concat'), # 'concat', 'mean'
        verbose= messagePassingConfig.get('verbose', False),


    )
    return messagePassing

# messagePassing = buildMessagePassingFromDict(config['messagePassing']).to(device)


def buildFFNFromDict(configDict):
    ffn = FeedForwardNetworkLayer(
        input_dim = configDict.get('input_dim'),
        output_dim = configDict.get('output_dim'),

        pre_norm = configDict.get('pre_norm', False),
        post_norm = configDict.get('post_norm', True),

        skip_connection = configDict.get('skip_connection', False),

        linear = configDict.get('linear', False),
        MLPDict= configDict.get('mlpDict', None),

        verbose = configDict.get('verbose', False),
    )
    return ffn


def buildLayersFromConfig(config):
    layers = []

    if config.get('inputEncoder', None) is not None:
        inputEncoder = buildEncoderFromDict(config['inputEncoder'])
        layers.append(inputEncoder)

    attentionMechanism = buildAttentionMechanismFromDict(config['attentionMechanism'])
    layers.append(attentionMechanism)

    messagePassing = buildMessagePassingFromDict(config['messagePassing'])
    layers.append(messagePassing)

    if config.get('ffn', None) is not None:
        ffn = buildFFNFromDict(config['ffn'])
        layers.append(ffn)

    return layers