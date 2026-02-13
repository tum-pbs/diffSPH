
import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

import matplotlib.pyplot as plt
import os
import torch
os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'

import torch
from gencase import *
from util import visualize, sampleParticles, generateInitialVariables, SamplingScheme
from sample import smoothState, addNoise, populateCGrid, populateUGrid, smoothValues
from util import plotState, plotInitialState
from simulation import runSimulation
from util import getCurrentTimestamp, copyWaveSystem
from argparse import ArgumentParser

import h5py
from dataset import *

parser = ArgumentParser()

parser.add_argument('--skipInitialSteps', type=int, default=0, help='Number of initial steps to skip when loading data')
parser.add_argument('--skipFinalSteps', type=int, default=0, help='Number of final steps to skip when loading data')
parser.add_argument('--temporalCoarseGrainingRate', type=int, default=1, help='Rate at which to temporally coarse grain the data')
parser.add_argument('--maxUnrollSteps', type=int, default=1, help='Maximum number of steps to unroll the simulation for')
parser.add_argument('--historyLength', type=int, default=0, help='Number of past states to include in the input')
parser.add_argument('--unrollIncrement', type=int, default=250, help='Number of iterations between increments of the unroll length')

parser.add_argument('--dataFolder', type=str, default='output', help='Folder containing the dataset to load')
parser.add_argument('--outputFolder', type=str, default='models', help='Folder to save the trained model and results to')
parser.add_argument('--includeC', action='store_true', help='Whether to include wave speed c in the input features for the GNN')
parser.add_argument('--includeDamping', action='store_true', help='Whether to include damping in the input features for the GNN')

parser.add_argument('--batchSize', type=int, default=1, help='Batch size for training')
parser.add_argument('--nIter', type=int, default=1000, help='Number of training iterations')

parser.add_argument('--initialLR', type=float, default=1e-3, help='Initial learning rate for training')
parser.add_argument('--lrStepSize', type=int, default=1000, help='Step size for learning rate scheduler')
parser.add_argument('--lrGamma', type=float, default=0.75, help='Gamma for learning rate scheduler')
parser.add_argument('--activation', type=str, default='gelu', help='Activation function to use')

parser.add_argument('--hiddenDim', type=int, default=128, help='Hidden dimension for the Transformer')
parser.add_argument('--nHead', type=int, default=4, help='Number of heads for the Transformer')
parser.add_argument('--nLayer', type=int, default=16, help='Number of layers for the Transformer')
parser.add_argument('--mlpRate', type=float, default=4.0, help='MLP expansion rate for the Transformer')
parser.add_argument('--latentFeatures', type=int, default=128, help='Number of latent features for the Transformer')

parser.add_argument('--patchSize', type=int, default=8, help='Patch size for the Transformer'  )
parser.add_argument('--windowingStyle', type=str, default='ballTree', help='Windowing style for the Transformer')

parser.add_argument('--verbose', action='store_true', help='Whether to print verbose output during simulation and training')

# import shlex
# cmdargs = '--dataFolder output/dataset_128_regular --verbose --historyLength 2 --maxUnrollSteps 4'

args = parser.parse_args()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

datasetProperties = DataSetProperties(
    skipInitialSteps= args.skipInitialSteps,
    skipFinalSteps= args.skipFinalSteps,
    temporalCoarseGrainingRate= args.temporalCoarseGrainingRate,
    unrollLength= args.maxUnrollSteps,
    historyLength= args.historyLength
)
if args.verbose:
    print(f'Dataset properties: {datasetProperties}')

simFolder = args.dataFolder
files, S, T, N, Nx, Ny = getDatasetProperties(simFolder)
if args.verbose:
    print(f'Found {len(files)} files in {simFolder}')
    print(f'Dataset has S={S} samples, T={T} timesteps, N={N} particles, Nx={Nx}, Ny={Ny}')

dataset = Dataset(datasetProperties, files, T, device)
batchSize = args.batchSize

datasetLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)
datasetIter = iter(datasetLoader)
batch = next(datasetIter)

data = next(datasetIter)
historyState, targetState, currentState, positions, densities, supports, volumes, dt, scheme, fileName, simIndex, startingPoint = data
waveSystem, config, integrator, dt, history, trajectory, current = batchToSimulation(data)

def processFeatures(historyState, targetState, currentState, includeC=False, includeDamping=False, verbose=False):
    # if verbose:
        # print(historyState.shape, targetState.shape, currentState.shape)
    B, H, N, D = historyState.shape
    if verbose:
        print(f'Batch size: {B}, History length: {H}, Number of particles: {N}, State dimension: {D}')
    B, T, N, D = targetState.shape
    if verbose:
        print(f'Batch size: {B}, Target sequence length: {T}, Number of particles: {N}, State dimension: {D}')
    B, N, D = currentState.shape
    if verbose:
        print(f'Batch size: {B}, Number of particles: {N}, State dimension: {D}')

    concatFeatures = torch.cat([historyState, currentState.unsqueeze(1)], dim=1).transpose(1, 2).reshape(B, N, (H+1), D)
    if verbose:
        print(f'Concatenated features shape: {concatFeatures.shape}')
    trajFeatures = targetState.transpose(1, 2).reshape(B, N, T, D)

    if not includeC and not includeDamping:
        inputFeatures = concatFeatures[:, :, :, :2]
        trajFeatures = trajFeatures[:, :, :, :2]
    elif includeC and not includeDamping:
        inputFeatures = concatFeatures[:, :, :, :3]
        trajFeatures = trajFeatures[:, :, :, :3]
    elif not includeC and includeDamping:
        inputFeatures = torch.cat([concatFeatures[:, :, :, :2], concatFeatures[:, :, :, 3:4]], dim=-1)
        trajFeatures = torch.cat([trajFeatures[:, :, :, :2], trajFeatures[:, :, :, 3:4]], dim=-1)
    elif includeC and includeDamping:
        inputFeatures = concatFeatures[:, :, :, :4]
        trajFeatures = trajFeatures[:, :, :, :4]
    inputFeatures = inputFeatures#.reshape(B, N, -1)
    trajFeatures = trajFeatures[:,:,:2]
    if verbose:
        print(f'Input features shape: {inputFeatures.shape}, Trajectory features shape: {trajFeatures.shape}')
    return inputFeatures, trajFeatures

inputFeatures, trajFeatures = processFeatures(historyState, targetState, currentState, includeC=args.includeC, includeDamping=args.includeDamping, verbose=args.verbose)

particleState = waveSystem.systemState
kernel = config['kernel']

neighborhood, neighbors = evaluateNeighborhood(particleState, config['domain'], kernel, verletScale = config['neighborhood']['verletScale'], mode = SupportScheme.SuperSymmetric, priorNeighborhood=None)
particleState.numNeighbors = coo_to_csr(filterNeighborhoodByKind(particleState, neighbors.neighbors, which = 'noghost')).rowEntries

outputDir = os.path.join(args.outputFolder, f'transformer_{args.windowingStyle}_{args.patchSize}_{args.maxUnrollSteps}_{args.historyLength}_{args.nIter}_{getCurrentTimestamp()}')

os.makedirs(outputDir, exist_ok=True)

import graphTransformers as dT

conditioning = None
# print(f'Positions: {positions.shape}, Input Features: {inputFeatures.shape}, GT: {trajectory.shape}')

dtype = torch.float32
dim = 2

domain = DomainDescription(
    min = torch.ones(dim, device=device, dtype=dtype) * -1,
    max = torch.ones(dim, device=device, dtype=dtype) * 1,
    periodic = torch.tensor([True for _ in range(dim)], device=device, dtype=torch.bool),
    dim = dim,
)

positions = waveSystem.systemState.positions
# inputFeatures = torch.cat([
#     waveSystem.waveState.u.view(-1, 1),
#     waveSystem.waveState.v.view(-1, 1),
# ], dim=-1)
node_volumes = waveSystem.systemState.masses
node_supports = waveSystem.systemState.supports
node_attr = [inputFeatures]
node_positions = [positions]
node_gt = trajectory

# p = 2
mode = args.windowingStyle
p = args.patchSize

windows = dT.windowifyNoAdjacency(
    domain,
    nodePositions = node_positions[0],
    nodeSupports = node_supports,
    nodeVolumes = node_volumes,
    windowConfig = dT.WindowConfig(
        window_size = p,
        mode = mode,
        shifted = False,
        spatial_dim = 2
    )
)

shifted_windows = dT.windowifyNoAdjacency(
    domain,
    nodePositions = node_positions[0],
    nodeSupports = node_supports,
    nodeVolumes = node_volumes,
    windowConfig = dT.WindowConfig(
        window_size = p,
        mode = mode,
        shifted = True,
        spatial_dim = 2
    )
)

from graphTransformers.layers.windowLayer import WindowLayer, WindowLayerConfig
from graphTransformers.layers.mlp import MLPConfig

in_features = inputFeatures.flatten(2).shape[-1] # v_i, v_j, h_i, h_j

out_features = trajFeatures.shape[-1]
# print(f'in features: {in_features}, out features: {out_features}')
embedding_dim = 0
historyLength = 0

attention_heads = args.nHead
# print(f'Using {attention_heads} attention heads, latent features: {args.latentFeatures}, hidden dim factor: {args.mlpRate}, activation: {args.activation}')

mlpConfig = MLPConfig(
    input_dim = -1,
    output_dim = -1,
    hidden_dim_factor = int(args.mlpRate),
    hidden_layers = 1,
    activation=args.activation,
    hidden_norm = False,
)

windowConfig = WindowLayerConfig(
    token_input_dim=in_features,
    token_output_dim=out_features,
    spatial_dim=dim,
    
    use_conditioning=False if embedding_dim == 0 else True,
    embedding_dim=embedding_dim,
    
    operation_type='window',
    window_operation='attention',
    
    use_ffn = True,
    ffn_linear = False,
    
    use_normalization=False,
    normalize_before = False,
    normalize_after = False,
    normalize_tokens = True,
    
    use_skip_connection=True,
    
    mlp_properties = mlpConfig,
    embedding_properties= mlpConfig if embedding_dim > 0 else None,
    
    attention_heads = attention_heads,
    attention_use_position_encoding=True,
    per_head_position_encoding=True,
    attention_variant = 'v2',
    
    position_encoding_type='linear',
    normalize_positions=True,
    
    attention_chunk_size=4
)

import copy
from typing import Optional
class SimpleModel(torch.nn.Module):
    def __init__(self,    
                 latent_features: int,
                 windowConfig: WindowLayerConfig,
                 num_transformers: int = 2,
                 verbose: bool = False
                 
                 ):
        super(SimpleModel, self).__init__()
        self.config = copy.deepcopy(windowConfig)

        self.token_input_dim = self.config.token_input_dim
        self.token_output_dim = self.config.token_output_dim
        self.latent_features = latent_features
        self.spatial_dim = self.config.spatial_dim
        self.edge_feature_dim = self.config.spatial_dim
        self.multi_heads = self.config.attention_heads
        self.embedding_dim = self.config.embedding_dim

        self.encoder = torch.nn.Linear(self.token_input_dim, self.latent_features, bias=False)
        self.decoder = torch.nn.Linear(self.latent_features, self.token_output_dim, bias=False)
        

        transformerLayers = []
        for _i in range(num_transformers):
             transformerLayers.append(
                WindowLayer(
                    token_input_dim=self.latent_features,
                    token_output_dim=self.latent_features,
                    token_latent_dim=self.latent_features,
                    
                    spatial_dim = self.spatial_dim,
                    windowConfig = copy.deepcopy(self.config),
                    verbose=verbose
                )
            )
        self.transformers = torch.nn.ModuleList(transformerLayers)
    
    def forward(self, 
                node_attr : torch.Tensor, 
                node_positions : torch.Tensor, 
                node_supports: torch.Tensor, 
                node_volumes: torch.Tensor, 
                
                domain : DomainDescription, 
                windows : dT.Windowing, 
                shifted_windows: dT.Windowing,
                conditioning_vector : Optional[torch.Tensor] = None):
        encoded = self.encoder(node_attr)
        # print(f'Encoded shape: {encoded.shape}')
        for i in range(len(self.transformers)):
            curWindows = shifted_windows if i % 2 == 1 else windows
            encoded = self.transformers[i](
                nodeTokens = encoded,
                nodePositions = node_positions,
                nodeSupports = node_supports,
                nodeMasses = node_volumes,
                
                domain = domain,
                windowing = curWindows,
                windowTokens = None,
                conditioning = conditioning_vector
            )
        decoded = self.decoder(encoded)
        return decoded

model = SimpleModel(
    num_transformers=args.nLayer,
    latent_features = args.latentFeatures,
    windowConfig = windowConfig,
    verbose = False
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
learningRateScheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
if args.verbose:
    print(f'Number of trainable parameters: {number_of_parameters}')

modelConfig = {
    'num_transformers': args.nLayer,
    'latent_features': args.latentFeatures,
    'verbose': args.verbose,
    'token_input_dim': in_features,
    'token_output_dim': out_features,
    'spatial_dim': dim,

    'use_conditioning': windowConfig.use_conditioning,
    'embedding_dim': windowConfig.embedding_dim,

    'operation_type': windowConfig.operation_type,
    'window_operation': windowConfig.window_operation,

    'use_ffn': windowConfig.use_ffn,
    'ffn_linear': windowConfig.ffn_linear,

    'use_normalization': windowConfig.use_normalization,
    'normalize_before': windowConfig.normalize_before,
    'normalize_after': windowConfig.normalize_after,
    'normalize_tokens': windowConfig.normalize_tokens,

    'use_skip_connection': windowConfig.use_skip_connection,

    'mlp_hidden_dim_factor': mlpConfig.hidden_dim_factor,
    'mlp_hidden_layers': mlpConfig.hidden_layers,
    'mlp_activation': mlpConfig.activation,

    'attention_heads': windowConfig.attention_heads,
    'attention_use_position_encoding': windowConfig.attention_use_position_encoding,
    'per_head_position_encoding': windowConfig.per_head_position_encoding,
    'attention_variant': windowConfig.attention_variant,

    'position_encoding_type': windowConfig.position_encoding_type,
    'normalize_positions': windowConfig.normalize_positions,
    'attention_chunk_size': windowConfig.attention_chunk_size,
}



dataSetConfig = {
    'skipInitialSteps': args.skipInitialSteps,
    'skipFinalSteps': args.skipFinalSteps,
    'temporalCoarseGrainingRate': args.temporalCoarseGrainingRate,
    'unrollLength': args.maxUnrollSteps,
    'historyLength': args.historyLength
}

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'learning_rate_scheduler_state_dict': learningRateScheduler.state_dict(),
    'args': vars(args),
    'number_of_parameters': number_of_parameters,
    'modelConfig': modelConfig,
    'dataSetConfig': dataSetConfig,
}, os.path.join(outputDir, 'initialModel.pt'))

import time

data = next(datasetIter)
eulerIntegrator = getIntegrator(IntegrationSchemeType.explicitEuler)
RK4Integrator = getIntegrator(IntegrationSchemeType.rungeKutta4)

trainIter = args.nIter
currentUnrollLength = 1
unrollLength = args.maxUnrollSteps
# tq = tqdm(total=trainIter, desc='Training', position=0, leave=True)
# sleepTime = 0.1
# time.sleep(0.1)
# tq2 = tqdm(total=currentUnrollLength, desc='Unroll', position=1, leave=True)

losses = []
uLosses = []
vLosses = []
for i in (tq := tqdm(range(trainIter))):
# for i in range(trainIter):
    if i % args.unrollIncrement == 0 and i > 0:
        currentUnrollLength = min(currentUnrollLength + 1, unrollLength)
        # tq2.n = currentUnrollLength
        # print(f'Current unroll length: {currentUnrollLength}')

    optimizer.zero_grad()

    try:
        data = next(datasetIter)
    except StopIteration:
        datasetIter = iter(datasetLoader)
        data = next(datasetIter)

    with torch.no_grad():
        waveSystem, config, integrator, dt, history, trajectory, current = batchToSimulation(data)

    with torch.no_grad():
        historyState, targetState, currentState, *_ = data
        inputFeatures, trajFeatures = processFeatures(historyState, targetState, currentState, includeC=args.includeC, includeDamping=args.includeDamping, verbose=False)
    # print(f'Input features shape: {inputFeatures.shape}, Trajectory features shape: {trajFeatures.shape}')

    inputFeatures2 = inputFeatures.flatten(0,1).flatten(-2)
    # print(f'Input features 2 shape: {inputFeatures2.shape}')
    inputFeatures = inputFeatures[:,:,:-1,:]
    # print(f'Input features shape: {inputFeatures.shape}, Trajectory features

    loss = 0

    waveSystemNext = copyWaveSystem(waveSystem)

    uuLosses = []
    vuLosses = []


    node_volumes = waveSystemNext.systemState.masses
    node_supports = waveSystemNext.systemState.supports
    node_positions = [positions]
    node_gt = trajectory

    # p = 2
    # mode = 'ballTree'
    # p = 16

    windows = dT.windowifyNoAdjacency(
        domain,
        nodePositions = node_positions[0],
        nodeSupports = node_supports,
        nodeVolumes = node_volumes,
        windowConfig = dT.WindowConfig(
            window_size = p,
            mode = mode,
            shifted = False,
            spatial_dim = 2
        )
    )

    shifted_windows = dT.windowifyNoAdjacency(
        domain,
        nodePositions = node_positions[0],
        nodeSupports = node_supports,
        nodeVolumes = node_volumes,
        windowConfig = dT.WindowConfig(
            window_size = p,
            mode = mode,
            shifted = True,
            spatial_dim = 2
        )
    )



    for t in range(currentUnrollLength):
        # waveSystemNext, updates =  eulerIntegrator.function(
        #     waveSystem_,
        #     dt = dt,
        #     f = waveSystemFunction,
        #     verbose = False,
        #     config = config,
        # )
        waveSystemNext = copyWaveSystem(waveSystemNext)

        positions = waveSystemNext.systemState.positions

        if args.includeC and args.includeDamping:
            currentFeatures = torch.cat([
                waveSystemNext.waveState.u.view(-1, 1),
                waveSystemNext.waveState.v.view(-1, 1),
                waveSystemNext.waveState.c.view(-1, 1),
                waveSystemNext.waveState.damping.view(-1, 1),
            ], dim=-1)
        elif args.includeC and not args.includeDamping:
            currentFeatures = torch.cat([
                waveSystemNext.waveState.u.view(-1, 1),
                waveSystemNext.waveState.v.view(-1, 1),
                waveSystemNext.waveState.c.view(-1, 1),
            ], dim=-1)
        elif not args.includeC and args.includeDamping:
            currentFeatures = torch.cat([
                waveSystemNext.waveState.u.view(-1, 1),
                waveSystemNext.waveState.v.view(-1, 1),
                waveSystemNext.waveState.damping.view(-1, 1),
            ], dim=-1)
        else:
            currentFeatures = torch.cat([
                waveSystemNext.waveState.u.view(-1, 1),
                waveSystemNext.waveState.v.view(-1, 1),
            ], dim=-1)
        currentFeatures = currentFeatures.unsqueeze(1).unsqueeze(0)
        # print(f'Current features shape: {currentFeatures.shape}')
        inputFeatures = torch.cat([inputFeatures, currentFeatures], dim=2)
        # print(f'Input features shape: {inputFeatures.shape}, Trajectory features shape: {trajFeatures.shape}')

        selectedFeatures = inputFeatures[:,:,-args.historyLength-1:,:]
        # print(f'Selected features shape: {selectedFeatures.shape}')
        flattenedSelectedFeatures = selectedFeatures.reshape(selectedFeatures.shape[0], selectedFeatures.shape[1], -1).flatten(0,1)
        # print(f'Flattened selected features shape: {flattenedSelectedFeatures.shape}')


        # inputFeatures = torch.cat([
        #     waveSystemNext.waveState.u.view(-1, 1),
        #     waveSystemNext.waveState.v.view(-1, 1),
        # ], dim=-1)
        node_attr = [flattenedSelectedFeatures]

        fi_nn = model(
            node_attr = node_attr[0],
            node_positions = node_positions[0],
            node_supports = node_supports,
            node_volumes = node_volumes,
            
            domain = domain,
            windows = windows,
            shifted_windows = shifted_windows
        )


        # fi_nn = GNN(waveSystemNext.systemState, waveSystemNext.neighborhood, features)
        # fi_nn = torch.zeros_like(features)
        dudt = fi_nn[:, 0]
        dvdt = fi_nn[:, 1]

        # print(f'Max dudt: {torch.max(dudt)}, Max dvdt: {torch.max(dvdt)}')
        # print(f'Requires Gradient: dudt {dudt.requires_grad}, dvdt {dvdt.requires_grad}')

        waveSystemNext.waveState.u = waveSystemNext.waveState.u + dudt
        waveSystemNext.waveState.v = waveSystemNext.waveState.v + dvdt # / dt

        uLoss = torch.mean((waveSystemNext.waveState.u - trajectory[t,:,0])**2)
        vLoss = torch.mean((waveSystemNext.waveState.v - trajectory[t,:,1])**2)

        uuLosses.append(uLoss)
        vuLosses.append(vLoss)

        # print(f'Iter {i}, uLoss: {uLoss} [type: {type(uLoss)}], vLoss: {vLoss}')
        # print(uLoss)
        # print(vLoss)

    uuLosses = torch.stack(uuLosses)
    vuLosses = torch.stack(vuLosses)

    uLossTotal = sum(uuLosses) / len(uuLosses)
    vLossTotal = sum(vuLosses) / len(vuLosses)

    # print(uLossTotal, vLossTotal)

    totalLoss = uLossTotal + vLossTotal
    # print(f'Total Loss:' , totalLoss)
    totalLoss.backward()
    optimizer.step()
    learningRateScheduler.step()

    losses.append(totalLoss.detach().item())
    uLosses.append(uLossTotal.detach().item())
    vLosses.append(vLossTotal.detach().item())
    
    tq.set_description(f'Training (Loss: {totalLoss.item():.6f}, uLoss: {uLossTotal.item():.6f}, vLoss: {vLossTotal.item():.6f}) [{data[-2].cpu().item()} | {data[-1].cpu().item()}]')
    
    # tq.set_postfix({'loss': totalLoss.detach().item(), 'uLoss': uLossTotal.detach().item(), 'vLoss': vLossTotal.detach().item(), 'lr': learningRateScheduler.get_last_lr()[0]})
    # tq.update()

import json

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'learning_rate_scheduler_state_dict': learningRateScheduler.state_dict(),
    'losses': losses,
    'uLosses': uLosses,
    'vLosses': vLosses,
    'args': vars(args),
    'number_of_parameters': number_of_parameters,
    'modelConfig': modelConfig,
    'dataSetConfig': dataSetConfig,
}, os.path.join(outputDir, 'model.pt'))

with open(os.path.join(outputDir, 'config.json'), 'w') as f:
    json.dump({
        'modelConfig': modelConfig,
        'dataSetConfig': dataSetConfig,
    }, f, indent=4)
    
with open(os.path.join(outputDir, 'training_log.txt'), 'w') as f:
    for i in range(len(losses)):
        f.write(f'Iter {i}, Loss: {losses[i]}, uLoss: {uLosses[i]}, vLoss: {vLosses[i]}\n')

fig, axis = plt.subplots(2,4, figsize=(16, 7), squeeze=False)

markerSize = 0.5

uInitial = visualizeParticles(fig, axis[0,0], waveSystem.systemState, config['domain'], waveSystem.waveState.u, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'managua', markerSize = markerSize, gridVisualization = False, title = 'u [Initial]')
vInitial = visualizeParticles(fig, axis[1,0], waveSystem.systemState, config['domain'], waveSystem.waveState.v, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'vanimo', markerSize = markerSize, gridVisualization = False, title = 'v [Initial]')

uPlotRk4 = visualizeParticles(fig, axis[0,1], waveSystem.systemState, config['domain'], waveSystemNext.waveState.u, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'managua', markerSize = markerSize, gridVisualization = False, title = 'u [Prediction]')
vPlotRk4 = visualizeParticles(fig, axis[1,1], waveSystem.systemState, config['domain'], waveSystemNext.waveState.v, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'vanimo', markerSize = markerSize, gridVisualization = False, title = 'v [Prediction]')


uDiff = waveSystem.waveState.u - trajectory[0,:,0]
vDiff = waveSystem.waveState.v - trajectory[0,:,1]

uDiff = trajectory[0,:,0]
vDiff = trajectory[0,:,1]

uPlotGT = visualizeParticles(fig, axis[0,2], waveSystem.systemState, config['domain'], uDiff, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'managua', markerSize = markerSize, gridVisualization = False, title = 'u [Ground Truth]')
vPlotGT = visualizeParticles(fig, axis[1,2], waveSystem.systemState, config['domain'], vDiff, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'vanimo', markerSize = markerSize, gridVisualization = False, title = 'v [Ground Truth]')

diffU = (waveSystemNext.waveState.u - trajectory[0,:,0])#.detach().cpu().numpy()
diffV = (waveSystemNext.waveState.v - trajectory[0,:,1])#.detach().cpu().numpy()

visualizeParticles(fig, axis[0,3], waveSystem.systemState, config['domain'], diffU, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'bwr', markerSize = markerSize, gridVisualization = False, title = 'u [Diff]')
visualizeParticles(fig, axis[1,3], waveSystem.systemState, config['domain'], diffV, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'bwr', markerSize = markerSize, gridVisualization = False, title = 'v [Diff]')


fig.tight_layout()
fig.savefig(os.path.join(outputDir, 'prediction.png'), dpi=300)

fig, axis = plt.subplots(1, 1)
axis.plot(losses, label='Total Loss')
axis.plot([u for u in uLosses], label='u Loss')
axis.plot([v for v in vLosses], label='v Loss')
axis.set_yscale('log')
axis.set_title('Loss')
axis.set_xlabel('Iteration')
axis.set_ylabel('Loss')
axis.grid(True)
axis.legend()

fig.tight_layout()
fig.savefig(os.path.join(outputDir, 'loss.png'), dpi=300)

frameStart = datasetLoader.dataset[24]

def unsq(t: Union[torch.Tensor, float, int], device) -> torch.Tensor:
    # print(f'Unsq input: {t}, type: {type(t)}')
    if isinstance(t, torch.Tensor):
        return t.unsqueeze(0)
    elif isinstance(t, str):
        return t
    else:
        return torch.tensor([t], device=device)
    

def batchifyFrame(frame):
    historyState, targetState, currentState, positions, densities, supports, volumes, dt, scheme, fileNames, simIndex, startingPoint = frame
    device = historyState.device
    
    batch = (
        unsq(historyState, device),
        unsq(targetState, device),
        unsq(currentState, device),
        unsq(positions, device),
        unsq(densities, device),
        unsq(supports, device),
        unsq(volumes, device),
        unsq(dt, device),
        unsq(scheme, device),
        unsq(fileNames, device),
        unsq(simIndex, device),        unsq(startingPoint, device),
    )
    
    return batch

batchedFrame = batchifyFrame(frameStart)

waveSystem, config, integrator, dt, history, trajectory, current = batchToSimulation(batchedFrame)
eulerIntegrator = getIntegrator(IntegrationSchemeType.explicitEuler)
RK4Integrator = getIntegrator(IntegrationSchemeType.rungeKutta4)

n = waveSystem.waveState.u.shape[0]
nx = int(n**0.5)
# print(f'Assuming square grid with nx={nx}, ny={nx}')
domainArea = (config['domain'].max[0] - config['domain'].min[0]) * (config['domain'].max[1] - config['domain'].min[1])
# print(f'Particle area: {domainArea/n}, particle spacing: {(domainArea/n)**0.5}')
dx = (domainArea/n)**0.5


waveSystemEuler = copyWaveSystem(waveSystem)
waveSystemRK4 = copyWaveSystem(waveSystem)
waveSystemNeural = copyWaveSystem(waveSystem)

with torch.no_grad():
    historyState, targetState, currentState, *_ = data
    inputFeatures, trajFeatures = processFeatures(historyState, targetState, currentState, includeC=args.includeC, includeDamping=args.includeDamping, verbose=False)
# print(f'Input features shape: {inputFeatures.shape}, Trajectory features shape: {trajFeatures.shape}')

# inputFeatures2 = inputFeatures.flatten(0,1).flatten(-2)
# inputFeatures = inputFeatures[:,:,:-1,:]

        # fi_nn = GNN(waveSystemNext.systemState, waveSystemNext.neighborhood, flattenedSelectedFeatures)



node_volumes = waveSystem.systemState.masses
node_supports = waveSystem.systemState.supports
node_positions = [positions]
node_gt = trajectory

# p = 2
# mode = 'ballTree'
# p = 16

windows = dT.windowifyNoAdjacency(
    domain,
    nodePositions = node_positions[0],
    nodeSupports = node_supports,
    nodeVolumes = node_volumes,
    windowConfig = dT.WindowConfig(
        window_size = p,
        mode = mode,
        shifted = False,
        spatial_dim = 2
    )
)

shifted_windows = dT.windowifyNoAdjacency(
    domain,
    nodePositions = node_positions[0],
    nodeSupports = node_supports,
    nodeVolumes = node_volumes,
    windowConfig = dT.WindowConfig(
        window_size = p,
        mode = mode,
        shifted = True,
        spatial_dim = 2
    )
)



for i in range(1):
    waveSystemEuler, updates =  eulerIntegrator.function(
        waveSystemEuler,
        dt = dt,
        f = waveSystemFunction,
        verbose = False,
        config = config,
    )
    waveSystemRK4, updates = RK4Integrator.function(
        waveSystemRK4,
        dt = dt,
        f = waveSystemFunction,
        verbose = False,
        config = config,    
    )
# 
    waveSystemNext = copyWaveSystem(waveSystemNeural)

    features = torch.cat([
        waveSystemNext.waveState.u.view(-1, 1),
        waveSystemNext.waveState.v.view(-1, 1),
    ], dim=-1)


    positions = waveSystemNext.systemState.positions

    if args.includeC and args.includeDamping:
        currentFeatures = torch.cat([
            waveSystemNext.waveState.u.view(-1, 1),
            waveSystemNext.waveState.v.view(-1, 1),
            waveSystemNext.waveState.c.view(-1, 1),
            waveSystemNext.waveState.damping.view(-1, 1),
        ], dim=-1)
    elif args.includeC and not args.includeDamping:
        currentFeatures = torch.cat([
            waveSystemNext.waveState.u.view(-1, 1),
            waveSystemNext.waveState.v.view(-1, 1),
            waveSystemNext.waveState.c.view(-1, 1),
        ], dim=-1)
    elif not args.includeC and args.includeDamping:
        currentFeatures = torch.cat([
            waveSystemNext.waveState.u.view(-1, 1),
            waveSystemNext.waveState.v.view(-1, 1),
            waveSystemNext.waveState.damping.view(-1, 1),
        ], dim=-1)
    else:
        currentFeatures = torch.cat([
            waveSystemNext.waveState.u.view(-1, 1),
            waveSystemNext.waveState.v.view(-1, 1),
        ], dim=-1)
    currentFeatures = currentFeatures.unsqueeze(1).unsqueeze(0)
    # print(f'Current features shape: {currentFeatures.shape}')
    inputFeatures = torch.cat([inputFeatures, currentFeatures], dim=2)
    # print(f'Input features shape: {inputFeatures.shape}, Trajectory features shape: {trajFeatures.shape}')

    selectedFeatures = inputFeatures[:,:,-args.historyLength-1:,:]
    # print(f'Selected features shape: {selectedFeatures.shape}')
    flattenedSelectedFeatures = selectedFeatures.reshape(selectedFeatures.shape[0], selectedFeatures.shape[1], -1).flatten(0,1)
    # print(f'Flattened selected features shape: {flattenedSelectedFeatures.shape}')


    # inputFeatures = torch.cat([
    #     waveSystemNext.waveState.u.view(-1, 1),
    #     waveSystemNext.waveState.v.view(-1, 1),
    # ], dim=-1)
    node_attr = [flattenedSelectedFeatures]

    fi_nn = model(
        node_attr = node_attr[0],
        node_positions = node_positions[0],
        node_supports = node_supports,
        node_volumes = node_volumes,
        
        domain = domain,
        windows = windows,
        shifted_windows = shifted_windows
    )

    dudt = fi_nn[:, 0]
    dvdt = fi_nn[:, 1]

    waveSystemNeural.waveState.u = waveSystemNext.waveState.u + dudt
    waveSystemNeural.waveState.v = waveSystemNext.waveState.v + dvdt #/ dt
    
fig, axis = plt.subplots(2, 5, figsize=(18,6), sharex=True, sharey=True)

markerSize = 1.5

uMaxRk4 = torch.max(torch.abs(waveSystemRK4.waveState.u)).cpu().detach().item()
vMaxRk4 = torch.max(torch.abs(waveSystemRK4.waveState.v)).cpu().detach().item()

uPlotRk4 = visualizeParticles(fig, axis[0,0], waveSystem.systemState, config['domain'], waveSystemRK4.waveState.u, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'managua', markerSize = markerSize, gridVisualization = False, title = 'u [Rk4]', vmin = -uMaxRk4, vmax = uMaxRk4)
vPlotRk4 = visualizeParticles(fig, axis[1,0], waveSystem.systemState, config['domain'], waveSystemRK4.waveState.v, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'vanimo', markerSize = markerSize, gridVisualization = False, title = 'v [Rk4]', vmin = -vMaxRk4, vmax = vMaxRk4)


umaxEuler = torch.max(torch.abs(waveSystemNeural.waveState.u)).cpu().detach().item()
vmaxEuler = torch.max(torch.abs(waveSystemNeural.waveState.v)).cpu().detach().item()

uPlotEuler = visualizeParticles(fig, axis[0,1], waveSystem.systemState, config['domain'], waveSystemNeural.waveState.u, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'managua', markerSize = markerSize, gridVisualization = False, title = 'u [Neural]', vmin = -umaxEuler, vmax = umaxEuler)
vPlotEuler = visualizeParticles(fig, axis[1,1], waveSystem.systemState, config['domain'], waveSystemNeural.waveState.v, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'vanimo', markerSize = markerSize, gridVisualization = False, title = 'v [Neural]', vmin = -vmaxEuler, vmax = vmaxEuler)

uDiff = (waveSystemRK4.waveState.u - waveSystemNeural.waveState.u)
vDiff = (waveSystemRK4.waveState.v - waveSystemNeural.waveState.v)

uDiff = torch.stack([u.dudt for u in updates], dim=0).mean(dim=0) * dt
vDiff = torch.stack([u.dvdt for u in updates], dim=0).mean(dim=0) * dt

uDiffVmax = torch.max(torch.abs(uDiff)).cpu().detach().item()
vDiffVmax = torch.max(torch.abs(vDiff)).cpu().detach().item()

uPlotDiff = visualizeParticles(fig, axis[0,2], waveSystem.systemState, config['domain'], uDiff, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'RdBu', markerSize = markerSize, gridVisualization = False, title = 'du [RK4]', vmin = -uDiffVmax, vmax = uDiffVmax)
vPlotDiff = visualizeParticles(fig, axis[1,2], waveSystem.systemState, config['domain'], vDiff, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'RdBu', markerSize = markerSize, gridVisualization = False, title = 'dv [RK4]', vmin = -vDiffVmax, vmax = vDiffVmax)

# uDiff = (waveSystemRK4.waveState.u - waveSystem.waveState.u)
# vDiff = (waveSystemRK4.waveState.v - waveSystem.waveState.v)
uDiffNN = fi_nn[:, 0]
vDiffNN = fi_nn[:, 1]

uDiffVmax = torch.max(torch.abs(uDiffNN)).cpu().detach().item()
vDiffVmax = torch.max(torch.abs(vDiffNN)).cpu().detach().item()

uPlotDiffInitial = visualizeParticles(fig, axis[0,3], waveSystem.systemState, config['domain'], uDiffNN, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'RdBu', markerSize = markerSize, gridVisualization = False, title = 'du [Prediction]', vmin = -uDiffVmax, vmax = uDiffVmax)
vPlotDiffInitial = visualizeParticles(fig, axis[1,3], waveSystem.systemState, config['domain'], vDiffNN, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'RdBu', markerSize = markerSize, gridVisualization = False, title = 'dv [Prediction]', vmin = -vDiffVmax, vmax = vDiffVmax)

errU = uDiff - uDiffNN
errV = vDiff - vDiffNN

errU = waveSystem.waveState.u
errV = waveSystem.waveState.v

errUmax = torch.max(torch.abs(errU)).cpu().detach().item()
errVmax = torch.max(torch.abs(errV)).cpu().detach().item()

uPlotError = visualizeParticles(fig, axis[0,4], waveSystem.systemState, config['domain'], errU, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'managua', markerSize = markerSize, gridVisualization = False, title = 'u [Init]', vmin = -errUmax, vmax = errUmax)
vPlotError = visualizeParticles(fig, axis[1,4], waveSystem.systemState, config['domain'], errV, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'vanimo', markerSize = markerSize, gridVisualization = False, title = 'v [Init]', vmin = -errVmax, vmax = errVmax)


fig.tight_layout()
fig.savefig(os.path.join(outputDir, 'prediction_comparison.png'), dpi=300)

waveSystem, config, integrator, dt, history, trajectory, current = batchToSimulation(batchedFrame)
eulerIntegrator = getIntegrator(IntegrationSchemeType.explicitEuler)
RK4Integrator = getIntegrator(IntegrationSchemeType.rungeKutta4)

n = waveSystem.waveState.u.shape[0]
nx = int(n**0.5)
# print(f'Assuming square grid with nx={nx}, ny={nx}')
domainArea = (config['domain'].max[0] - config['domain'].min[0]) * (config['domain'].max[1] - config['domain'].min[1])
# print(f'Particle area: {domainArea/n}, particle spacing: {(domainArea/n)**0.5}')
dx = (domainArea/n)**0.5


waveSystemEuler = copyWaveSystem(waveSystem)
waveSystemRK4 = copyWaveSystem(waveSystem)
waveSystemNeural = copyWaveSystem(waveSystem)

with torch.no_grad():
    historyState, targetState, currentState, *_ = data
    inputFeatures, trajFeatures = processFeatures(historyState, targetState, currentState, includeC=args.includeC, includeDamping=args.includeDamping, verbose=False)
# print(f'Input features shape: {inputFeatures.shape}, Trajectory features shape: {trajFeatures.shape}')

# inputFeatures2 = inputFeatures.flatten(0,1).flatten(-2)
# inputFeatures = inputFeatures[:,:,:-1,:]

        # fi_nn = GNN(waveSystemNext.systemState, waveSystemNext.neighborhood, flattenedSelectedFeatures)



node_volumes = waveSystem.systemState.masses
node_supports = waveSystem.systemState.supports
node_positions = [positions]
node_gt = trajectory

# p = 2
# mode = 'ballTree'
# p = 16

windows = dT.windowifyNoAdjacency(
    domain,
    nodePositions = node_positions[0],
    nodeSupports = node_supports,
    nodeVolumes = node_volumes,
    windowConfig = dT.WindowConfig(
        window_size = p,
        mode = mode,
        shifted = False,
        spatial_dim = 2
    )
)

shifted_windows = dT.windowifyNoAdjacency(
    domain,
    nodePositions = node_positions[0],
    nodeSupports = node_supports,
    nodeVolumes = node_volumes,
    windowConfig = dT.WindowConfig(
        window_size = p,
        mode = mode,
        shifted = True,
        spatial_dim = 2
    )
)



for i in range(16):
    waveSystemEuler, updates =  eulerIntegrator.function(
        waveSystemEuler,
        dt = dt,
        f = waveSystemFunction,
        verbose = False,
        config = config,
    )
    waveSystemRK4, updates = RK4Integrator.function(
        waveSystemRK4,
        dt = dt,
        f = waveSystemFunction,
        verbose = False,
        config = config,    
    )
# 
    waveSystemNext = copyWaveSystem(waveSystemNeural)

    features = torch.cat([
        waveSystemNext.waveState.u.view(-1, 1),
        waveSystemNext.waveState.v.view(-1, 1),
    ], dim=-1)


    positions = waveSystemNext.systemState.positions

    if args.includeC and args.includeDamping:
        currentFeatures = torch.cat([
            waveSystemNext.waveState.u.view(-1, 1),
            waveSystemNext.waveState.v.view(-1, 1),
            waveSystemNext.waveState.c.view(-1, 1),
            waveSystemNext.waveState.damping.view(-1, 1),
        ], dim=-1)
    elif args.includeC and not args.includeDamping:
        currentFeatures = torch.cat([
            waveSystemNext.waveState.u.view(-1, 1),
            waveSystemNext.waveState.v.view(-1, 1),
            waveSystemNext.waveState.c.view(-1, 1),
        ], dim=-1)
    elif not args.includeC and args.includeDamping:
        currentFeatures = torch.cat([
            waveSystemNext.waveState.u.view(-1, 1),
            waveSystemNext.waveState.v.view(-1, 1),
            waveSystemNext.waveState.damping.view(-1, 1),
        ], dim=-1)
    else:
        currentFeatures = torch.cat([
            waveSystemNext.waveState.u.view(-1, 1),
            waveSystemNext.waveState.v.view(-1, 1),
        ], dim=-1)
    currentFeatures = currentFeatures.unsqueeze(1).unsqueeze(0)
    # print(f'Current features shape: {currentFeatures.shape}')
    inputFeatures = torch.cat([inputFeatures, currentFeatures], dim=2)
    # print(f'Input features shape: {inputFeatures.shape}, Trajectory features shape: {trajFeatures.shape}')

    selectedFeatures = inputFeatures[:,:,-args.historyLength-1:,:]
    # print(f'Selected features shape: {selectedFeatures.shape}')
    flattenedSelectedFeatures = selectedFeatures.reshape(selectedFeatures.shape[0], selectedFeatures.shape[1], -1).flatten(0,1)
    # print(f'Flattened selected features shape: {flattenedSelectedFeatures.shape}')


    # inputFeatures = torch.cat([
    #     waveSystemNext.waveState.u.view(-1, 1),
    #     waveSystemNext.waveState.v.view(-1, 1),
    # ], dim=-1)
    node_attr = [flattenedSelectedFeatures]

    fi_nn = model(
        node_attr = node_attr[0],
        node_positions = node_positions[0],
        node_supports = node_supports,
        node_volumes = node_volumes,
        
        domain = domain,
        windows = windows,
        shifted_windows = shifted_windows
    )

    dudt = fi_nn[:, 0]
    dvdt = fi_nn[:, 1]

    waveSystemNeural.waveState.u = waveSystemNext.waveState.u + dudt
    waveSystemNeural.waveState.v = waveSystemNext.waveState.v + dvdt #/ dt
    
fig, axis = plt.subplots(2, 5, figsize=(18,6), sharex=True, sharey=True)

markerSize = 1.5

uMaxRk4 = torch.max(torch.abs(waveSystemRK4.waveState.u)).cpu().detach().item()
vMaxRk4 = torch.max(torch.abs(waveSystemRK4.waveState.v)).cpu().detach().item()

uPlotRk4 = visualizeParticles(fig, axis[0,0], waveSystem.systemState, config['domain'], waveSystemRK4.waveState.u, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'managua', markerSize = markerSize, gridVisualization = False, title = 'u [Rk4]', vmin = -uMaxRk4, vmax = uMaxRk4)
vPlotRk4 = visualizeParticles(fig, axis[1,0], waveSystem.systemState, config['domain'], waveSystemRK4.waveState.v, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'vanimo', markerSize = markerSize, gridVisualization = False, title = 'v [Rk4]', vmin = -vMaxRk4, vmax = vMaxRk4)


umaxEuler = torch.max(torch.abs(waveSystemNeural.waveState.u)).cpu().detach().item()
vmaxEuler = torch.max(torch.abs(waveSystemNeural.waveState.v)).cpu().detach().item()

uPlotEuler = visualizeParticles(fig, axis[0,1], waveSystem.systemState, config['domain'], waveSystemNeural.waveState.u, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'managua', markerSize = markerSize, gridVisualization = False, title = 'u [Neural]', vmin = -umaxEuler, vmax = umaxEuler)
vPlotEuler = visualizeParticles(fig, axis[1,1], waveSystem.systemState, config['domain'], waveSystemNeural.waveState.v, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'vanimo', markerSize = markerSize, gridVisualization = False, title = 'v [Neural]', vmin = -vmaxEuler, vmax = vmaxEuler)

uDiff = (waveSystemRK4.waveState.u - waveSystemNeural.waveState.u)
vDiff = (waveSystemRK4.waveState.v - waveSystemNeural.waveState.v)

uDiff = torch.stack([u.dudt for u in updates], dim=0).mean(dim=0) * dt
vDiff = torch.stack([u.dvdt for u in updates], dim=0).mean(dim=0) * dt

uDiffVmax = torch.max(torch.abs(uDiff)).cpu().detach().item()
vDiffVmax = torch.max(torch.abs(vDiff)).cpu().detach().item()

uPlotDiff = visualizeParticles(fig, axis[0,2], waveSystem.systemState, config['domain'], uDiff, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'RdBu', markerSize = markerSize, gridVisualization = False, title = 'du [RK4]', vmin = -uDiffVmax, vmax = uDiffVmax)
vPlotDiff = visualizeParticles(fig, axis[1,2], waveSystem.systemState, config['domain'], vDiff, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'RdBu', markerSize = markerSize, gridVisualization = False, title = 'dv [RK4]', vmin = -vDiffVmax, vmax = vDiffVmax)

# uDiff = (waveSystemRK4.waveState.u - waveSystem.waveState.u)
# vDiff = (waveSystemRK4.waveState.v - waveSystem.waveState.v)
uDiffNN = fi_nn[:, 0]
vDiffNN = fi_nn[:, 1]

uDiffVmax = torch.max(torch.abs(uDiffNN)).cpu().detach().item()
vDiffVmax = torch.max(torch.abs(vDiffNN)).cpu().detach().item()

uPlotDiffInitial = visualizeParticles(fig, axis[0,3], waveSystem.systemState, config['domain'], uDiffNN, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'RdBu', markerSize = markerSize, gridVisualization = False, title = 'du [Prediction]', vmin = -uDiffVmax, vmax = uDiffVmax)
vPlotDiffInitial = visualizeParticles(fig, axis[1,3], waveSystem.systemState, config['domain'], vDiffNN, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'RdBu', markerSize = markerSize, gridVisualization = False, title = 'dv [Prediction]', vmin = -vDiffVmax, vmax = vDiffVmax)

errU = uDiff - uDiffNN
errV = vDiff - vDiffNN

errU = waveSystem.waveState.u
errV = waveSystem.waveState.v

errUmax = torch.max(torch.abs(errU)).cpu().detach().item()
errVmax = torch.max(torch.abs(errV)).cpu().detach().item()

uPlotError = visualizeParticles(fig, axis[0,4], waveSystem.systemState, config['domain'], errU, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'managua', markerSize = markerSize, gridVisualization = False, title = 'u [Init]', vmin = -errUmax, vmax = errUmax)
vPlotError = visualizeParticles(fig, axis[1,4], waveSystem.systemState, config['domain'], errV, config['kernel'], which = 'both', visualizeBoth = True, cbar = True, cmap = 'vanimo', markerSize = markerSize, gridVisualization = False, title = 'v [Init]', vmin = -errVmax, vmax = errVmax)


fig.tight_layout()
fig.savefig(os.path.join(outputDir, 'prediction_comparison_16steps.png'), dpi=300)




















