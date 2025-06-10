import torch
import warnings
import os
import copy
import subprocess
import shlex
import matplotlib.pyplot as plt
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm
from torch.profiler import profile,  ProfilerActivity

# os.environ['TORCH_CUDA_ARCH_LIST'] = f'{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}'

from sphMath.sampling import buildDomainDescription, sampleDivergenceFreeNoise
from sphMath.modules.adaptiveSmoothingASPH import n_h_to_nH
from sphMath.plotting import visualizeParticles, updatePlot
from sphMath.integration import getIntegrator, getIntegrationEnum
from sphMath.util import volumeToSupport
from sphMath.boundary import sampleDomainSDF
from sphMath.kernels import Kernel_Scale, getKernelEnum
from sphMath.sdf import getSDF, sdfFunctions, operatorDict, sampleSDF
from sphMath.regions import buildRegion, filterRegion, plotRegions
from sphMath.modules.timestep import computeTimestep
from sphMath.schemes.initializers import initializeSimulation, updateBodyParticles
from sphMath.schemes.deltaSPH import deltaPlusSPHScheme, DeltaPlusSPHSystem
from sphMath.schema import getSimulationScheme
from sphMath.enums import *
import math
import argparse

def setupExampleSimulation(simulationName, scheme, particleState, config, regions, stacked = 'horizontal', figsize = (12, 4), markerSize= 1):
    dt = computeTimestep(scheme, 1e-2, particleState, config, None)
    particles = copy.deepcopy(particleState)
    particleSystem = DeltaPlusSPHSystem(config['domain'], None, 0., copy.deepcopy(particleState), 'momentum', None, rigidBodies = config['rigidBodies'], regions = config['regions'], config = config)

    for rigidBody in config['rigidBodies']:
        particleState = updateBodyParticles(scheme, particleState, rigidBody)

    ## Setup values for plotting 

    kineticEnergy = 0.5 * particleState.densities * (particleState.velocities ** 2).sum(1)

    t = 0
    E_k0 = 0.5 * particleState.masses * torch.linalg.norm(particleState.velocities, dim = -1)**2
    E_k = 0.5 * particleState.masses * torch.linalg.norm(particleState.velocities, dim = -1)**2

    rhoMin = particleState.densities.min().detach().cpu().item() / config['fluid']['rho0']
    rhoMean = particleState.densities.mean().detach().cpu().item() / config['fluid']['rho0']
    rhoMax = particleState.densities.max().detach().cpu().item() / config['fluid']['rho0']

    initialVelocity = particleState.velocities.clone()
    initialDensities = particleState.densities.clone()
    initialPotentialEnergy = None

    if config['gravity']['active']:
        if 'mode' in config['gravity'] and config['gravity']['mode'] == 'potential':
            B = config['gravity']['magnitude']
            initialPotentialEnergy = 0.5 * B**2 * particleState.masses / config['fluid']['rho0'] * particleState.densities * torch.linalg.norm(particleState.positions, dim = -1)**2

    initialKineticEnergy = 0.5 * particleState.masses / config['fluid']['rho0'] * particleState.densities * torch.linalg.norm(initialVelocity, dim = -1)**2
    initialEnergy = initialKineticEnergy + initialPotentialEnergy if initialPotentialEnergy is not None else initialKineticEnergy

    if stacked == 'horizontal':
        fig, axis = plt.subplots(1, 3, figsize = figsize, squeeze=False)
    elif stacked == 'vertical':
        fig, axis = plt.subplots(3, 1, figsize = figsize, squeeze=False)
    else:
        raise ValueError(f"Invalid value for 'stacked': {stacked}. Expected 'horizontal' or 'vertical'.")

    axes = axis.flatten()
    
    velocityPlot = visualizeParticles(fig, axes[0], 
                particles = particleSystem.systemState, 
                domain = config['domain'], 
                quantity = particleSystem.systemState.velocities, 
                which = 'fluid',
                mapping = 'L2',
                cmap = 'viridis',
                visualizeBoth=True,
                kernel = config['kernel'],
                plotDomain = False,
                gridVisualization=False, markerSize=markerSize, streamLines=False)

    densityPlot = visualizeParticles(fig, axes[1],
                particles = particleSystem.systemState, 
                domain = config['domain'], 
                quantity = particleSystem.systemState.densities, 
                which = 'both',
                # operation = 'curl',
                mapping = '.y',
                cmap = 'magma',
                visualizeBoth=True,
                kernel = config['kernel'],
                plotDomain = False,
                gridVisualization=False, markerSize=markerSize)

    uidPlot = visualizeParticles(fig, axes[2],
                particles = particleSystem.systemState, 
                domain = config['domain'], 
                quantity = particleSystem.systemState.UIDs, 
                which = 'fluid',
                mapping = '.x',
                cmap = 'twilight_r',
                visualizeBoth=True,
                kernel = config['kernel'],
                plotDomain = False,
                gridVisualization=False, markerSize=markerSize, streamLines=False)
    axis[0,0].set_title(r'$|\mathbf{u}|$')
    axis[0,1].set_title(r'$\rho$')
    axis[0,2].set_title(r'UID')

    totalInitialEnergy = (initialEnergy).sum().detach().cpu().item()
    totalEnergy = (initialEnergy).sum().detach().cpu().item()

    fig.suptitle(f'{simulationName}, ptcls = {particleState.positions.shape[0]}, kernel = {config["kernel"].name}, neighbors = {config["targetNeighbors"]:.2g}, $c_s$ = {config["fluid"]["c_s"]:.1f}\n$\\rho$ = [{rhoMin:.4g} | {rhoMean:.4g} | {rhoMax:.4g}], $E_0$ = {totalInitialEnergy:.4g}, $E$ = {totalEnergy:.4g}, $\\Delta E$ = {totalEnergy - totalInitialEnergy:.4g}, $t$ = {particleSystem.t:.4g}, $\Delta t$ = {dt:.2e}')

    fig.tight_layout()
    
    return fig, axis, velocityPlot, densityPlot, uidPlot, particleSystem, dt, initialKineticEnergy, initialPotentialEnergy, initialEnergy

## Run simulation

# imagePrefix = f'./images/{exportName}/'
# os.makedirs(imagePrefix, exist_ok = True)
# fig.savefig(f'{imagePrefix}frame_{0:05d}.png', dpi = 100)

# particles = copy.deepcopy(particleState)
# particleSystem = DeltaPlusSPHSystem(domain, None, 0., copy.deepcopy(particleState), 'momentum', None, rigidBodies = rigidBodies, regions = config['regions'], config = config)
from sphMath.io import saveState

def runSimulation(simulationName, particleSystem, integrationScheme, simulationScheme, timesteps, dt, config, fig, axis, velocityPlot, densityPlot, uidPlot, initialEnergy, imagePrefix, plotInterval = 1, callBackFn = None, plotCallbackFn = None, outFile = None, exportInterval = 1):

    exportStatic = False
    for region in config['regions']:
        if region['type'] == 'inlet' or region['type'] == 'outlet':
            exportStatic = True
            break
    if outFile is not None:
        outGroup = outFile.create_group('simulationData')
    if outFile is not None:
        frameGroup = outGroup.create_group(f'{0:06d}')
        saveState(particleSystem.systemState, particleSystem.systemState.kinds != 2, frameGroup, config, static = exportStatic)
        # if torch.any(particleSystem.systemState.kinds > 0):
        #     frameGroup.create_dataset('normals', data = particleSystem.systemState.ghostOffsets[particleSystem.systemState.kinds != 2].cpu().numpy())
        frameGroup.attrs['time'] = particleSystem.t.cpu().item() if isinstance(particleSystem.t, torch.Tensor) else particleSystem.t
        frameGroup.attrs['dt'] = dt.cpu().item() if isinstance(dt, torch.Tensor) else dt
        frameGroup.attrs['timestep'] = 0
    for i in (tq:=tqdm(range(timesteps))):
        particleSystem, currentState, updates = integrationScheme.function(particleSystem, dt, simulationScheme, config, priorStep = particleSystem.priorStep, verbose = False)
        
        if callBackFn is not None:
            particleSystem = callBackFn(particleSystem)
        
        if i % exportInterval == 0:
            if outFile is not None:
                frameGroup = outGroup.create_group(f'{i + 1:06d}')
                saveState(particleSystem.systemState, particleSystem.systemState.kinds != 2, frameGroup, config, static = exportStatic)
                # if torch.any(particleSystem.systemState.kinds > 0):
                #     frameGroup.create_dataset('normals', data = particleSystem.systemState.ghostOffsets[particleSystem.systemState.kinds != 2].cpu().numpy())
                frameGroup.attrs['time'] = particleSystem.t.cpu().item()
                frameGroup.attrs['dt'] = dt.cpu().item()
                frameGroup.attrs['timestep'] = i + 1

        if i % plotInterval == plotInterval - 1 or i == timesteps - 1:

            rhoMin = particleSystem.systemState.densities.min().detach().cpu().item() / config['fluid']['rho0']
            rhoMean = particleSystem.systemState.densities.mean().detach().cpu().item() / config['fluid']['rho0']
            rhoMax = particleSystem.systemState.densities.max().detach().cpu().item() / config['fluid']['rho0']

            kineticEnergy = 0.5 * particleSystem.systemState.masses / config['fluid']['rho0'] * particleSystem.systemState.densities * torch.linalg.norm(particleSystem.systemState.velocities, dim = -1)**2
            potentialEnergy = None
            if config['gravity']['active']:
                if 'mode' in config['gravity'] and config['gravity']['mode'] == 'potential':
                    B = config['gravity']['magnitude']
                    potentialEnergy = 0.5 * B**2 * particleSystem.systemState.masses / config['fluid']['rho0'] * particleSystem.systemState.densities * torch.linalg.norm(particleSystem.systemState.positions, dim = -1)**2
                    
            combinedEnergy = kineticEnergy + potentialEnergy if potentialEnergy is not None else kineticEnergy
            totalInitialEnergy = (initialEnergy).sum().detach().cpu().item()
            totalEnergy = (combinedEnergy).sum().detach().cpu().item()

            fig.suptitle(f'{simulationName}, ptcls = {particleSystem.systemState.positions.shape[0]}, kernel = {config["kernel"].name}, neighbors = {config["targetNeighbors"]:.2g}, $c_s$ = {config["fluid"]["c_s"]:.1f}\n$\\rho$ = [{rhoMin:.4g} | {rhoMean:.4g} | {rhoMax:.4g}], $E_0$ = {totalInitialEnergy:.4g}, $E$ = {totalEnergy:.4g}, $\\Delta E$ = {totalEnergy - totalInitialEnergy:.4g}, $t$ = {particleSystem.t:.4g}, $\Delta t$ = {dt:.2e}')
            updatePlot(velocityPlot, particleSystem.systemState, particleSystem.systemState.velocities)
            updatePlot(densityPlot, particleSystem.systemState, particleSystem.systemState.densities)
            updatePlot(uidPlot, particleSystem.systemState, particleSystem.systemState.UIDs)
            if plotCallbackFn is not None:
                plotCallbackFn(fig, axis, particleSystem, i)
            fig.canvas.draw()
            fig.canvas.flush_events()
            fig.savefig(f'{imagePrefix}frame_{i:05d}.png', dpi = 100)

    ## build output
    
def postProcess(imagePrefix, fps, timesteps, exportName):
    output = 'timestamp'
    scale = 1280

    command = f'/usr/bin/ffmpeg -loglevel warning -hide_banner -y -framerate {fps} -f image2 -pattern_type glob -i '+ imagePrefix + f'/frame_*.png -c:v libx264 -b:v 20M -r {fps} ' + imagePrefix + '/output.mp4'
    commandB = f'/usr/bin/ffmpeg -loglevel warning -hide_banner -y -i {imagePrefix}/output.mp4 -vf "fps={fps},scale={scale}:-1:flags=lanczos,palettegen" {imagePrefix}/palette.png'
    commandC = f'/usr/bin/ffmpeg -loglevel warning -hide_banner -y -i {imagePrefix}/output.mp4 -i {imagePrefix}/palette.png -filter_complex "fps={fps},scale={scale}:-1:flags=lanczos[x];[x][1:v]paletteuse" {imagePrefix}/output.gif'

    # command = f'ffmpeg -loglevel warning -hide_banner -y -framerate {fps} -f image2 -pattern_type glob -i '+ imagePrefix + f'/frame_*.png -c:v libx264 -b:v 20M -r {fps} ' + imagePrefix + '/output.mp4'
    # commandB = f'ffmpeg -loglevel warning -hide_banner -y -i {imagePrefix}/output.mp4 -vf "fps={fps},scale={scale}:-1:flags=lanczos,palettegen" {imagePrefix}/palette.png'
    # commandC = f'ffmpeg -loglevel warning -hide_banner -y -i {imagePrefix}/output.mp4 -i {imagePrefix}/palette.png -filter_complex "fps={fps},scale={scale}:-1:flags=lanczos[x];[x][1:v]paletteuse" {imagePrefix}/output.gif'

    # print('Creating video from  frames (frame count: {})'.format(len(os.listdir(imagePrefix))))
    subprocess.run(shlex.split(command))
    subprocess.run(shlex.split(commandB))
    subprocess.run(shlex.split(commandC))

    os.makedirs(f'./videos/', exist_ok= True)
    # subprocess.run(shlex.split(f'cp {imagePrefix}/output.mp4 ./videos/{exportName}.mp4'))
    subprocess.run(shlex.split(f'cp {imagePrefix}/output.gif ./videos/{exportName}.gif'))
    lastFrameFile = f'{imagePrefix}/frame_{timesteps - 1:05d}.png'
    subprocess.run(shlex.split(f'cp {lastFrameFile} ./videos/{exportName}.png'))


import imageio.v2 as imageio
from skimage.transform import resize, rescale
from skimage.io import imread

def postProcess(imagePrefix, fps, timesteps, exportName):
    fileList = os.listdir(imagePrefix)
    fileList = [f for f in fileList if f.endswith('.png') and f.startswith('frame_')]
    fileList = sorted(fileList)

    writer = imageio.get_writer(f'{imagePrefix}/output.mp4', fps=50, bitrate='10M')
    for image in fileList:
        writer.append_data(imageio.imread(imagePrefix + image))
    writer.close()

    images = []
    for image in fileList:
        images.append(imread(imagePrefix + image))

    currentImageSize = images[0].shape
    currentLongEdge = max(images[0].shape[0], images[0].shape[1])
    targetLongEdge = 800
    ratio = targetLongEdge / currentLongEdge

    images = [(rescale(image, ratio, anti_aliasing=True, channel_axis=-1)*255).astype('uint8') for image in images]

    # for image in images:
        # print(image.shape)
        # break

    # images = [resizeImageLongEdge(i, 600) for i in images]

    imageio.mimsave(f'{imagePrefix}/output.gif', images, fps=50, loop=0)
    
    print('Copying video to videos folder')
    os.makedirs(f'./videos/', exist_ok= True)
    # subprocess.run(shlex.split(f'cp {imagePrefix}/output.mp4 ./videos/{exportName}.mp4'))
    subprocess.run(shlex.split(f'cp {imagePrefix}/output.gif ./videos/{exportName}.gif'))
    lastFrameFile = f'{imagePrefix}/frame_{timesteps - 1:05d}.png'
    subprocess.run(shlex.split(f'cp {lastFrameFile} ./videos/{exportName}.png'))
    print('Done!')