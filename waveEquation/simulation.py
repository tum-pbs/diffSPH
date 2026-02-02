from util import getCurrentTimestamp
from diffSPH.simple import *

import h5py
import datetime
import os
import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
from tqdm.autonotebook import tqdm
from damping import apply_spectral_filter

import shlex
import subprocess

def runSimulation(fig, particleState, uPlot, vPlot, waveSystem, waveSystemFunction, integrator, nx, dt, nIter, kernel, config, export, plotInterval, exportImages = True):

    timestamp = getCurrentTimestamp()
    if export:
        os.makedirs('output', exist_ok=True)

        fileName = f"output/waveEqn_{timestamp}.h5"
        print(f"Saving to {fileName}")

        outFile = h5py.File(fileName, 'w')

        particleGroup = outFile.create_group('particles')
        particleGroup.create_dataset('positions', data = waveSystem.systemState.positions.cpu().numpy())
        particleGroup.create_dataset('masses', data = waveSystem.systemState.masses.cpu().numpy())
        particleGroup.create_dataset('densities', data = waveSystem.systemState.densities.cpu().numpy())
        particleGroup.create_dataset('supports', data = waveSystem.systemState.supports.cpu().numpy())
        particleGroup.create_dataset('numNeighbors', data = waveSystem.systemState.numNeighbors.cpu().numpy())

        initialWaveGroup = outFile.create_group('initialWaveState')
        initialWaveGroup.create_dataset('u', data = waveSystem.waveState.u.cpu().numpy())
        initialWaveGroup.create_dataset('v', data = waveSystem.waveState.v.cpu().numpy())
        initialWaveGroup.create_dataset('c', data = waveSystem.waveState.c.cpu().numpy())
        initialWaveGroup.create_dataset('damping', data = waveSystem.waveState.damping.cpu().numpy())

        simulationGroup = outFile.create_group('simulation')
        simulationGroup.attrs['dt'] = dt
        simulationGroup.attrs['nIter'] = nIter
        simulationGroup.attrs['kernel'] = kernel.name
        simulationGroup.attrs['targetNeighbors'] = n_h_to_nH(4, 2)
        simulationGroup.attrs['integrationScheme'] = config['integrationScheme'].name
        
    if exportImages:
        imagePath = f'output/images/waveEqn_{timestamp}'
        os.makedirs(imagePath, exist_ok=True)
        fig.savefig(f'{imagePath}/waveEqn_0000.png', dpi = 200)
    
    import numpy as np
    # Optional: Apply spectral filtering instead of (or in addition to) global damping
    # Uncomment these lines in the integration loop below to use spectral filtering
    use_spectral_filter = False  # Set to True to enable
    k_cutoff_fraction = 0.7      # Start damping at 70% of max wavenumber
    spectral_power = 4           # Sharpness of spectral cutoff

    t = 0.0
    # plotInterval = 50
    if export:
        us = []
        vs = []

        dudts = []
        dvdts = []

    for i in tqdm(range(nIter), leave = False):
        waveSystem, updates =  integrator.function(
            waveSystem,
            dt = dt,
            f = waveSystemFunction,
            verbose = False,
            config = config,
        )
        # Damping is now applied within the wave equation itself (PML-style or global)
        
        # Optional: Apply spectral filtering for periodic domains (alternative to global damping)
        if use_spectral_filter:
            waveSystem.waveState.u = apply_spectral_filter(
                waveSystem.waveState.u, nx, 2, k_cutoff_fraction, spectral_power)
            waveSystem.waveState.v = apply_spectral_filter(
                waveSystem.waveState.v, nx, 2, k_cutoff_fraction, spectral_power)
        
        t += dt

        if export:
            dudt = []
            dvdt = []
            for j in range(len(updates)):
                dudt.append(updates[j].dudt.cpu().numpy())
                dvdt.append(updates[j].dvdt.cpu().numpy())
            dudt = np.stack([torch.tensor(d) for d in dudt], axis=0).T
            dvdt = np.stack([torch.tensor(d) for d in dvdt], axis=0).T

            us.append(waveSystem.waveState.u.view(-1,1).cpu().numpy())
            vs.append(waveSystem.waveState.v.view(-1,1).cpu().numpy())
            dudts.append(dudt)
            dvdts.append(dvdt)

        if i % plotInterval == 0 or i == nIter - 1:
            updatePlot(uPlot, particleState, waveSystem.waveState.u)
            updatePlot(vPlot, particleState, waveSystem.waveState.v)

            fig.canvas.draw()
            fig.canvas.flush_events()
            if exportImages:    
                fig.savefig(f'{imagePath}/waveEqn_{i+1:04d}.png', dpi = 200)
            
    if exportImages:
        output = 'timestamp'
        scale = 1280

        command = '/usr/bin/ffmpeg -loglevel warning -hide_banner -y -framerate 50 -f image2 -pattern_type glob -i '+ imagePath + '/waveEqn_*.png -c:v libx264 -b:v 20M -r 50 ' + imagePath + '/output.mp4'
        commandB = f'ffmpeg -loglevel warning -hide_banner -y -i {imagePath}/output.mp4 -vf "fps=50,scale={scale}:-1:flags=lanczos,palettegen" {imagePath}/palette.png'
        commandC = f'ffmpeg -loglevel warning -hide_banner -y -i {imagePath}/output.mp4 -i {imagePath}/palette.png -filter_complex "fps=50,scale={scale}:-1:flags=lanczos[x];[x][1:v]paletteuse" {imagePath}/output.gif'

        print('Creating video from  frames (frame count: {})'.format(len(os.listdir(imagePath))))
        subprocess.run(shlex.split(command))
        print('Creating gif palette')
        subprocess.run(shlex.split(commandB))
        print('Creating gif')
        subprocess.run(shlex.split(commandC))
        print('Done')

    if export:        
        dudt_stacked = np.stack(dudts, axis=0)
        dvdt_stacked = np.stack(dvdts, axis=0)
        u_stacked = np.stack(us, axis=0)
        v_stacked = np.stack(vs, axis=0)

        simulationGroup.create_dataset('u', data = u_stacked)
        simulationGroup.create_dataset('v', data = v_stacked)
        simulationGroup.create_dataset('dudt', data = dudt_stacked)
        simulationGroup.create_dataset('dvdt', data = dvdt_stacked)

        outFile.close()