import torch
from sphMath.sphOperations.shared import getTerms, compute_xij
from sphMath.kernels import Kernel_xi
from sphMath.enums import ViscositySwitch

def getAlphas(particles_a, particles_b, neighborhood, solverConfig):
    viscositySwitch     = solverConfig.get('diffusionSwitch',{}).get('scheme', None)
    if viscositySwitch is not None and viscositySwitch != ViscositySwitch.NoneSwitch:
        alpha_i = particles_a.alphas[neighborhood.row]
        alpha_j = particles_b.alphas[neighborhood.col]
        return alpha_i, alpha_j
    else:
        return torch.ones(neighborhood.row.shape[0], dtype = particles_a.densities.dtype, device = particles_a.densities.device), torch.ones(neighborhood.col.shape[0], dtype = particles_b.densities.dtype, device = particles_b.densities.device) #* solverConfig.get('alpha', 1.0)
    
def compute_Pi(particles_a, particles_b, domain, neighborhood, config, useJ = False, thermalConductivity = False):
    if 'diffusion' not in config:
        config['diffusion'] = {}
    correctXi           = config.get('diffusion',{}).get('correctXi', False)
    viscosityTerm       = config.get('diffusion',{}).get('viscosityTerm', None)
    switch              = config.get('diffusion',{}).get('monaghanSwitch', True)
    use_rho_bar         = config.get('diffusion',{}).get('use_rho_bar', None)
    use_c_bar           = config.get('diffusion',{}).get('use_cbar', None)
    use_h_bar           = config.get('diffusion',{}).get('use_hbar', None)
    C_l                 = config.get('diffusion',{}).get('C_l', 1)
    C_q                 = config.get('diffusion',{}).get('C_q', 2)
    K                   = config.get('diffusion',{}).get('K', None)   
    switchScaling       = config.get('diffusion',{}).get('switchScaling', 'ij')
    viscosityFormulation= config.get('diffusion',{}).get('viscosityFormulation', 'MonaghanGingold1983')
    if thermalConductivity:
        viscosityFormulation = config['diffusion']['thermalConductivityFormulation'] if 'thermalConductivityFormulation' in config['diffusion'] else viscosityFormulation
        C_l = config['diffusion']['Cu_l'] if 'Cu_l' in config['diffusion'] else C_l
        C_q = config['diffusion']['Cu_q'] if 'Cu_q' in config['diffusion'] else C_q
    scaleBeta           = config['diffusion'].get('scaleBeta', False)

    if viscosityFormulation == 'MonaghanGingold1983':
        K               = 1 if K is None else K
        use_rho_bar     = True if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1992'
    elif viscosityFormulation == 'Cleary1998':
        K               = 1 if K is None else K
        use_rho_bar     = False if use_rho_bar is None else use_rho_bar
        use_c_bar       = False if use_c_bar is None else use_c_bar
        use_h_bar       = False if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1997'
    elif viscosityFormulation == 'Monaghan1992':
        K               = 1 if K is None else K
        use_rho_bar     = True if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1992'
    elif viscosityFormulation in ['Monaghan1997', 'Dukowicz', 'Price2012', 'Price2012_98', 'Price2008', 'Wadsley2008']:
        K               = 1 if K is None else K
        use_rho_bar     = True if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1997'
    elif viscosityFormulation == 'delta':
        K = 1 if K is None else K
        use_rho_bar     = False if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1992'
    

    rho_i, rho_j, rho_bar   = getTerms(neighborhood.row, neighborhood.col, (particles_a.densities, particles_b.densities))
    c_i, c_j, c_bar         = getTerms(neighborhood.row, neighborhood.col, (particles_a.soundspeeds, particles_b.soundspeeds))
    h_i, h_j, h_bar         = getTerms(neighborhood.row, neighborhood.col, (particles_a.supports, particles_b.supports))
    alpha_i, alpha_j        = getAlphas(particles_a, particles_b, neighborhood, config)

    d   = particles_a.positions.shape[1]
    rho = rho_bar   if use_rho_bar  else (rho_j if useJ else rho_i)
    c   = c_bar     if use_c_bar    else (c_j   if useJ else c_i)
    h   = h_bar     if use_h_bar    else (h_j   if useJ else h_i)
    
    # if viscosityFormulation == 'delta':
        # rho = rho_i / solverConfig['fluid']['rho0']
        # c = solverConfig['fluid']['c_s']
    
    kernel = config['kernel']

    xi  = Kernel_xi(kernel, particles_a.positions.shape[1]) if correctXi else 1.0

    # print(f'alpha_i: {alpha_i.shape}, min: {alpha_i.min()}, max: {alpha_i.max()}')
    # print(f'alpha_j: {alpha_j.shape}, min: {alpha_j.min()}, max: {alpha_j.max()}')    
    
    C_l = 1/2 * (alpha_i + alpha_j) * C_l
    C_q = 1/2 * (alpha_i + alpha_j) * C_q
    if switchScaling == 'i':
        # print('Scaling with i')
        C_l = C_l * alpha_i
        C_q = C_q * alpha_i
    elif switchScaling == 'j':
        # print('Scaling with j')
        C_l = C_l * alpha_j
        C_q = C_q * alpha_j
    
        
    if scaleBeta:
        # print('Scaling beta')
        C_q = C_q * C_l
        
    # print(solverConfig)
    # print(f'C_l: {C_l.shape}, min: {C_l.min()}, max: {C_l.max()}')
    # print(f'C_q: {C_q.shape}, min: {C_q.min()}, max: {C_q.max()}')
        

    x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)
    u_ij        = particles_a.velocities[neighborhood.row] - particles_b.velocities[neighborhood.col]
    ux_ij       = torch.einsum('ij,ij->i', u_ij, x_ij)

    if viscosityTerm == 'Monaghan' or viscosityTerm == 'Monaghan1992':
        mu_ij = ux_ij / (r_ij**2 + 1e-14 * h**2) * h / xi
    else: # 'Monaghan1997' also referred to as 'j' in the paper
        mu_ij = ux_ij / (r_ij + 1e-14 * h)
    
    # print(f'mu_ij: {mu_ij.shape}, min: {mu_ij.min()}, max: {mu_ij.max()}')
    
    if switch:
        mu_ij[ux_ij > 0] = 0

    # There is some confusion regarding the terms here.
    # Monaghan 2005 prescribes several potential terms for the viscosity
    # Notationally we use Pi_ab = - K / rho v_sig mu_ab
    
    if viscosityFormulation == 'MonaghanGingold1983':
    # Monaghan and Gingold 1983: The terms are given in (8.3) and (8.4)  of Monaghan 2005 and are
    # Pi_ab = -nu ( v_ab \cdot r_ab) / (r_ab^2 + epsilon^2 h_ab^2)
    # nu = alpha h_bar c_bar / rho_bar
    # Rewording this slightly we get the 'Monaghan1992' viscosity Term (with xi correction)
    # combined with using c_bar, rho_bar and h_bar. 
    # Consequently this uses 
    # v_sig = c_bar
    # K = 1
        v_sig = c_bar
    elif viscosityFormulation == 'Cleary1998':
    # Cleary 1998: The terms are given in (8.8) and (8.9) of Monaghan 2005 and are
    # mu_a = 1/8 alpha_a h_a c_a rho_a
    # Pi_ab = - 16 mu_a mu_b / (rho_a rho_b (mu_a + mu_b)) mu_ij
        f = 1/(2*(d+2)) # Based on estimations based on Monaghan 2005, not given for 1D
        mu_i = f * alpha_i * C_l * h_i * c_i * rho_i / xi
        mu_j = f * alpha_j * C_l * h_j * c_j * rho_j / xi
        # 19.8 based on Cleary and Ha 2002
        v_sig = 19.8 * mu_i * mu_j / (rho_i * rho_j * (mu_i + mu_j)) / (r_ij + 1e-14 * h)
    elif viscosityFormulation == 'Monaghan1992':
    # Monaghan 1992: The term is given in (8.10) of Monaghan 2005 and is
    # mu = h / rho ( alpha c - beta mu_ij)
    # This uses the Monaghan 1992 viscosity term with alpha = 1 and beta = 2
        v_sig = C_l * c - C_q * mu_ij
    elif viscosityFormulation == 'Monaghan1997a':
    # Monaghan 1997: The term is given in (8.11) of Monaghan 2005 and is very similar
    # to the Monaghan1992 term but uses the Monaghan1997 viscosity term. denoted as j
    # in the 1997 paper and has a strange wording in 2005 of using 1/2 instead of 1 for K
    # c_i + c_j instead of c_bar and beta = 4. Cancelling these terms out gives the normal
    # c_bar term with alpha = 1 and beta = 4! This is also eq 3.7 in Monaghan1997
        v_sig = C_l * c - C_q * mu_ij
    elif viscosityFormulation == 'Monaghan1997b':
    # Based on Monaghan 1997 eq 4.7:
        v_sig = (c_i**2 + C_q * mu_ij**2)**0.5 + (c_j**2 + C_q * mu_ij**2)**0.5 - C_q * mu_ij
    elif viscosityFormulation == 'Dukowicz':
    # The term is given in (4.8) of Monaghan 1997 and is simply the 1997a term with a 3/4 factor
        v_sig = C_l * c - 3/4 * C_q * mu_ij
    # Next are the formulations based on Price's SPMHD paper from 2012
    elif viscosityFormulation == 'Price2012_98':
    # This term is identical to Monaghan 1992, equation 98 in Price 2012
        v_sig = C_l * c - C_q * mu_ij
    elif viscosityFormulation == 'Price2012':
    # Based on equation 103
        v_sig = C_l * c - C_q / 2 * mu_ij
    elif viscosityFormulation == 'Price2008':
    # This formulation and the next are only mentioned in the Price 2012 after equation 103, no explicit equation numbers
        P_i, P_j    = particles_a.pressures[neighborhood.row], particles_b.pressures[neighborhood.col]
        rho_bar     = (rho_i + rho_j) / 2
        v_sig       = C_l * torch.sqrt(torch.abs(P_i - P_j) / (rho_bar + 1e-14 * h))
    elif viscosityFormulation == 'Wadsley2008':
        v_sig = C_l * torch.abs(mu_ij)
    else:
        v_sig = C_l * c - C_q * mu_ij
    if thermalConductivity:
        return -K / rho * v_sig * (particles_a.internalEnergies[neighborhood.row] - particles_b.internalEnergies[neighborhood.col])
    return -K / rho * v_sig * mu_ij
    

    return -K/ rho * C_l * (c * mu_ij - C_q * mu_ij**2)


def compute_Pi_v2(particles_a, particles_b, domain, neighborhood, config, useJ = False, thermalConductivity = False):
    if 'diffusion' not in config:
        config['diffusion'] = {}
    correctXi           = config.get('diffusion',{}).get('correctXi', False)
    viscosityTerm       = config.get('diffusion',{}).get('viscosityTerm', None)
    switch              = config.get('diffusion',{}).get('monaghanSwitch', True)
    use_rho_bar         = config.get('diffusion',{}).get('use_rho_bar', None)
    use_c_bar           = config.get('diffusion',{}).get('use_cbar', None)
    use_h_bar           = config.get('diffusion',{}).get('use_hbar', None)
    C_l                 = config.get('diffusion',{}).get('C_l', 1)
    C_q                 = config.get('diffusion',{}).get('C_q', 2)
    K                   = config.get('diffusion',{}).get('K', None)   
    switchScaling       = config.get('diffusion',{}).get('switchScaling', 'ij')
    viscosityFormulation= config.get('diffusion',{}).get('viscosityFormulation', 'MonaghanGingold1983')
    if thermalConductivity:
        viscosityFormulation = config['diffusion']['thermalConductivityFormulation'] if 'thermalConductivityFormulation' in config['diffusion'] else viscosityFormulation
        C_l = config['diffusion']['Cu_l'] if 'Cu_l' in config['diffusion'] else C_l
        C_q = config['diffusion']['Cu_q'] if 'Cu_q' in config['diffusion'] else C_q
    scaleBeta           = config['diffusion'].get('scaleBeta', False)

    if viscosityFormulation == 'MonaghanGingold1983':
        K               = 1 if K is None else K
        use_rho_bar     = True if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1992'
    elif viscosityFormulation == 'Cleary1998':
        K               = 1 if K is None else K
        use_rho_bar     = False if use_rho_bar is None else use_rho_bar
        use_c_bar       = False if use_c_bar is None else use_c_bar
        use_h_bar       = False if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1997'
    elif viscosityFormulation == 'Monaghan1992':
        K               = 1 if K is None else K
        use_rho_bar     = True if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1992'
    elif viscosityFormulation in ['Monaghan1997', 'Dukowicz', 'Price2012', 'Price2012_98', 'Price2008', 'Wadsley2008']:
        K               = 1 if K is None else K
        use_rho_bar     = True if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1997'
    elif viscosityFormulation == 'delta':
        K = 1 if K is None else K
        use_rho_bar     = False if use_rho_bar is None else use_rho_bar
        use_c_bar       = True if use_c_bar is None else use_c_bar
        use_h_bar       = True if use_h_bar is None else use_h_bar
        viscosityTerm   = 'Monaghan1992'
    

    rho_i, rho_j, rho_bar   = getTerms(neighborhood.row, neighborhood.col, (particles_a.densities, particles_b.densities))
    c_i, c_j, c_bar         = getTerms(neighborhood.row, neighborhood.col, (particles_a.soundspeeds, particles_b.soundspeeds))
    h_i, h_j, h_bar         = getTerms(neighborhood.row, neighborhood.col, (particles_a.supports, particles_b.supports))
    alpha_i, alpha_j        = getAlphas(particles_a, particles_b, neighborhood, config)

    d   = particles_a.positions.shape[1]
    rho = rho_bar   if use_rho_bar  else (rho_j if useJ else rho_i)
    c   = c_bar     if use_c_bar    else (c_j   if useJ else c_i)
    h   = h_bar     if use_h_bar    else (h_j   if useJ else h_i)
    
    # if viscosityFormulation == 'delta':
        # rho = rho_i / solverConfig['fluid']['rho0']
        # c = solverConfig['fluid']['c_s']
    
    kernel = config['kernel']

    xi  = Kernel_xi(kernel, particles_a.positions.shape[1]) if correctXi else 1.0

    # print(f'alpha_i: {alpha_i.shape}, min: {alpha_i.min()}, max: {alpha_i.max()}')
    # print(f'alpha_j: {alpha_j.shape}, min: {alpha_j.min()}, max: {alpha_j.max()}')    
    
    C_l = 1/2 * (alpha_i + alpha_j) * C_l
    C_q = 1/2 * (alpha_i + alpha_j) * C_q
    if switchScaling == 'i':
        # print('Scaling with i')
        C_l = C_l * alpha_i
        C_q = C_q * alpha_i
    elif switchScaling == 'j':
        # print('Scaling with j')
        C_l = C_l * alpha_j
        C_q = C_q * alpha_j
    
        
    if scaleBeta:
        # print('Scaling beta')
        C_q = C_q * C_l
        
    # print(solverConfig)
    # print(f'C_l: {C_l.shape}, min: {C_l.min()}, max: {C_l.max()}')
    # print(f'C_q: {C_q.shape}, min: {C_q.min()}, max: {C_q.max()}')
        

    x_ij, r_ij  = compute_xij(particles_a, particles_b, neighborhood, domain)
    u_ij        = particles_a.velocities[neighborhood.row] - particles_b.velocities[neighborhood.col]
    ux_ij       = torch.einsum('ij,ij->i', u_ij, x_ij)

    if viscosityTerm == 'Monaghan' or viscosityTerm == 'Monaghan1992':
        mu_ij = ux_ij / (r_ij**2 + 1e-14 * h**2) * h / xi
    else: # 'Monaghan1997' also referred to as 'j' in the paper
        mu_ij = ux_ij / (r_ij + 1e-14 * h)
    
    # print(f'mu_ij: {mu_ij.shape}, min: {mu_ij.min()}, max: {mu_ij.max()}')
    
    if switch:
        mu_ij[ux_ij > 0] = 0

    # There is some confusion regarding the terms here.
    # Monaghan 2005 prescribes several potential terms for the viscosity
    # Notationally we use Pi_ab = - K / rho v_sig mu_ab
    
    if viscosityFormulation == 'MonaghanGingold1983':
    # Monaghan and Gingold 1983: The terms are given in (8.3) and (8.4)  of Monaghan 2005 and are
    # Pi_ab = -nu ( v_ab \cdot r_ab) / (r_ab^2 + epsilon^2 h_ab^2)
    # nu = alpha h_bar c_bar / rho_bar
    # Rewording this slightly we get the 'Monaghan1992' viscosity Term (with xi correction)
    # combined with using c_bar, rho_bar and h_bar. 
    # Consequently this uses 
    # v_sig = c_bar
    # K = 1
        v_sig = c_bar
    elif viscosityFormulation == 'Cleary1998':
    # Cleary 1998: The terms are given in (8.8) and (8.9) of Monaghan 2005 and are
    # mu_a = 1/8 alpha_a h_a c_a rho_a
    # Pi_ab = - 16 mu_a mu_b / (rho_a rho_b (mu_a + mu_b)) mu_ij
        f = 1/(2*(d+2)) # Based on estimations based on Monaghan 2005, not given for 1D
        mu_i = f * alpha_i * C_l * h_i * c_i * rho_i / xi
        mu_j = f * alpha_j * C_l * h_j * c_j * rho_j / xi
        # 19.8 based on Cleary and Ha 2002
        v_sig = 19.8 * mu_i * mu_j / (rho_i * rho_j * (mu_i + mu_j)) / (r_ij + 1e-14 * h)
    elif viscosityFormulation == 'Monaghan1992':
    # Monaghan 1992: The term is given in (8.10) of Monaghan 2005 and is
    # mu = h / rho ( alpha c - beta mu_ij)
    # This uses the Monaghan 1992 viscosity term with alpha = 1 and beta = 2
        v_sig = C_l * c - C_q * mu_ij
    elif viscosityFormulation == 'Monaghan1997a':
    # Monaghan 1997: The term is given in (8.11) of Monaghan 2005 and is very similar
    # to the Monaghan1992 term but uses the Monaghan1997 viscosity term. denoted as j
    # in the 1997 paper and has a strange wording in 2005 of using 1/2 instead of 1 for K
    # c_i + c_j instead of c_bar and beta = 4. Cancelling these terms out gives the normal
    # c_bar term with alpha = 1 and beta = 4! This is also eq 3.7 in Monaghan1997
        v_sig = C_l * c - C_q * mu_ij
    elif viscosityFormulation == 'Monaghan1997b':
    # Based on Monaghan 1997 eq 4.7:
        v_sig = (c_i**2 + C_q * mu_ij**2)**0.5 + (c_j**2 + C_q * mu_ij**2)**0.5 - C_q * mu_ij
    elif viscosityFormulation == 'Dukowicz':
    # The term is given in (4.8) of Monaghan 1997 and is simply the 1997a term with a 3/4 factor
        v_sig = C_l * c - 3/4 * C_q * mu_ij
    # Next are the formulations based on Price's SPMHD paper from 2012
    elif viscosityFormulation == 'Price2012_98':
    # This term is identical to Monaghan 1992, equation 98 in Price 2012
        v_sig = C_l * c - C_q * mu_ij
    elif viscosityFormulation == 'Price2012':
    # Based on equation 103
        v_sig = C_l * c - C_q / 2 * mu_ij
    elif viscosityFormulation == 'Price2008':
    # This formulation and the next are only mentioned in the Price 2012 after equation 103, no explicit equation numbers
        P_i, P_j    = particles_a.pressures[neighborhood.row], particles_b.pressures[neighborhood.col]
        rho_bar     = (rho_i + rho_j) / 2
        v_sig       = C_l * torch.sqrt(torch.abs(P_i - P_j) / (rho_bar + 1e-14 * h))
    elif viscosityFormulation == 'Wadsley2008':
        v_sig = C_l * torch.abs(mu_ij)
    else:
        v_sig = C_l * c - C_q * mu_ij
    val = K / rho * v_sig #* mu_ij
    if thermalConductivity:
        val = K / rho * v_sig #* (particles_a.internalEnergies[neighborhood.row] - particles_b.internalEnergies[neighborhood.col])
    # else:
    if viscosityTerm == 'Monaghan' or viscosityTerm == 'Monaghan1992':
        val = val * h / xi
    else:
        val = val * r_ij
    
    if switch and not thermalConductivity:
        val[ux_ij > 0] = 0
    return 1/2 * val
    

    return -K/ rho * C_l * (c * mu_ij - C_q * mu_ij**2)
