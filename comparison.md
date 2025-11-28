Feature | diffSPH | Jax-SPH | DualSPHysics | spheral | pySPH
---|---|---|---|---|---
Maximum Particle Count | 8M on 12GB | 3.3M just for neighborhood on 48GB | 160M on 80GB (small neighborhoods) | / | /
GPU Support                     | ✅ | ✅ | ✅ | ❌ | ✅
Backwards Pass                  | ✅ | ✅ | ❌ | ❌ | ❌
Inverse Problem                 | 800, 832 | 36 particles, 100 steps | ❌ | ❌| ❌
Dynamic Particle Count          | ✅ | ❌ | ✅ | ✅ | ✅
Variable Neighborhood Sizes     | ✅ | ❌ | ✅ | ✅ | ✅
Adaptive support radii          | ✅ | ❌ | ❌ | ✅ | ✅
grad-h corrections              | ✅ | ❌ | ❌ | ✅ | ✅
moving-boundaries               | ✅ | ❌ (technically maybe possible, not well documented)| ✅ | ❌ | ❌
Verlet List Support             | ✅ | ❌ | ❌ | ❌ | ❌
Inlet/Outlet                    | ✅ | ❌ | ✅ | ✅ | ✅
Optimized Backward Operations   | ✅ | ❌ | ❌ | ❌ | ❌
Compressible Dynamics           | ✅ (various schemes) | ❌ | ❌ | ✅ (various schemes) | ✅ (various schemes)
Self Gravity                    | ❌ | ❌ | ❌ | ✅ | ✅
Weakly Compressible SPH         | ✅ (delta-SPH) | ✅ (delta) | ✅ (delta) | ✅ (delta) | ✅ (delta)
Incompressible SPH              | ✅ (DFSPH) | ❌ | ❌ | ❌ | ✅ (various)
Particle Shifting               | ✅ (delta + implicit) | ❌ | ✅ (delta) | ❌ | ✅ (delta)
Boundary Handling               | ✅ (mDBC) | ✅ (Riemann Problem) | ✅ mDBC + DBC | ✅ Ghost Particles | ✅ Ghost Particles
Variable Kernels                | ✅ (all but self-adaptive) | ✅ (all but self-adaptive) | ❌ | ✅ (all but self adaptive) | ✅
Variable Time Integrator        | ✅ (all explicit integrators supported) | ❌ (only semi-implicit Euler) | ✅ (verlet + euler) | ✅ | ✅
Variable Viscosity Terms        | ✅ (various formulations) | ❌ (only adami 2012) | ❌ (Monaghan 92 + Laminar) | ✅ | ✅
Variable Viscosity Switches     | ✅ (various formulations) | ❌ (no compresisble) |  ❌ (no compresisble) | ✅ | ✅

python main.py config=cases/tgv.yaml solver.name=SPH solver.tvf=0.0 ->
2500 ptcls, no shifting, simple integration, 11000 timestepos, 9 seconds -> 3M/sec

256K ptcls, no shifting, 2252 timesteps, 56 seconds -> 10M/sec (4.2GB) (SPH solver NOT delta) 64s with delta



256K -> 2252 timesteps, 3:40m -> 3.9 (2.7GB) -> 3.43x
2500 -> 16ms -> 156k/sec