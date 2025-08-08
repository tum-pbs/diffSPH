nx | obstacle | domain boundary | $\max\{\mathbf{v}\}$ | $\Delta t$ | $c_s$ CFL | 
---|---|---|---|---|---
32 | <span style="color:red;">&#10007;</span>   | <span style="color:red;">&#10007;</span>   | 
32 | <span style="color:red;">&#10007;</span>   | <span style="color:green;">&#10003;</span> | 
32 | <span style="color:green;">&#10003;</span> | <span style="color:red;">&#10007;</span>   | 
32 | <span style="color:green;">&#10003;</span> | <span style="color:green;">&#10003;</span> | 
64 | <span style="color:red;">&#10007;</span>   | <span style="color:red;">&#10007;</span>   | 
64 | <span style="color:red;">&#10007;</span>   | <span style="color:green;">&#10003;</span> | 
64 | <span style="color:green;">&#10003;</span> | <span style="color:red;">&#10007;</span>   | 
64 | <span style="color:green;">&#10003;</span> | <span style="color:green;">&#10003;</span> | 
128| <span style="color:red;">&#10007;</span>   | <span style="color:red;">&#10007;</span>   | 
128| <span style="color:red;">&#10007;</span>   | <span style="color:green;">&#10003;</span> | 
128| <span style="color:green;">&#10003;</span> | <span style="color:red;">&#10007;</span>   | 
128| <span style="color:green;">&#10003;</span> | <span style="color:green;">&#10003;</span> | 
256| <span style="color:red;">&#10007;</span>   | <span style="color:red;">&#10007;</span>   | 
256| <span style="color:red;">&#10007;</span>   | <span style="color:green;">&#10003;</span> | 
256| <span style="color:green;">&#10003;</span> | <span style="color:red;">&#10007;</span>   | 
256| <span style="color:green;">&#10003;</span> | <span style="color:green;">&#10003;</span> | 



v_max = 1 -> c_s = 20 (very conservative)
nx = 256 -> dt = 2.5e-4 (c_s CFL = 20.27)
nx = 128 -> dt = 5e-4   (c_s CFL = 20.27)
nx =  64 -> dt = 1e-3   (c_s CFL = 20.27)
nx =  32 -> dt = 2e-3   (c_s CFL = 20.27)


c_s = 30 ->
nx = 256 -> dt = 1.5e-4 (c_s CFL = 20.27)
nx = 128 -> dt = 5e-4   (c_s CFL = 20.27)
nx =  64 -> dt = 1e-3   (c_s CFL = 20.27)
nx =  32 -> dt = 2e-3   (c_s CFL = 20.27)


python wcflows.py --normalizeEnergy --TGV 2 --cs 30 --domainBoundary --obstacle --verbose --nx 256 --dt 1e-4 --velocityNoise --seed 24395732 