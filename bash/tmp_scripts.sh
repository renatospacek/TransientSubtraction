#!/bin/bash

threadNo = $1
eta = $2
M = $3

nohup /libre/spacekr/julia-1.10.4/bin/julia --threads=$threadNo transient_LJ_mobility.jl $eta $M > mobility_${eta}_${M}.out &







#!/bin/bash

threadNo = $1
eta = $2
M = $3

nohup /libre/spacekr/julia-1.10.4/bin/julia --threads=$threadNo transient_LJ_shear.jl $eta $M > shear_${eta}_${M}.out &





