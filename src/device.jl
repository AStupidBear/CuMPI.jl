using CUDAdrv

function device!(n)
    ngpu = length(CUDAdrv.devices())
    CUDAdrv.CuContext(CUDAdrv.CuDevice(mod1(n, ngpu) - 1))
end

function Init()
    !MPI.Initialized() && MPI.Init()
    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    device!(rank)
end