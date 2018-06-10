using MPI, CuArrays, CuMPI

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

dev = CUDAdrv.CuDevice(rank)
ctx = CUDAdrv.CuContext(dev)

send_arr = cu(rand(Float32, 100))
recv_arr = cu(zeros(Float32, 100))
MPI.Allreduce!(send_arr, recv_arr, MPI.SUM, comm)

CUDAdrv.destroy!(ctx)

MPI.Finalize()