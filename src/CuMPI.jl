__precompile__(false)

module CuMPI

using MPI, CuArrays
import MPI: Op, Comm, mpitype
import MPI: MPI_ALLGATHER, MPI_GATHERV, MPI_ALLGATHERV, MPI_ALLTOALLV
import MPI: Send, Isend, Recv!, Irecv!, Bcast!, Reduce, 
            Gather, Scatter, Scatterv, Scan, Exscan
import MPI: Allreduce!, allreduce, Allgather, Gatherv, Allgatherv, Alltoall, Alltoallv
import MPI: user_op

include("device.jl")

for fun in [:Send, :Isend, :Recv!, :Irecv!, :Bcast!, :Reduce,
            :Gather, :Scatter, :Scatterv, :Scan, :Exscan]
    @eval function $fun(buf::CuArray{T}, args...) where T
        $fun(Ptr{T}(buf.buf.ptr), args...)
    end
end

for fun in [:Send, :Isend, :Rev!, :Irecv!]
    @eval function $fun(buf::CuArray{T}, dest::Integer, tag::Integer, comm::Comm) where T
        $fun(buf, length(buf), dest, tag, comm)
    end
end

for fun in [:Bcast!, :Gather]
    @eval function $fun(buffer::CuArray{T}, root::Integer, comm::Comm) where T
        $fun(buffer, length(buffer), root, comm)
    end
end

function Reduce(sendbuf::CuArray{T}, op::Union{Op,Function}, root::Integer, comm::Comm) where T
    Reduce(sendbuf, length(sendbuf), op, root, comm)
end

function Allreduce!(sendbuf::CuArray{T}, recvbuf::CuArray{T}, args...) where T
    Allreduce!(Ptr{T}(sendbuf.buf.ptr), Ptr{T}(recvbuf.buf.ptr), args...)
end

function Allreduce!(sendbuf::CuArray{T}, recvbuf::CuArray{T},
                   op::Union{Op,Function}, comm::Comm) where T
    Allreduce!(sendbuf, recvbuf, length(recvbuf), op, comm)
end

# allocate receive buffer automatically
function allreduce(sendbuf::CuArray{T}, op::Union{Op,Function}, comm::Comm) where T
    recvbuf = similar(sendbuf)
    Allreduce!(sendbuf, recvbuf, length(recvbuf), op, comm)
end

function Allgather{T}(sendbuf::CuArray{T}, count::Integer, comm::Comm)
    recvbuf = CuArray{T}(Comm_size(comm) * count)
    ccall(MPI_ALLGATHER, Void,
          (Ptr{T}, Ptr{Cint}, Ptr{Cint}, Ptr{T}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
          sendbuf.buf.ptr, &count, &mpitype(T), recvbuf.buf.ptr, &count, &mpitype(T), &comm.val, &0)
    recvbuf
end

function Gatherv{T}(sendbuf::CuArray{T}, counts::Vector{Cint}, root::Integer, comm::Comm)
    isroot = Comm_rank(comm) == root
    displs = cumsum(counts) - counts
    sendcnt = counts[Comm_rank(comm) + 1]
    recvbuf = CuArray{T}(isroot ? sum(counts) : 0)
    ccall(MPI_GATHERV, Void,
          (Ptr{T}, Ptr{Cint}, Ptr{Cint}, Ptr{T}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
          sendbuf, &sendcnt, &mpitype(T), recvbuf, counts, displs, &mpitype(T), &root, &comm.val, &0)
    isroot ? recvbuf : nothing
end

function Allgatherv(sendbuf::CuArray{T}, counts::Vector{Cint}, comm::Comm) where T
    displs = cumsum(counts) - counts
    sendcnt = counts[Comm_rank(comm) + 1]
    recvbuf = CuArray{T}(sum(counts))
    ccall(MPI_ALLGATHERV, Void,
          (Ptr{T}, Ptr{Cint}, Ptr{Cint}, Ptr{T}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
          sendbuf.buf.ptr, &sendcnt, &mpitype(T), recvbuf.buf.ptr, counts, displs, &mpitype(T), &comm.val, &0)
    recvbuf
end

function Alltoall(sendbuf::CuArray{T}, count::Integer, comm::Comm) where T
    recvbuf = CuArray{T}(Comm_size(comm)*count)
    ccall(MPI_ALLTOALL, Void,
          (Ptr{T}, Ptr{Cint}, Ptr{Cint}, Ptr{T}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
          sendbuf.buf.ptr, &count, &mpitype(T), recvbuf.buf.ptr, &count, &mpitype(T), &comm.val, &0)
    recvbuf
end

function Alltoallv(sendbuf::CuArray{T}, scounts::Vector{Cint}, rcounts::Vector{Cint}, comm::Comm) where T
    recvbuf = CuArray{T}(sum(rcounts))
    sdispls = cumsum(scounts) - scounts
    rdispls = cumsum(rcounts) - rcounts
    ccall(MPI_ALLTOALLV, Void,
          (Ptr{T}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{T}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}, Ptr{Cint}),
          sendbuf.buf.ptr, scounts, sdispls, &mpitype(T), recvbuf.buf.ptr, rcounts, rdispls, &mpitype(T), &comm.val, &0)
    recvbuf
end

Allreduce!(sendbuf::CuArray{T}, recvbuf::CuArray{T},
           count::Integer, opfunc::Function, comm::Comm) where {T} =
    Allreduce!(sendbuf, recvbuf, count, user_op(opfunc), comm)

Reduce(sendbuf::CuArray{T}, count::Integer,
       opfunc::Function, root::Integer, comm::Comm) where {T} =
    Reduce(sendbuf, count, user_op(opfunc), root, comm)

end