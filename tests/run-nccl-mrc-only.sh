#!/bin/bash

print_usage() {
    echo "Usage: $0 <NUM_NODES> <BENCH>"
    echo "Example: $0 2 sendrecv"
    echo "Environment variables:"
    echo "  PPN: Processes per node (default: 4)."
    echo "  HOSTFILE: Path to the MPI hostfile (default: ./hostfile)."
    echo "  MPI_MODULE: MPI module to load via 'module load' (default: mpi/hpcx). Set to empty to skip."
    echo "  NCCL_MRC_PLUGIN_PATH: Directory containing libnccl-net-mrc.so (default: /opt/microsoft/mrc/mrc-nccl-plugin/)."
    echo "  NCCL_TESTS_DIR: Directory containing built nccl-tests binaries (default: /opt/nccl-tests/build)."
    echo "  MPI_NETDEV: Frontend network device for MPI/UCX/NCCL bootstrap (default: enP22p1s0f1)."
    echo "  CUDA_VISIBLE_DEVICES: GPU ordering (default: 0,1,2,3)."
    echo "  NCCL_IB_HCA: IB HCA list (default: =mlx5_1,mlx5_0,mlx5_3,mlx5_2)."
    echo "  GID_INDEX: RoCE GID index (default: 3)."
    echo "  OUTER_ITER: Number of outer iterations (default: 1)."
    echo "  INNER_ITER: Number of inner iterations (default: 50)."
    echo "  WARMUP_ITER: Number of warmup iterations (default: 50)."
    echo "  BEGIN_MSG_SIZE: Starting message size (default: 4K)."
    echo "  END_MSG_SIZE: Ending message size (default: 4G). In long-running mode, the size held indefinitely."
    echo "  LONG_RUN: 0 or 1 (default: 0). When 1, run indefinitely at END_MSG_SIZE."
    echo "  DEBUG: 0 or 1 (default: 0). When 1, enables NCCL_DEBUG=TRACE with NCCL_DEBUG_SUBSYS=ALL."
    echo "  ADDNL_MPI_ARGS: Additional arguments to pass to mpirun (default: empty)."
    echo "  ADDNL_NCCL_ENV: Additional env vars to pass to NCCL, format '-x VAR=value' (default: empty)."
}

set -u

if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    print_usage
    exit 0
fi

if [ "$#" -ne 2 ]; then
    print_usage
    exit 1
fi

NUM_NODES=${1:?"Error: NUM_NODES argument is required"}
BENCH=${2:?"Error: BENCH argument is required"}

PPN=${PPN:-4}
HOSTFILE=${HOSTFILE:-"./hostfile"}
MPI_MODULE=${MPI_MODULE:-"mpi/hpcx"}
NCCL_MRC_PLUGIN_PATH=${NCCL_MRC_PLUGIN_PATH:-/opt/microsoft/mrc/mrc-nccl-plugin/}
NCCL_TESTS_DIR=${NCCL_TESTS_DIR:-/opt/nccl-tests/build}
MPI_NETDEV=${MPI_NETDEV:-enP22p1s0f1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
NCCL_IB_HCA=${NCCL_IB_HCA:-"=mlx5_1,mlx5_0,mlx5_3,mlx5_2"}
GID_INDEX=${GID_INDEX:-3}
OUTER_ITER=${OUTER_ITER:-1}
INNER_ITER=${INNER_ITER:-50}
WARMUP_ITER=${WARMUP_ITER:-50}
BEGIN_MSG_SIZE=${BEGIN_MSG_SIZE:-4K}
END_MSG_SIZE=${END_MSG_SIZE:-4G}
LONG_RUN=${LONG_RUN:-0}
DEBUG=${DEBUG:-0}
ADDNL_MPI_ARGS=${ADDNL_MPI_ARGS:-""}
ADDNL_NCCL_ENV=${ADDNL_NCCL_ENV:-""}

source /etc/profile.d/modules.sh 2>/dev/null || true
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

echo "Running NCCL test on $NUM_NODES nodes with $PPN GPUs per node"

if [ -n "$MPI_MODULE" ]; then
    module load "$MPI_MODULE"
fi

export NCCL_MRC_PLUGIN_PATH
export LD_LIBRARY_PATH=$NCCL_MRC_PLUGIN_PATH:$LD_LIBRARY_PATH

COLL=$(realpath "$NCCL_TESTS_DIR/${BENCH}_perf")
if [ ! -x "$COLL" ]; then
    echo "Error: NCCL test binary '$COLL' not found or not executable."
    exit 1
fi

# Set COLL_ARGS based on LONG_RUN mode
if [ "$LONG_RUN" -eq 1 ]; then
    echo "Long running mode enabled: running at $END_MSG_SIZE indefinitely"
    COLL_ARGS="-w $WARMUP_ITER -n $INNER_ITER -b $END_MSG_SIZE -e $END_MSG_SIZE -g1 -c 1 -R 1 -N 0"
else
    COLL_ARGS="-b $BEGIN_MSG_SIZE -f2 -e $END_MSG_SIZE -g1 -c 1 -R 1 -w $WARMUP_ITER -n $INNER_ITER -N $OUTER_ITER"
fi

NCCL_ENV="
  --allow-run-as-root \
  -mca plm_rsh_no_tree_spawn 1 -mca plm_rsh_num_concurrent 8192 \
  --map-by ppr:2:numa --bind-to none \
  --hostfile $HOSTFILE \
  -x UCX_TLS=tcp \
  -x LD_LIBRARY_PATH \
  -mca coll_hcoll_enable 0 \
  --mca btl tcp,vader,self \
  --mca pml ob1 \
  --mca btl_tcp_if_include $MPI_NETDEV \
  -x UCX_NET_DEVICES=$MPI_NETDEV \
  -x NCCL_SOCKET_IFNAME=$MPI_NETDEV \
  -x NCCL_NET_PLUGIN=libnccl-net-mrc.so \
  -x NCCL_TUNER_PLUGIN=none \
  -x NCCL_IB_DISABLE=0 \
  -x NCCL_SHM_DISABLE=1 \
  -x NCCL_P2P_DISABLE=1 \
  -x NCCL_MNNVL_ENABLE=0 \
  -x CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  -x NCCL_IB_HCA=$NCCL_IB_HCA \
  -x NCCL_IB_ECE_ENABLE=0 \
  -x NCCL_IB_GID_INDEX=$GID_INDEX \
  -x NCCL_NVLS_ENABLE=0 \
  -x NCCL_GDR_FLUSH_DISABLE=1 \
  -x NCCL_GDRCOPY_ENABLE=1 \
  -x NCCL_GDRCOPY_SYNC_ENABLE=1 \
  -x NCCL_IB_QPS_PER_CONNECTION=2 \
  -x NCCL_IB_SPLIT_DATA_ON_QPS=1 \
  -x NCCL_IB_TC=$((1 << 2)) -x NCCL_IB_FIFO_TC=$((3 << 2)) \
  -x NV_MRC_POST_SEND_PREFER_BF=1 \
  $ADDNL_MPI_ARGS \
  $ADDNL_NCCL_ENV"

if [ "$PPN" -eq 4 ]; then
       NCCL_ENV+=" -x NCCL_TESTS_SPLIT_MASK=0x3"
elif [ "$PPN" -eq 2 ]; then
       NCCL_ENV+=" -x NCCL_TESTS_SPLIT_MASK=0x1"
elif [ "$PPN" -eq 1 ]; then
       NCCL_ENV+=" -x NCCL_TESTS_SPLIT_MASK=0x0"
else
       echo "NCCL_TESTS_SPLIT_MASK cannot be set for PPN = $PPN. Exiting."
       exit 1
fi

if [ "$DEBUG" -eq 0 ]; then
        NCCL_ENV+=" -x NCCL_DEBUG=WARN"
else
        NCCL_ENV+=" -x NCCL_DEBUG=TRACE -x NCCL_DEBUG_SUBSYS=ALL"
fi

set -x
date
echo "mpirun -np $((NUM_NODES*PPN)) $NCCL_ENV $COLL $COLL_ARGS"
mpirun -np $((NUM_NODES*PPN)) \
        $NCCL_ENV \
        $COLL $COLL_ARGS
set +x
