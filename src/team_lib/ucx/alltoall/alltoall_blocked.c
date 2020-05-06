#include "config.h"
#include "xccl_ucx_lib.h"
#include "alltoall.h"
#include "xccl_ucx_sendrecv.h"
#include "utils/mem_component.h"
#include <stdlib.h>
#include <string.h>

/* Blocked AllToAll Algorithm */
/* ========================== */
/* The AllToAll matrix consists of rows and columns. The number of rows and */
/* the number columns is equal. */
/* Each row contains the data of a single rank (that its rank ID is the */
/* same as the row index) that should be sent during the AllToAll collective */
/* operation to all other ranks (each rank should get the correct) segment */
/* of the data. Therefore the rows can be called SRs (Source Ranks). */
/* Each column of the AllToAll matrix contains the data of a single rank */
/* (that its rank ID is the same as the column index), that the rank will */
/* have in the end of the AllToAll collective operation. Therefore the column */
/* can be called DRs (Destination Ranks). */

/* The Blocked AllToAll algorithm process the AllToAll collective operation */
/* by dividing the AllToAll matrix into equal blocks, by getting as an */
/* input for the algorithm the amount of SRs and DRs that should be in every */
/* block. In case the AllToAll matrix cannot be divided into block of equal */
/* size, because the number of DRs or SRs cannot be divided without a */
/* reminder, the last DRs and SRs will be divided into blocks that are */
/* smaller than requested. The matrix that is formed after dividing the */
/* AllToAll matrix into blocks is called "Blocked AllToAll" matrix. */

/* Each block in the Blocked AllToAll matrix has a rank that is called */
/* Aggregation Send Rank (ASR) and this rank is responsible for collecting */
/* (aggregating) the data from all the SRs (the rows) of the block and then */
/* spread the data between all the DRs (the columns) of the block in a way */
/* that each DR get the data it should get from the SRs in the AllToAll */
/* collective operation */
/* Notice that when the ASR collects the data from the SRs it gets the data */
/* in "row-view" of the blocks row, but when the ASR wants to spread the */
/* data to the DRs, it spreads it as in "column-view". In order to optimize */
/* the process of spreading the data to the DRs, the ASR transposes the data */
/* it got from the SRs and only then distributes it to the DRs. Note that */
/* after the transpose operation, the data stored in the memory in a */
/* convenient form to distribute. It can be noticed that all the blocks can */
/* work in parallel.*/

/* Note: Each rank can calculate all the information regarding all the blocks,
/* and their parameters (like amount of SRs, amount of SRs, ASR etc.) */

/* Assignment of ASR to Block */
/* ===========================*/
/* The algorithm tries to assign different ASRs for all the blocks, but in */
/* case there are more blocks than ranks, then a single rank can serve as */
/* an ASR of several blocks. */
/* The assignment of ASR to block is done as follows: */
/* Each block within a single row of blocks in the Blocked AllToAll matrix */
/* will be assigned with an ASR that is chosen from the SRs of the block. */
/* This way, all the blocks within a single row in the Blocked AllToAll */
/* matrix share the same "pool" of possible ASRs. The first block (column */
/* index 0) of every row will get the first SR (smallest rank) and the */
/* following block get the next SR. In case there are more blocks in a row */
/* than SRs, than the assignment will begin again from the first SR. */
/* Note: this distribution guarantees that all the SRs that serve as */
/* ASRs will be ASRs of blocks that have the same amount of SRs. */

/* macros */
#define N_INSTANCES_CEIL(total_len, segment_len) ((total_len)+(segment_len)-1)/(segment_len)
#define N_INSTANCES_FLOOR(total_len, segment_len) (total_len)/(segment_len)
#define WRAPAROUND_DOWN(value, max_val) (value)%(max_val)
#define WRAPAROUND_UP(value, max_val) (value+max_val)%(max_val)

/* Data structure to hold the communication parameters for the Blocked */
/* AllToAll algorithm. */
struct comm_params {
    /* The buffer that contains the data that the rank should send during */
    /* the AllToAll collective operation */
    void *send_buf;

    /* The buffer that should contain the data that the rank received */
    /* during the AllToAll collective operation */
    void *recv_buf;

    /* Holds the maximal number of outstanding send requests that a process */
    /* manages in all times */
    int max_send;

    /* Holds the maximal number of outstanding receive requests that a */
    /* process manages in all times */
    int max_recv;

    /* List of outstanding send requests that a process manages */
    xccl_ucx_request_t **send_requests;

    /* List of outstanding receive requests that a process manages */
    xccl_ucx_request_t **recv_requests;

    /* How many ranks are part of the AllToAll collective operation */
    int job_size;

    /* Rank ID within the MPI_COMM_WORLD communicator */
    int job_rank;

    /* Holds the amount of rows in the Blocked AllToAll matrix */
    int n_src_blocks;

    /* Holds the amount of columns in the Blocked AllToAll matrix */
    int n_dst_blocks;

    /* Holds the row index of the block that contains this rank as SR */
    int src_block_id;

    /* Holds the column index of the block that contains this rank as DR */
    int dst_block_id;

    /* The amount of AllToAll matrix rows within a single block in the */
    /* Blocked AllToAll matrix which is the matrix that is formed after */
    /* dividing the AllToAll matrix into blocks. This is the required size */
    /* that the algorithm gets as an input. */
    /* This also can be thought as the number of SRs (Source Ranks) that will
    /* send data to the ASR (Aggregation Send Rank) of the block. */
    int src_block_size;

    /* The amount of AllToAll matrix columns within a single block in the */
    /* Blocked AllToAll matrix which is the matrix that is formed after */
    /* dividing the AllToAll matrix into blocks. This is the required size */
    /* that the algorithm gets as an input. */
    /* This also can be thought as the number of DRs (Destination Ranks) */
    /* that the will receive the transposed data to their receive buffer */
    /* from the ASR */
    int dst_block_size;

    /* Holds the number of AllToAll matrix rows in the last row of the */
    /* Blocked AllToAll matrix. In case the rows of AllToAll matrix cannot */
    /* be divided to the number of blocks that are required, the last block */
    /* will contain less rows. */
    int last_src_block_size;

    /* Holds the number of AllToAll matrix columns in the last column of the */
    /* Blocked AllToAll matrix. In case the columns of AllToAll matrix */
    /* cannot be divided to the number of blocks that are required, the last */
    /* block will contain less columns. */
    int last_dst_block_size;

    /* Holds the number of rows in the block that contains this rank as SR */
    int my_src_block_size;

    /* Holds the number of columns in the block that contains this rank as */
    /* DR */
    int my_dst_block_size;

    /* The length (in bytes) of the buffer that each rank that serves as an */
    /* ASR (Send Aggregator Rank) will use for the aggregation phase. */
    /* For each time that the rank serves as an ASR, it will have different */
    /* buffer with this size */
    size_t asr_buf_len;

    /* Number of times that a rank has to serve as an ASR of a block */
    int asr_cycles;

    /* Information that the ranks saves in case it cannot proceed the */
    /* algorithm and that will be restored after the rank will continue */

    /* Current phase in which the algorithm is located for the current rank */
    int phase;

    /* The amount outstanding send request that currently the rank manages */
    int outstanding_sends;

    /* The amount outstanding receive request that currently the rank */
    /* manages */
    int outstanding_recvs;

    /* Holds how many times till now the rank has served as an ASR. */
    /* For each time the rank serves as a ASR, it uses a different buffer */
    /* so all the blocks under this rank responsibilty as an ASR could be */
    /* executed in parallel */
    int asr_cycle;

    int i_dst_block;
    int i;
    int count;

};
typedef struct comm_params comm_params_t;

enum {
    PHASE_00,
    PHASE_01,
    PHASE_02,
    PHASE_03,
    PHASE_04,
    PHASE_05,
    PHASE_06,
};

/* Initialize the communication parameters */
static int comm_params_init(xccl_ucx_collreq_t *req, comm_params_t **job_config,
                            int src_block_size, int dst_block_size)
{
    size_t data_size      = req->args.buffer_info.len;
    int    group_rank     = req->team->params.oob.rank;
    int    group_size     = req->team->params.oob.size;
    ptrdiff_t sbuf        = (ptrdiff_t)req->args.buffer_info.src_buffer;
    ptrdiff_t rbuf        = (ptrdiff_t)req->args.buffer_info.dst_buffer;
    xccl_ucx_team_t *team = ucs_derived_of(req->team, xccl_ucx_team_t);
    comm_params_t job_cfg;
    xccl_ucx_request_t **requests;

    memset(&job_cfg, 0, sizeof(job_cfg));
    job_cfg.phase = PHASE_00;

    job_cfg.job_size = group_size;
    job_cfg.job_rank = group_rank;

    job_cfg.send_buf = req->args.buffer_info.src_buffer;
    job_cfg.recv_buf = req->args.buffer_info.dst_buffer;

    job_cfg.src_block_size = src_block_size;
    job_cfg.dst_block_size = dst_block_size;

    job_cfg.max_send = 8; //TODO config
    job_cfg.max_recv = 8; //TODO config

    job_cfg.n_src_blocks = N_INSTANCES_CEIL(job_cfg.job_size, job_cfg.src_block_size);
    job_cfg.n_dst_blocks = N_INSTANCES_CEIL(job_cfg.job_size, job_cfg.dst_block_size);

    job_cfg.src_block_id = job_cfg.job_rank / job_cfg.src_block_size;
    job_cfg.dst_block_id = job_cfg.job_rank / job_cfg.dst_block_size;

    job_cfg.last_src_block_size = (job_cfg.job_size % job_cfg.src_block_size != 0) ?
        job_cfg.job_size % job_cfg.src_block_size : job_cfg.src_block_size;
    job_cfg.last_dst_block_size = (job_cfg.job_size % job_cfg.dst_block_size != 0) ?
        job_cfg.job_size % job_cfg.dst_block_size : job_cfg.dst_block_size;

    job_cfg.my_src_block_size = (job_cfg.src_block_id != job_cfg.n_src_blocks - 1) ?
        job_cfg.src_block_size : job_cfg.last_src_block_size;
    job_cfg.my_dst_block_size = (job_cfg.dst_block_id != job_cfg.n_dst_blocks - 1) ?
        job_cfg.dst_block_size : job_cfg.last_dst_block_size;

    /* Allocate UCX requests */
    requests = calloc((job_cfg.max_recv + job_cfg.max_send), sizeof(*requests));
    if (!requests) {
        goto err;
    }
    /* Map recv and send requests to total requests that were allocated */
    /* recv requests starting from 0, send requests follow */
    job_cfg.recv_requests = requests;
    job_cfg.send_requests = requests + job_cfg.max_recv;

    /* The the amount of blocks that will have this rank as theirs ASR */
    job_cfg.asr_cycles = N_INSTANCES_CEIL(job_cfg.n_dst_blocks, job_cfg.my_src_block_size);

    job_cfg.asr_buf_len = data_size * job_cfg.dst_block_size * job_cfg.src_block_size;

    /* Each rank must allocated enough memory to hold the context */
    /* information of the AllToAll algorithm during the run. The context */
    /* must hold the following information: */
    /*   1. The communication parameters (comm_params_t). */
    /*   2. "ASR Memory": Memory that will be used for all the communication */
    /*      the rank will do as an ASR. Notice that a rank that serves as an */
    /*      ASR needs enough memory to store the data it receives from other */
    /*      ranks and because the rank has to transpose the data (which does */
    /*      not occur in place) it has to allocate another buffer for it,  */
    /*      thus allocate twice the size that was initially needed. */

    int asr_memory = 2*job_cfg.asr_cycles*job_cfg.asr_buf_len;

    int to_alloc = asr_memory + sizeof(comm_params_t);
    *job_config = (comm_params_t*)malloc(to_alloc);
    if (!(*job_config)) {
        goto clean_reqs;
    }
    memcpy((void*)(*job_config), (void*)&job_cfg, sizeof(comm_params_t));
    return XCCL_OK;
clean_reqs:
    free(requests);
err:
    return XCCL_ERR_NO_MEMORY;
}

/* For details please refer to the code during init stage when buffers */
/* are allocated. */
#define ASR_RECV_BUF(_i) (void*)(((ptrdiff_t)job_config) + sizeof(comm_params_t) + job_config->asr_buf_len*(_i))
#define ASR_SEND_BUF(_i) (void*)(((ptrdiff_t)job_config) + sizeof(comm_params_t) + job_config->asr_buf_len*(_i + job_config->asr_cycles))

#define SAVE_STATE(_phase) do {                                                \
    job_config->i_dst_block       = i_dst_block;                               \
    job_config->i                 = i;                                         \
    job_config->phase             = _phase;                                    \
    job_config->asr_cycle         = asr_cycle;                                 \
    job_config->count             = count;                                     \
    job_config->outstanding_sends = outstanding_sends;                         \
    job_config->outstanding_recvs = outstanding_recvs;                         \
} while(0)

#define RESTORE_STATE() do {                                                   \
    i_dst_block       = job_config->i_dst_block;                               \
    asr_cycle         = job_config->asr_cycle;                                 \
    count             = job_config->count;                                     \
    i                 = job_config->i;                                         \
    outstanding_sends = job_config->outstanding_sends;                         \
    outstanding_recvs = job_config->outstanding_recvs;                         \
} while(0)

#define CHECK_PHASE(_p) case _p: goto _p; break;

#define GOTO_PHASE(_phase) do {                                                \
    switch (_phase) {                                                          \
        CHECK_PHASE(PHASE_00);                                                 \
        CHECK_PHASE(PHASE_01);                                                 \
        CHECK_PHASE(PHASE_02);                                                 \
        CHECK_PHASE(PHASE_03);                                                 \
        CHECK_PHASE(PHASE_04);                                                 \
        CHECK_PHASE(PHASE_05);                                                 \
        CHECK_PHASE(PHASE_06);                                                 \
    };                                                                         \
} while(0)

/* Block algorithm with multiple ASRs */
static int alltoall_block_multi(xccl_ucx_collreq_t *req, comm_params_t *job_config)
{
    int i, j;
    int rc;
    int outstanding_recvs, outstanding_sends;
    int src, dst;
    void *buf;
    int i_dst_block;
    int asr_rank;
    int i_am_asr;
    int asr_cycle;          /* Holds the current ASR cycle for the ASR */
    int n_ranks;            /* Number of ranks an ASR is responsible for */
    int count;              /* Number of bytes a rank has to process */
    int n_sends;            /* Number of sends an ASR has to send to DRs */

    size_t len;
    size_t in_offset, out_offset;
    xccl_ucx_request_t **request;
    size_t data_size      = req->args.buffer_info.len;
    int    group_rank     = req->team->params.oob.rank;
    int    group_size     = req->team->params.oob.size;
    ptrdiff_t sbuf        = (ptrdiff_t)req->args.buffer_info.src_buffer;
    ptrdiff_t rbuf        = (ptrdiff_t)req->args.buffer_info.dst_buffer;
    xccl_ucx_team_t *team = ucs_derived_of(req->team, xccl_ucx_team_t);
    /* Tag to be used for p2p for this collective */
    int tag = req->tag;

    RESTORE_STATE();
    GOTO_PHASE(job_config->phase);

  PHASE_00:
    /* 1. Send Aggregation Phase */
    /* During this phase, each rank serves as an ASR for the local block. */
    /* Loop through the destination direction */

    outstanding_sends = 0;
    outstanding_recvs = 0;

    /* Going through all the ASRs */
    for (i_dst_block = 0; i_dst_block < job_config->n_dst_blocks; i_dst_block++) {
      PHASE_01:
        /* The i-th cycle for the ASR */
        asr_cycle = N_INSTANCES_FLOOR(i_dst_block, job_config->my_src_block_size);

        /* Calculate the ASR of the block with row index */
        /* job_config->src_block_id and column index i_dst_block */
        asr_rank = (job_config->src_block_id * job_config->src_block_size) + (i_dst_block % job_config->my_src_block_size);

        i_am_asr = (job_config->job_rank == asr_rank) ? 1 : 0;

        /* Number DRs in the current block */
        n_ranks = (i_dst_block != job_config->n_dst_blocks - 1) ? job_config->dst_block_size : job_config->last_dst_block_size;

        /* real data size to be send and receive */
        count = data_size * n_ranks;

        /* SR sends data to the ASR */
        buf = job_config->send_buf + data_size * job_config->dst_block_size * i_dst_block;
        dst = asr_rank;
        request = xccl_ucx_get_free_req_nb(team, job_config->send_requests,
                                           &outstanding_sends, job_config->max_send);
        if (!request) {
            SAVE_STATE(PHASE_01);
            return XCCL_OK;
        }
        xccl_ucx_send_nb(buf, count, dst, team, tag, request);

        /* aggregator receives */
        if (i_am_asr) {
            for (i = 0; i < job_config->my_src_block_size; i++) {
              PHASE_02:
                /* for the i-th cycle, use the i-th buffer */
                buf = ASR_RECV_BUF(asr_cycle) + data_size * job_config->dst_block_size * i;
                src = job_config->src_block_id * job_config->src_block_size + i;
                request = xccl_ucx_get_free_req_nb(team, job_config->recv_requests,
                                                   &outstanding_recvs, job_config->max_recv);
                if (!request) {
                    SAVE_STATE(PHASE_02);
                    return XCCL_OK;
                }
                xccl_ucx_recv_nb(buf, count, src, team, tag, request);
            }
        }
    }

  PHASE_03:
    /* Check that all the send/recv operation were finished before */
    /* continuing to the next phase of the operation. */
    if (XCCL_INPROGRESS == xccl_ucx_testall(team, job_config->send_requests,
                                            job_config->max_send) ||
        XCCL_INPROGRESS == xccl_ucx_testall(team, job_config->recv_requests,
                                            job_config->max_recv)) {
        SAVE_STATE(PHASE_03);
        return XCCL_OK;
    }

    /* 2. Transpose The Data In the ASR Phase */
    /* The purpose of the phase is to make the send buffers contiguous */
    /* during the distribution phase */
    /* TODO: Potential optimization with an O(n) algorithm (in-shuffle problem) */
    for (asr_cycle = 0; asr_cycle < job_config->asr_cycles; asr_cycle++) {
        /* TODO: need to cut down cost on unnecessary memcpy() due to empty cycles */
        for (i = 0; i < job_config->my_src_block_size; i++) {
            for (j = 0; j < job_config->dst_block_size; j++) {
                in_offset = data_size * (i * job_config->dst_block_size + j);
                out_offset = data_size * (j * job_config->my_src_block_size + i);
                memcpy(ASR_SEND_BUF(asr_cycle) + out_offset, ASR_RECV_BUF(asr_cycle) + in_offset, data_size);
            }
        }
    }

    /* 3. Distribution Phase */

    /* Reset send and receive slots list */
    outstanding_sends = 0;
    outstanding_recvs = 0;

    /* Going through all the ASRs */
    for (i_dst_block = 0; i_dst_block < job_config->n_dst_blocks; i_dst_block++) {
        /* The i-th cycle for the ASR */
        asr_cycle = N_INSTANCES_FLOOR(i_dst_block, job_config->my_src_block_size);

        /* Calculate the ASR of the block with row index */
        /* job_config->src_block_id and column index i_dst_block */
        asr_rank = job_config->src_block_id * job_config->src_block_size + (i_dst_block % job_config->my_src_block_size);

        i_am_asr = (job_config->job_rank == asr_rank) ? 1 : 0;

        // ASR sends data to DR directly
        if (i_am_asr) {
            i = 0;
          PHASE_04:
            n_sends = (i_dst_block != job_config->n_dst_blocks - 1) ? job_config->dst_block_size : job_config->last_dst_block_size;

            /* The number of elements that the ASR will send to each DR is */
            /* the total amount of elements it got from all the SRs. */
            /* Multiplying it by the number of bytes within each element */
            /* results in the size in bytes of each transaction. */
            count = job_config->my_src_block_size * data_size;

            for (; i < n_sends; i++) {
                /* for the i-th cycle, use the i-th buffer */
                buf = ASR_SEND_BUF(asr_cycle) + data_size * job_config->my_src_block_size * i;
                dst = (asr_cycle * job_config->my_src_block_size + job_config->job_rank % job_config->src_block_size) * job_config->dst_block_size + i;
                request = xccl_ucx_get_free_req_nb(team, job_config->send_requests,
                                                   &outstanding_sends, job_config->max_send);
                if (!request) {
                    SAVE_STATE(PHASE_04);
                    return XCCL_OK;
                }
                xccl_ucx_send_nb(buf, count, dst, team, tag, request);
            }
        }

        // DR Receives data from ASR
        if (job_config->dst_block_id == i_dst_block) {
            for (i = 0; i < job_config->n_src_blocks; i++) {
              PHASE_05:
                /* Number of ranks that sent the ASR data from the DR */
                n_ranks = (i != job_config->n_src_blocks - 1) ? job_config->src_block_size : job_config->last_src_block_size;

                /* Real data size to be received in bytes */
                count = data_size * n_ranks;

                buf = job_config->recv_buf + (data_size * job_config->src_block_size * i);
                src = (i != job_config->n_src_blocks - 1) ?
                    i * job_config->src_block_size + job_config->dst_block_id % job_config->src_block_size :
                    i * job_config->src_block_size + job_config->dst_block_id % job_config->last_src_block_size;

                request = xccl_ucx_get_free_req_nb(team, job_config->recv_requests,
                                                   &outstanding_recvs, job_config->max_recv);
                if (!request) {
                    SAVE_STATE(PHASE_05);
                    return XCCL_OK;
                }
                xccl_ucx_recv_nb(buf, count, src, team, tag, request);
            }
        }
    }


  PHASE_06:
    /* Check that all the send/recv operation were finished before finishing */
    /* the algorithm. */
    if (XCCL_INPROGRESS == xccl_ucx_testall(team, job_config->send_requests,
                                            job_config->max_send) ||
        XCCL_INPROGRESS == xccl_ucx_testall(team, job_config->recv_requests,
                                            job_config->max_recv)) {
        SAVE_STATE(PHASE_06);
        return XCCL_OK;
    }

    free(job_config->recv_requests);
    free(job_config);
    req->complete = XCCL_OK;
    return XCCL_OK;
}

xccl_status_t xccl_ucx_alltoall_blocked_progress(xccl_ucx_collreq_t *req)
{
    comm_params_t *job_config = (comm_params_t*)req->alltoall_blocked.config;
    return alltoall_block_multi(req, job_config);
}

xccl_status_t xccl_ucx_alltoall_blocked_start(xccl_ucx_collreq_t *req)
{
    size_t data_size      = req->args.buffer_info.len;
    int    group_rank     = req->team->params.oob.rank;
    int    group_size     = req->team->params.oob.size;
    ptrdiff_t sbuf        = (ptrdiff_t)req->args.buffer_info.src_buffer;
    ptrdiff_t rbuf        = (ptrdiff_t)req->args.buffer_info.dst_buffer;
    xccl_ucx_team_t *team = ucs_derived_of(req->team, xccl_ucx_team_t);
    comm_params_t *job_config;
    if (XCCL_OK != comm_params_init(req, &job_config,
                                    TEAM_UCX_CTX_REQ(req)->alltoall_src_block_size,
                                    TEAM_UCX_CTX_REQ(req)->alltoall_dst_block_size)) {
        return XCCL_ERR_NO_MESSAGE;
    }
    req->alltoall_blocked.config = (void*)job_config;
    req->progress = xccl_ucx_alltoall_blocked_progress;
    return xccl_ucx_alltoall_blocked_progress(req);
}
