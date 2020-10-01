#include "config.h"
#include "xccl_ucx_lib.h"
#include "allreduce.h"
#include "allreduce_knomial.h"
#include "xccl_ucx_sendrecv.h"
#include "utils/reduce.h"
#include "utils/mem_component.h"
#include <stdlib.h>
#include <string.h>

xccl_status_t
xccl_ucx_allreduce_knomial_progress(xccl_ucx_schedule_t *schedule, int radix,
                                    xccl_coll_op_args_t *coll)
{
    int full_tree_size, pow_k_sup, n_full_subtrees, full_size, node_type;
    int iteration, radix_pow, active_reqs, k, step_size, peer;
    ptrdiff_t recv_offset;
    void *dst_buffer;
    void *src_buffer;
    xccl_tl_team_t *team = &schedule->team->super;
    size_t data_size     = coll->buffer_info.len;
    int myrank           = team->params.oob.rank;
    int group_size       = team->params.oob.size;
    int radix_pow        = 1;
    void *scratch        = schedule->scratch;
    xccl_ucx_task_t *task;
    /* fprintf(stderr, "AR, radix %d, data_size %zd, count %d\n",
 radix, data_size, args->allreduce.count); */
    KN_RECURSIVE_SETUP(radix, myrank, group_size, pow_k_sup, full_tree_size,
                       n_full_subtrees, full_size, node_type);

    if (KN_EXTRA == node_type) {
        peer = KN_RECURSIVE_GET_PROXY(myrank, full_size);
        task = get_next_task_and_subscribe(schedule, XCCL_UCX_TASK_SEND_RECV);
        task->sr.n_sends    = task->sr.n_recvs    = 1;
        task->sr.s_peers[0] = task->sr.r_peers[0] = peer;
        task->sr.s_lens[0]  = task->sr.r_lens[0]  = data_size;
        task->sr.s_bufs[0]  = coll->buffer_info.src_buffer;
        task->sr.r_bufs[0]  = coll->buffer_info.dst_buffer;
        return;
    }

    if (KN_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        task = get_next_task_and_subscribe(schedule, XCCL_UCX_TASK_SEND_RECV);
        task->sr.n_recvs    = 1;
        task->sr.r_peers[0] = peer;
        task->sr.r_lens[0]  = data_size;
        task->sr.r_bufs[0]  = scratch;
        
        task = get_next_task_and_subscribe(schedule, XCCL_UCX_TASK_REDUCE);
        task->reduce.sbuf1    = coll->buffer_info.src_buffer;
        task->reduce.sbuf2    = scratch;
        task->reduce.rbuf     = coll->buffer_info.dst_buffer;
        task->reduce.count    = 1;
        task->reduce.size     = coll->reduce_info.count;
        task->reduce.stride   = 0;
        task->reduce.dtype    = coll->reduce_info.dt;
        task->reduce.op       = coll->reduce_info.op;
        task->reduce.mem_type = mem_type;
    }

    if (KN_PROXY == node_type || KN_EXTRA == node_type) {
        if (XCCL_INPROGRESS == xccl_ucx_testall((xccl_ucx_team_t *)team,
                                                   reqs, active_reqs)) {
            SAVE_STATE(PHASE_EXTRA);
            return XCCL_OK;
        }
        if (KN_EXTRA == node_type) {
            goto completion;
        } else {
            xccl_mem_component_reduce(req->args.buffer_info.src_buffer,
                                      scratch,
                                      req->args.buffer_info.dst_buffer,
                                      req->args.reduce_info.count,
                                      req->args.reduce_info.dt,
                                      req->args.reduce_info.op,
                                      req->mem_type);
        }
    }

    for (; iteration < pow_k_sup; iteration++) {
        src_buffer  = ((iteration == 0) && (node_type == KN_BASE)) ?
                      req->args.buffer_info.src_buffer:
                      req->args.buffer_info.dst_buffer;        
        dst_buffer  = req->args.buffer_info.dst_buffer;
        step_size   = radix_pow * radix;
        active_reqs = 0;
        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size
                + (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            xccl_ucx_send_nb(src_buffer, data_size, peer,
                            (xccl_ucx_team_t *)team, req->tag, &reqs[active_reqs]);
            active_reqs++;
        }

        recv_offset = 0;
        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size
                + (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            xccl_ucx_recv_nb((void*)((ptrdiff_t)scratch + recv_offset), data_size,
                             peer, (xccl_ucx_team_t *)team, req->tag, &reqs[active_reqs]);
            active_reqs++;
            recv_offset += data_size;
        }
        radix_pow *= radix;
        if (active_reqs) {
        PHASE_1:
            src_buffer = ((iteration == 0) && (node_type == KN_BASE)) ?
                         req->args.buffer_info.src_buffer:
                         req->args.buffer_info.dst_buffer;        
            dst_buffer = req->args.buffer_info.dst_buffer;
            if (XCCL_INPROGRESS == xccl_ucx_testall((xccl_ucx_team_t *)team,
                                                       reqs, active_reqs)) {
                SAVE_STATE(PHASE_1);
                return XCCL_OK;
            }
            assert(active_reqs % 2 == 0);
 
            xccl_mem_component_reduce_multi(src_buffer, scratch, dst_buffer,
                                            active_reqs/2,
                                            req->args.reduce_info.count,
                                            data_size,
                                            req->args.reduce_info.dt,
                                            req->args.reduce_info.op,
                                            req->mem_type);
        }
    }
    if (KN_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        xccl_ucx_send_nb(req->args.buffer_info.dst_buffer, data_size, peer,
                        (xccl_ucx_team_t *)team, req->tag, &reqs[0]);
        active_reqs = 1;
        goto PHASE_PROXY;
    } else {
        goto completion;
    }

PHASE_PROXY:
    if (XCCL_INPROGRESS == xccl_ucx_testall((xccl_ucx_team_t *)team,
                                               reqs, active_reqs)) {
        SAVE_STATE(PHASE_PROXY);
        return XCCL_OK;
    }

completion:
    /* fprintf(stderr, "Complete reduce, level %d frag %d and full coll arg\n", */
    /*         COLL_ID_IN_SCHEDULE(bcol_args), bcol_args->next_frag-1); */
    req->complete = XCCL_OK;
    if (req->allreduce.scratch) {
        xccl_mem_component_free(req->allreduce.scratch, req->mem_type);
    }
    return XCCL_OK;
}

xccl_status_t xccl_ucx_allreduce_knomial_start(xccl_ucx_team_t *team,
                                               xccl_coll_op_args_t *coll,
                                               xccl_ucx_schedule_t **sched,
                                               ucs_memory_type_t mem_type)
{
    size_t data_size              = req->args.buffer_info.len;
    int radix                     = TEAM_UCX_CTX_REQ(req)->allreduce_kn_radix;
    xccl_ucx_schedule_t *schedule = malloc(sizeof(*schedule));
    schedule->is_static = 0;
    if (radix > team->super.params.oob.size) {
        radix = team->super.params.oob.size;
    }


    schedule->tasks = malloc(16*sizeof(xccl_ucx_task_t));//todo: compute how many
    schedule->team = team;
    schedule->n_tasks = 0;
    ucc_schedule_init(&schedule->super, team->super.ctx);
    xccl_mem_component_alloc(&schedule->scratch,
                             (radix-1)*data_size, mem_type);
    schedule->mem_type = mem_type;
    xccl_ucx_allreduce_knomial_progress(schedule, coll, radix);
    *sched = schedule;
    return XCCL_OK;
}
