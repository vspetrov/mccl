#include "config.h"
#include "xccl_ucx_lib.h"
#include "barrier.h"
#include "xccl_ucx_sendrecv.h"
#include "utils/reduce.h"
#include <stdlib.h>
#include <string.h>

enum {
    KN_BASE,
    KN_PROXY,
    KN_EXTRA
};

#define CALC_POW_K_SUP(_size, _radix, _pow_k_sup, _full_tree_size) do{  \
        int pk = 1;                                                     \
        int fs = _radix;                                                \
        while (fs < _size) {                                            \
            pk++; fs*=_radix;                                           \
        }                                                               \
        _pow_k_sup = pk;                                                \
        _full_tree_size = (fs != _size) ? fs/_radix : fs;               \
    }while(0)

#define KN_RECURSIVE_SETUP(__radix, __myrank, __size, __pow_k_sup,      \
                           __full_tree_size, __n_full_subtrees,         \
                           __full_size, __node_type) do{                \
        CALC_POW_K_SUP(__size, __radix, __pow_k_sup, __full_tree_size); \
        __n_full_subtrees = __size / __full_tree_size;                  \
        __full_size = __n_full_subtrees*__full_tree_size;               \
        __node_type = __myrank >= __full_size ? KN_EXTRA :              \
            (__size > __full_size && __myrank < __size - __full_size ?  \
             KN_PROXY : KN_BASE);                                       \
    }while(0)

#define KN_RECURSIVE_GET_PROXY(__myrank, __full_size) (__myrank - __full_size)
#define KN_RECURSIVE_GET_EXTRA(__myrank, __full_size) (__myrank + __full_size)

static void xccl_ucx_task_sr_send_cb(void *request, ucs_status_t status,
                                     void *user_data) {
    xccl_ucx_task_t *t = (xccl_ucx_task_t *)user_data;
    ucp_request_free(request);
    t->sr.completed++;
    if (t->sr.completed == (t->sr.n_sends + t->sr.n_recvs)) {
        t->super.state = UCC_TASK_STATE_COMPLETED;
    }
}

static void xccl_ucx_task_sr_recv_cb(void *request, ucs_status_t status,
                                     const ucp_tag_recv_info_t *tag_info, void *user_data) {
    xccl_ucx_task_t *t = (xccl_ucx_task_t *)user_data;
    ucp_request_free(request);
    t->sr.completed++;
    if (t->sr.completed == (t->sr.n_sends + t->sr.n_recvs)) {
        t->super.state = UCC_TASK_STATE_COMPLETED;
    }
}

static void xccl_ucx_send_recv_task_start(ucc_coll_task_t *task)
{
    int i;
    xccl_ucx_task_t *t = ucs_derived_of(task, xccl_ucx_task_t);
    xccl_ucx_schedule_t *s = ucs_derived_of(task->schedule, xccl_ucx_schedule_t);
    int immediate;
    int n_sends = t->sr.n_sends;
    int n_recvs = t->sr.n_recvs;
    /* start task if completion event received */
    task->state = UCC_TASK_STATE_INPROGRESS;
    for (i=0; i<n_recvs; i++) {
        xccl_ucx_recv_nbx(t->sr.r_bufs[i], t->sr.r_lens[i], t->sr.r_peers[i],
                          s->team, s->tag, task,
                          xccl_ucx_task_sr_recv_cb, &immediate);
        if (immediate) t->sr.completed++;
    }
    for (i=0; i<n_sends; i++) {
        xccl_ucx_send_nbx(t->sr.s_bufs[i], t->sr.s_lens[i], t->sr.s_peers[i],
                          s->team, s->tag, task,
                          xccl_ucx_task_sr_send_cb, &immediate);
        if (immediate) t->sr.completed++;
    }

    if (t->sr.completed == (n_sends + n_recvs)) {
        t->super.state = UCC_TASK_STATE_COMPLETED;
        ucc_event_manager_notify(&t->super.em, UCC_EVENT_COMPLETED);
    } else {
        xccl_task_enqueue(task->schedule->tl_ctx->pq, task);
    }
}

static inline xccl_ucx_task_t*
get_next_task_and_subscribe(xccl_ucx_schedule_t *schedule, int type) {
    int curr_task = schedule->n_tasks;
    xccl_ucx_task_t *t = &schedule->tasks[curr_task];
    ucc_coll_task_init(&t->super);
    t->super.progress = NULL;
    t->type = type;
    if (type == XCCL_UCX_TASK_SEND_RECV) {
        t->sr.n_sends = t->sr.n_recvs = t->sr.completed = 0;
        t->super.handlers[UCC_EVENT_COMPLETED] = xccl_ucx_send_recv_task_start;
        t->super.handlers[UCC_EVENT_SCHEDULE_STARTED] = xccl_ucx_send_recv_task_start;
    }
    if (curr_task > 0) {
        ucc_event_manager_subscribe(&schedule->tasks[curr_task-1].super.em,
                                    UCC_EVENT_COMPLETED, &t->super);
    } else {
        ucc_event_manager_subscribe(&schedule->super.super.em,
                                    UCC_EVENT_SCHEDULE_STARTED, &t->super);
    }
    ucc_schedule_add_task(&schedule->super, &t->super);
    schedule->n_tasks++;
    return t;
}

static void xccl_ucx_barrier_knomial_progress(xccl_ucx_schedule_t *schedule, int radix)
{
    int full_tree_size, pow_k_sup, n_full_subtrees, full_size, node_type;
    int iteration, k, step_size, peer;
    xccl_tl_team_t *team = &schedule->team->super;
    int myrank           = team->params.oob.rank;
    int group_size       = team->params.oob.size;
    int radix_pow = 1;
    xccl_ucx_task_t *task;

    KN_RECURSIVE_SETUP(radix, myrank, group_size, pow_k_sup, full_tree_size,
                       n_full_subtrees, full_size, node_type);

    if (KN_EXTRA == node_type) {
        peer = KN_RECURSIVE_GET_PROXY(myrank, full_size);
        task = get_next_task_and_subscribe(schedule, XCCL_UCX_TASK_SEND_RECV);
        task->sr.n_sends = task->sr.n_recvs = 1;
        task->sr.s_peers[0] = task->sr.r_peers[0] = peer;
        task->sr.s_lens[0] = task->sr.r_lens[0] = 0;
        return;
    }

    if (KN_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        task = get_next_task_and_subscribe(schedule, XCCL_UCX_TASK_SEND_RECV);
        task->sr.n_recvs = 1;
        task->sr.r_peers[0] = peer;
        task->sr.r_lens[0] = 0;
    }

    for (iteration = 0; iteration < pow_k_sup; iteration++) {
        step_size = radix_pow * radix;
        task = get_next_task_and_subscribe(schedule, XCCL_UCX_TASK_SEND_RECV);

        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size
                + (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            task->sr.s_peers[task->sr.n_sends] = peer;
            task->sr.s_lens[task->sr.n_sends] = 0;
            task->sr.n_sends++;
        }

        for (k=1; k < radix; k++) {
            peer = (myrank + k*radix_pow) % step_size
                + (myrank - myrank % step_size);
            if (peer >= full_size) continue;
            task->sr.r_peers[task->sr.n_recvs] = peer;
            task->sr.r_lens[task->sr.n_recvs] = 0;
            task->sr.n_recvs++;
        }
        radix_pow *= radix;
    }

    if (KN_PROXY == node_type) {
        peer = KN_RECURSIVE_GET_EXTRA(myrank, full_size);
        task = get_next_task_and_subscribe(schedule, XCCL_UCX_TASK_SEND_RECV);
        task->sr.n_sends = 1;
        task->sr.s_peers[0] = peer;
        task->sr.s_lens[0] = 0;
    }
}

xccl_status_t xccl_ucx_barrier_knomial_start(xccl_ucx_team_t *team,
                                             xccl_coll_op_args_t *coll,
                                             xccl_ucx_schedule_t **sched)
{
    size_t data_size     = coll->buffer_info.len;
    int radix   = TEAM_UCX_CTX(team)->barrier_kn_radix;
    xccl_ucx_schedule_t *schedule = &team->barrier_schedule;
    if (radix > team->super.params.oob.size) {
        radix = team->super.params.oob.size;
    }
    if (-1 == team->barrier_schedule.is_static) {
        team->barrier_schedule.is_static = 1;
        schedule->tasks = malloc(16*sizeof(xccl_ucx_task_t));
        schedule->team = team;
        schedule->n_tasks = 0;
        ucc_schedule_init(&schedule->super, team->super.ctx);
        xccl_ucx_barrier_knomial_progress(schedule, radix);
    } else {
        ucc_schedule_reset(&schedule->super);
        int i;
        for (i=0; i<schedule->n_tasks; i++) {
            schedule->tasks[i].sr.completed = 0;
            schedule->tasks[i].super.state = UCC_TASK_STATE_NOT_READY;
        }
    }
    *sched = schedule;
    return XCCL_OK;
}
