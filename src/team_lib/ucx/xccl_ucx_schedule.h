/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef XCCL_UCX_SCHEDULE_H_
#define XCCL_UCX_SCHEDULE_H_
#include "xccl_team_lib.h"
#include <ucp/api/ucp.h>
#include <ucs/memory/memory_type.h>

enum {
    XCCL_UCX_TASK_SEND_RECV,
    XCCL_UCX_TASK_REDUCE,
};

#define XCCL_UCX_TASK_MAX_PEERS 16
typedef struct xccl_ucx_task {
    ucc_coll_task_t     super;
    xccl_coll_req_h     req;
    int type;
    union {
        struct {
            int n_sends;
            int n_recvs;
            int completed;
            int s_peers[XCCL_UCX_TASK_MAX_PEERS];
            int r_peers[XCCL_UCX_TASK_MAX_PEERS];
            void *s_bufs[XCCL_UCX_TASK_MAX_PEERS];
            void *r_bufs[XCCL_UCX_TASK_MAX_PEERS];
            size_t s_lens[XCCL_UCX_TASK_MAX_PEERS];
            size_t r_lens[XCCL_UCX_TASK_MAX_PEERS];
        } sr;
    };
} xccl_ucx_task_t;

typedef struct xccl_ucx_team_t xccl_ucx_team_t;

typedef struct xccl_ucx_schedule {
    ucc_schedule_t     super;
    xccl_tl_coll_req_t req;
    xccl_ucx_task_t   *tasks;
    int                n_tasks;
    xccl_ucx_team_t *team;
    int tag;
    int is_static;
} xccl_ucx_schedule_t;

void xccl_ucx_task_sr_send_cb(void *request, ucs_status_t status,
                              void *user_data);
void xccl_ucx_task_sr_recv_cb(void *request, ucs_status_t status,
                              const ucp_tag_recv_info_t *tag_info, void *user_data);
void xccl_ucx_send_recv_task_start(ucc_coll_task_t *task);

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

#endif
