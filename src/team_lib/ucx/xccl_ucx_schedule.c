#include "config.h"
#include "xccl_ucx_schedule.h"
#include "xccl_ucx_sendrecv.h"
void xccl_ucx_task_sr_send_cb(void *request, ucs_status_t status,
                              void *user_data)
{
    xccl_ucx_task_t *t = (xccl_ucx_task_t *)user_data;
    ucp_request_free(request);
    t->sr.completed++;
    if (t->sr.completed == (t->sr.n_sends + t->sr.n_recvs)) {
        t->super.state = UCC_TASK_STATE_COMPLETED;
    }
}

void xccl_ucx_task_sr_recv_cb(void *request, ucs_status_t status,
                              const ucp_tag_recv_info_t *tag_info, void *user_data)
{
    xccl_ucx_task_t *t = (xccl_ucx_task_t *)user_data;
    ucp_request_free(request);
    t->sr.completed++;
    if (t->sr.completed == (t->sr.n_sends + t->sr.n_recvs)) {
        t->super.state = UCC_TASK_STATE_COMPLETED;
    }
}

void xccl_ucx_send_recv_task_start(ucc_coll_task_t *task)
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
