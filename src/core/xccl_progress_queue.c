#include <xccl_progress_queue.h>
#include <xccl_team_lib.h>
#include <xccl_schedule.h>
#include <tasks_pool.h>


typedef struct xccl_progress_queue {
    context st;
} xccl_progress_queue_t;


xccl_status_t xccl_ctx_progress_queue_init(xccl_progress_queue_t **q) {
    xccl_progress_queue_t *pq = malloc(sizeof(*pq));
    xccl_status_t status = tasks_pool_init(&pq->st);
    if (status != XCCL_OK) {
        return status;
    }
    *q = pq;
    return XCCL_OK;

}

xccl_status_t xccl_task_enqueue(xccl_progress_queue_t *q,
                                ucc_coll_task_t *task) {
    task->was_progressed = 0;
    return tasks_pool_insert(&(q->st), task);
}

xccl_status_t xccl_ctx_progress_queue(xccl_tl_context_t *tl_ctx) {
    ucc_coll_task_t *popped_task;
    xccl_status_t status = tasks_pool_pop(&tl_ctx->pq->st, &popped_task,1);
    if (status != XCCL_OK) {
        return status;
    }
    if (popped_task) {
        if (popped_task->progress) {
            popped_task->progress(popped_task);
        }
        if (UCC_TASK_STATE_COMPLETED == popped_task->state) {
            ucc_event_manager_notify(&popped_task->em, UCC_EVENT_COMPLETED);
            /* task cleanup ? */
        } else {
            tasks_pool_insert(&tl_ctx->pq->st, popped_task);
        }
    }
    return XCCL_OK;
}

int xccl_ctx_progress_queue_is_empty(xccl_tl_context_t *tl_ctx) {
    return tasks_pool_is_empty(&tl_ctx->pq->st);
}

xccl_status_t xccl_ctx_progress_queue_destroy(xccl_progress_queue_t **q) {
    return tasks_pool_cleanup(&((*q)->st));
}
