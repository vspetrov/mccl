#include <xccl_progress_queue.h>
#include <xccl_team_lib.h>
#include <xccl_schedule.h>
#include <pthread.h>


#ifdef PTRARRAY
#include "ptr_array.h"
#else
#include <liblfds711.h>
#endif

typedef struct xccl_progress_queue {
#ifdef PTRARRAY
    ucs_ptr_array_locked_t pa;
#else    
    struct lfds711_queue_umm_state st;
#endif
} xccl_progress_queue_t;


xccl_status_t xccl_ctx_progress_queue_init(xccl_progress_queue_t **q)
{
    xccl_progress_queue_t *pq = malloc(sizeof(*pq));
#ifdef PTRARRAY
    ucs_ptr_array_locked_init(&pq->pa, "progres_queue");
#else
    struct lfds711_queue_umm_element *qe = malloc(sizeof(*qe));
    lfds711_queue_umm_init_valid_on_current_logical_core(&pq->st, qe, NULL);
#endif    
    *q = pq;
    return XCCL_OK;

}
static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

xccl_status_t xccl_task_enqueue(xccl_progress_queue_t *q,
                                ucc_coll_task_t *task)
{
#ifdef PTRARRAY
    ucs_ptr_array_locked_insert(&q->pa, (void*)task);
#else
    struct lfds711_queue_umm_element *qe = malloc(sizeof(*qe));
    LFDS711_QUEUE_UMM_SET_VALUE_IN_ELEMENT(*qe, (void*)task);
    lfds711_queue_umm_enqueue( &q->st, qe);
#endif

    return XCCL_OK;
}

xccl_status_t xccl_ctx_progress_queue(xccl_tl_context_t *tl_ctx)
{
    ucc_coll_task_t *task;    
#ifdef PTRARRAY
    int i;
    ucs_ptr_array_t *p_array = &tl_ctx->pq->pa.super;
    ucs_ptr_array_locked_for_each(task, i, &tl_ctx->pq->pa) {
        if (__sync_fetch_and_add(&task->busy, 1) == 0) {
            if (XCCL_OK == task->progress(task)) {
                ucs_ptr_array_locked_remove(&tl_ctx->pq->pa,i);
            }
            __sync_fetch_and_add(&task->busy, -1);
        }
    }
#else
    struct lfds711_queue_umm_element *qe;
    if (lfds711_queue_umm_dequeue(&tl_ctx->pq->st, &qe)) {
        task = LFDS711_QUEUE_UMM_GET_VALUE_FROM_ELEMENT(*qe);
        if (XCCL_OK == task->progress(task)) {
            free(qe);
        } else {
            lfds711_queue_umm_enqueue(&tl_ctx->pq->st, qe);
        }
    }
    
#endif
    return XCCL_OK;
}