#ifndef BARRIER_H_
#define BARRIER_H_
#include "../xccl_ucx_lib.h"
#include "../xccl_ucx_schedule.h"

xccl_status_t xccl_ucx_barrier_knomial_start(xccl_ucx_team_t *team,
                                             xccl_coll_op_args_t *coll,
                                             xccl_ucx_schedule_t **sched);
#endif
