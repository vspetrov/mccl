/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef XCCL_TEAM_LIB_MHBA_H_
#define XCCL_TEAM_LIB_MHBA_H_
#include "xccl_team_lib.h"

typedef struct xccl_team_lib_mhba_config {
    xccl_team_lib_config_t super;
} xccl_team_lib_mhba_config_t;

typedef struct xccl_tl_mhba_context_config {
    xccl_tl_context_config_t super;
    char                     *device;
} xccl_tl_mhba_context_config_t;

typedef struct xccl_team_lib_mhba {
    xccl_team_lib_t             super;
    xccl_team_lib_mhba_config_t config;
} xccl_team_lib_mhba_t;

extern xccl_team_lib_mhba_t xccl_team_lib_mhba;

#define xccl_team_mhba_log_component(_level, _fmt, ...) \
    do { \
        ucs_log_component(_level, &xccl_team_lib_mhba.config.super.log_component, _fmt, ## __VA_ARGS__); \
    } while (0)

#define xccl_mhba_error(_fmt, ...)       xccl_team_mhba_log_component(UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define xccl_mhba_warn(_fmt, ...)        xccl_team_mhba_log_component(UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define xccl_mhba_info(_fmt, ...)        xccl_team_mhba_log_component(UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define xccl_mhba_debug(_fmt, ...)       xccl_team_mhba_log_component(UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define xccl_mhba_trace(_fmt, ...)       xccl_team_mhba_log_component(UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define xccl_mhba_trace_req(_fmt, ...)   xccl_team_mhba_log_component(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define xccl_mhba_trace_data(_fmt, ...)  xccl_team_mhba_log_component(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define xccl_mhba_trace_async(_fmt, ...) xccl_team_mhba_log_component(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define xccl_mhba_trace_func(_fmt, ...)  xccl_team_mhba_log_component(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ## __VA_ARGS__)
#define xccl_mhba_trace_poll(_fmt, ...)  xccl_team_mhba_log_component(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)

typedef struct xccl_mhba_context {
    xccl_tl_context_t super;
} xccl_mhba_context_t;

typedef struct xccl_mhba_team {
    xccl_tl_team_t super;
} xccl_mhba_team_t;

#endif