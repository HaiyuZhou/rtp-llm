cc_library(
    name = "execution_recorder",
    srcs = ["ExecutionRecorder.cc"],
    hdrs = ["ExecutionRecorder.h"],
    deps = ["@havenask//aios/alog:alog"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)

cc_test(
    name = "execution_recorder_test",
    copts = ["-UNDEBUG"],
    srcs = ["ExecutionRecorderTest.cc"],
    deps = [":execution_recorder"],
)
