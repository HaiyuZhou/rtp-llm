#pragma once

#include <string>

namespace rtp_llm {

struct ProfileStatus {
    bool        supported = false;
    std::string state     = "finished";
    std::string trace_name;
    bool        stop_requested      = false;
    bool        collection_complete = false;
    bool        export_complete     = false;
    std::string collection_error;
    std::string export_error;
    std::string file_name;
};

}  // namespace rtp_llm
