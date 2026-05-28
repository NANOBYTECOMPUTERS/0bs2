#pragma once

#include <map>
#include <string>
#include <vector>

struct JsonRequest
{
    std::string command;
    std::map<std::string, std::string> strings;
    std::vector<std::string> images;
    int classCount = 0;
    double confidenceThreshold = 0.25;
    double nmsThreshold = 0.45;
};

std::string jsonEscape(const std::string& value);
bool parseJsonRequest(const std::string& line, JsonRequest& request, std::string& error);
std::string okObject(const std::string& body);
std::string errorObject(const std::string& message);
