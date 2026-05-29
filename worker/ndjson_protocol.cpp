#include "ndjson_protocol.h"

#include <cstdlib>
#include <sstream>

std::string jsonEscape(const std::string& value)
{
    std::ostringstream out;
    for (char ch : value)
    {
        switch (ch)
        {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << ch; break;
        }
    }
    return out.str();
}

static size_t skipWhitespace(const std::string& line, size_t pos)
{
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t' || line[pos] == '\r' || line[pos] == '\n'))
    {
        ++pos;
    }
    return pos;
}

static int hexValue(char ch)
{
    if (ch >= '0' && ch <= '9') return ch - '0';
    if (ch >= 'a' && ch <= 'f') return 10 + ch - 'a';
    if (ch >= 'A' && ch <= 'F') return 10 + ch - 'A';
    return -1;
}

static bool readHexCodeUnit(const std::string& line, size_t& pos, unsigned int& codeUnit)
{
    if (pos + 4 > line.size()) return false;

    codeUnit = 0;
    for (int i = 0; i < 4; ++i)
    {
        int digit = hexValue(line[pos + i]);
        if (digit < 0) return false;
        codeUnit = (codeUnit << 4) | static_cast<unsigned int>(digit);
    }
    pos += 4;
    return true;
}

static void appendUtf8(std::string& value, unsigned int codePoint)
{
    if (codePoint <= 0x7F)
    {
        value.push_back(static_cast<char>(codePoint));
    }
    else if (codePoint <= 0x7FF)
    {
        value.push_back(static_cast<char>(0xC0 | ((codePoint >> 6) & 0x1F)));
        value.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
    else if (codePoint <= 0xFFFF)
    {
        value.push_back(static_cast<char>(0xE0 | ((codePoint >> 12) & 0x0F)));
        value.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        value.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
    else
    {
        value.push_back(static_cast<char>(0xF0 | ((codePoint >> 18) & 0x07)));
        value.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
        value.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
        value.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
    }
}

static bool readJsonString(const std::string& line, size_t& pos, std::string& value)
{
    pos = skipWhitespace(line, pos);
    if (pos >= line.size() || line[pos] != '"') return false;
    ++pos;
    value.clear();
    while (pos < line.size())
    {
        char ch = line[pos++];
        if (ch == '"') return true;
        if (ch == '\\')
        {
            if (pos >= line.size()) return false;
            char escaped = line[pos++];
            switch (escaped)
            {
            case '"': value.push_back('"'); break;
            case '\\': value.push_back('\\'); break;
            case '/': value.push_back('/'); break;
            case 'b': value.push_back('\b'); break;
            case 'f': value.push_back('\f'); break;
            case 'n': value.push_back('\n'); break;
            case 'r': value.push_back('\r'); break;
            case 't': value.push_back('\t'); break;
            case 'u':
            {
                unsigned int codeUnit = 0;
                if (!readHexCodeUnit(line, pos, codeUnit)) return false;

                if (codeUnit >= 0xD800 && codeUnit <= 0xDBFF)
                {
                    if (pos + 6 > line.size() || line[pos] != '\\' || line[pos + 1] != 'u') return false;
                    pos += 2;

                    unsigned int lowSurrogate = 0;
                    if (!readHexCodeUnit(line, pos, lowSurrogate)) return false;
                    if (lowSurrogate < 0xDC00 || lowSurrogate > 0xDFFF) return false;

                    const unsigned int highPart = codeUnit - 0xD800;
                    const unsigned int lowPart = lowSurrogate - 0xDC00;
                    appendUtf8(value, 0x10000 + ((highPart << 10) | lowPart));
                }
                else if (codeUnit >= 0xDC00 && codeUnit <= 0xDFFF)
                {
                    return false;
                }
                else
                {
                    appendUtf8(value, codeUnit);
                }
                break;
            }
            default: return false;
            }
        }
        else
        {
            value.push_back(ch);
        }
    }
    return false;
}

static bool findValueStart(const std::string& line, const std::string& key, size_t& pos)
{
    const std::string needle = "\"" + key + "\"";
    pos = line.find(needle);
    if (pos == std::string::npos) return false;
    pos = line.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    pos = skipWhitespace(line, pos + 1);
    return true;
}

static bool extractString(const std::string& line, const std::string& key, std::string& value)
{
    size_t pos = 0;
    if (!findValueStart(line, key, pos)) return false;
    return readJsonString(line, pos, value);
}

static bool extractDouble(const std::string& line, const std::string& key, double& value)
{
    size_t pos = 0;
    if (!findValueStart(line, key, pos)) return false;
    char* end = nullptr;
    value = std::strtod(line.c_str() + pos, &end);
    return end && end != line.c_str() + pos;
}

static bool extractInt(const std::string& line, const std::string& key, int& value)
{
    size_t pos = 0;
    if (!findValueStart(line, key, pos)) return false;
    char* end = nullptr;
    long parsed = std::strtol(line.c_str() + pos, &end, 10);
    if (!end || end == line.c_str() + pos) return false;
    value = static_cast<int>(parsed);
    return true;
}

static bool extractImages(const std::string& line, std::vector<std::string>& images, std::string& error)
{
    size_t pos = 0;
    if (!findValueStart(line, "images", pos)) return true;
    if (pos >= line.size() || line[pos] != '[')
    {
        error = "images must be an array";
        return false;
    }

    ++pos;
    while (pos < line.size())
    {
        pos = skipWhitespace(line, pos);
        if (pos < line.size() && line[pos] == ']') return true;

        std::string imagePath;
        if (!readJsonString(line, pos, imagePath))
        {
            error = "images array contains a non-string item";
            return false;
        }
        images.push_back(imagePath);

        pos = skipWhitespace(line, pos);
        if (pos < line.size() && line[pos] == ',')
        {
            ++pos;
            continue;
        }
        if (pos < line.size() && line[pos] == ']') return true;
    }

    error = "images array is not closed";
    return false;
}

bool parseJsonRequest(const std::string& line, JsonRequest& request, std::string& error)
{
    if (!extractString(line, "cmd", request.command))
    {
        error = "request is missing cmd";
        return false;
    }

    std::string modelPath;
    if (extractString(line, "model_path", modelPath)) request.strings["model_path"] = modelPath;

    std::string backend;
    if (extractString(line, "backend", backend)) request.strings["backend"] = backend;

    if (!extractImages(line, request.images, error)) return false;
    extractInt(line, "class_count", request.classCount);
    extractDouble(line, "confidence_threshold", request.confidenceThreshold);
    extractDouble(line, "nms_threshold", request.nmsThreshold);
    return true;
}

std::string okObject(const std::string& body)
{
    return "{\"ok\":true" + (body.empty() ? std::string("") : "," + body) + "}";
}

std::string errorObject(const std::string& message)
{
    return "{\"ok\":false,\"error\":\"" + jsonEscape(message) + "\"}";
}
