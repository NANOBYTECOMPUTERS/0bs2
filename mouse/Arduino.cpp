#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include "0BS_box_2.h"
#include "Arduino.h"

Arduino::Arduino(const std::string& port, unsigned int baud_rate, ArduinoProtocol protocol)
    : protocol_(protocol),
    button_mask_(0),
    is_open_(false),
    timer_running_(false),
    listening_(false),
    aiming_active(false),
    shooting_active(false),
    zooming_active(false)
{
    try
    {
        serial_.setPort(port);
        serial_.setBaudrate(baud_rate);
        serial_.open();

        if (serial_.isOpen())
        {
            is_open_ = true;
            std::cout << "[Arduino] Connected! PORT: " << port << std::endl;

            if (config.arduino_enable_keys || protocol_ == ArduinoProtocol::Teensy41)
            {
                startListening();
            }
        }
        else
        {
            std::cerr << "[Arduino] Unable to connect to the port: " << port << std::endl;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << "[Arduino] Error: " << e.what() << std::endl;
    }
}

Arduino::~Arduino()
{
    listening_ = false;
    if (serial_.isOpen())
    {
        try { serial_.close(); }
        catch (...) {}
    }
    if (listening_thread_.joinable())
    {
        listening_thread_.join();
    }
    is_open_ = false;
}

bool Arduino::isOpen() const
{
    return is_open_;
}

void Arduino::write(const std::string& data)
{
    std::lock_guard<std::mutex> lock(write_mutex_);
    if (is_open_)
    {
        try
        {
            serial_.write(data);
        }
        catch (...)
        {

        }
    }
}

std::string Arduino::read()
{
    if (!is_open_)
        return std::string();

    std::string result;
    try
    {
        result = serial_.readline(65536, "\n");
    }
    catch (...)
    {
        is_open_ = false;
    }
    return result;
}

void Arduino::click()
{
    if (protocol_ == ArduinoProtocol::Teensy41)
    {
        press();
        release();
        return;
    }

    sendCommand("c");
}

void Arduino::press()
{
    if (protocol_ == ArduinoProtocol::Teensy41)
    {
        button_mask_ |= 1;
        sendButtons();
        return;
    }

    sendCommand("p");
}

void Arduino::release()
{
    if (protocol_ == ArduinoProtocol::Teensy41)
    {
        button_mask_ &= static_cast<uint8_t>(~1u);
        sendButtons();
        return;
    }

    sendCommand("r");
}

void Arduino::move(int x, int y)
{
    if (!is_open_)
        return;

    if (x == 0 && y == 0)
    {
        return;
    }

    if (protocol_ == ArduinoProtocol::Teensy41)
    {
        write("move " + std::to_string(x) + " " + std::to_string(y) + " 0 0\n");
        return;
    }

    if (config.arduino_16_bit_mouse)
    {
        std::string data = "m" + std::to_string(x) + "," + std::to_string(y) + "\n";
        write(data);
    }
    else
    {
        std::vector<int> x_parts = splitValue(x);
        std::vector<int> y_parts = splitValue(y);

        size_t max_splits = std::max(x_parts.size(), y_parts.size());
        while (x_parts.size() < max_splits) x_parts.push_back(0);
        while (y_parts.size() < max_splits) y_parts.push_back(0);

        for (size_t i = 0; i < max_splits; ++i)
        {
            std::string data = "m" + std::to_string(x_parts[i]) + "," + std::to_string(y_parts[i]) + "\n";
            write(data);
        }
    }
}

void Arduino::sendCommand(const std::string& command)
{
    write(command + "\n");
}

void Arduino::sendButtons()
{
    write("buttons " + std::to_string(button_mask_) + "\n");
}

std::vector<int> Arduino::splitValue(int value)
{
    std::vector<int> values;
    int sign = (value < 0) ? -1 : 1;
    int absVal = (value < 0) ? -value : value;

    if (value == 0)
    {
        values.push_back(0);
        return values;
    }

    while (absVal > 127)
    {
        values.push_back(sign * 127);
        absVal -= 127;
    }

    if (absVal != 0)
    {
        values.push_back(sign * absVal);
    }

    return values;
}

void Arduino::timerThreadFunc()
{
    while (timer_running_)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!is_open_)
            continue;

        bool arduino_enable_keys_local;
        {
            arduino_enable_keys_local = config.arduino_enable_keys;
        }

        if (arduino_enable_keys_local || protocol_ == ArduinoProtocol::Teensy41)
        {
            if (!listening_)
            {
                startListening();
            }
        }
        else
        {
            if (listening_)
            {
                listening_ = false;
                if (listening_thread_.joinable())
                {
                    listening_thread_.join();
                }
            }
        }
    }
}

void Arduino::startListening()
{
    listening_ = true;
    if (listening_thread_.joinable())
        listening_thread_.join();

    listening_thread_ = std::thread(&Arduino::listeningThreadFunc, this);
}

void Arduino::listeningThreadFunc()
{
    std::string buffer;
    while (listening_ && is_open_)
    {
        try
        {
            size_t available = serial_.available();
            if (available > 0)
            {
                std::string data = serial_.read(available);
                buffer += data;
                size_t pos;
                while ((pos = buffer.find('\n')) != std::string::npos)
                {
                    std::string line = buffer.substr(0, pos);
                    buffer.erase(0, pos + 1);
                    if (!line.empty() && line.back() == '\r')
                        line.pop_back();
                    processIncomingLine(line);
                }
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        catch (...)
        {
            is_open_ = false;
            break;
        }
    }
}

void Arduino::processIncomingLine(const std::string& line)
{
    try
    {
        auto applyButtonState = [&](uint16_t buttonId, bool pressed)
        {
            switch (buttonId)
            {
            case 1:
                shooting_active = pressed;
                shooting.store(pressed);
                break;
            case 2:
                if (protocol_ == ArduinoProtocol::Teensy41)
                {
                    zooming_active = pressed;
                    zooming.store(pressed);
                }
                else
                {
                    aiming_active = pressed;
                    aiming.store(pressed);
                }
                break;
            case 5:
                aiming_active = pressed;
                aiming.store(pressed);
                break;
            default:
                break;
            }
        };

        if (line.rfind("BD:", 0) == 0)
        {
            uint16_t buttonId = static_cast<uint16_t>(std::stoi(line.substr(3)));
            applyButtonState(buttonId, true);
        }
        else if (line.rfind("BU:", 0) == 0)
        {
            uint16_t buttonId = static_cast<uint16_t>(std::stoi(line.substr(3)));
            applyButtonState(buttonId, false);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Arduino] Error processing line '" << line << "': " << e.what() << std::endl;
    }
}
