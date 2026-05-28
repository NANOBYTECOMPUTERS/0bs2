#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <windows.h>
#include <algorithm>
#include <cctype>
#include <iostream>

#include "Makcu.h"
#include "0BS_box_2.h"

namespace
{
constexpr unsigned int MakcuDefaultBaudRate = 4000000;

std::string normalizeMakcuPort(const std::string& port)
{
    const auto first = port.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
        return "";

    const auto last = port.find_last_not_of(" \t\r\n");
    std::string trimmed = port.substr(first, last - first + 1);
    std::string upper = trimmed;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });

    if (upper.empty() || port == "COM0" || port == "AUTO" || upper == "COM0" || upper == "AUTO")
        return "";

    return trimmed;
}

std::string describeMakcuPort(const std::string& normalized_port)
{
    return normalized_port.empty() ? "AUTO" : normalized_port;
}

makcu::MouseButton toMakcuButton(int button)
{
    switch (button)
    {
    case 1:
        return makcu::MouseButton::RIGHT;
    case 2:
        return makcu::MouseButton::MIDDLE;
    case 3:
        return makcu::MouseButton::SIDE1;
    case 4:
        return makcu::MouseButton::SIDE2;
    case 0:
    default:
        return makcu::MouseButton::LEFT;
    }
}
}

MakcuConnection::MakcuConnection(const std::string& port, unsigned int baud_rate)
    : is_open_(false)
    , aiming_active(false)
    , shooting_active(false)
    , zooming_active(false)
{
    try
    {
        device_.setMouseButtonCallback([this](makcu::MouseButton button, bool pressed) {
            onButtonCallback(button, pressed);
        });

        const std::string normalized_port = normalizeMakcuPort(port);

        if (device_.connect(normalized_port))
        {
            if (baud_rate > 0 && baud_rate != MakcuDefaultBaudRate)
            {
                if (!device_.setBaudRate(baud_rate, true))
                {
                    std::cerr << "[Makcu] Failed to set baud rate to " << baud_rate
                        << "." << std::endl;
                    device_.disconnect();
                    return;
                }
            }

            if (!device_.enableButtonMonitoring(true))
            {
                std::cerr << "[Makcu] Failed to enable button monitoring." << std::endl;
                device_.disconnect();
                return;
            }

            is_open_ = true;
            std::cout << "[Makcu] Connected! PORT: " << describeMakcuPort(normalized_port) << std::endl;
        }
        else
        {
            std::cerr << "[Makcu] Unable to connect to the port: " << describeMakcuPort(normalized_port) << std::endl;
        }
    }
    catch (const makcu::MakcuException& e)
    {
        std::cerr << "[Makcu] Error: " << e.what() << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[Makcu] Error: " << e.what() << std::endl;
    }
}

MakcuConnection::~MakcuConnection()
{
    try
    {
        device_.disconnect();
    }
    catch (...)
    {
    }
    is_open_ = false;
}

bool MakcuConnection::isOpen() const
{
    return is_open_ && device_.isConnected();
}

bool MakcuConnection::move(int x, int y)
{
    if (!is_open_)
        return false;

    std::lock_guard<std::mutex> lock(write_mutex_);
    try
    {
        bool ok = device_.mouseMove(x, y);
        if (!ok)
            is_open_ = false;
        return ok;
    }
    catch (...)
    {
        is_open_ = false;
        return false;
    }
}

bool MakcuConnection::click(int button)
{
    if (!is_open_)
        return false;

    std::lock_guard<std::mutex> lock(write_mutex_);
    try
    {
        bool ok = device_.click(toMakcuButton(button));
        if (!ok)
            is_open_ = false;
        return ok;
    }
    catch (...)
    {
        is_open_ = false;
        return false;
    }
}

bool MakcuConnection::press(int button)
{
    if (!is_open_)
        return false;

    std::lock_guard<std::mutex> lock(write_mutex_);
    try
    {
        bool ok = device_.mouseDown(toMakcuButton(button));
        if (!ok)
            is_open_ = false;
        return ok;
    }
    catch (...)
    {
        is_open_ = false;
        return false;
    }
}

bool MakcuConnection::release(int button)
{
    if (!is_open_)
        return false;

    std::lock_guard<std::mutex> lock(write_mutex_);
    try
    {
        bool ok = device_.mouseUp(toMakcuButton(button));
        if (!ok)
            is_open_ = false;
        return ok;
    }
    catch (...)
    {
        is_open_ = false;
        return false;
    }
}

void MakcuConnection::onButtonCallback(makcu::MouseButton button, bool pressed)
{
    switch (button)
    {
    case makcu::MouseButton::LEFT:
        // LMB = shooting
        shooting_active = pressed;
        shooting.store(pressed);
        break;

    case makcu::MouseButton::RIGHT:
        // RMB = zooming
        zooming_active = pressed;
        zooming.store(pressed);
        break;

    case makcu::MouseButton::MIDDLE:
        // MMB - not used for now
        break;

    case makcu::MouseButton::SIDE1:
        // Mouse4 (side button 1) - not used
        break;

    case makcu::MouseButton::SIDE2:
        // Mouse5 (side button 2) = aiming
        aiming_active = pressed;
        aiming.store(pressed);
        break;
    }
}
