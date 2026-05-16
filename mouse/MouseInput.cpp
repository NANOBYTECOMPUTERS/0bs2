#define WIN32_LEAN_AND_MEAN
#define _WINSOCKAPI_
#include <winsock2.h>
#include <Windows.h>

#include "MouseInput.h"

#include <iostream>
#include <memory>

#include "Arduino.h"
#include "KmboxAConnection.h"
#include "KmboxNetConnection.h"
#include "Makcu.h"
#include "config.h"
#include "ghub.h"
#include "rzctl.h"

namespace
{
bool sendWin32Move(int dx, int dy)
{
    INPUT input{ 0 };
    input.type = INPUT_MOUSE;
    input.mi.dx = dx;
    input.mi.dy = dy;
    input.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_VIRTUALDESK;
    return SendInput(1, &input, sizeof(INPUT)) == 1;
}

bool sendWin32Click(DWORD flag)
{
    INPUT input{ 0 };
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = flag;
    return SendInput(1, &input, sizeof(INPUT)) == 1;
}

class Win32MouseInput final : public IMouseInput
{
public:
    const char* name() const override { return "WIN32"; }
    bool isOpen() const override { return true; }
    bool move(int dx, int dy) override { return sendWin32Move(dx, dy); }
    bool leftDown() override { return sendWin32Click(MOUSEEVENTF_LEFTDOWN); }
    bool leftUp() override { return sendWin32Click(MOUSEEVENTF_LEFTUP); }
};

class ArduinoMouseInput final : public IMouseInput
{
public:
    ArduinoMouseInput(const std::string& port, unsigned int baudrate)
        : device_(std::make_unique<Arduino>(port, baudrate))
    {
    }

    const char* name() const override { return "ARDUINO"; }
    bool isOpen() const override { return device_ && device_->isOpen(); }
    bool move(int dx, int dy) override
    {
        if (!isOpen())
            return sendWin32Move(dx, dy);
        device_->move(dx, dy);
        return true;
    }
    bool leftDown() override
    {
        if (!isOpen())
            return sendWin32Click(MOUSEEVENTF_LEFTDOWN);
        device_->press();
        return true;
    }
    bool leftUp() override
    {
        if (!isOpen())
            return sendWin32Click(MOUSEEVENTF_LEFTUP);
        device_->release();
        return true;
    }
    Arduino* arduino() override { return device_.get(); }

private:
    std::unique_ptr<Arduino> device_;
};

class GHubMouseInput final : public IMouseInput
{
public:
    GHubMouseInput()
        : device_(std::make_unique<GhubMouse>())
    {
    }

    ~GHubMouseInput() override
    {
        if (device_)
            device_->mouse_close();
    }

    const char* name() const override { return "GHUB"; }
    bool isOpen() const override { return device_ != nullptr; }
    bool move(int dx, int dy) override { return device_ ? device_->mouse_xy(dx, dy) : sendWin32Move(dx, dy); }
    bool leftDown() override { return device_ ? device_->mouse_down() : sendWin32Click(MOUSEEVENTF_LEFTDOWN); }
    bool leftUp() override { return device_ ? device_->mouse_up() : sendWin32Click(MOUSEEVENTF_LEFTUP); }
    GhubMouse* ghub() override { return device_.get(); }

private:
    std::unique_ptr<GhubMouse> device_;
};

class RazerMouseInput final : public IMouseInput
{
public:
    RazerMouseInput()
        : device_(std::make_unique<RzctlMouse>())
    {
    }

    ~RazerMouseInput() override
    {
        if (device_)
            device_->mouse_close();
    }

    const char* name() const override { return "RAZER"; }
    bool isOpen() const override { return device_ && device_->isOpen(); }
    bool move(int dx, int dy) override { return device_ ? device_->mouse_xy(dx, dy) : sendWin32Move(dx, dy); }
    bool leftDown() override { return device_ ? device_->mouse_down() : sendWin32Click(MOUSEEVENTF_LEFTDOWN); }
    bool leftUp() override { return device_ ? device_->mouse_up() : sendWin32Click(MOUSEEVENTF_LEFTUP); }
    RzctlMouse* razer() override { return device_.get(); }

private:
    std::unique_ptr<RzctlMouse> device_;
};

class KmboxNetMouseInput final : public IMouseInput
{
public:
    KmboxNetMouseInput(const std::string& ip, const std::string& port, const std::string& uuid)
        : device_(std::make_unique<KmboxNetConnection>(ip, port, uuid))
    {
    }

    const char* name() const override { return "KMBOX_NET"; }
    bool isOpen() const override { return device_ && device_->isOpen(); }
    bool move(int dx, int dy) override
    {
        if (!isOpen())
            return sendWin32Move(dx, dy);
        device_->move(dx, dy);
        return true;
    }
    bool leftDown() override
    {
        if (!isOpen())
            return sendWin32Click(MOUSEEVENTF_LEFTDOWN);
        device_->leftDown();
        return true;
    }
    bool leftUp() override
    {
        if (!isOpen())
            return sendWin32Click(MOUSEEVENTF_LEFTUP);
        device_->leftUp();
        return true;
    }
    KmboxNetConnection* kmboxNet() override { return device_.get(); }

private:
    std::unique_ptr<KmboxNetConnection> device_;
};

class KmboxAMouseInput final : public IMouseInput
{
public:
    explicit KmboxAMouseInput(const std::string& pidvid)
        : device_(std::make_unique<KmboxAConnection>(pidvid))
    {
    }

    const char* name() const override { return "KMBOX_A"; }
    bool isOpen() const override { return device_ && device_->isOpen(); }
    bool move(int dx, int dy) override
    {
        if (!isOpen())
            return sendWin32Move(dx, dy);
        device_->move(dx, dy);
        return true;
    }
    bool leftDown() override
    {
        if (!isOpen())
            return sendWin32Click(MOUSEEVENTF_LEFTDOWN);
        device_->leftDown();
        return true;
    }
    bool leftUp() override
    {
        if (!isOpen())
            return sendWin32Click(MOUSEEVENTF_LEFTUP);
        device_->leftUp();
        return true;
    }
    KmboxAConnection* kmboxA() override { return device_.get(); }

private:
    std::unique_ptr<KmboxAConnection> device_;
};

class MakcuMouseInput final : public IMouseInput
{
public:
    MakcuMouseInput(const std::string& port, unsigned int baudrate)
        : device_(std::make_unique<MakcuConnection>(port, baudrate))
    {
    }

    const char* name() const override { return "MAKCU"; }
    bool isOpen() const override { return device_ && device_->isOpen(); }
    bool move(int dx, int dy) override
    {
        if (!isOpen())
            return sendWin32Move(dx, dy);
        if (device_->move(dx, dy))
            return true;
        return sendWin32Move(dx, dy);
    }
    bool leftDown() override
    {
        if (!isOpen())
            return sendWin32Click(MOUSEEVENTF_LEFTDOWN);
        if (device_->press(0))
            return true;
        return sendWin32Click(MOUSEEVENTF_LEFTDOWN);
    }
    bool leftUp() override
    {
        if (!isOpen())
            return sendWin32Click(MOUSEEVENTF_LEFTUP);
        if (device_->release(0))
            return true;
        return sendWin32Click(MOUSEEVENTF_LEFTUP);
    }
    MakcuConnection* makcu() override { return device_.get(); }

private:
    std::unique_ptr<MakcuConnection> device_;
};
}

std::optional<MouseInputMethod> ParseMouseInputMethod(const std::string& method)
{
    if (method == "WIN32")
        return MouseInputMethod::Win32;
    if (method == "GHUB")
        return MouseInputMethod::GHub;
    if (method == "RAZER")
        return MouseInputMethod::Razer;
    if (method == "ARDUINO")
        return MouseInputMethod::Arduino;
    if (method == "KMBOX_NET")
        return MouseInputMethod::KmboxNet;
    if (method == "KMBOX_A")
        return MouseInputMethod::KmboxA;
    if (method == "MAKCU")
        return MouseInputMethod::Makcu;
    return std::nullopt;
}

std::string MouseInputMethodName(MouseInputMethod method)
{
    switch (method)
    {
    case MouseInputMethod::GHub: return "GHUB";
    case MouseInputMethod::Razer: return "RAZER";
    case MouseInputMethod::Arduino: return "ARDUINO";
    case MouseInputMethod::KmboxNet: return "KMBOX_NET";
    case MouseInputMethod::KmboxA: return "KMBOX_A";
    case MouseInputMethod::Makcu: return "MAKCU";
    case MouseInputMethod::Win32:
    default:
        return "WIN32";
    }
}

std::unique_ptr<IMouseInput> CreateMouseInputDevice(const Config& config)
{
    const MouseInputMethod method = ParseMouseInputMethod(config.input_method).value_or(MouseInputMethod::Win32);
    switch (method)
    {
    case MouseInputMethod::Arduino:
        return std::make_unique<ArduinoMouseInput>(config.arduino_port, static_cast<unsigned int>(config.arduino_baudrate));
    case MouseInputMethod::GHub:
        return std::make_unique<GHubMouseInput>();
    case MouseInputMethod::Razer:
        return std::make_unique<RazerMouseInput>();
    case MouseInputMethod::KmboxNet:
        return std::make_unique<KmboxNetMouseInput>(config.kmbox_net_ip, config.kmbox_net_port, config.kmbox_net_uuid);
    case MouseInputMethod::KmboxA:
        return std::make_unique<KmboxAMouseInput>(config.kmbox_a_pidvid);
    case MouseInputMethod::Makcu:
        return std::make_unique<MakcuMouseInput>(config.makcu_port, static_cast<unsigned int>(config.makcu_baudrate));
    case MouseInputMethod::Win32:
    default:
        return std::make_unique<Win32MouseInput>();
    }
}
