#ifndef MOUSE_INPUT_H
#define MOUSE_INPUT_H

#include <memory>
#include <optional>
#include <string>

class Arduino;
class Config;
class GhubMouse;
class KmboxAConnection;
class KmboxNetConnection;
class MakcuConnection;
class RzctlMouse;

enum class MouseInputMethod
{
    Win32,
    GHub,
    Razer,
    Arduino,
    KmboxNet,
    KmboxA,
    Makcu
};

std::optional<MouseInputMethod> ParseMouseInputMethod(const std::string& method);
std::string MouseInputMethodName(MouseInputMethod method);

class IMouseInput
{
public:
    virtual ~IMouseInput() = default;

    virtual const char* name() const = 0;
    virtual bool isOpen() const = 0;
    virtual bool move(int dx, int dy) = 0;
    virtual bool leftDown() = 0;
    virtual bool leftUp() = 0;

    virtual Arduino* arduino() { return nullptr; }
    virtual GhubMouse* ghub() { return nullptr; }
    virtual RzctlMouse* razer() { return nullptr; }
    virtual KmboxNetConnection* kmboxNet() { return nullptr; }
    virtual KmboxAConnection* kmboxA() { return nullptr; }
    virtual MakcuConnection* makcu() { return nullptr; }
};

std::unique_ptr<IMouseInput> CreateMouseInputDevice(const Config& config);

#endif // MOUSE_INPUT_H
