#ifndef RZCTL_H
#define RZCTL_H

#include <filesystem>
#include <windows.h>

class RzctlMouse
{
public:
    RzctlMouse();
    ~RzctlMouse();

    bool isOpen() const { return rzctlOk; }
    bool mouse_xy(int x, int y);
    bool mouse_down(int key = 1);
    bool mouse_up(int key = 1);
    bool mouse_close();

private:
    using InitFn = bool (*)();
    using MouseMoveFn = void (*)(int, int, bool);
    using MouseMoveStatusFn = BOOL (*)(int, int, BOOL);
    using MouseClickFn = void (*)(int);
    using MouseClickStatusFn = BOOL (*)(int);
    using KeyboardInputFn = void (*)(short, int);

    std::filesystem::path dllPath;
    HMODULE rzctl = nullptr;
    bool rzctlOk = false;

    InitFn init = nullptr;
    MouseMoveFn mouseMove = nullptr;
    MouseMoveStatusFn mouseMoveStatus = nullptr;
    MouseClickFn mouseClick = nullptr;
    MouseClickStatusFn mouseClickStatus = nullptr;
    KeyboardInputFn keyboardInput = nullptr;

    static std::filesystem::path resolveDllPath();
    static bool sendInputMove(int x, int y);
    static bool sendInputClick(DWORD flag);
    static int downFlagForKey(int key);
    static int upFlagForKey(int key);
};

#endif // RZCTL_H
