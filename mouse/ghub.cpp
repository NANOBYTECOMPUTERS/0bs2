#include <iostream>
#include <string>

#include "ghub.h"

GhubMouse::GhubMouse()
{
    char buffer[MAX_PATH];
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    basedir = std::filesystem::path(buffer).parent_path();
    dlldir = basedir / "ghub_mouse.dll";
    gm = LoadLibraryA(dlldir.string().c_str());
    if (gm == NULL)
    {
        std::cerr << "[Ghub] Failed to load DLL" << std::endl;
        gmok = false;
    }
    else
    {
        auto mouse_open = reinterpret_cast<bool(*)()>(GetProcAddress(gm, "mouse_open"));
        if (mouse_open == NULL)
        {
            std::cerr << "[Ghub] Failed to open mouse!" << std::endl;
            gmok = false;
        }
        else
        {
            gmok = mouse_open();
        }
    }
}

GhubMouse::~GhubMouse()
{
    if (gm != NULL)
    {
        FreeLibrary(gm);
    }
}

bool GhubMouse::mouse_xy(int x, int y)
{
    if (!gmok)
        return false;

    auto moveR = reinterpret_cast<bool(*)(int, int)>(GetProcAddress(gm, "moveR"));
    return moveR != NULL && moveR(x, y);
}

bool GhubMouse::mouse_down(int key)
{
    if (!gmok)
        return false;

    auto press = reinterpret_cast<bool(*)(int)>(GetProcAddress(gm, "press"));
    return press != NULL && press(key);
}

bool GhubMouse::mouse_up(int key)
{
    if (!gmok)
        return false;

    auto release = reinterpret_cast<bool(*)()>(GetProcAddress(gm, "release"));
    return release != NULL && release();
}

bool GhubMouse::mouse_close()
{
    if (gmok)
    {
        auto mouse_close = reinterpret_cast<bool(*)()>(GetProcAddress(gm, "mouse_close"));
        if (mouse_close != NULL)
        {
            return mouse_close();
        }
    }
    return false;
}
