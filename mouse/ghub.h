#ifndef GHUB_H
#define GHUB_H

#include <filesystem>
#include <windows.h>

class GhubMouse
{
public:
    GhubMouse();
    ~GhubMouse();
    bool isOpen() const { return gmok; }
    bool mouse_xy(int x, int y);
    bool mouse_down(int key = 1);
    bool mouse_up(int key = 1);
    bool mouse_close();

private:
    std::filesystem::path basedir;
    std::filesystem::path dlldir;
    HMODULE gm;
    bool gmok;
};

#endif // GHUB_H
