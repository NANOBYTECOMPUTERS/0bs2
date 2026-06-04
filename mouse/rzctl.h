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
    // NOTE: All loader-related members (HMODULE, function pointers, dllPath) have been
    // removed. The implementation is now fully in-process / direct to the RZCONTROL
    // device (stealth-hardened: dynamic NT resolution, runtime string/IOCTL hiding,
    // lazy device open). The public API surface is unchanged.

    bool rzctlOk = false;   // reflects last known success of a send

    // Local mapping helpers (kept as before for exact behavioral compatibility).
    static int downFlagForKey(int key);
    static int upFlagForKey(int key);
};

#endif // RZCTL_H
