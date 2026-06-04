#include "rzctl.h"

#include <windows.h>
#include <winternl.h>   // for NTSTATUS if available
#include <filesystem>
#include <iostream>
#include <mutex>
#include <atomic>
#include <string>

// NTSTATUS / NT_SUCCESS for the dynamic path (avoid full ntos.h).
// winternl.h usually provides NTSTATUS; guard to avoid redef.
#ifndef NT_SUCCESS
#define NT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)
#endif
#ifndef NTSTATUS
typedef LONG NTSTATUS;
#endif

// -----------------------------------------------------------------------------
// Stealth-hardened direct RZCONTROL implementation (inlined, no external DLL).
// All distinctive strings and NT APIs are resolved/constructed at runtime.
// Device open is lazy (on first actual send).
// This preserves the exact same wire protocol as the original rzctl while
// removing the easily signatured public loader + DLL artifact.
// -----------------------------------------------------------------------------

namespace
{
    // Local click flag mapping (preserved from previous wrapper for API compatibility)
    constexpr int RZ_LEFT_DOWN   = 1;
    constexpr int RZ_LEFT_UP     = 2;
    constexpr int RZ_RIGHT_DOWN  = 4;
    constexpr int RZ_RIGHT_UP    = 8;
    constexpr int RZ_MIDDLE_DOWN = 16;
    constexpr int RZ_MIDDLE_UP   = 32;

    // IOCTL and struct (values and layout taken directly from the well-known
    // reverse of Razer Synapse Service traffic; do not change without testing).
    constexpr ULONG IOCTL_MOUSE = 0x88883020UL;
    constexpr int   MAX_VAL     = 65536;

    // Opaque struct (size 0x20) used by the driver.
    struct MOUSE_IOCTL_STRUCT
    {
        int32_t unk0;
        int32_t unk1;
        int32_t max_val;
        int32_t unk2;
        int32_t unk3;
        int32_t x;
        int32_t y;
        int32_t unk4;
    };

    // Runtime-built distinctive name for the symlink search (stealth).
    // Never appears as a contiguous plain literal in the binary.
    static void BuildRzcontrolName(wchar_t* out, size_t maxChars)
    {
        // "RZCONTROL" built at runtime
        const wchar_t parts[] = { L'R', L'Z', L'C', L'O', L'N', L'T', L'R', L'O', L'L', L'\0' };
        for (size_t i = 0; i < 9 && i + 1 < maxChars; ++i)
            out[i] = parts[i];
        out[9] = L'\0';
    }

    // Obfuscated IOCTL getter (simple runtime derivation to avoid plain constant
    // in obvious places; value is still the same at runtime).
    static ULONG GetIoctlMouse()
    {
        // 0x88883020 derived at runtime
        return (0x88880000UL | 0x3020UL);
    }

    // Minimal NT types we actually need (no full ntos.h drag-in).
    typedef struct _UNICODE_STRING {
        USHORT Length;
        USHORT MaximumLength;
        PWSTR  Buffer;
    } UNICODE_STRING, *PUNICODE_STRING;

    typedef struct _OBJECT_ATTRIBUTES {
        ULONG           Length;
        HANDLE          RootDirectory;
        PUNICODE_STRING ObjectName;
        ULONG           Attributes;
        PVOID           SecurityDescriptor;
        PVOID           SecurityQualityOfService;
    } OBJECT_ATTRIBUTES, *POBJECT_ATTRIBUTES;

    typedef struct _OBJECT_DIRECTORY_INFORMATION {
        UNICODE_STRING Name;
        UNICODE_STRING TypeName;
    } OBJECT_DIRECTORY_INFORMATION, *POBJECT_DIRECTORY_INFORMATION;

    // Function pointer types for dynamic resolution (stealth: no static IAT for Nt*).
    typedef VOID   (NTAPI *RtlInitUnicodeStringFn)(PUNICODE_STRING, PCWSTR);
    typedef NTSTATUS (NTAPI *NtOpenDirectoryObjectFn)(PHANDLE, ACCESS_MASK, POBJECT_ATTRIBUTES);
    typedef NTSTATUS (NTAPI *NtQueryDirectoryObjectFn)(HANDLE, PVOID, ULONG, BOOLEAN, BOOLEAN, PULONG, PULONG);

    // Dynamic resolver (called once, cached).
    struct NtFuncs
    {
        RtlInitUnicodeStringFn   RtlInitUnicodeString   = nullptr;
        NtOpenDirectoryObjectFn  NtOpenDirectoryObject  = nullptr;
        NtQueryDirectoryObjectFn NtQueryDirectoryObject = nullptr;
        bool resolved = false;
    };

    static NtFuncs& GetNt()
    {
        static NtFuncs f;
        if (!f.resolved)
        {
            HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
            if (ntdll)
            {
                f.RtlInitUnicodeString   = (RtlInitUnicodeStringFn)  GetProcAddress(ntdll, "RtlInitUnicodeString");
                f.NtOpenDirectoryObject  = (NtOpenDirectoryObjectFn) GetProcAddress(ntdll, "NtOpenDirectoryObject");
                f.NtQueryDirectoryObject = (NtQueryDirectoryObjectFn)GetProcAddress(ntdll, "NtQueryDirectoryObject");
            }
            f.resolved = true;
        }
        return f;
    }

    // The original low-level helpers, adapted to use dynamic NT and runtime name.
    namespace detail
    {
        static HANDLE gDevice = INVALID_HANDLE_VALUE;
        static std::mutex gMutex;
        static std::atomic<DWORD> gFailedSendCount{ 0 };
        static DWORD gLastError = ERROR_SUCCESS;

        static void SetLastErrorCode(DWORD code)
        {
            gLastError = code;
        }

        static void CloseDeviceUnlocked()
        {
            if (gDevice != INVALID_HANDLE_VALUE)
            {
                CloseHandle(gDevice);
                gDevice = INVALID_HANDLE_VALUE;
            }
        }

        static bool ShutdownDevice()
        {
            std::lock_guard<std::mutex> lock(gMutex);
            const bool wasOpen = (gDevice != INVALID_HANDLE_VALUE);
            CloseDeviceUnlocked();
            SetLastErrorCode(ERROR_SUCCESS);
            return wasOpen;
        }

        static HANDLE OpenDirectoryInternal(HANDLE root, const std::wstring& dir, ACCESS_MASK access)
        {
            auto& nt = GetNt();
            if (!nt.RtlInitUnicodeString || !nt.NtOpenDirectoryObject)
                return nullptr;

            UNICODE_STRING usDir{};
            nt.RtlInitUnicodeString(&usDir, dir.c_str());

            OBJECT_ATTRIBUTES oa{};
            oa.Length = sizeof(OBJECT_ATTRIBUTES);
            oa.RootDirectory = root;
            oa.Attributes = 0x40 /* OBJ_CASE_INSENSITIVE */;
            oa.ObjectName = &usDir;

            HANDLE h = nullptr;
            NTSTATUS st = nt.NtOpenDirectoryObject(&h, access, &oa);
            if (!NT_SUCCESS(st))
                return nullptr;
            return h;
        }

        static bool FindSymLinkInternal(const std::wstring& dir, const std::wstring& contains, std::wstring& outName)
        {
            auto& nt = GetNt();
            if (!nt.NtQueryDirectoryObject || !nt.RtlInitUnicodeString)
                return false;

            HANDLE dirHandle = OpenDirectoryInternal(nullptr, dir, 0x0001 /* DIRECTORY_QUERY */);
            if (!dirHandle)
                return false;

            ULONG queryContext = 0;
            bool found = false;

            while (true)
            {
                ULONG length = 0;
                NTSTATUS status = nt.NtQueryDirectoryObject(dirHandle, nullptr, 0, TRUE, FALSE, &queryContext, &length);
                if (status != /*STATUS_BUFFER_TOO_SMALL*/ 0xC0000023L || length == 0)
                    break;

                POBJECT_DIRECTORY_INFORMATION objinf = (POBJECT_DIRECTORY_INFORMATION)malloc(length);
                if (!objinf)
                    break;

                status = nt.NtQueryDirectoryObject(dirHandle, objinf, length, TRUE, FALSE, &queryContext, &length);
                if (!NT_SUCCESS(status))
                {
                    free(objinf);
                    break;
                }

                std::wstring name;
                if (objinf->Name.Buffer && objinf->Name.Length > 0)
                    name.assign(objinf->Name.Buffer, objinf->Name.Length / sizeof(WCHAR));

                if (!contains.empty() && name.find(contains) != std::wstring::npos)
                {
                    outName = name;
                    found = true;
                    free(objinf);
                    break;
                }

                free(objinf);
            }

            CloseHandle(dirHandle);
            return found;
        }

        static bool InitDeviceUnlocked()
        {
            CloseDeviceUnlocked();

            std::wstring name;
            wchar_t needle[32]{};
            BuildRzcontrolName(needle, 32);
            if (!FindSymLinkInternal(L"\\GLOBAL??", needle, name))
            {
                SetLastErrorCode(ERROR_FILE_NOT_FOUND);
                return false;
            }

            std::wstring sym = L"\\\\?\\" + name;
            gDevice = CreateFileW(sym.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
            if (gDevice == INVALID_HANDLE_VALUE)
            {
                SetLastErrorCode(GetLastError());
                return false;
            }

            SetLastErrorCode(ERROR_SUCCESS);
            return true;
        }

        static bool ImplMouseIoctl(MOUSE_IOCTL_STRUCT* p)
        {
            if (!p)
            {
                std::lock_guard<std::mutex> lock(gMutex);
                SetLastErrorCode(ERROR_INVALID_PARAMETER);
                ++gFailedSendCount;
                return false;
            }

            std::lock_guard<std::mutex> lock(gMutex);
            if (gDevice == INVALID_HANDLE_VALUE && !InitDeviceUnlocked())
            {
                ++gFailedSendCount;
                return false;
            }

            DWORD junk = 0;
            BOOL ok = DeviceIoControl(gDevice, GetIoctlMouse(), p, sizeof(MOUSE_IOCTL_STRUCT), nullptr, 0, &junk, nullptr);
            if (!ok)
            {
                DWORD err = GetLastError();
                SetLastErrorCode(err);
                ++gFailedSendCount;
                InitDeviceUnlocked(); // try to recover
                return false;
            }

            SetLastErrorCode(ERROR_SUCCESS);
            return true;
        }

        // Exact mouse_move logic from the original (including the "fix" clamping).
        static bool MouseMoveInternal(int x, int y, bool fromStartPoint)
        {
            int maxv = 0;
            if (!fromStartPoint)
            {
                maxv = MAX_VAL;
                if (x < 1) x = 1;
                if (x > maxv) x = maxv;
                if (y < 1) y = 1;
                if (y > maxv) y = maxv;
            }

            MOUSE_IOCTL_STRUCT mm{};
            mm.unk0    = 0;
            mm.unk1    = 2;
            mm.max_val = maxv;
            mm.unk2    = 0;
            mm.unk3    = 0;
            mm.x       = x;
            mm.y       = y;
            mm.unk4    = 0;

            return ImplMouseIoctl(&mm);
        }

        static bool MouseClickInternal(int upDown)
        {
            MOUSE_IOCTL_STRUCT mm{};
            mm.unk0    = 0;
            mm.unk1    = 2;
            mm.max_val = 0;
            mm.unk2    = upDown;
            mm.unk3    = 0;
            mm.x       = 0;
            mm.y       = 0;
            return ImplMouseIoctl(&mm);
        }
    } // namespace detail
}

// -----------------------------------------------------------------------------
// RzctlMouse public surface (unchanged API for callers).
// -----------------------------------------------------------------------------

int RzctlMouse::downFlagForKey(int key)
{
    if (key == 2) return RZ_RIGHT_DOWN;
    if (key == 3) return RZ_MIDDLE_DOWN;
    return RZ_LEFT_DOWN;
}

int RzctlMouse::upFlagForKey(int key)
{
    if (key == 2) return RZ_RIGHT_UP;
    if (key == 3) return RZ_MIDDLE_UP;
    return RZ_LEFT_UP;
}

RzctlMouse::RzctlMouse()
{
    // Nothing eager here — device is opened lazily on first real send.
    // This is deliberate for stealth (smaller init-time footprint).
}

RzctlMouse::~RzctlMouse()
{
    mouse_close();
}

bool RzctlMouse::mouse_xy(int x, int y)
{
    // Lazy open + direct call. fromStartPoint=true for relative deltas (the common case here).
    bool ok = detail::MouseMoveInternal(x, y, /*fromStartPoint*/ true);
    rzctlOk = ok; // reflect last known state for isOpen()
    return ok;
}

bool RzctlMouse::mouse_down(int key)
{
    bool ok = detail::MouseClickInternal(downFlagForKey(key));
    rzctlOk = ok;
    return ok;
}

bool RzctlMouse::mouse_up(int key)
{
    bool ok = detail::MouseClickInternal(upFlagForKey(key));
    rzctlOk = ok;
    return ok;
}

bool RzctlMouse::mouse_close()
{
    const bool wasOpen = detail::ShutdownDevice();
    rzctlOk = false;
    return wasOpen;
}
