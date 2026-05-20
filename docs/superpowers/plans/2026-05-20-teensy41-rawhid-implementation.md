# Teensy 4.1 Generic HID RawHID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `TEENSY41_HID` input method that sends mouse movement/buttons through RawHID with no COM port, while leaving `ARDUINO` and serial `TEENSY41` unchanged.

**Architecture:** Add a focused RawHID bridge class beside the existing mouse input devices, then expose it through `IMouseInput` and the keyboard listener's physical button-state path. Store generic HID defaults in config/GUI, and put the firmware in a separate sketch folder so serial Teensy remains a rollback option.

**Tech Stack:** C++17, Win32, existing `mouse/hidapi.h`/`mouse/hid.c`, ImGui, Python `unittest`, Teensy Arduino `USBHost_t36` + RawHID.

---

## File Structure

- Create `mouse/Teensy41RawHid.h`: packet constants, fixed 64-byte packet structs, RAII class declaration, physical button state accessors.
- Create `mouse/Teensy41RawHid.cpp`: hidapi open/filter logic, send movement/button packets, read thread, physical transition mapping.
- Modify `mouse/MouseInput.h`: add `Teensy41Hid` enum and optional generic button-state methods on `IMouseInput`.
- Modify `mouse/MouseInput.cpp`: add `Teensy41RawHidMouseInput`, parse/name/factory wiring.
- Modify `keyboard/keyboard_listener.cpp`: read optional physical button state from `activeMouseInputOwner`, not only `arduinoSerial`.
- Modify `0BS_box_2.cpp` and `0BS_box_2.h`: expose `activeMouseInputOwner` for keyboard listener use and refresh HID settings on config reload.
- Modify `config/config.h` and `config/config.cpp`: add RawHID/default descriptor settings, load/save/merge fields.
- Modify `overlay/draw_mouse.cpp`: add `TEENSY41_HID` selection, hide serial controls for it, show Generic HID advanced defaults with reset buttons.
- Modify `0BS_box_2.vcxproj` and `cuda/0BS_cuda.vcxproj`: include new RawHID C++/header files.
- Create `TeensyMouseRawHidBridge/TeensyMouseRawHidBridge.ino`: no-Serial RawHID firmware sketch with generic identity constants and physical mouse passthrough.
- Create/modify `training/tests/test_teensy41_hid_contract.py`: contract tests for all new wiring.
- Keep `training/tests/test_teensy41_mouse_contract.py`: serial Teensy tests must keep passing.

---

### Task 1: Contract Tests For RawHID Wiring

**Files:**
- Create: `training/tests/test_teensy41_hid_contract.py`

- [ ] **Step 1: Write failing contract tests**

Create `training/tests/test_teensy41_hid_contract.py`:

```python
import unittest
from pathlib import Path


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "mouse" / "MouseInput.cpp").exists()
)


class Teensy41HidContractTests(unittest.TestCase):
    def read(self, relative_path):
        return (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    def test_teensy41_hid_input_method_is_wired(self):
        config_h = self.read("config/config.h")
        config_cpp = self.read("config/config.cpp")
        mouse_h = self.read("mouse/MouseInput.h")
        mouse_cpp = self.read("mouse/MouseInput.cpp")
        draw_mouse = self.read("overlay/draw_mouse.cpp")

        self.assertIn("Teensy41Hid", mouse_h)
        self.assertIn('method == "TEENSY41_HID"', mouse_cpp)
        self.assertIn('return "TEENSY41_HID"', mouse_cpp)
        self.assertIn("Teensy41RawHidMouseInput", mouse_cpp)
        self.assertIn('"TEENSY41_HID"', draw_mouse)
        self.assertIn("teensy_hid_manufacturer", config_h)
        self.assertIn("teensy_hid_usage_page", config_h)
        self.assertIn("teensy_hid_vid_filter", config_h)
        self.assertIn("MERGE_FIELD(\"teensy_hid_manufacturer\"", config_cpp)

    def test_rawhid_packet_contract_is_fixed_64_bytes(self):
        raw_h = self.read("mouse/Teensy41RawHid.h")

        self.assertIn("constexpr size_t Teensy41RawHidPacketSize = 64", raw_h)
        self.assertIn("struct Teensy41RawHidHostPacket", raw_h)
        self.assertIn("struct Teensy41RawHidDevicePacket", raw_h)
        self.assertIn("static_assert(sizeof(Teensy41RawHidHostPacket) == Teensy41RawHidPacketSize", raw_h)
        self.assertIn("static_assert(sizeof(Teensy41RawHidDevicePacket) == Teensy41RawHidPacketSize", raw_h)
        self.assertIn("int16_t dx", raw_h)
        self.assertIn("uint8_t buttonMask", raw_h)

    def test_generic_button_state_is_available_without_arduino_serial(self):
        mouse_h = self.read("mouse/MouseInput.h")
        keyboard_cpp = self.read("keyboard/keyboard_listener.cpp")

        self.assertIn("virtual bool aimingActive() const", mouse_h)
        self.assertIn("virtual bool shootingActive() const", mouse_h)
        self.assertIn("virtual bool zoomingActive() const", mouse_h)
        self.assertIn("activeMouseInputOwner && activeMouseInputOwner->isOpen()", keyboard_cpp)
        self.assertIn("activeMouseInputOwner->aimingActive()", keyboard_cpp)
        self.assertIn("activeMouseInputOwner->shootingActive()", keyboard_cpp)
        self.assertIn("activeMouseInputOwner->zoomingActive()", keyboard_cpp)

    def test_gui_hides_serial_controls_and_exposes_hid_defaults(self):
        draw_mouse = self.read("overlay/draw_mouse.cpp")

        self.assertIn('config.input_method == "TEENSY41_HID"', draw_mouse)
        self.assertIn("Generic HID Mouse", draw_mouse)
        self.assertIn("Manufacturer", draw_mouse)
        self.assertIn("Product", draw_mouse)
        self.assertIn("VID filter", draw_mouse)
        self.assertIn("PID filter", draw_mouse)
        self.assertIn("Reset HID defaults", draw_mouse)

    def test_firmware_uses_rawhid_and_not_serial_commands(self):
        sketch = self.read("TeensyMouseRawHidBridge/TeensyMouseRawHidBridge.ino")

        self.assertIn("#include <USBHost_t36.h>", sketch)
        self.assertIn("RawHID.recv", sketch)
        self.assertIn("RawHID.send", sketch)
        self.assertIn("USB HID Mouse", sketch)
        self.assertIn("applyButtons()", sketch)
        self.assertIn("emitButtonTransition", sketch)
        self.assertNotIn("Serial.available()", sketch)
        self.assertNotIn('sscanf(line, "move', sketch)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
python -m unittest training.tests.test_teensy41_hid_contract
```

Expected: failure because `mouse/Teensy41RawHid.h`, `TEENSY41_HID`, and the firmware sketch do not exist.

- [ ] **Step 3: Commit failing tests**

```powershell
git add training/tests/test_teensy41_hid_contract.py
git commit -m "test: add teensy rawhid contracts"
```

---

### Task 2: Config Defaults And Persistence

**Files:**
- Modify: `config/config.h`
- Modify: `config/config.cpp`

- [ ] **Step 1: Add config fields**

Add fields under the Arduino/input device settings in `config/config.h`:

```cpp
// Teensy 4.1 RawHID generic mouse bridge
std::string teensy_hid_manufacturer;
std::string teensy_hid_product;
std::string teensy_hid_serial;
std::string teensy_hid_vid_filter;
std::string teensy_hid_pid_filter;
int teensy_hid_usage_page;
int teensy_hid_usage_id;
int teensy_hid_open_index;
int teensy_hid_packet_timeout_ms;
int teensy_hid_reconnect_interval_ms;
```

- [ ] **Step 2: Add defaults in both default config creation and normal load**

In `config/config.cpp`, set these defaults in the block that creates a new config and in the normal load path:

```cpp
teensy_hid_manufacturer = get_string("teensy_hid_manufacturer", "Generic");
teensy_hid_product = get_string("teensy_hid_product", "USB HID Mouse");
teensy_hid_serial = get_string("teensy_hid_serial", "AUTO");
teensy_hid_vid_filter = get_string("teensy_hid_vid_filter", "AUTO");
teensy_hid_pid_filter = get_string("teensy_hid_pid_filter", "AUTO");
teensy_hid_usage_page = get_long("teensy_hid_usage_page", 0xFFAB);
teensy_hid_usage_id = get_long("teensy_hid_usage_id", 0x0200);
teensy_hid_open_index = get_long("teensy_hid_open_index", 0);
teensy_hid_packet_timeout_ms = get_long("teensy_hid_packet_timeout_ms", 2);
teensy_hid_reconnect_interval_ms = get_long("teensy_hid_reconnect_interval_ms", 500);
```

Use literal assignments in the "file does not exist" default block:

```cpp
teensy_hid_manufacturer = "Generic";
teensy_hid_product = "USB HID Mouse";
teensy_hid_serial = "AUTO";
teensy_hid_vid_filter = "AUTO";
teensy_hid_pid_filter = "AUTO";
teensy_hid_usage_page = 0xFFAB;
teensy_hid_usage_id = 0x0200;
teensy_hid_open_index = 0;
teensy_hid_packet_timeout_ms = 2;
teensy_hid_reconnect_interval_ms = 500;
```

- [ ] **Step 3: Clamp values in `validate()`**

Add:

```cpp
if (teensy_hid_manufacturer.empty()) teensy_hid_manufacturer = "Generic";
if (teensy_hid_product.empty()) teensy_hid_product = "USB HID Mouse";
if (teensy_hid_serial.empty()) teensy_hid_serial = "AUTO";
if (teensy_hid_vid_filter.empty()) teensy_hid_vid_filter = "AUTO";
if (teensy_hid_pid_filter.empty()) teensy_hid_pid_filter = "AUTO";
teensy_hid_usage_page = std::clamp(teensy_hid_usage_page, 1, 0xFFFF);
teensy_hid_usage_id = std::clamp(teensy_hid_usage_id, 1, 0xFFFF);
teensy_hid_open_index = std::clamp(teensy_hid_open_index, 0, 32);
teensy_hid_packet_timeout_ms = std::clamp(teensy_hid_packet_timeout_ms, 0, 100);
teensy_hid_reconnect_interval_ms = std::clamp(teensy_hid_reconnect_interval_ms, 50, 10000);
```

- [ ] **Step 4: Add merge and save fields**

Add to `loadConfigMerged()`:

```cpp
MERGE_FIELD("teensy_hid_manufacturer", teensy_hid_manufacturer);
MERGE_FIELD("teensy_hid_product", teensy_hid_product);
MERGE_FIELD("teensy_hid_serial", teensy_hid_serial);
MERGE_FIELD("teensy_hid_vid_filter", teensy_hid_vid_filter);
MERGE_FIELD("teensy_hid_pid_filter", teensy_hid_pid_filter);
MERGE_FIELD("teensy_hid_usage_page", teensy_hid_usage_page);
MERGE_FIELD("teensy_hid_usage_id", teensy_hid_usage_id);
MERGE_FIELD("teensy_hid_open_index", teensy_hid_open_index);
MERGE_FIELD("teensy_hid_packet_timeout_ms", teensy_hid_packet_timeout_ms);
MERGE_FIELD("teensy_hid_reconnect_interval_ms", teensy_hid_reconnect_interval_ms);
```

Add to `saveConfig()` near Arduino settings:

```cpp
file << "# Teensy 4.1 RawHID generic mouse bridge\n"
     << "teensy_hid_manufacturer = " << teensy_hid_manufacturer << "\n"
     << "teensy_hid_product = " << teensy_hid_product << "\n"
     << "teensy_hid_serial = " << teensy_hid_serial << "\n"
     << "teensy_hid_vid_filter = " << teensy_hid_vid_filter << "\n"
     << "teensy_hid_pid_filter = " << teensy_hid_pid_filter << "\n"
     << "teensy_hid_usage_page = " << teensy_hid_usage_page << "\n"
     << "teensy_hid_usage_id = " << teensy_hid_usage_id << "\n"
     << "teensy_hid_open_index = " << teensy_hid_open_index << "\n"
     << "teensy_hid_packet_timeout_ms = " << teensy_hid_packet_timeout_ms << "\n"
     << "teensy_hid_reconnect_interval_ms = " << teensy_hid_reconnect_interval_ms << "\n\n";
```

- [ ] **Step 5: Update input method validation text**

Change comments and emitted config text from:

```cpp
// "WIN32", "GHUB", "RAZER", "ARDUINO", "TEENSY41", "KMBOX_NET", "KMBOX_A", "MAKCU"
```

to:

```cpp
// "WIN32", "GHUB", "RAZER", "ARDUINO", "TEENSY41", "TEENSY41_HID", "KMBOX_NET", "KMBOX_A", "MAKCU"
```

- [ ] **Step 6: Run config-focused contracts**

Run:

```powershell
python -m unittest training.tests.test_teensy41_hid_contract training.tests.test_config_gui_control_contract
```

Expected: config assertions pass; GUI and RawHID assertions still fail until Task 3 and Task 5 are complete.

- [ ] **Step 7: Commit config changes**

```powershell
git add config/config.h config/config.cpp
git commit -m "feat: add teensy rawhid config defaults"
```

---

### Task 3: RawHID Packet And Device Class

**Files:**
- Create: `mouse/Teensy41RawHid.h`
- Create: `mouse/Teensy41RawHid.cpp`

- [ ] **Step 1: Create packet header and class declaration**

Create `mouse/Teensy41RawHid.h`:

```cpp
#ifndef TEENSY41_RAWHID_H
#define TEENSY41_RAWHID_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

#include "hidapi.h"

class Config;

constexpr size_t Teensy41RawHidPacketSize = 64;
constexpr uint16_t Teensy41RawHidHostMagic = 0x3448;
constexpr uint16_t Teensy41RawHidDeviceMagic = 0x4834;
constexpr uint8_t Teensy41RawHidVersion = 1;

enum class Teensy41RawHidCommand : uint8_t
{
    Move = 1,
    Buttons = 2,
    Ping = 3,
    Status = 4
};

enum class Teensy41RawHidEvent : uint8_t
{
    Button = 1,
    Pong = 2,
    Status = 3,
    Error = 4
};

#pragma pack(push, 1)
struct Teensy41RawHidHostPacket
{
    uint16_t magic = Teensy41RawHidHostMagic;
    uint8_t version = Teensy41RawHidVersion;
    uint8_t command = 0;
    uint8_t buttonMask = 0;
    int16_t dx = 0;
    int16_t dy = 0;
    int16_t wheel = 0;
    int16_t wheelH = 0;
    uint32_t sequence = 0;
    uint8_t reserved[49] = {};
};

struct Teensy41RawHidDevicePacket
{
    uint16_t magic = Teensy41RawHidDeviceMagic;
    uint8_t version = Teensy41RawHidVersion;
    uint8_t event = 0;
    uint8_t buttonId = 0;
    uint8_t pressed = 0;
    uint8_t hostButtonMask = 0;
    uint32_t sequenceAck = 0;
    uint8_t reserved[53] = {};
};
#pragma pack(pop)

static_assert(sizeof(Teensy41RawHidHostPacket) == Teensy41RawHidPacketSize, "Host RawHID packet must be 64 bytes");
static_assert(sizeof(Teensy41RawHidDevicePacket) == Teensy41RawHidPacketSize, "Device RawHID packet must be 64 bytes");

class Teensy41RawHid
{
public:
    explicit Teensy41RawHid(const Config& config);
    ~Teensy41RawHid();

    Teensy41RawHid(const Teensy41RawHid&) = delete;
    Teensy41RawHid& operator=(const Teensy41RawHid&) = delete;

    bool isOpen() const;
    bool move(int dx, int dy, int wheel = 0, int wheelH = 0);
    bool setButtons(uint8_t mask);
    bool leftDown();
    bool leftUp();

    bool aimingActive() const { return aiming_active.load(); }
    bool shootingActive() const { return shooting_active.load(); }
    bool zoomingActive() const { return zooming_active.load(); }

private:
    bool openFromConfig(const Config& config);
    bool writePacket(const Teensy41RawHidHostPacket& packet);
    void readLoop();
    void processPacket(const Teensy41RawHidDevicePacket& packet);
    void clearButtonState();

    hid_device* device_ = nullptr;
    std::atomic<bool> open_{ false };
    std::atomic<bool> reading_{ false };
    std::thread read_thread_;
    mutable std::mutex write_mutex_;
    uint8_t app_button_mask_ = 0;
    uint32_t sequence_ = 1;
    int packet_timeout_ms_ = 2;

    std::atomic<bool> aiming_active{ false };
    std::atomic<bool> shooting_active{ false };
    std::atomic<bool> zooming_active{ false };
};

#endif
```

- [ ] **Step 2: Create implementation**

Create `mouse/Teensy41RawHid.cpp`:

```cpp
#include "Teensy41RawHid.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "0BS_box_2.h"
#include "config.h"

namespace
{
bool parseHexOrAuto(const std::string& text, unsigned short& out)
{
    if (text.empty() || text == "AUTO")
        return false;

    char* end = nullptr;
    const unsigned long value = std::strtoul(text.c_str(), &end, 16);
    if (end == text.c_str() || value > 0xFFFF)
        return false;

    out = static_cast<unsigned short>(value);
    return true;
}
}

Teensy41RawHid::Teensy41RawHid(const Config& config)
{
    packet_timeout_ms_ = std::clamp(config.teensy_hid_packet_timeout_ms, 0, 100);
    open_ = openFromConfig(config);
    if (open_)
    {
        reading_ = true;
        read_thread_ = std::thread(&Teensy41RawHid::readLoop, this);
    }
}

Teensy41RawHid::~Teensy41RawHid()
{
    reading_ = false;
    if (read_thread_.joinable())
        read_thread_.join();
    if (device_)
    {
        hid_close(device_);
        device_ = nullptr;
    }
    clearButtonState();
    open_ = false;
}

bool Teensy41RawHid::isOpen() const
{
    return open_.load();
}

bool Teensy41RawHid::openFromConfig(const Config& config)
{
    unsigned short vid = 0;
    unsigned short pid = 0;
    const bool hasVid = parseHexOrAuto(config.teensy_hid_vid_filter, vid);
    const bool hasPid = parseHexOrAuto(config.teensy_hid_pid_filter, pid);
    const unsigned short usagePage = static_cast<unsigned short>(std::clamp(config.teensy_hid_usage_page, 1, 0xFFFF));
    const unsigned short usage = static_cast<unsigned short>(std::clamp(config.teensy_hid_usage_id, 1, 0xFFFF));
    const int wantedIndex = std::clamp(config.teensy_hid_open_index, 0, 32);

    hid_device_info* devices = hid_enumerate(hasVid ? vid : 0, hasPid ? pid : 0);
    hid_device_info* selected = nullptr;
    int matchIndex = 0;

    for (hid_device_info* cur = devices; cur; cur = cur->next)
    {
        if (cur->usage_page != usagePage || cur->usage != usage)
            continue;
        if (matchIndex == wantedIndex)
        {
            selected = cur;
            break;
        }
        ++matchIndex;
    }

    if (selected)
        device_ = hid_open_path(selected->path);

    hid_free_enumeration(devices);
    return device_ != nullptr;
}

bool Teensy41RawHid::writePacket(const Teensy41RawHidHostPacket& packet)
{
    if (!device_)
        return false;

    std::lock_guard<std::mutex> lock(write_mutex_);
    const int written = hid_write(device_, reinterpret_cast<const unsigned char*>(&packet), Teensy41RawHidPacketSize);
    if (written == static_cast<int>(Teensy41RawHidPacketSize))
        return true;

    open_ = false;
    return false;
}

bool Teensy41RawHid::move(int dx, int dy, int wheel, int wheelH)
{
    if (!isOpen() || (dx == 0 && dy == 0 && wheel == 0 && wheelH == 0))
        return false;

    Teensy41RawHidHostPacket packet;
    packet.command = static_cast<uint8_t>(Teensy41RawHidCommand::Move);
    packet.buttonMask = app_button_mask_;
    packet.dx = static_cast<int16_t>(std::clamp(dx, -32767, 32767));
    packet.dy = static_cast<int16_t>(std::clamp(dy, -32767, 32767));
    packet.wheel = static_cast<int16_t>(std::clamp(wheel, -32767, 32767));
    packet.wheelH = static_cast<int16_t>(std::clamp(wheelH, -32767, 32767));
    packet.sequence = sequence_++;
    return writePacket(packet);
}

bool Teensy41RawHid::setButtons(uint8_t mask)
{
    if (!isOpen())
        return false;

    app_button_mask_ = mask;
    Teensy41RawHidHostPacket packet;
    packet.command = static_cast<uint8_t>(Teensy41RawHidCommand::Buttons);
    packet.buttonMask = app_button_mask_;
    packet.sequence = sequence_++;
    return writePacket(packet);
}

bool Teensy41RawHid::leftDown()
{
    return setButtons(static_cast<uint8_t>(app_button_mask_ | 0x01));
}

bool Teensy41RawHid::leftUp()
{
    return setButtons(static_cast<uint8_t>(app_button_mask_ & static_cast<uint8_t>(~0x01)));
}

void Teensy41RawHid::readLoop()
{
    while (reading_ && device_)
    {
        Teensy41RawHidDevicePacket packet;
        const int read = hid_read_timeout(
            device_,
            reinterpret_cast<unsigned char*>(&packet),
            Teensy41RawHidPacketSize,
            packet_timeout_ms_);

        if (read == static_cast<int>(Teensy41RawHidPacketSize))
        {
            processPacket(packet);
        }
        else if (read < 0)
        {
            open_ = false;
            clearButtonState();
            break;
        }
    }
}

void Teensy41RawHid::processPacket(const Teensy41RawHidDevicePacket& packet)
{
    if (packet.magic != Teensy41RawHidDeviceMagic || packet.version != Teensy41RawHidVersion)
        return;
    if (packet.event != static_cast<uint8_t>(Teensy41RawHidEvent::Button))
        return;

    const bool pressed = packet.pressed != 0;
    switch (packet.buttonId)
    {
    case 1:
        shooting_active.store(pressed);
        shooting.store(pressed);
        break;
    case 2:
        zooming_active.store(pressed);
        zooming.store(pressed);
        break;
    case 5:
        aiming_active.store(pressed);
        aiming.store(pressed);
        break;
    default:
        break;
    }
}

void Teensy41RawHid::clearButtonState()
{
    aiming_active.store(false);
    shooting_active.store(false);
    zooming_active.store(false);
}
```

- [ ] **Step 3: Run packet contract**

Run:

```powershell
python -m unittest training.tests.test_teensy41_hid_contract
```

Expected: packet contract passes; input method, GUI, and firmware checks still fail.

- [ ] **Step 4: Commit RawHID class**

```powershell
git add mouse/Teensy41RawHid.h mouse/Teensy41RawHid.cpp
git commit -m "feat: add teensy rawhid packet bridge"
```

---

### Task 4: Mouse Input And Button-State Integration

**Files:**
- Modify: `mouse/MouseInput.h`
- Modify: `mouse/MouseInput.cpp`
- Modify: `0BS_box_2.h`
- Modify: `0BS_box_2.cpp`
- Modify: `keyboard/keyboard_listener.cpp`

- [ ] **Step 1: Extend `IMouseInput`**

In `mouse/MouseInput.h`, add the forward declaration and enum entry:

```cpp
class Teensy41RawHid;
```

```cpp
Teensy41Hid,
```

Add optional methods to `IMouseInput`:

```cpp
virtual Teensy41RawHid* teensy41RawHid() { return nullptr; }
virtual bool aimingActive() const { return false; }
virtual bool shootingActive() const { return false; }
virtual bool zoomingActive() const { return false; }
```

- [ ] **Step 2: Add `Teensy41RawHidMouseInput`**

In `mouse/MouseInput.cpp`, include the new header:

```cpp
#include "Teensy41RawHid.h"
```

Add the wrapper class near `Teensy41MouseInput`:

```cpp
class Teensy41RawHidMouseInput final : public IMouseInput
{
public:
    explicit Teensy41RawHidMouseInput(const Config& config)
        : device_(std::make_unique<Teensy41RawHid>(config))
    {
    }

    const char* name() const override { return "TEENSY41_HID"; }
    bool isOpen() const override { return device_ && device_->isOpen(); }
    bool move(int dx, int dy) override
    {
        if (!isOpen())
            return sendWin32Move(dx, dy);
        return device_->move(dx, dy);
    }
    bool leftDown() override
    {
        if (!isOpen())
            return sendWin32Click(MOUSEEVENTF_LEFTDOWN);
        return device_->leftDown();
    }
    bool leftUp() override
    {
        if (!isOpen())
            return sendWin32Click(MOUSEEVENTF_LEFTUP);
        return device_->leftUp();
    }
    Teensy41RawHid* teensy41RawHid() override { return device_.get(); }
    bool aimingActive() const override { return device_ && device_->aimingActive(); }
    bool shootingActive() const override { return device_ && device_->shootingActive(); }
    bool zoomingActive() const override { return device_ && device_->zoomingActive(); }

private:
    std::unique_ptr<Teensy41RawHid> device_;
};
```

Add parse/name/factory wiring:

```cpp
if (method == "TEENSY41_HID")
    return MouseInputMethod::Teensy41Hid;
```

```cpp
case MouseInputMethod::Teensy41Hid: return "TEENSY41_HID";
```

```cpp
case MouseInputMethod::Teensy41Hid:
    return std::make_unique<Teensy41RawHidMouseInput>(config);
```

- [ ] **Step 3: Expose active input owner**

In `0BS_box_2.h`, add:

```cpp
extern std::unique_ptr<IMouseInput> activeMouseInputOwner;
```

Make sure `0BS_box_2.h` includes `<memory>` and `MouseInput.h` or forward declares `class IMouseInput;`.

- [ ] **Step 4: Refresh on HID settings changes**

In `RefreshRuntimeAfterConfigLoad()` in `0BS_box_2.cpp`, extend the input-method change condition:

```cpp
previousConfig.teensy_hid_manufacturer != config.teensy_hid_manufacturer ||
previousConfig.teensy_hid_product != config.teensy_hid_product ||
previousConfig.teensy_hid_serial != config.teensy_hid_serial ||
previousConfig.teensy_hid_vid_filter != config.teensy_hid_vid_filter ||
previousConfig.teensy_hid_pid_filter != config.teensy_hid_pid_filter ||
previousConfig.teensy_hid_usage_page != config.teensy_hid_usage_page ||
previousConfig.teensy_hid_usage_id != config.teensy_hid_usage_id ||
previousConfig.teensy_hid_open_index != config.teensy_hid_open_index ||
previousConfig.teensy_hid_packet_timeout_ms != config.teensy_hid_packet_timeout_ms ||
previousConfig.teensy_hid_reconnect_interval_ms != config.teensy_hid_reconnect_interval_ms
```

- [ ] **Step 5: Update keyboard listener**

In `keyboard/keyboard_listener.cpp`, make serial Teensy detection exact:

```cpp
bool isTeensy41SerialInputMethod()
{
    return config.input_method == "TEENSY41";
}
```

Add helper:

```cpp
bool hasActiveMouseInputButtonState()
{
    return activeMouseInputOwner && activeMouseInputOwner->isOpen() &&
        (activeMouseInputOwner->aimingActive() ||
         activeMouseInputOwner->shootingActive() ||
         activeMouseInputOwner->zoomingActive());
}
```

In `isAnyKeyPressed()`, treat the active `IMouseInput` as a physical device if it is open and provides the pressed key:

```cpp
else if (activeMouseInputOwner && activeMouseInputOwner->isOpen())
{
    usePhysicalDevice = true;
}
```

Before Win32 fallback in `isAnyKeyPressed()`, add:

```cpp
if (!pressed && activeMouseInputOwner && activeMouseInputOwner->isOpen())
{
    if (key_name == "LeftMouseButton")       pressed = activeMouseInputOwner->shootingActive();
    else if (key_name == "RightMouseButton") pressed = activeMouseInputOwner->zoomingActive();
    else if (key_name == "X2MouseButton")    pressed = activeMouseInputOwner->aimingActive();
}
```

In `keyboardListener()`, OR the active input state into the three runtime states:

```cpp
(activeMouseInputOwner && activeMouseInputOwner->isOpen() && activeMouseInputOwner->aimingActive())
```

```cpp
(activeMouseInputOwner && activeMouseInputOwner->isOpen() && activeMouseInputOwner->shootingActive())
```

```cpp
(activeMouseInputOwner && activeMouseInputOwner->isOpen() && activeMouseInputOwner->zoomingActive())
```

- [ ] **Step 6: Run integration contract**

Run:

```powershell
python -m unittest training.tests.test_teensy41_hid_contract training.tests.test_teensy41_mouse_contract training.tests.test_razer_input_contract
```

Expected: input method and button state assertions pass; GUI and firmware checks still fail.

- [ ] **Step 7: Commit integration**

```powershell
git add mouse/MouseInput.h mouse/MouseInput.cpp 0BS_box_2.h 0BS_box_2.cpp keyboard/keyboard_listener.cpp
git commit -m "feat: wire teensy rawhid input method"
```

---

### Task 5: GUI Controls And Defaults Reset

**Files:**
- Modify: `overlay/draw_mouse.cpp`

- [ ] **Step 1: Add method option**

Update input methods:

```cpp
std::vector<std::string> inputMethods = {
    "WIN32", "GHUB", "RAZER", "ARDUINO", "TEENSY41", "TEENSY41_HID", "KMBOX_NET", "KMBOX_A", "MAKCU"
};
```

- [ ] **Step 2: Split serial and HID UI branches**

Change the Arduino/Teensy branch:

```cpp
if (config.input_method == "ARDUINO" || config.input_method == "TEENSY41")
{
    // existing COM port and baud-rate UI remains here
}
else if (config.input_method == "TEENSY41_HID")
{
    ImGui::TextColored(
        ImVec4(255, 255, 255, 255),
        activeMouseInputOwner && activeMouseInputOwner->isOpen()
            ? "Generic HID mouse bridge connected"
            : "Generic HID mouse bridge not found");
}
```

- [ ] **Step 3: Add reset helper**

Near the mouse UI helpers in `overlay/draw_mouse.cpp`, add:

```cpp
static void ResetTeensyHidDefaults()
{
    config.teensy_hid_manufacturer = "Generic";
    config.teensy_hid_product = "USB HID Mouse";
    config.teensy_hid_serial = "AUTO";
    config.teensy_hid_vid_filter = "AUTO";
    config.teensy_hid_pid_filter = "AUTO";
    config.teensy_hid_usage_page = 0xFFAB;
    config.teensy_hid_usage_id = 0x0200;
    config.teensy_hid_open_index = 0;
    config.teensy_hid_packet_timeout_ms = 2;
    config.teensy_hid_reconnect_interval_ms = 500;
}
```

- [ ] **Step 4: Add advanced settings block**

Inside the `TEENSY41_HID` UI branch, add:

```cpp
if (OverlayUI::BeginSection("Generic HID Mouse", "mouse_section_teensy_hid"))
{
    char manufacturer[128];
    char product[128];
    char serial[128];
    char vidFilter[32];
    char pidFilter[32];
    std::snprintf(manufacturer, sizeof(manufacturer), "%s", config.teensy_hid_manufacturer.c_str());
    std::snprintf(product, sizeof(product), "%s", config.teensy_hid_product.c_str());
    std::snprintf(serial, sizeof(serial), "%s", config.teensy_hid_serial.c_str());
    std::snprintf(vidFilter, sizeof(vidFilter), "%s", config.teensy_hid_vid_filter.c_str());
    std::snprintf(pidFilter, sizeof(pidFilter), "%s", config.teensy_hid_pid_filter.c_str());

    bool changed = false;
    changed |= ImGui::InputText("Manufacturer", manufacturer, sizeof(manufacturer));
    changed |= ImGui::InputText("Product", product, sizeof(product));
    changed |= ImGui::InputText("Serial", serial, sizeof(serial));
    changed |= ImGui::InputText("VID filter", vidFilter, sizeof(vidFilter));
    changed |= ImGui::InputText("PID filter", pidFilter, sizeof(pidFilter));
    changed |= ImGui::SliderInt("RawHID usage page", &config.teensy_hid_usage_page, 1, 0xFFFF);
    changed |= ImGui::SliderInt("RawHID usage ID", &config.teensy_hid_usage_id, 1, 0xFFFF);
    changed |= ImGui::SliderInt("HID open index", &config.teensy_hid_open_index, 0, 32);
    changed |= ImGui::SliderInt("Packet timeout (ms)", &config.teensy_hid_packet_timeout_ms, 0, 100);
    changed |= ImGui::SliderInt("Reconnect interval (ms)", &config.teensy_hid_reconnect_interval_ms, 50, 10000);

    ImGui::TextWrapped("Use only VID/PID values you own or values provided by the firmware/toolchain. Descriptor strings change after firmware rebuild and reflash.");

    if (ImGui::Button("Reset HID defaults"))
    {
        ResetTeensyHidDefaults();
        changed = true;
    }

    if (changed)
    {
        config.teensy_hid_manufacturer = manufacturer;
        config.teensy_hid_product = product;
        config.teensy_hid_serial = serial;
        config.teensy_hid_vid_filter = vidFilter;
        config.teensy_hid_pid_filter = pidFilter;
        OverlayConfig_MarkDirty();
        input_method_changed.store(true);
    }

    OverlayUI::EndSection();
}
```

- [ ] **Step 5: Add previous-value tracking**

Add `prev_teensy_hid_*` globals and include them in the refresh check at the bottom of `draw_mouse()`, mirroring the existing Arduino fields. Use this exact pattern for each field:

```cpp
std::string prev_teensy_hid_manufacturer = config.teensy_hid_manufacturer;
```

```cpp
prev_teensy_hid_manufacturer != config.teensy_hid_manufacturer ||
```

```cpp
prev_teensy_hid_manufacturer = config.teensy_hid_manufacturer;
```

- [ ] **Step 6: Run GUI/config contract**

Run:

```powershell
python -m unittest training.tests.test_teensy41_hid_contract training.tests.test_config_gui_control_contract
```

Expected: GUI and config assertions pass; firmware may still fail.

- [ ] **Step 7: Commit GUI changes**

```powershell
git add overlay/draw_mouse.cpp
git commit -m "feat: add generic hid mouse controls"
```

---

### Task 6: RawHID Firmware Sketch

**Files:**
- Create: `TeensyMouseRawHidBridge/TeensyMouseRawHidBridge.ino`

- [ ] **Step 1: Create firmware sketch**

Create `TeensyMouseRawHidBridge/TeensyMouseRawHidBridge.ino`:

```cpp
#include <USBHost_t36.h>

#if !defined(MOUSE_INTERFACE)
#error "Select a USB Type that includes Mouse"
#endif

#if !defined(RAWHID_INTERFACE)
#error "Select a USB Type that includes RawHID"
#endif

// Generic descriptor-facing defaults. Changing these requires firmware rebuild/reflash.
static const char GENERIC_MANUFACTURER[] = "Generic";
static const char GENERIC_PRODUCT[] = "USB HID Mouse";
static const char GENERIC_SERIAL[] = "HIDMOUSE001";

constexpr uint16_t HOST_MAGIC = 0x3448;
constexpr uint16_t DEVICE_MAGIC = 0x4834;
constexpr uint8_t PROTOCOL_VERSION = 1;
constexpr uint8_t CMD_MOVE = 1;
constexpr uint8_t CMD_BUTTONS = 2;
constexpr uint8_t EVT_BUTTON = 1;
constexpr size_t PACKET_SIZE = 64;

struct HostPacket {
  uint16_t magic;
  uint8_t version;
  uint8_t command;
  uint8_t buttonMask;
  int16_t dx;
  int16_t dy;
  int16_t wheel;
  int16_t wheelH;
  uint32_t sequence;
  uint8_t reserved[49];
} __attribute__((packed));

struct DevicePacket {
  uint16_t magic;
  uint8_t version;
  uint8_t event;
  uint8_t buttonId;
  uint8_t pressed;
  uint8_t hostButtonMask;
  uint32_t sequenceAck;
  uint8_t reserved[53];
} __attribute__((packed));

static_assert(sizeof(HostPacket) == PACKET_SIZE, "HostPacket must be 64 bytes");
static_assert(sizeof(DevicePacket) == PACKET_SIZE, "DevicePacket must be 64 bytes");

USBHost myusb;
USBHub hub1(myusb);
USBHub hub2(myusb);
USBHIDParser hid1(myusb);
USBHIDParser hid2(myusb);
USBHIDParser hid3(myusb);
USBHIDParser hid4(myusb);
MouseController mouse1(myusb);

uint8_t hostButtons = 0;
uint8_t appButtons = 0;
uint32_t lastSequence = 0;

int8_t clampMouseStep(int value) {
  if (value > 127) return 127;
  if (value < -127) return -127;
  return (int8_t)value;
}

void moveMouseInSteps(int dx, int dy, int wheel, int wheelH) {
  while (dx != 0 || dy != 0 || wheel != 0 || wheelH != 0) {
    int8_t stepX = clampMouseStep(dx);
    int8_t stepY = clampMouseStep(dy);
    int8_t stepWheel = clampMouseStep(wheel);
    int8_t stepWheelH = clampMouseStep(wheelH);
    Mouse.move(stepX, stepY, stepWheel, stepWheelH);
    dx -= stepX;
    dy -= stepY;
    wheel -= stepWheel;
    wheelH -= stepWheelH;
  }
}

void setMouseButtons(uint8_t buttons) {
  Mouse.set_buttons(
    buttons & MOUSE_LEFT,
    buttons & MOUSE_MIDDLE,
    buttons & MOUSE_RIGHT,
    buttons & MOUSE_BACK,
    buttons & MOUSE_FORWARD
  );
}

void applyButtons() {
  setMouseButtons(hostButtons | appButtons);
}

uint8_t buttonIdForMask(uint8_t mask) {
  if (mask == MOUSE_LEFT) return 1;
  if (mask == MOUSE_RIGHT) return 2;
  if (mask == MOUSE_MIDDLE) return 3;
  if (mask == MOUSE_BACK) return 4;
  if (mask == MOUSE_FORWARD) return 5;
  return 0;
}

void emitButtonTransition(uint8_t previousButtons, uint8_t currentButtons, uint8_t mask) {
  bool wasDown = (previousButtons & mask) != 0;
  bool isDown = (currentButtons & mask) != 0;
  if (wasDown == isDown) return;

  uint8_t buttonId = buttonIdForMask(mask);
  if (buttonId == 0) return;

  DevicePacket packet = {};
  packet.magic = DEVICE_MAGIC;
  packet.version = PROTOCOL_VERSION;
  packet.event = EVT_BUTTON;
  packet.buttonId = buttonId;
  packet.pressed = isDown ? 1 : 0;
  packet.hostButtonMask = currentButtons;
  packet.sequenceAck = lastSequence;
  RawHID.send(&packet, 0);
}

void emitButtonTransitions(uint8_t previousButtons, uint8_t currentButtons) {
  emitButtonTransition(previousButtons, currentButtons, MOUSE_LEFT);
  emitButtonTransition(previousButtons, currentButtons, MOUSE_RIGHT);
  emitButtonTransition(previousButtons, currentButtons, MOUSE_MIDDLE);
  emitButtonTransition(previousButtons, currentButtons, MOUSE_BACK);
  emitButtonTransition(previousButtons, currentButtons, MOUSE_FORWARD);
}

void handleHostPacket(const HostPacket& packet) {
  if (packet.magic != HOST_MAGIC || packet.version != PROTOCOL_VERSION)
    return;

  lastSequence = packet.sequence;
  if (packet.command == CMD_MOVE) {
    appButtons = packet.buttonMask;
    applyButtons();
    moveMouseInSteps(packet.dx, packet.dy, packet.wheel, packet.wheelH);
  } else if (packet.command == CMD_BUTTONS) {
    appButtons = packet.buttonMask;
    applyButtons();
  }
}

void readRawHidCommands() {
  HostPacket packet;
  int bytes = RawHID.recv(&packet, 0);
  if (bytes == (int)sizeof(packet)) {
    handleHostPacket(packet);
  }
}

void forwardHostMouse() {
  if (!mouse1.available()) return;

  int dx = mouse1.getMouseX();
  int dy = mouse1.getMouseY();
  int wheel = mouse1.getWheel();
  int wheelH = mouse1.getWheelH();
  uint8_t buttons = mouse1.getButtons();

  moveMouseInSteps(dx, dy, wheel, wheelH);

  if (buttons != hostButtons) {
    emitButtonTransitions(hostButtons, buttons);
    hostButtons = buttons;
    applyButtons();
  }

  mouse1.mouseDataClear();
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  myusb.begin();
  Mouse.begin();
}

void loop() {
  myusb.Task();
  forwardHostMouse();
  readRawHidCommands();
}
```

- [ ] **Step 2: Run firmware contract**

Run:

```powershell
python -m unittest training.tests.test_teensy41_hid_contract
```

Expected: firmware assertions pass.

- [ ] **Step 3: Commit firmware**

```powershell
git add TeensyMouseRawHidBridge/TeensyMouseRawHidBridge.ino
git commit -m "firmware: add teensy rawhid mouse bridge"
```

---

### Task 7: Project Files And Build Wiring

**Files:**
- Modify: `0BS_box_2.vcxproj`
- Modify: `cuda/0BS_cuda.vcxproj`

- [ ] **Step 1: Add new source/header to DML project**

In `0BS_box_2.vcxproj`, add:

```xml
<ClCompile Include="mouse\Teensy41RawHid.cpp" />
```

near other `mouse\*.cpp` entries, and:

```xml
<ClInclude Include="mouse\Teensy41RawHid.h" />
```

near other `mouse\*.h` entries.

- [ ] **Step 2: Add new source/header to CUDA project**

In `cuda/0BS_cuda.vcxproj`, add:

```xml
<ClCompile Include="..\mouse\Teensy41RawHid.cpp" />
```

and:

```xml
<ClInclude Include="..\mouse\Teensy41RawHid.h" />
```

- [ ] **Step 3: Run contract tests**

Run:

```powershell
python -m unittest training.tests.test_teensy41_hid_contract training.tests.test_teensy41_mouse_contract training.tests.test_razer_input_contract training.tests.test_config_gui_control_contract
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit project wiring**

```powershell
git add 0BS_box_2.vcxproj cuda/0BS_cuda.vcxproj
git commit -m "build: include teensy rawhid sources"
```

---

### Task 8: Full Verification

**Files:**
- No source changes in this task unless verification reveals compile errors from the previous tasks.

- [ ] **Step 1: Run targeted Python tests**

Run:

```powershell
python -m unittest training.tests.test_teensy41_hid_contract training.tests.test_teensy41_mouse_contract training.tests.test_config_gui_control_contract training.tests.test_razer_input_contract training.tests.test_control_loop_synthetic training.tests.test_inneraim_tracking_contract
```

Expected: all selected tests pass.

- [ ] **Step 2: Build DML**

Run:

```powershell
& 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\amd64\MSBuild.exe' .\0BS_box_2.vcxproj /p:Configuration=DML /p:Platform=x64 /m /v:minimal
```

Expected: `0BS_box_2.vcxproj -> C:\Users\donar\OneDrive\Desktop\0BS\x64\DML\0BS.exe`.

- [ ] **Step 3: Build CUDA**

Run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\cuda\build-cuda.ps1
```

Expected: `0BS_cuda.vcxproj -> C:\Users\donar\OneDrive\Desktop\0BS\x64\CUDA\0BS.exe`.

- [ ] **Step 4: Run diff check**

Run:

```powershell
git diff --check -- mouse\Teensy41RawHid.h mouse\Teensy41RawHid.cpp mouse\MouseInput.h mouse\MouseInput.cpp keyboard\keyboard_listener.cpp 0BS_box_2.h 0BS_box_2.cpp config\config.h config\config.cpp overlay\draw_mouse.cpp 0BS_box_2.vcxproj cuda\0BS_cuda.vcxproj TeensyMouseRawHidBridge\TeensyMouseRawHidBridge.ino training\tests\test_teensy41_hid_contract.py
```

Expected: no whitespace errors.

- [ ] **Step 5: Commit final verification adjustments**

If Task 8 required compile-fix edits, commit them:

```powershell
git add mouse\Teensy41RawHid.h mouse\Teensy41RawHid.cpp mouse\MouseInput.h mouse\MouseInput.cpp keyboard\keyboard_listener.cpp 0BS_box_2.h 0BS_box_2.cpp config\config.h config\config.cpp overlay\draw_mouse.cpp 0BS_box_2.vcxproj cuda\0BS_cuda.vcxproj TeensyMouseRawHidBridge\TeensyMouseRawHidBridge.ino training\tests\test_teensy41_hid_contract.py
git commit -m "fix: complete teensy rawhid verification"
```

If no files changed during Task 8, do not create an empty commit.

---

## Manual Hardware Check

- [ ] Flash `TeensyMouseRawHidBridge/TeensyMouseRawHidBridge.ino` using a Mouse + RawHID USB type.
- [ ] Set app input method to `TEENSY41_HID`.
- [ ] Leave VID/PID filters as `AUTO` and open index as `0`.
- [ ] Confirm Windows does not create a COM port for this firmware mode.
- [ ] Confirm physical mouse movement passes through.
- [ ] Hold physical right mouse button and confirm app zooming activates.
- [ ] Hold physical forward mouse button and confirm app aiming activates.
- [ ] Let the app send movement and confirm relative motion arrives through the Teensy.
- [ ] Switch back to serial `TEENSY41` and confirm the existing COM-port path still works.
