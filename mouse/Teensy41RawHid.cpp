#include "Teensy41RawHid.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include "config.h"
#include "hidapi.h"

namespace
{
uint16_t clampToUint16(int value, uint16_t fallback)
{
    if (value < 1 || value > 0xFFFF)
        return fallback;
    return static_cast<uint16_t>(value);
}

bool isAuto(const std::string& value)
{
    if (value.empty())
        return true;

    std::string upper = value;
    std::transform(upper.begin(), upper.end(), upper.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return upper == "AUTO";
}

bool parseHexFilter(const std::string& value, uint16_t& parsed)
{
    if (isAuto(value))
        return false;

    char* end = nullptr;
    unsigned long result = std::strtoul(value.c_str(), &end, 16);
    if (end == value.c_str() || *end != '\0' || result > 0xFFFF)
        return false;

    parsed = static_cast<uint16_t>(result);
    return true;
}

std::string narrow(const wchar_t* value)
{
    if (!value)
        return {};

    std::string out;
    while (*value)
    {
        wchar_t wc = *value++;
        out.push_back(wc >= 0 && wc <= 0x7F ? static_cast<char>(wc) : '?');
    }
    return out;
}

int clampInt16(int value)
{
    return std::clamp(value,
        static_cast<int>(std::numeric_limits<int16_t>::min()),
        static_cast<int>(std::numeric_limits<int16_t>::max()));
}
}

void Teensy41RawHid::HidDeviceDeleter::operator()(hid_device* device) const
{
    if (device)
        hid_close(device);
}

Teensy41RawHid::Teensy41RawHid(const Config& config)
    : usagePage_(clampToUint16(config.teensy_hid_usage_page, 0xFFAB)),
      usageId_(clampToUint16(config.teensy_hid_usage_id, 0x0200)),
      openIndex_(std::clamp(config.teensy_hid_open_index, 0, 32)),
      packetTimeoutMs_(std::clamp(config.teensy_hid_packet_timeout_ms, 0, 100)),
      reconnectIntervalMs_(std::clamp(config.teensy_hid_reconnect_interval_ms, 50, 10000)),
      serialFilter_(config.teensy_hid_serial),
      vidFilter_(config.teensy_hid_vid_filter),
      pidFilter_(config.teensy_hid_pid_filter)
{
    connected_.store(open(), std::memory_order_release);
    readerThread_ = std::thread(&Teensy41RawHid::readerLoop, this);
}

Teensy41RawHid::~Teensy41RawHid()
{
    stopReader_.store(true, std::memory_order_release);
    if (readerThread_.joinable())
        readerThread_.join();
    device_.reset();
}

bool Teensy41RawHid::isOpen() const
{
    if (!connected_.load(std::memory_order_acquire))
        return false;

    std::lock_guard<std::mutex> lock(writeMutex_);
    return device_ != nullptr;
}

bool Teensy41RawHid::move(int dx, int dy, int wheel, int wheelH)
{
    return sendPacket(Teensy41RawHidCommand::Move, dx, dy, wheel, wheelH, hostButtons_.load(std::memory_order_acquire));
}

bool Teensy41RawHid::press()
{
    return setHostButtons(static_cast<uint8_t>(hostButtons_.load(std::memory_order_acquire) | Teensy41RawHidButtonLeft));
}

bool Teensy41RawHid::release()
{
    return setHostButtons(static_cast<uint8_t>(hostButtons_.load(std::memory_order_acquire) & ~Teensy41RawHidButtonLeft));
}

bool Teensy41RawHid::setHostButtons(uint8_t mask)
{
    hostButtons_.store(mask, std::memory_order_release);
    return sendPacket(Teensy41RawHidCommand::Buttons, 0, 0, 0, 0, mask);
}

bool Teensy41RawHid::aimingActive() const
{
    return aimingActive_.load(std::memory_order_acquire);
}

bool Teensy41RawHid::shootingActive() const
{
    return shootingActive_.load(std::memory_order_acquire);
}

bool Teensy41RawHid::zoomingActive() const
{
    return zoomingActive_.load(std::memory_order_acquire);
}

bool Teensy41RawHid::open()
{
    uint16_t vid = 0;
    uint16_t pid = 0;
    const bool useVid = parseHexFilter(vidFilter_, vid);
    const bool usePid = parseHexFilter(pidFilter_, pid);

    hid_device_info* devices = hid_enumerate(useVid ? vid : 0, usePid ? pid : 0);
    int matchedIndex = 0;
    for (hid_device_info* cur = devices; cur; cur = cur->next)
    {
        if (cur->usage_page != usagePage_ || cur->usage != usageId_)
            continue;
        if (useVid && cur->vendor_id != vid)
            continue;
        if (usePid && cur->product_id != pid)
            continue;
        if (!isAuto(serialFilter_) && serialFilter_ != narrow(cur->serial_number))
            continue;

        if (matchedIndex++ != openIndex_)
            continue;

        hid_device* opened = hid_open_path(cur->path);
        if (opened)
        {
            device_.reset(opened);
            hid_set_nonblocking(device_.get(), 1);
            hid_free_enumeration(devices);
            return true;
        }
    }

    hid_free_enumeration(devices);
    return false;
}

bool Teensy41RawHid::sendPacket(Teensy41RawHidCommand command, int dx, int dy, int wheel, int wheelH, uint8_t buttonMask)
{
    if (!connected_.load(std::memory_order_acquire))
        return false;

    Teensy41RawHidHostPacket packet;
    packet.command = static_cast<uint8_t>(command);
    packet.buttonMask = buttonMask;
    packet.dx = static_cast<int16_t>(clampInt16(dx));
    packet.dy = static_cast<int16_t>(clampInt16(dy));
    packet.wheel = static_cast<int16_t>(clampInt16(wheel));
    packet.wheelH = static_cast<int16_t>(clampInt16(wheelH));
    std::array<unsigned char, Teensy41RawHidPacketSize + 1> report{};

    std::lock_guard<std::mutex> lock(writeMutex_);
    if (!device_)
        return false;

    packet.sequence = ++sequence_;
    report[0] = 0;
    std::memcpy(report.data() + 1, &packet, sizeof(packet));

    int written = hid_write(device_.get(), report.data(), report.size());
    if (written < 0)
    {
        connected_.store(false, std::memory_order_release);
        return false;
    }
    return written == static_cast<int>(report.size());
}

void Teensy41RawHid::readerLoop()
{
    while (!stopReader_.load(std::memory_order_acquire))
    {
        if (!isOpen())
        {
            {
                std::lock_guard<std::mutex> lock(writeMutex_);
                device_.reset();
                connected_.store(open(), std::memory_order_release);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(reconnectIntervalMs_));
            continue;
        }

        std::array<unsigned char, Teensy41RawHidPacketSize + 1> buffer{};
        int read = hid_read_timeout(device_.get(), buffer.data(), buffer.size(), packetTimeoutMs_);
        if (read == 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (read < 0)
        {
            connected_.store(false, std::memory_order_release);
            continue;
        }
        const size_t offset = (read == static_cast<int>(Teensy41RawHidPacketSize + 1) && buffer[0] == 0) ? 1u : 0u;
        if (read - static_cast<int>(offset) < static_cast<int>(sizeof(Teensy41RawHidDevicePacket)))
            continue;

        Teensy41RawHidDevicePacket packet;
        std::memcpy(&packet, buffer.data() + offset, sizeof(packet));
        if (packet.magic != Teensy41RawHidDeviceMagic || packet.version != Teensy41RawHidVersion)
            continue;

        if (packet.event == static_cast<uint8_t>(Teensy41RawHidEvent::Button))
            applyButtonEvent(packet.buttonId, packet.pressed != 0);
    }
}

void Teensy41RawHid::applyButtonEvent(uint8_t buttonId, bool pressed)
{
    switch (buttonId)
    {
    case 1:
        shootingActive_.store(pressed, std::memory_order_release);
        break;
    case 2:
        zoomingActive_.store(pressed, std::memory_order_release);
        break;
    case 5:
        aimingActive_.store(pressed, std::memory_order_release);
        break;
    default:
        break;
    }
}
