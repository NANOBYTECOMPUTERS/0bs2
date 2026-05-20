# Teensy 4.1 Generic HID Mouse RawHID Design

## Goal

Add a no-COM-port Teensy 4.1 transport that behaves like the current serial Teensy bridge while exposing a generic HID mouse identity to the OS. The host app sends relative movement and PC-generated button state through a RawHID control endpoint. The Teensy forwards the physical mouse plugged into its USB host port, merges physical and app-generated buttons, and reports physical button transitions back to the app.

The existing `ARDUINO` serial protocol and current serial `TEENSY41` mode remain untouched for rollback.

## Identity Boundary

The device should avoid user-facing `Teensy` naming, but it must not copy a real third-party vendor identity. The firmware will use neutral strings such as:

- Manufacturer: `Generic`
- Product: `USB HID Mouse`
- Serial: a stable generic value or value derived from the Teensy unique ID

VID/PID handling must stay within IDs legitimately available to the Teensy firmware/toolchain or a user-owned VID/PID if supplied later. The implementation will not use Razer, Wuxi China Resources Semico, or any other real third-party VID/PID or product identity.

The app should expose these values as editable advanced fields with safe defaults and a reset-to-default action. Editing is intended for compatibility and recovery, not impersonation. The UI should label VID/PID fields as "use only values you own or values provided by the firmware/toolchain" and should not ship presets for real third-party devices.

## Architecture

Add a new input method, `TEENSY41_HID`, displayed in the GUI as a generic HID mouse option. This avoids changing the meaning of the existing serial `TEENSY41` method.

Host-side components:

- `Teensy41RawHid`: RAII wrapper over existing `mouse/hidapi.h` and `mouse/hid.c`.
- `Teensy41RawHidMouseInput`: `IMouseInput` implementation for movement and left-button commands.
- Keyboard listener bridge: reads button state from the new RawHID device the same way it currently reads serial Teensy button state.

Firmware components:

- New sketch folder for RawHID firmware, separate from `TeensyMouseSerialPassthrough`.
- Mouse + RawHID USB type with no CDC/Serial dependency.
- USB host physical mouse forwarding remains based on `USBHost_t36`.

## Packet Contract

Use fixed 64-byte packets.

Host to Teensy:

- `magic`: 16-bit constant, little-endian
- `version`: protocol version
- `command`: move, buttons, ping, status
- `buttonMask`: merged app-side button mask for left/middle/right/back/forward
- `dx`, `dy`, `wheel`, `wheelH`: signed 16-bit relative movement values
- `sequence`: monotonic counter for diagnostics
- remaining bytes reserved and zeroed

Teensy to host:

- `magic`: same family, distinct direction value
- `version`
- `event`: physical button down/up, pong, status, error
- `buttonId`: 1 left, 2 right, 3 middle, 4 back, 5 forward
- `pressed`: 0/1
- `hostButtonMask`: current physical mouse button mask
- `sequenceAck`: last accepted host packet sequence
- remaining bytes reserved and zeroed

Movement is fire-and-forget for low latency. Status/ping packets can use sequence acknowledgement for troubleshooting.

## Button Behavior

The firmware maintains two masks:

- `hostButtons`: physical mouse buttons from the USB host port
- `appButtons`: buttons requested by the PC app

The final emitted USB mouse button state is `hostButtons | appButtons`.

Physical button transitions are reported to the app:

- Button 1 left maps to shooting
- Button 2 right maps to zoom
- Button 5 forward maps to aiming

This preserves the current serial Teensy behavior.

## Error Handling

Host:

- If no RawHID device opens, the input method reports closed and the app falls back the same way other `IMouseInput` devices do.
- Packet writes validate packet size and device handle.
- Read thread exits cleanly on disconnect and clears active physical button states.
- No file I/O in movement hot path.

Firmware:

- Reject packets with wrong magic/version.
- Clamp relative movement into USB mouse report-sized substeps.
- Keep forwarding physical mouse movement even when no RawHID host packets arrive.
- Avoid printing/logging in the hot path because there is no Serial transport in this mode.

## GUI and Config

Add `TEENSY41_HID` as a new input method option. For this mode, the GUI should not show COM port or baud-rate controls. It should show concise connection status, for example:

- `Generic HID mouse bridge connected`
- `Generic HID mouse bridge not found`

Serial `TEENSY41` continues to show COM port settings.

Add an advanced `Generic HID Mouse` settings block for `TEENSY41_HID`. Each field should have a visible default value and a reset button that restores the project default. These values should be saved in `config.ini` and merged by Load Config:

- Manufacturer string, default `Generic`
- Product string, default `USB HID Mouse`
- Serial string, default `HIDMOUSE001` or `AUTO`
- RawHID usage page, default project value
- RawHID usage ID, default project value
- HID open index, default `0`, for systems with multiple matching bridges
- Optional VID and PID filters, default `AUTO`
- Packet timeout in milliseconds, default `2`
- Reconnect interval in milliseconds, default `500`

RawHID does not use a COM port, but the settings block should still provide an editable "device selector" concept through VID/PID filters and open index. If the fields are left at `AUTO`, the app should find the first matching generic bridge using the default RawHID usage page/usage ID.

The firmware sketch should define the same defaults in one small constants block so the user can change strings and descriptor-related values in one place before flashing. Runtime edits in the app can affect host-side filtering and packet timing, but descriptor strings and VID/PID only change after firmware rebuild/reflash.

## Tests

Add or update tests for:

- `TEENSY41_HID` parse/name/factory wiring.
- RawHID packet struct size is 64 bytes and uses fixed-width fields.
- Button mask merge and transition mapping are preserved.
- GUI hides serial controls for `TEENSY41_HID`.
- GUI exposes editable Generic HID defaults and reset-to-default controls.
- Config load/save/merge preserves the Generic HID settings.
- Serial `TEENSY41` and `ARDUINO` contract tests still pass.
- Firmware sketch contains RawHID receive/send logic and no Serial command dependency.

## Non-Goals

- No Razer identity cloning.
- No use of another real vendor's VID/PID or product strings.
- No custom single-interface stealth descriptor in this update.
- No changes to the DML/CUDA inference path.
- No changes to model paths.

## Open Implementation Notes

The exact Teensy USB type may require a local board/core profile if Arduino IDE does not provide a ready Mouse + RawHID option for Teensy 4.1. If that is required, it should be kept as a separate firmware support folder and not mixed into the PC app source.
