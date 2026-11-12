import depthai as dai

print("Checking for Luxonis cameras...")

try:
    devices = dai.Device.getAllAvailableDevices()

    if len(devices) == 0:
        print("NO CAMERA DETECTED BY PYTHON")
    else:
        print(f"Found {len(devices)} camera(s)!")

        for device in devices:
            print("-------------------------")
            print("Device ID:", device.getDeviceId())
            print("Device name:", device.name)
            print("Device state:", device.state)

except Exception as e:
    print("ERROR:", e)