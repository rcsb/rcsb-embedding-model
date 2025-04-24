

def arg_devices(devices):
    if len(devices) == 1:
        return devices[0] if devices[0] == "auto" else int(devices[0])
    return [int(x) for x in devices]



