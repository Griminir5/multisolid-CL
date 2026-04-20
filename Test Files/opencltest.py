from daetools.pyDAE.evaluator_opencl import pyEvaluator_OpenCL

openclPlatforms = pyEvaluator_OpenCL.AvailableOpenCLPlatforms()
openclDevices   = pyEvaluator_OpenCL.AvailableOpenCLDevices()
print('Available OpenCL platforms:')
for platform in openclPlatforms:
    print('  Platform: %s' % platform.Name)
    print('    PlatformID: %d' % platform.PlatformID)
    print('    Vendor:     %s' % platform.Vendor)
    print('    Version:    %s' % platform.Version)
    #print('    Profile:    %s' % platform.Profile)
    #print('    Extensions: %s' % platform.Extensions)
    print('')
print('Available OpenCL devices:')
for device in openclDevices:
    print('  Device: %s' % device.Name)
    print('    PlatformID:      %d' % device.PlatformID)
    print('    DeviceID:        %d' % device.DeviceID)
    print('    DeviceVersion:   %s' % device.DeviceVersion)
    print('    DriverVersion:   %s' % device.DriverVersion)
    print('    OpenCLVersion:   %s' % device.OpenCLVersion)
    print('    MaxComputeUnits: %d' % device.MaxComputeUnits)
    print('')