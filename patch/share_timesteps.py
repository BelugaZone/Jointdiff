current_timestep = None
camera_intrinsic = None
camera_E = None

def get_current_timestep():
    global current_timestep
    return current_timestep

def set_current_timestep(timestep):
    global current_timestep
    current_timestep = timestep



def get_current_camera_intrinsic():
    global camera_intrinsic
    return camera_intrinsic

def set_current_camera_intrinsic(intrinsic):
    global camera_intrinsic
    camera_intrinsic = intrinsic


def get_current_E():
    global camera_E
    return camera_E

def set_current_E(E):
    global camera_E
    camera_E = E