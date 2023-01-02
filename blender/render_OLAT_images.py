from pathlib import Path
import bpy
from mathutils import Matrix
from mathutils import Vector
import random, math
from math import degrees
import numpy as np
import json
import os

# SETTING ENVIRONMENT VARIABLES

PROJECTS_FOLDER = r"C:\Users\eliaf\Desktop\NRTF"
SCENE_NAME = 'nerfactor_ficus'  # string, name of the scene's folder inside "<project_root_folder>\data" containing training/validation data
RAND_CAM_NUM = 100  # number of camera positions to be generated randomly around the object
OLAT_TRAIN_REND_NUM = 2000  # number of random olat captures to be generated for training
OLAT_VAL_REND_NUM = 200  # number of random olat captures to be generated for validation
SKIP_FIRST_LAST_ROW = True  # if True first and last row of the envmap will not be used to generate olat captures
SEMISPHERE = True  # if True, bottom half (rows from 8 to 15) of the envmap will not be used to generate olat captures
W_RESOLUTION = 512  # resolution of rendered images (width)
H_RESOLUTION = 512  # resolution of rendered images (height)
SAMPLES = 50  # number of samples per pixel to be computed in the rendering process (path tracing)


def set_up_blender(root_path: str, samples_num: int, resolution: tuple, sRGB: bool = True):
    """
    :param root_path: string,  root folder's path of the project
    :param samples_num: int, number of samples per pixel to be computed in the rendering process (path tracing)
    :param resolution: resolution of rendered images, as an int tuple [width,height]
    :param sRGB: bool, if True the renderings will be saved as sRGB ".png" files, HDR ".exr" otherwise
    :return: None
    """
    # Setting renderer settings
    bpy.ops.file.pack_all()
    scene = bpy.context.scene
    scene.world.use_nodes = True
    scene.render.engine = 'CYCLES'
    scene.render.film_transparent = True
    scene.cycles.device = 'GPU'
    scene.cycles.samples = samples_num
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 8
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100


    # Setting image output
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '16'

    if sRGB:
        # PNG output with sRGB tonemapping
        scene.display_settings.display_device = 'sRGB'
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.render.image_settings.file_format = 'PNG'
    else:
        # OpenEXR output, no tonemapping applied
        scene.display_settings.display_device = 'None'
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
        scene.render.image_settings.file_format = 'OPEN_EXR'

    # Importing object mesh

    imported_object = bpy.ops.import_scene.obj(filepath=root_path + fr"\out\{SCENE_NAME}\mesh\mesh.obj",
                                               axis_forward='-Z', axis_up='Y')
    obj_object = bpy.context.selected_objects[0]

    # Setting material graph

    material = obj_object.active_material
    bsdf = material.node_tree.nodes["Principled BSDF"]
    image_node_ks = bsdf.inputs["Specular"].links[0].from_node
    separate_node = material.node_tree.nodes.new(type="ShaderNodeSeparateRGB")
    separate_node.name = "SeparateKs"
    material.node_tree.links.new(image_node_ks.outputs[0], separate_node.inputs[0])
    material.node_tree.links.new(separate_node.outputs[2], bsdf.inputs["Metallic"])
    material.node_tree.links.new(separate_node.outputs[1], bsdf.inputs["Roughness"])
    normal_map_node = bsdf.inputs["Normal"].links[0].from_node
    texture_n_node = normal_map_node.inputs["Color"].links[0].from_node
    material.node_tree.links.remove(normal_map_node.inputs["Color"].links[0])
    normal_separate_node = material.node_tree.nodes.new(type="ShaderNodeSeparateRGB")
    normal_separate_node.name = "SeparateNormal"
    normal_combine_node = material.node_tree.nodes.new(type="ShaderNodeCombineRGB")
    normal_combine_node.name = "CombineNormal"
    normal_invert_node = material.node_tree.nodes.new(type="ShaderNodeMath")
    normal_invert_node.name = "InvertNormal"
    normal_invert_node.operation = 'SUBTRACT'
    normal_invert_node.inputs[0].default_value = 1.0
    material.node_tree.links.new(texture_n_node.outputs[0], normal_separate_node.inputs['Image'])
    material.node_tree.links.new(normal_separate_node.outputs['R'], normal_combine_node.inputs['R'])
    material.node_tree.links.new(normal_separate_node.outputs['G'], normal_invert_node.inputs[1])
    material.node_tree.links.new(normal_invert_node.outputs[0], normal_combine_node.inputs['G'])
    material.node_tree.links.new(normal_separate_node.outputs['B'], normal_combine_node.inputs['B'])
    material.node_tree.links.new(normal_combine_node.outputs[0], normal_map_node.inputs["Color"])
    material.node_tree.links.remove(bsdf.inputs["Specular"].links[0])
    bsdf.inputs["Specular"].default_value = 0.5
    bsdf.inputs["Specular Tint"].default_value = 0.0
    bsdf.inputs["Sheen Tint"].default_value = 0.0
    bsdf.inputs["Clearcoat Roughness"].default_value = 0.0

    # Importing envmap

    texcoord = scene.world.node_tree.nodes.new(type="ShaderNodeTexCoord")
    mapping = scene.world.node_tree.nodes.new(type="ShaderNodeMapping")
    envmap = scene.world.node_tree.nodes.new(type="ShaderNodeTexEnvironment")

    envmap.image = bpy.data.images.load(root_path + r"\data\OLAT_envmaps\60.exr")

    scene.world.node_tree.links.new(envmap.outputs['Color'], scene.world.node_tree.nodes['Background'].inputs['Color'])
    scene.world.node_tree.links.new(texcoord.outputs['Generated'], mapping.inputs['Vector'])
    scene.world.node_tree.links.new(mapping.outputs['Vector'], envmap.inputs['Vector'])


def get_calibration_matrix_K_from_blender(camd):
    """
    3x4 P matrix from Blender camera
    Builds intrinsic camera parameters from Blender camera data.
    See notes on this in blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
    :param camd:
    :return: calibration matrix K
    """
    scene = bpy.context.scene
    assert scene.render.resolution_percentage == 100
    # assume angles describe the horizontal field of view
    assert camd.sensor_fit != 'VERTICAL'

    f_in_mm = camd.lens
    sensor_width_in_mm = camd.sensor_width

    w = scene.render.resolution_x
    h = scene.render.resolution_y

    pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

    f_x = f_in_mm / sensor_width_in_mm * w
    f_y = f_x * pixel_aspect

    # Parameters of intrinsic calibration matrix K
    c_x = w * (0.5 - camd.shift_x)
    c_y = (h / 2) + (camd.shift_y * w)

    K = Matrix(
        ((f_x, 0, c_x, 0),
         (0, f_y, c_y, 0),
         (0, 0, 1, 0),
         (0, 0, 0, 1)))

    return K



def get_3x4_RT_matrix_from_blender(cam):
    """
    Returns camera rotation and translation matrices from Blender.

    There are 3 coordinate systems involved:
    1. The World coordinates: "world"
       - right-handed
    2. The Blender camera coordinates: "bcam"
       - x is horizontal
       - y is up
       - right-handed: negative z look-at direction
    3. The desired computer vision camera coordinates: "cv"
       - x is horizontal
       - y is down (to align to the actual pixel coordinates
         used in digital images)
       - right-handed: positive z look-at direction
    :param cam: Blender's camera, a Blender's object
    :return: RT rotation matrix
    """
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    # Transpose since the rotation is object rotation, and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix()  # .transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = location  # -1*R_world2bcam @

    # Build the coordinate transform matrix from world to computer vision camera
    # NOTE: Use * instead of @ here for older versions of Blender

    R_world2cv = R_world2bcam @ R_bcam2cv  # R_bcam2cv@
    T_world2cv = T_world2bcam

    # put into 4x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],),
        (0, 0, 0, 1)
    ))

    return RT


def get_3x4_P_matrix_from_blender(cam):
    """
    Extracts P, K, RT matrices from Blender's camera object
    :param cam: Blender's camera, a Blender's object
    :return: P, K, RT matrices
    """
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K @ RT, K, RT


def import_cameras_from_json(path: str, counter: int):
    """
    Imports cameras' settings into Blender, reading them from json file (compatible with
    nerf synthetic and nerfactor datasets)
    :param path: string, path to the json file containing the camera parameters
    :param counter: int, counter used to count the total amount of cameras loaded into Blender
    :return: int, the counter after adding all cameras
    """
    # reading json file
    f = open(path)
    data = json.load(f)
    fov = data['camera_angle_x']
    # iterating through all cameras
    for cam in data['frames']:
        # importing the camera into Blender
        wmat = cam['transform_matrix']
        camera_data = bpy.data.cameras.new(name='Camera.%d' % counter)
        camera = bpy.data.objects.new('Camera.%d' % counter, camera_data)
        bpy.context.scene.collection.objects.link(camera)

        # setting the new camera as active
        for i in range(4):
            for j in range(4):
                camera.matrix_world[i][j] = wmat[i][j]
        camera.data.angle = fov
        counter = counter + 1
    return counter


def export_cameras(path: str, start_counter: int):
    """
    Exports cameras' settings from Blender to '.npz' file
    :param path: string, path to the output file (that will be created), containing the camera parameters
    :param start_counter: int, number of the first camera in the scene that has to be exported
    :return: None
    """
    scene = bpy.context.scene

    Intri = []
    Extri = []
    FOV = []
    for ob in scene.objects:
        # iterating through all blender scene's cameras
        if ob.type == 'CAMERA':
            # extracting camera's index from the name of the object
            index = ob.name.split('.')[-1]
            if index == 'Camera':
                index = '000'
            index = (int)(index)
            # skipping cameras previous to start_counter
            if index < start_counter:
                continue
            # extracting P, K and RT matrix from blender's coordinate system
            P, K, RT = get_3x4_P_matrix_from_blender(ob)

            K = np.matrix(K)
            RT = np.matrix(RT)
            fov = degrees(ob.data.angle)

            Intri.append(K)
            Extri.append(RT)
            FOV.append(fov)
    # saving all to file
    np.savez(path, Intri=Intri,
             Extri=Extri, FOV=FOV)


def generate_rnd_olat_cams(cam_path, counter: int, rand_cam_num: int, radius: float = None ):
    """
    Generates random cameras around the object, pointing at it
    :param cam_path: path to the '.npz' file (that will be created) to store the cameras' parameters
    :param counter: int, counter used to count the total amount of cameras loaded into Blender
    :param rand_cam_num: int, number of the cameras to be generated
    :param radius: float, cameras will be generated randomly at a distance ranging between [radius, radius + 0.5*radius].
    Default is None, in such case it will be calculated automatically from training cams
    :return: int, the counter after adding all cameras
    """
    camera_dict = np.load(cam_path)

    Intri_all = camera_dict['Intri']
    Extri_all = camera_dict['Extri']

    K = Intri_all[0]

    scale = 1

    if radius is None:
        first_cam = bpy.data.objects["Camera.0"].location
        radius = (first_cam - Vector((0, 0, 0))).length

    for _ in range(rand_cam_num):

        rnd_radius = np.random.uniform(radius, radius + 0.5*radius)

        theta = math.pi * random.uniform(0., 1.)
        phi = 2 * math.pi * random.uniform(0., 1.)

        # Randomly place the camera on a circle around the object at the same height as the main camera
        new_camera_pos = Vector((rnd_radius * math.sin(theta) * math.cos(phi), rnd_radius * math.sin(theta) * math.sin(phi),
                                 abs(rnd_radius * math.cos(theta))))

        theta1 = math.pi * random.uniform(0., 1.)
        phi1 = 2 * math.pi * random.uniform(0., 1.)

        # Randomly place the camera on a circle around the object at the same height as the main camera
        new_track_pos = Vector(
            (0.1 * math.sin(theta1) * math.cos(phi1), 0.1 * math.sin(theta1) * math.sin(phi1), 0.1 * math.cos(theta1)))

        direction = new_track_pos - new_camera_pos
        rot_quat = direction.to_track_quat('-Z', 'Y')

        rotation_euler = rot_quat.to_euler()

        R = rotation_euler.to_matrix()

        camera_data = bpy.data.cameras.new(name='Camera.%d' % counter)
        camera = bpy.data.objects.new('Camera.%d' % counter, camera_data)
        bpy.context.scene.collection.objects.link(camera)

        scene = bpy.context.scene
        sensor_width_in_mm = K[1, 1] * K[0, 2] / (K[0, 0] * K[1, 2])
        sensor_height_in_mm = 1  # doesn't matter
        resolution_x_in_px = K[0, 2] * 2  # principal point assumed at the center
        resolution_y_in_px = K[1, 2] * 2  # principal point assumed at the center

        s_u = resolution_x_in_px / sensor_width_in_mm
        s_v = resolution_y_in_px / sensor_height_in_mm
        # TODO include aspect ratio
        f_in_mm = K[0, 0] / s_u

        # Set the new camera as active
        for i in range(3):
            camera.matrix_world[i][3] = new_camera_pos[i]
            for j in range(3):
                camera.matrix_world[i][j] = R[i][j]
        #    camera.data.lens_unit = 'FOV'
        camera.data.type = 'PERSP'
        camera.data.lens = f_in_mm
        camera.data.lens_unit = 'MILLIMETERS'
        camera.data.sensor_width = sensor_width_in_mm
        counter = counter + 1

    return counter


def render_olat_captures(output_path: str, envmaps_path: str, olat_train_rend_num: int, olat_val_rend_num: int, counter_before_val: int,
                         semisphere: bool, skip_first_last_row: bool):
    """
    Renders olat_train_rend_num+olat_val_rend_num OLAT images and stores them in output_path
    :param output_path: string, path to the folder where rendered images will be stored
    :param envmaps_path: string, path to the folder containing ".exr" OLAT's envmaps
    :param olat_train_rend_num: int, number of training images to be rendered that will be used for training
    :param olat_val_rend_num: int, number of training images to be rendered that will be used for validation
    :param counter_before_val: int, value of the counter (number of cameras imported in Blender's scene) before adding validation cameras
    :param semisphere: bool, if True bottom half (rows from 8 to 15) of the envmap will not be used to generate olat captures
    :param skip_first_last_row: bool, if True first and last row of the envmap will not be used to generate olat captures
    :return: None
    """
    C = bpy.context
    scn = C.scene

    # Getting the environment node tree of the current scene
    node_tree = scn.world.node_tree
    tree_nodes = node_tree.nodes

    Env_node = tree_nodes['Environment Texture']


    save_rnd = []
    rnd_dict = {}

    # Setting the amount of envamp's pixels to be used in OLAT generations
    if semisphere:
        envmap_pix_num = 256
    else:
        if skip_first_last_row:
            envmap_pix_num = 512 - 32
        else:
            envmap_pix_num = 512

    if skip_first_last_row:
        start = 32
    else:
        start = 0

    for i in range(olat_train_rend_num + olat_val_rend_num):
        # choosing a random envmap and a random camera among the available
        rnd_envmap = int(random.uniform(0., 1.) * (envmap_pix_num - start - 1)) + start
        rnd_cam = int(random.uniform(0., 1.) * (counter_before_val - 1))
        # if the combination was already chosen, generate a new random combination
        while ((rnd_envmap in rnd_dict.keys()) and (rnd_cam in rnd_dict[rnd_envmap])):
            rnd_envmap = int(random.uniform(0., 1.) * (envmap_pix_num - start - 1)) + start
            rnd_cam = int(random.uniform(0., 1.) * (counter_before_val - 1))
        # activate the camera and load the envmap
        bpy.context.scene.camera = bpy.data.objects["Camera.{}".format(rnd_cam)]
        envmap = bpy.data.images.load("{}/{}.exr".format(envmaps_path, rnd_envmap))
        Env_node.image = envmap
        file = "{}/{}.png".format(output_path, i)
        bpy.context.scene.render.filepath = file
        # render tha image
        bpy.ops.render.render(write_still=True)
        save_rnd.append([rnd_cam, rnd_envmap])
        # saving random combination used to generate the image
        if rnd_envmap in rnd_dict.keys():
            rnd_dict[rnd_envmap].append(rnd_cam)
        else:
            rnd_dict[rnd_envmap] = [rnd_cam]
    # saving the random combinations of [camera, envmap] used to generate the images to '.txt' file
    with open(output_path + '/cam_env_idx.txt', 'w') as filehandle:
        for listitem in save_rnd:
            filehandle.write(f'{str(listitem)}\n')


# Gettin project's root folder path
root_dir = PROJECTS_FOLDER
data_dir = root_dir + fr"\data\{SCENE_NAME}"

# Setting up blender scene
set_up_blender(root_dir, SAMPLES, [W_RESOLUTION, H_RESOLUTION])

# Importing cameras used to generate dataset's training images into Blender,converting the coordinate system to Blender's one

counter = 0
counter = import_cameras_from_json(data_dir + r'\transforms_train.json', counter)

# Exporting cameras used to generate dataset's training images to ".npz" file
export_cameras(data_dir + r'\cameras.npz', 0)

# Generating additional cameras (OLAT cameras) around the object in random positions
cam_file = data_dir + r'\cameras.npz'
counter_before_olat = counter
counter = generate_rnd_olat_cams(cam_file, counter, RAND_CAM_NUM)

# Exporting randomly generated OLAT cameras to ".npz" file
export_cameras(data_dir + r'\cameras_olat.npz', counter_before_olat)

# Importing cameras used to generate dataset's validation images into Blender,converting the coordinate system to Blender's one
counter_before_val = counter
counter = import_cameras_from_json(data_dir + r'\transforms_val.json', counter)

# Exporting cameras used to generate dataset's validation images to ".npz" file
export_cameras(data_dir + r'\cameras_test.npz', counter_before_val)

# Rendering OLAT_TRAIN_REND_NUM + OLAT_VAL_REND_NUM random OLAT captures
render_olat_captures(data_dir + r"\_OLAT", root_dir + r"\data\OLAT_envmaps", OLAT_TRAIN_REND_NUM, OLAT_VAL_REND_NUM,
                     counter_before_val, SEMISPHERE, SKIP_FIRST_LAST_ROW)
