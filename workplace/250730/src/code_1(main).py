import glfw
from OpenGL.GL import *
import numpy as np
from pyrr import Matrix44
import trimesh
import time
from PIL import Image
import ctypes

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

import imgui
from imgui.integrations.glfw import GlfwRenderer

from pycuda.compiler import SourceModule

def load_cuda_kernel(filepath, funcname):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    return SourceModule(code).get_function(funcname)

cuda_func = load_cuda_kernel("../cuda_program/cal_physics.cu", "update_particles_cuda")

boundary_mode = 0
i_switch = 0.0

# === GLFW 初期化 ===
if not glfw.init():
    raise Exception("GLFWの初期化に失敗しました")

camera_theta = 3.14 / 4.0
camera_phi = np.radians(30.0)

monitor = glfw.get_monitors()[1]
mode = glfw.get_video_mode(monitor)

xpos, ypos = glfw.get_monitor_pos(monitor)
glfw.window_hint(glfw.DECORATED, glfw.FALSE)

window = glfw.create_window(mode.size.width, mode.size.height, "10K粒子シミュレーション with GUI", None, None)
if not window:
    glfw.terminate()
    raise Exception("ウィンドウの作成に失敗しました")

glfw.set_window_pos(window, xpos, ypos)
glfw.make_context_current(window)

imgui.create_context()
impl = GlfwRenderer(window)

current_bg_color = [1, 1, 1, 1]
glClearColor(*current_bg_color)
bg_color_flag = False

print("Renderer:", glGetString(GL_RENDERER).decode())
print("Vendor:", glGetString(GL_VENDOR).decode())
print("Version:", glGetString(GL_VERSION).decode())

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 instancePos;
layout(location = 3) in float instanceScale;
layout(location = 4) in vec3 instanceColor;

out vec3 fragNormal;
out vec3 fragPos;
out vec3 fragColor;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    vec3 scaledPos = position * instanceScale;
    vec4 worldPos = vec4(scaledPos + instancePos, 1.0);
    
    fragPos = worldPos.xyz;
    fragNormal = normal;
    fragColor = instanceColor;
    
    gl_Position = projection * view * worldPos;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 fragNormal;
in vec3 fragPos;
in vec3 fragColor;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 viewPos;

void main()
{
    vec3 norm = normalize(fragNormal);
    vec3 lightDir = normalize(lightPos - fragPos);
    
    // 環境光
    float ambient = 0.3;
    
    // 拡散光
    float diff = max(dot(norm, lightDir), 0.0);
    
    // 鏡面反射
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    
    vec3 result = (ambient + diff + spec * 0.5) * fragColor;
    FragColor = vec4(result, 1.0);
}
"""

BBOX_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    gl_Position = projection * view * vec4(position, 1.0);
}
"""

BBOX_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

uniform vec3 lineColor;

void main()
{
    FragColor = vec4(lineColor, 1.0);
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

shader_program = glCreateProgram()
vs = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
fs = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
glAttachShader(shader_program, vs)
glAttachShader(shader_program, fs)
glLinkProgram(shader_program)
glDeleteShader(vs)
glDeleteShader(fs)

bbox_shader_program = glCreateProgram()
bbox_vs = compile_shader(BBOX_VERTEX_SHADER, GL_VERTEX_SHADER)
bbox_fs = compile_shader(BBOX_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
glAttachShader(bbox_shader_program, bbox_vs)
glAttachShader(bbox_shader_program, bbox_fs)
glLinkProgram(bbox_shader_program)
glDeleteShader(bbox_vs)
glDeleteShader(bbox_fs)

glUseProgram(shader_program)
view_loc = glGetUniformLocation(shader_program, "view")
proj_loc = glGetUniformLocation(shader_program, "projection")
light_pos_loc = glGetUniformLocation(shader_program, "lightPos")
view_pos_loc = glGetUniformLocation(shader_program, "viewPos")

glUseProgram(bbox_shader_program)
bbox_view_loc = glGetUniformLocation(bbox_shader_program, "view")
bbox_proj_loc = glGetUniformLocation(bbox_shader_program, "projection")
bbox_color_loc = glGetUniformLocation(bbox_shader_program, "lineColor")

color_positive = np.array([0.4, 0.8, 1.0], dtype=np.float32)
color_negative = np.array([1.0, 0.5, 0.0], dtype=np.float32)

def create_bounding_box():
    size = 4.3
    half = size / 2
    
    corners = np.array([
        [-half, -half, -half], [half, -half, -half], [half, half, -half], [-half, half, -half],
        [-half, -half, half], [half, -half, half], [half, half, half], [-half, half, half],
    ], dtype=np.float32)
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 奥面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 手前面
        (0, 4), (1, 5), (2, 6), (3, 7),  # 奥-手前
    ]
    
    num_divisions = 3
    step_size = size / num_divisions
    steps = [-half + step_size, half - step_size]
    
    vertices = []
    
    for start, end in edges:
        vertices.extend([corners[start], corners[end]])
    
    for z in [-half, half]:
        for x in steps:
            vertices.extend([[x, -half, z], [x, half, z]])
        for y in steps:
            vertices.extend([[-half, y, z], [half, y, z]])
    
    for x in [-half, half]:
        for y in steps:
            vertices.extend([[x, y, -half], [x, y, half]])
        for z in steps:
            vertices.extend([[x, -half, z], [x, half, z]])
    
    for y in [-half, half]:
        for z in steps:
            vertices.extend([[-half, y, z], [half, y, z]])
        for x in steps:
            vertices.extend([[x, y, -half], [x, y, half]])
    
    return np.array(vertices, dtype=np.float32)

bbox_vertices = create_bounding_box()
bbox_VAO = glGenVertexArrays(1)
bbox_VBO = glGenBuffers(1)

glBindVertexArray(bbox_VAO)
glBindBuffer(GL_ARRAY_BUFFER, bbox_VBO)
glBufferData(GL_ARRAY_BUFFER, bbox_vertices.nbytes, bbox_vertices, GL_STATIC_DRAW)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
glBindVertexArray(0)

BOX_SIZE = 4.2
grid_divisions = 50  
cell_size = BOX_SIZE / grid_divisions
grid_size = (grid_divisions, grid_divisions, grid_divisions)

grid_origin = np.array([-BOX_SIZE/2, -BOX_SIZE/2, -BOX_SIZE/2], dtype=np.float32)
potential_grid = np.zeros(grid_size, dtype=np.float32)
E_field = np.zeros(grid_size + (3,), dtype=np.float32)

print(f"電場グリッド設定:")
print(f"  グリッドサイズ: {grid_size}")
print(f"  セルサイズ: {cell_size}")
print(f"  グリッド原点: {grid_origin}")
print(f"  ボックスサイズ: {BOX_SIZE}")

gui_params = {
    'num_particles': 5000,
    'max_particles': 10000,
    'E_field_x': 0.0,
    'E_field_y': 15.0, 
    'E_field_z': 0.0,
    'positive_charge': 2.0,
    'negative_charge': -2.0,
    'particle_mass': 1.0,
    'dt': 0.01,
    'scale_factor': 0.03,
    'boundary_mode_name': "ワープ",
    'show_bbox': True,
    'simulation_paused': False,
    'positive_color': [0.4, 0.8, 1.0],
    'negative_color': [1.0, 0.5, 0.0],
    'warp_x_negative': False,
    'warp_x_positive': False,
    'warp_y_negative': True,
    'warp_y_positive': True,
    'warp_z_negative': False,
    'warp_z_positive': False,
}

def update_electric_field():
    E_field[:, :, :, 0] = gui_params['E_field_x']
    E_field[:, :, :, 1] = gui_params['E_field_y']
    E_field[:, :, :, 2] = gui_params['E_field_z']
    
    print(f"電場更新: E=({gui_params['E_field_x']}, {gui_params['E_field_y']}, {gui_params['E_field_z']})")

update_electric_field()

class ChargedParticle:
    def __init__(self, position, velocity, charge, mass):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.charge = charge
        self.mass = mass

def create_particles(num_particles):
    particles = []
    
    spawn_range = 1.0
    
    for _ in range(num_particles // 2):
        position = np.random.uniform(-spawn_range, spawn_range, 3)
        velocity = np.zeros(3)  
        charge = gui_params['positive_charge']
        mass = gui_params['particle_mass']
        particles.append(ChargedParticle(position, velocity, charge, mass))

    for _ in range(num_particles - num_particles // 2):
        position = np.random.uniform(-spawn_range, spawn_range, 3)
        velocity = np.zeros(3)  
        charge = gui_params['negative_charge']
        mass = gui_params['particle_mass']
        particles.append(ChargedParticle(position, velocity, charge, mass))
    
    print(f"粒子生成完了: {len(particles)}個")
    return particles

particles = create_particles(gui_params['num_particles'])
dt = gui_params['dt']

def update_particles_optimized(particles, E_field_array):
    if gui_params['simulation_paused']:
        return
        
    N = len(particles)
    if N == 0:
        return
        
    positions_np = np.array([tuple(p.position) for p in particles], dtype=np.float32)
    velocities_np = np.array([tuple(p.velocity) for p in particles], dtype=np.float32)
    charges_np = np.array([p.charge for p in particles], dtype=np.float32)
    masses_np = np.array([p.mass for p in particles], dtype=np.float32)
    
    E_field_flat = E_field_array.reshape(-1, 3).astype(np.float32)

    warp_flags = np.array([
        gui_params['warp_x_negative'], gui_params['warp_x_positive'],
        gui_params['warp_y_negative'], gui_params['warp_y_positive'],
        gui_params['warp_z_negative'], gui_params['warp_z_positive']
    ], dtype=np.int32)

    if not hasattr(update_particles_optimized, 'gpu_arrays') or update_particles_optimized.last_N != N:
        if hasattr(update_particles_optimized, 'gpu_arrays'):
            for arr in update_particles_optimized.gpu_arrays.values():
                arr.free()
        
        update_particles_optimized.gpu_arrays = {
            'pos': cuda.mem_alloc(positions_np.nbytes),
            'vel': cuda.mem_alloc(velocities_np.nbytes),
            'chg': cuda.mem_alloc(charges_np.nbytes),
            'mas': cuda.mem_alloc(masses_np.nbytes),
            'ef': cuda.mem_alloc(E_field_flat.nbytes),
            'warp': cuda.mem_alloc(warp_flags.nbytes)
        }
        update_particles_optimized.last_N = N
    
    gpu_arrays = update_particles_optimized.gpu_arrays
    
    cuda.memcpy_htod(gpu_arrays['pos'], positions_np)
    cuda.memcpy_htod(gpu_arrays['vel'], velocities_np)
    cuda.memcpy_htod(gpu_arrays['chg'], charges_np)
    cuda.memcpy_htod(gpu_arrays['mas'], masses_np)
    cuda.memcpy_htod(gpu_arrays['ef'], E_field_flat)
    cuda.memcpy_htod(gpu_arrays['warp'], warp_flags)

    block_size = 256
    grid_size_cuda = (N + block_size - 1) // block_size

    cuda_func(
        gpu_arrays['pos'], gpu_arrays['vel'], gpu_arrays['chg'], gpu_arrays['mas'],
        gpu_arrays['ef'], 
        np.int32(grid_size[0]), np.int32(grid_size[1]), np.int32(grid_size[2]),
        np.float32(cell_size), np.int32(N), 
        np.float32(1.0),  
        np.float32(0.1),  
        np.float32(gui_params['dt']), 
        np.float32(BOX_SIZE),
        np.int32(boundary_mode), 
        gpu_arrays['warp'],
        block=(block_size, 1, 1), grid=(grid_size_cuda, 1)
    )

    cuda.memcpy_dtoh(positions_np, gpu_arrays['pos'])
    cuda.memcpy_dtoh(velocities_np, gpu_arrays['vel'])
    
    for i, p in enumerate(particles):
        p.position = positions_np[i]
        p.velocity = velocities_np[i]

sphere_mesh = trimesh.load("../3D_models/sphere.obj", force='mesh')
sphere_vertices = np.array(sphere_mesh.vertices, dtype=np.float32)
sphere_normals = np.array(sphere_mesh.vertex_normals, dtype=np.float32)
sphere_indices = np.array(sphere_mesh.faces, dtype=np.uint32).flatten()

sphere_VAO = glGenVertexArrays(1)
sphere_VBO = glGenBuffers(1)
sphere_EBO = glGenBuffers(1)
instance_VBO = glGenBuffers(1)

glBindVertexArray(sphere_VAO)

sphere_vertex_data = np.hstack([sphere_vertices, sphere_normals])
glBindBuffer(GL_ARRAY_BUFFER, sphere_VBO)
glBufferData(GL_ARRAY_BUFFER, sphere_vertex_data.nbytes, sphere_vertex_data, GL_STATIC_DRAW)

glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_indices.nbytes, sphere_indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

max_instance_data = np.zeros((gui_params['max_particles'], 7), dtype=np.float32)
glBindBuffer(GL_ARRAY_BUFFER, instance_VBO)
glBufferData(GL_ARRAY_BUFFER, max_instance_data.nbytes, max_instance_data, GL_DYNAMIC_DRAW)

glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))
glVertexAttribDivisor(2, 1)

glEnableVertexAttribArray(3)
glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))
glVertexAttribDivisor(3, 1)

glEnableVertexAttribArray(4)
glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(16))
glVertexAttribDivisor(4, 1)

glBindVertexArray(0)

projection = Matrix44.perspective_projection(45.0, mode.size.width/mode.size.height, 0.1, 100.0)

glEnable(GL_DEPTH_TEST)
glEnable(GL_CULL_FACE)

angle_speed = 0.02
phi_limit = np.radians(89.0)
radius = 10
zoom_speed = 0.1
min_radius = 1.0
max_radius = 20.0
c_switch = 0.0

print("=== 操作方法 ===")
print("矢印キー: カメラ回転")
print("W/S: ズーム")
print("C: 背景色切り替え")
print("I: 境界条件切り替え（跳ね返り/ワープ）")
print("GUI: 右側のパネルで各種設定")
print("===============")

frame_times = []
frame_count = 0
last_particle_count = gui_params['num_particles']
last_electric_field = [gui_params['E_field_x'], gui_params['E_field_y'], gui_params['E_field_z']]

while not glfw.window_should_close(window):
    start_time = time.time()
    
    glfw.poll_events()
    impl.process_inputs()
    
    if gui_params['num_particles'] != last_particle_count:
        particles = create_particles(gui_params['num_particles'])
        last_particle_count = gui_params['num_particles']
        if hasattr(update_particles_optimized, 'gpu_arrays'):
            delattr(update_particles_optimized, 'gpu_arrays')
    
    current_electric_field = [gui_params['E_field_x'], gui_params['E_field_y'], gui_params['E_field_z']]
    if current_electric_field != last_electric_field:
        update_electric_field()
        last_electric_field = current_electric_field.copy()
    
    for p in particles:
        if p.charge > 0:
            p.charge = gui_params['positive_charge']
        else:
            p.charge = gui_params['negative_charge']
        p.mass = gui_params['particle_mass']
    
    update_particles_optimized(particles, E_field)
    
    if i_switch >= 0:
        i_switch -= 0.1
        
    if glfw.get_key(window, glfw.KEY_I) == glfw.PRESS:
        if i_switch < 0:
            boundary_mode = 1 - boundary_mode
            gui_params['boundary_mode_name'] = "跳ね返り" if boundary_mode == 1 else "ワープ"
            print(f"境界条件: {gui_params['boundary_mode_name']}")
            i_switch = 2
    
    if c_switch >= 0:
        c_switch -= 0.1
        
    if glfw.get_key(window, glfw.KEY_C) == glfw.PRESS:
        if c_switch < 0:
            if bg_color_flag == False:
                current_bg_color = [0, 0, 0, 1.0]
                bg_color_flag = True
                c_switch = 2
            else:
                current_bg_color = [1, 1, 1, 1.0]
                bg_color_flag = False
                c_switch = 2
    
    glClearColor(*current_bg_color)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        camera_theta -= angle_speed
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        camera_theta += angle_speed
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        camera_phi += angle_speed
        camera_phi = min(camera_phi, phi_limit)
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        camera_phi -= angle_speed
        camera_phi = max(camera_phi, -phi_limit)
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        radius -= zoom_speed
        radius = max(radius, min_radius)
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        radius += zoom_speed
        radius = min(radius, max_radius)

    camX = radius * np.cos(camera_phi) * np.sin(camera_theta)
    camY = radius * np.sin(camera_phi)
    camZ = radius * np.cos(camera_phi) * np.cos(camera_theta)
    eye = np.array([camX, camY, camZ])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    view = Matrix44.look_at(eye, target, up)

    if gui_params['show_bbox'] and bg_color_flag == False:
        glUseProgram(bbox_shader_program)
        glUniformMatrix4fv(bbox_view_loc, 1, GL_FALSE, view.astype(np.float32))
        glUniformMatrix4fv(bbox_proj_loc, 1, GL_FALSE, projection.astype(np.float32))
        glUniform3fv(bbox_color_loc, 1, np.array([0, 0, 0], dtype=np.float32))
        
        glBindVertexArray(bbox_VAO)
        glDrawArrays(GL_LINES, 0, len(bbox_vertices))
    
    if len(particles) > 0:
        glUseProgram(shader_program)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view.astype(np.float32))
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection.astype(np.float32))
        glUniform3fv(light_pos_loc, 1, np.array([5.0, 5.0, 5.0], dtype=np.float32))
        glUniform3fv(view_pos_loc, 1, eye.astype(np.float32))
        
        instance_data = np.zeros((len(particles), 7), dtype=np.float32)
        for i, p in enumerate(particles):
            instance_data[i, 0:3] = p.position
            if p.charge > 0:
                instance_data[i, 3] = gui_params['scale_factor']
                instance_data[i, 4:7] = gui_params['positive_color']
            else:
                instance_data[i, 3] = gui_params['scale_factor'] * 0.8
                instance_data[i, 4:7] = gui_params['negative_color']
        
        glBindBuffer(GL_ARRAY_BUFFER, instance_VBO)
        glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
        glBindVertexArray(sphere_VAO)
        glDrawElementsInstanced(GL_TRIANGLES, len(sphere_indices), GL_UNSIGNED_INT, None, len(particles))
    
    imgui.new_frame()
    
    imgui.begin("Simulation Control", True)
    
    if imgui.collapsing_header("Particle Settings")[0]:
        changed, gui_params['num_particles'] = imgui.slider_int(
            "num of particle", gui_params['num_particles'], 100, gui_params['max_particles']
        )
        
        changed, gui_params['positive_charge'] = imgui.slider_float(
            "positive charge", gui_params['positive_charge'], 0.1, 10.0
        )
        
        changed, gui_params['negative_charge'] = imgui.slider_float(
            "negative charge", gui_params['negative_charge'], -10.0, -0.1
        )
        
        changed, gui_params['particle_mass'] = imgui.slider_float(
            "mass of particle", gui_params['particle_mass'], 0.1, 5.0
        )
        
        changed, gui_params['scale_factor'] = imgui.slider_float(
            "size of particle", gui_params['scale_factor'], 0.01, 0.1
        )
    
    if imgui.collapsing_header("Electric Field Settings")[0]:
        imgui.text(f"current E_field: ({gui_params['E_field_x']:.1f}, {gui_params['E_field_y']:.1f}, {gui_params['E_field_z']:.1f})")
        
        changed, gui_params['E_field_x'] = imgui.slider_float(
            "Field_E X", gui_params['E_field_x'], -50.0, 50.0
        )
        
        changed, gui_params['E_field_y'] = imgui.slider_float(
            "Field_E Y", gui_params['E_field_y'], -50.0, 50.0
        )
        
        changed, gui_params['E_field_z'] = imgui.slider_float(
            "Field_E Z", gui_params['E_field_z'], -50.0, 50.0
        )
        
        imgui.separator()
        if imgui.button("Field_E reset"):
            gui_params['E_field_x'] = 0.0
            gui_params['E_field_y'] = 0.0
            gui_params['E_field_z'] = 0.0
        
        imgui.same_line()
        if imgui.button("Y+direction"):
            gui_params['E_field_x'] = 0.0
            gui_params['E_field_y'] = 20.0
            gui_params['E_field_z'] = 0.0
        
        imgui.same_line()
        if imgui.button("Y-direction"):
            gui_params['E_field_x'] = 0.0
            gui_params['E_field_y'] = -20.0
            gui_params['E_field_z'] = 0.0
    
    if imgui.collapsing_header("Boundary Conditions")[0]:
        imgui.text(f"current boundary mode: {gui_params['boundary_mode_name']}")
        if imgui.button("switch"):
            boundary_mode = 1 - boundary_mode
            gui_params['boundary_mode_name'] = "rebounding" if boundary_mode == 1 else "warp"
        
        imgui.separator()
        imgui.text("Warp settings for each surface:")
        
        imgui.text("X-axis boundary:")
        changed, gui_params['warp_x_negative'] = imgui.checkbox(
            "warp_x_negative", gui_params['warp_x_negative']
        )
        changed, gui_params['warp_x_positive'] = imgui.checkbox(
            "warp_x_positive", gui_params['warp_x_positive']
        )
        
        imgui.text("Y-axis boundary:")
        changed, gui_params['warp_y_negative'] = imgui.checkbox(
            "warp_y_negative", gui_params['warp_y_negative']
        )
        changed, gui_params['warp_y_positive'] = imgui.checkbox(
            "warp_y_positive", gui_params['warp_y_positive']
        )
        
        imgui.text("Z-axis boundary:")
        changed, gui_params['warp_z_negative'] = imgui.checkbox(
            "warp_z_negative", gui_params['warp_z_negative']
        )
        changed, gui_params['warp_z_positive'] = imgui.checkbox(
            "warp_z_positive", gui_params['warp_z_positive']
        )
        
        imgui.separator()
        if imgui.button("All warp"):
            gui_params['warp_x_negative'] = True
            gui_params['warp_x_positive'] = True
            gui_params['warp_y_negative'] = True
            gui_params['warp_y_positive'] = True
            gui_params['warp_z_negative'] = True
            gui_params['warp_z_positive'] = True
        
        imgui.same_line()
        if imgui.button("All rebounding"):
            gui_params['warp_x_negative'] = False
            gui_params['warp_x_positive'] = False
            gui_params['warp_y_negative'] = False
            gui_params['warp_y_positive'] = False
            gui_params['warp_z_negative'] = False
            gui_params['warp_z_positive'] = False
        
        imgui.same_line()
        if imgui.button("X-axis only"):
            gui_params['warp_x_negative'] = True
            gui_params['warp_x_positive'] = True
            gui_params['warp_y_negative'] = False
            gui_params['warp_y_positive'] = False
            gui_params['warp_z_negative'] = False
            gui_params['warp_z_positive'] = False
        
        imgui.same_line()
        if imgui.button("Y-axis only"):
            gui_params['warp_x_negative'] = False
            gui_params['warp_x_positive'] = False
            gui_params['warp_y_negative'] = True
            gui_params['warp_y_positive'] = True
            gui_params['warp_z_negative'] = False
            gui_params['warp_z_positive'] = False
        
        imgui.same_line()
        if imgui.button("Z-axis only"):
            gui_params['warp_x_negative'] = False
            gui_params['warp_x_positive'] = False
            gui_params['warp_y_negative'] = False
            gui_params['warp_y_positive'] = False
            gui_params['warp_z_negative'] = True
            gui_params['warp_z_positive'] = True
    
    if imgui.collapsing_header("Simulation Settings")[0]:
        changed, gui_params['dt'] = imgui.slider_float(
            "time step", gui_params['dt'], 0.001, 0.1
        )
        
        changed, gui_params['simulation_paused'] = imgui.checkbox(
            "simulation paused", gui_params['simulation_paused']
        )
        
        if imgui.button("reset particles"):
            particles = create_particles(gui_params['num_particles'])
            if hasattr(update_particles_optimized, 'gpu_arrays'):
                delattr(update_particles_optimized, 'gpu_arrays')
            print("粒子をリセットしました")
        
        imgui.same_line()
        if imgui.button("reset velocity"):
            for p in particles:
                p.velocity = np.zeros(3)
            print("全粒子の速度をリセットしました")
    
    if imgui.collapsing_header("display settings")[0]:
        changed, gui_params['show_bbox'] = imgui.checkbox(
            "bounding box display", gui_params['show_bbox']
        )
        
        changed, gui_params['positive_color'] = imgui.color_edit3(
            "positive color", *gui_params['positive_color']
        )
        
        changed, gui_params['negative_color'] = imgui.color_edit3(
            "negative color", *gui_params['negative_color']
        )
        
        if imgui.button("reset color"):
            gui_params['positive_color'] = [0.4, 0.8, 1.0]
            gui_params['negative_color'] = [1.0, 0.5, 0.0]
    
    if imgui.collapsing_header("statistics information")[0]:
        imgui.text(f"Statistics Current Particle Count: {len(particles)}")
        positive_count = sum(1 for p in particles if p.charge > 0)
        negative_count = len(particles) - positive_count
        imgui.text(f"positive charge: {positive_count}")
        imgui.text(f"negative charge: {negative_count}")
        
        if len(particles) > 0:
            avg_speed = np.mean([np.linalg.norm(p.velocity) for p in particles])
            imgui.text(f"average velocity: {avg_speed:.2f}")
        
        if len(frame_times) > 0:
            avg_frame_time = np.mean(frame_times[-60:]) if len(frame_times) >= 60 else np.mean(frame_times)
            fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
            imgui.text(f"FPS: {fps:.1f}")
            imgui.text(f"frame time: {avg_frame_time:.2f}ms")
        
        imgui.separator()
        imgui.text("E-field grid information:")
        imgui.text(f"grid division: {grid_divisions}×{grid_divisions}×{grid_divisions}")
        imgui.text(f"cell size: {cell_size:.3f}")
    
    if imgui.collapsing_header("preset")[0]:
        if imgui.button("default settings"):
            gui_params.update({
                'num_particles': 3000,
                'E_field_x': 0.0,
                'E_field_y': 15.0,
                'E_field_z': 0.0,
                'positive_charge': 2.0,
                'negative_charge': -2.0,
                'particle_mass': 1.0,
                'dt': 0.01,
                'scale_factor': 0.03,
                'positive_color': [0.4, 0.8, 1.0],
                'negative_color': [1.0, 0.5, 0.0]
            })
            particles = create_particles(gui_params['num_particles'])
        
        if imgui.button("Y+ E-field test"):
            gui_params.update({
                'num_particles': 2000,
                'E_field_x': 0.0,
                'E_field_y': 25.0,
                'E_field_z': 0.0,
                'positive_charge': 2.0,
                'negative_charge': -2.0,
                'particle_mass': 1.0,
                'dt': 0.008,
                'warp_x_negative': False, 'warp_x_positive': False,
                'warp_y_negative': True, 'warp_y_positive': True,
                'warp_z_negative': False, 'warp_z_positive': False
            })
            particles = create_particles(gui_params['num_particles'])
            print("Y+電場テスト設定を適用")
        
        if imgui.button("Y- E-field test"):
            gui_params.update({
                'num_particles': 2000,
                'E_field_x': 0.0,
                'E_field_y': -25.0,
                'E_field_z': 0.0,
                'positive_charge': 2.0,
                'negative_charge': -2.0,
                'particle_mass': 1.0,
                'dt': 0.008,
                'warp_x_negative': False, 'warp_x_positive': False,
                'warp_y_negative': True, 'warp_y_positive': True,
                'warp_z_negative': False, 'warp_z_positive': False
            })
            particles = create_particles(gui_params['num_particles'])
            print("Y-電場テスト設定を適用")
        
        if imgui.button("gravity field simulation"):
            gui_params.update({
                'num_particles': 3000,
                'E_field_x': 0.0,
                'E_field_y': -20.0,
                'E_field_z': 0.0,
                'positive_charge': 1.0,
                'negative_charge': -1.0,
                'particle_mass': 2.0,
                'dt': 0.008
            })
            particles = create_particles(gui_params['num_particles'])
        
        if imgui.button("spiral E-field"):
            gui_params.update({
                'num_particles': 4000,
                'E_field_x': 10.0,
                'E_field_y': 0.0,
                'E_field_z': 30.0,
                'positive_charge': 3.0,
                'negative_charge': -1.5,
                'particle_mass': 0.8,
                'dt': 0.005
            })
            particles = create_particles(gui_params['num_particles'])
        
        if imgui.button("High density simulation"):
            gui_params.update({
                'num_particles': 8000,
                'E_field_x': 5.0,
                'E_field_y': 5.0,
                'E_field_z': 25.0,
                'positive_charge': 1.5,
                'negative_charge': -2.5,
                'particle_mass': 1.2,
                'dt': 0.006,
                'scale_factor': 0.02
            })
            particles = create_particles(gui_params['num_particles'])
    
    imgui.end()
    
    imgui.begin("操作ヘルプ", True)
    imgui.text("カメラ操作:")
    imgui.text("  矢印キー: カメラ回転")
    imgui.text("  W/S: ズームイン/アウト")
    imgui.text("  C: 背景色切り替え")
    imgui.text("  I: 境界条件切り替え")
    imgui.separator()
    imgui.text("電場テスト方法:")
    imgui.text("1. 'Y+電場テスト'プリセットを選択")
    imgui.text("2. 正電荷が+Y方向に移動することを確認")
    imgui.text("3. 'Y-電場テスト'プリセットを選択")
    imgui.text("4. 正電荷が-Y方向に移動することを確認")
    imgui.separator()
    imgui.text("境界条件:")
    imgui.text("  ワープ: 粒子が反対側から出現")
    imgui.text("  跳ね返り: 粒子が壁で反射")
    imgui.text("  Y軸のみワープで電場効果を観察")
    imgui.end()
    
    imgui.render()
    impl.render(imgui.get_draw_data())
    
    glfw.swap_buffers(window)
    
    end_time = time.time()
    frame_time = (end_time - start_time) * 1000
    frame_times.append(frame_time)
    frame_count += 1
    
    if frame_count % 60 == 0:
        avg_frame_time = np.mean(frame_times[-60:])
        fps = 1000.0 / avg_frame_time
        print(f"平均フレーム時間: {avg_frame_time:.2f}ms, FPS: {fps:.1f}")
    
    target_frame_time = 1000.0 / 60.0
    if frame_time < target_frame_time:
        time.sleep((target_frame_time - frame_time) / 1000.0)

impl.shutdown()
glfw.terminate()