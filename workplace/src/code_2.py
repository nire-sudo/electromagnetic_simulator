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

from pycuda.compiler import SourceModule

def load_cuda_kernel(filepath, funcname):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    return SourceModule(code).get_function(funcname)

cuda_func = load_cuda_kernel("./cuda_program/cal_physics.cu", "update_particles_cuda")
update_particles_kernel = load_cuda_kernel("./cuda_program/update_particles_gpu.cu", "update_particles_gpu")

# 境界条件フラグ（0: 跳ね返り, 1: ワープ）
boundary_mode = 0
i_switch = 0.0

# === GLFW 初期化 ===
if not glfw.init():
    raise Exception("GLFWの初期化に失敗しました")

# カメラの角度
camera_theta = 3.14 / 4.0
camera_phi = np.radians(30.0)

monitor = glfw.get_monitors()[1]
mode = glfw.get_video_mode(monitor)

xpos, ypos = glfw.get_monitor_pos(monitor)
glfw.window_hint(glfw.DECORATED, glfw.FALSE)

window = glfw.create_window(mode.size.width, mode.size.height, "10K粒子シミュレーション", None, None)
if not window:
    glfw.terminate()
    raise Exception("ウィンドウの作成に失敗しました")

glfw.set_window_pos(window, xpos, ypos)
glfw.make_context_current(window)

current_bg_color = [1, 1, 1, 1]
glClearColor(*current_bg_color)
bg_color_flag = False

# === OpenGL情報出力 ===
print("Renderer:", glGetString(GL_RENDERER).decode())
print("Vendor:", glGetString(GL_VENDOR).decode())
print("Version:", glGetString(GL_VERSION).decode())

# === インスタンシング用シェーダー ===
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

# === バウンディングボックス用シェーダー ===
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

# === メインシェーダープログラム ===
shader_program = glCreateProgram()
vs = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
fs = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
glAttachShader(shader_program, vs)
glAttachShader(shader_program, fs)
glLinkProgram(shader_program)
glDeleteShader(vs)
glDeleteShader(fs)

# === バウンディングボックスシェーダープログラム ===
bbox_shader_program = glCreateProgram()
bbox_vs = compile_shader(BBOX_VERTEX_SHADER, GL_VERTEX_SHADER)
bbox_fs = compile_shader(BBOX_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
glAttachShader(bbox_shader_program, bbox_vs)
glAttachShader(bbox_shader_program, bbox_fs)
glLinkProgram(bbox_shader_program)
glDeleteShader(bbox_vs)
glDeleteShader(bbox_fs)

# === Uniform位置取得 ===
glUseProgram(shader_program)
view_loc = glGetUniformLocation(shader_program, "view")
proj_loc = glGetUniformLocation(shader_program, "projection")
light_pos_loc = glGetUniformLocation(shader_program, "lightPos")
view_pos_loc = glGetUniformLocation(shader_program, "viewPos")

glUseProgram(bbox_shader_program)
bbox_view_loc = glGetUniformLocation(bbox_shader_program, "view")
bbox_proj_loc = glGetUniformLocation(bbox_shader_program, "projection")
bbox_color_loc = glGetUniformLocation(bbox_shader_program, "lineColor")

# === 色定義 ===
color_positive = np.array([0.4, 0.8, 1.0], dtype=np.float32)
color_negative = np.array([1.0, 0.5, 0.0], dtype=np.float32)

# === バウンディングボックス作成 ===
def create_bounding_box():
    size = 4.3
    half = size / 2
    
    # 立方体の8頂点
    corners = np.array([
        [-half, -half, -half], [half, -half, -half], [half, half, -half], [-half, half, -half],
        [-half, -half, half], [half, -half, half], [half, half, half], [-half, half, half],
    ], dtype=np.float32)
    
    # 立方体のエッジ
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 奥面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 手前面
        (0, 4), (1, 5), (2, 6), (3, 7),  # 奥-手前
    ]
    
    # 分割線
    num_divisions = 3
    step_size = size / num_divisions
    steps = [-half + step_size, half - step_size]
    
    vertices = []
    
    # エッジ
    for start, end in edges:
        vertices.extend([corners[start], corners[end]])
    
    # 分割線
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

# === 電磁気シミュレーション ===
grid_size = (80, 80, 80)
cell_size = 0.5
grid_origin = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
potential_grid = np.zeros(grid_size, dtype=np.float32)
E_field = np.zeros(grid_size + (3,), dtype=np.float32)
E_field[:, :, :, :] = [0.0, 0, 15]

class ChargedParticle:
    def __init__(self, position, velocity, charge, mass):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.charge = charge
        self.mass = mass

# === 10,000個の粒子生成 ===
num_particles = 5000
particles = []

# 正電荷 (5000個)
for _ in range(num_particles // 2):
    position = np.random.uniform(-2, 2, 3)
    velocity = np.zeros(3)
    charge = 2.0
    mass = 1.0
    particles.append(ChargedParticle(position, velocity, charge, mass))

# 負電荷 (5000個)
for _ in range(num_particles // 2):
    position = np.random.uniform(-2, 2, 3)
    velocity = np.zeros(3)
    charge = -2.0
    mass = 1.0
    particles.append(ChargedParticle(position, velocity, charge, mass))

dt = 0.01

# === GPU最適化された粒子更新 ===
def update_particles_optimized(particles, E_field):
    N = len(particles)
    positions_np = np.array([tuple(p.position) for p in particles], dtype=np.float32)
    velocities_np = np.array([tuple(p.velocity) for p in particles], dtype=np.float32)
    charges_np = np.array([p.charge for p in particles], dtype=np.float32)
    masses_np = np.array([p.mass for p in particles], dtype=np.float32)
    E_field_np = np.array(E_field, dtype=np.float32).reshape(-1, 3)

    # GPUメモリ割り当て（一度だけ）
    if not hasattr(update_particles_optimized, 'gpu_arrays'):
        update_particles_optimized.gpu_arrays = {
            'pos': cuda.mem_alloc(positions_np.nbytes),
            'vel': cuda.mem_alloc(velocities_np.nbytes),
            'chg': cuda.mem_alloc(charges_np.nbytes),
            'mas': cuda.mem_alloc(masses_np.nbytes),
            'ef': cuda.mem_alloc(E_field_np.nbytes)
        }
    
    gpu_arrays = update_particles_optimized.gpu_arrays
    
    # データ転送
    cuda.memcpy_htod(gpu_arrays['pos'], positions_np)
    cuda.memcpy_htod(gpu_arrays['vel'], velocities_np)
    cuda.memcpy_htod(gpu_arrays['chg'], charges_np)
    cuda.memcpy_htod(gpu_arrays['mas'], masses_np)
    cuda.memcpy_htod(gpu_arrays['ef'], E_field_np)

    # 最適化されたブロックサイズ
    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    # カーネル実行
    cuda_func(
        gpu_arrays['pos'], gpu_arrays['vel'], gpu_arrays['chg'], gpu_arrays['mas'],
        gpu_arrays['ef'], np.int32(E_field.shape[0]), np.int32(E_field.shape[1]), np.int32(E_field.shape[2]),
        np.float32(cell_size), np.int32(N), np.float32(1.0), np.float32(0.1), np.float32(dt), np.float32(4.2),
        np.int32(boundary_mode),  # 境界条件フラグを追加
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )

    # 結果取得
    cuda.memcpy_dtoh(positions_np, gpu_arrays['pos'])
    cuda.memcpy_dtoh(velocities_np, gpu_arrays['vel'])
    
    # 粒子データ更新
    for i, p in enumerate(particles):
        p.position = positions_np[i]
        p.velocity = velocities_np[i]

# === 球モデル読み込み ===
sphere_mesh = trimesh.load("./3D_models/sphere.obj", force='mesh')
sphere_vertices = np.array(sphere_mesh.vertices, dtype=np.float32)
sphere_normals = np.array(sphere_mesh.vertex_normals, dtype=np.float32)
sphere_indices = np.array(sphere_mesh.faces, dtype=np.uint32).flatten()

# === インスタンシング用VAO設定 ===
sphere_VAO = glGenVertexArrays(1)
sphere_VBO = glGenBuffers(1)
sphere_EBO = glGenBuffers(1)
instance_VBO = glGenBuffers(1)

glBindVertexArray(sphere_VAO)

# 頂点データ
sphere_vertex_data = np.hstack([sphere_vertices, sphere_normals])
glBindBuffer(GL_ARRAY_BUFFER, sphere_VBO)
glBufferData(GL_ARRAY_BUFFER, sphere_vertex_data.nbytes, sphere_vertex_data, GL_STATIC_DRAW)

# インデックス
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_indices.nbytes, sphere_indices, GL_STATIC_DRAW)

# 頂点属性
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

# インスタンスデータ用バッファ
instance_data = np.zeros((num_particles, 7), dtype=np.float32)  # pos(3) + scale(1) + color(3)
glBindBuffer(GL_ARRAY_BUFFER, instance_VBO)
glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)

# インスタンス属性
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

# === 投影行列 ===
projection = Matrix44.perspective_projection(45.0, mode.size.width/mode.size.height, 0.1, 100.0)

# === 設定 ===
glEnable(GL_DEPTH_TEST)
glEnable(GL_CULL_FACE)  # 裏面カリング有効化

angle_speed = 0.02
phi_limit = np.radians(89.0)
radius = 10
scale_factor = 0.03
zoom_speed = 0.1
min_radius = 1.0
max_radius = 20.0
c_switch = 0.0

# === 操作説明 ===
print("=== 操作方法 ===")
print("矢印キー: カメラ回転")
print("W/S: ズーム")
print("C: 背景色切り替え")
print("I: 境界条件切り替え（跳ね返り/ワープ）")
print("===============")

# === パフォーマンス計測 ===
frame_times = []
frame_count = 0

# === メインループ ===
while not glfw.window_should_close(window):
    start_time = time.time()
    
    glfw.poll_events()
    
    # 物理更新
    update_particles_optimized(particles, E_field)
    
    # 境界条件切り替え
    if i_switch >= 0:
        i_switch -= 0.1
        
    if glfw.get_key(window, glfw.KEY_I) == glfw.PRESS:
        if i_switch < 0:
            boundary_mode = 1 - boundary_mode  # 0→1, 1→0
            mode_text = "跳ね返り" if boundary_mode == 1 else "ワープ"
            print(f"境界条件: {mode_text}")
            i_switch = 2
    
    # 背景色切り替え
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
    
    # カメラ操作
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

    # カメラ行列計算
    camX = radius * np.cos(camera_phi) * np.sin(camera_theta)
    camY = radius * np.sin(camera_phi)
    camZ = radius * np.cos(camera_phi) * np.cos(camera_theta)
    eye = np.array([camX, camY, camZ])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    view = Matrix44.look_at(eye, target, up)

    # === バウンディングボックス描画 ===
    if bg_color_flag == False:
        glUseProgram(bbox_shader_program)
        glUniformMatrix4fv(bbox_view_loc, 1, GL_FALSE, view.astype(np.float32))
        glUniformMatrix4fv(bbox_proj_loc, 1, GL_FALSE, projection.astype(np.float32))
        glUniform3fv(bbox_color_loc, 1, np.array([0, 0, 0], dtype=np.float32))
        
        glBindVertexArray(bbox_VAO)
        glDrawArrays(GL_LINES, 0, len(bbox_vertices))
    
    # === 粒子描画（インスタンシング） ===
    glUseProgram(shader_program)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view.astype(np.float32))
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection.astype(np.float32))
    glUniform3fv(light_pos_loc, 1, np.array([5.0, 5.0, 5.0], dtype=np.float32))
    glUniform3fv(view_pos_loc, 1, eye.astype(np.float32))
    
    # インスタンスデータ更新
    for i, p in enumerate(particles):
        instance_data[i, 0:3] = p.position  # 位置
        if p.charge > 0:
            instance_data[i, 3] = scale_factor  # スケール
            instance_data[i, 4:7] = color_positive  # 色
        else:
            instance_data[i, 3] = scale_factor * 0.8  # スケール
            instance_data[i, 4:7] = color_negative  # 色
    
    # インスタンスデータをGPUに転送
    glBindBuffer(GL_ARRAY_BUFFER, instance_VBO)
    glBufferSubData(GL_ARRAY_BUFFER, 0, instance_data.nbytes, instance_data)
    
    # インスタンス描画
    glBindVertexArray(sphere_VAO)
    glDrawElementsInstanced(GL_TRIANGLES, len(sphere_indices), GL_UNSIGNED_INT, None, num_particles)
    
    glfw.swap_buffers(window)
    
    # === パフォーマンス計測 ===
    end_time = time.time()
    frame_time = (end_time - start_time) * 1000
    frame_times.append(frame_time)
    frame_count += 1
    
    if frame_count % 60 == 0:
        avg_frame_time = np.mean(frame_times[-60:])
        fps = 1000.0 / avg_frame_time
        print(f"平均フレーム時間: {avg_frame_time:.2f}ms, FPS: {fps:.1f}")
    
    # フレームレート制限（必要に応じて）
    target_frame_time = 1000.0 / 60.0
    if frame_time < target_frame_time:
        time.sleep((target_frame_time - frame_time) / 1000.0)

glfw.terminate()