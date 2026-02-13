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

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp

import matplotlib.pyplot as plt

import pycuda.gl as cuda_gl
from pycuda.gl import graphics_map_flags


boundary_mode = 0
i_switch = 0.0
vbo_resource = None  # CUDA-OpenGL共有リソース
cuda_instance_data = None  # CUDA側の共有データポインタ

# === CUDAをロード ===
def load_cuda_kernel(filepath, funcname):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    return SourceModule(code).get_function(funcname)

cuda_func = load_cuda_kernel("./cuda_program/cal_physics_test.cu", "update_particles_cuda")
charge_density_func = load_cuda_kernel("./cuda_program/cal_physics_test.cu", "compute_charge_density")
electric_field_func = load_cuda_kernel("./cuda_program/cal_physics_test.cu", "compute_electric_field_from_potential")

# === GLFW 初期化 ===
if not glfw.init():
    raise Exception("GLFWの初期化に失敗しました")

camera_theta = 3.14 / 4.0
camera_phi = np.radians(30.0)

monitor = glfw.get_monitors()[0]
mode = glfw.get_video_mode(monitor)


xpos, ypos = glfw.get_monitor_pos(monitor)
glfw.window_hint(glfw.DECORATED, glfw.FALSE)

window = glfw.create_window(mode.size.width, mode.size.height, "Poisson Equation Particle Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("ウィンドウの作成に失敗しました")

glfw.set_window_pos(window, xpos, ypos)
glfw.make_context_current(window)
glfw.swap_interval(0)

imgui.create_context()
impl = GlfwRenderer(window)

current_bg_color = [1, 1, 1, 1]
glClearColor(*current_bg_color)
bg_color_flag = False

print("Renderer:", glGetString(GL_RENDERER).decode())
print("Vendor:", glGetString(GL_VENDOR).decode())
print("Version:", glGetString(GL_VERSION).decode())

# === GLSLシェーダープログラム ===
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;      //モデルの頂点座標
layout(location = 1) in vec3 normal;        //頂点法線
layout(location = 2) in vec3 instancePos;   //インスタンスごとの位置
layout(location = 3) in float instanceScale;//インスタンスごとのスケール
layout(location = 4) in vec3 instanceColor; //インスタンスごとの色

out vec3 fragNormal;
out vec3 fragPos;
out vec3 fragColor;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    vec3 scaledPos = position * instanceScale;         //モデルをスケーリング
    vec4 worldPos = vec4(scaledPos + instancePos, 1.0);//ワールド座標系に変換
    
    fragPos = worldPos.xyz;   //フラグメントシェーダー用のワールド座標
    fragNormal = normal;      //法線ベクトル
    fragColor = instanceColor;//インスタンスの色
    
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
    vec3 norm = normalize(fragNormal);            //正規化された法線
    vec3 lightDir = normalize(lightPos - fragPos);//ライトから頂点への方向ベクトル
    
    // 環境光
    float ambient = 0.3;
    
    // 拡散光
    float diff = max(dot(norm, lightDir), 0.0);
    
    // 鏡面反射
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    
    //環境光、拡散光、鏡面反射を合成し、インスタンスの色を掛け合わせて出力
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

VECTOR_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 instancePos;
layout(location = 3) in vec3 instanceDir;
layout(location = 4) in float instanceScale;
layout(location = 5) in vec3 instanceColor;

out vec3 fragNormal;
out vec3 fragPos;
out vec3 fragColor;

uniform mat4 projection;
uniform mat4 view;

mat3 lookAtMatrix(vec3 direction) {
    vec3 up = abs(direction.y) < 0.9 ? vec3(0, 1, 0) : vec3(1, 0, 0);
    vec3 right = normalize(cross(up, direction));
    up = cross(direction, right);
    return mat3(right, up, direction);
}

void main()
{
    mat3 rotation = lookAtMatrix(normalize(instanceDir));
    vec3 rotatedPos = rotation * (position * instanceScale);
    vec4 worldPos = vec4(rotatedPos + instancePos, 1.0);
    
    fragPos = worldPos.xyz;
    fragNormal = rotation * normal;
    fragColor = instanceColor;
    
    gl_Position = projection * view * worldPos;
}
"""

EQUIPOTENTIAL_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

out vec3 fragColor;

uniform mat4 projection;
uniform mat4 view;

void main()
{
    fragColor = color;
    gl_Position = projection * view * vec4(position, 1.0);
}
"""

EQUIPOTENTIAL_FRAGMENT_SHADER = """
#version 330 core
in vec3 fragColor;
out vec4 FragColor;

void main()
{
    FragColor = vec4(fragColor, 1.0);
}
"""

# === シェーダープログラムのセットアップ ===
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)              # 空のシェーダーオブジェクト作成
    glShaderSource(shader, source)                    # GLSLソースコードをシェーダーにセット
    glCompileShader(shader)                           # コンパイル
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):  # コンパイル結果の確認
        raise RuntimeError(glGetShaderInfoLog(shader).decode())  # エラー内容を表示
    return shader

# 通常描画用シェーダープログラム
shader_program = glCreateProgram()
vs = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)   # 頂点シェーダーをコンパイル
fs = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER) # フラグメントシェーダーをコンパイル
glAttachShader(shader_program, vs)                     # プログラムにアタッチ
glAttachShader(shader_program, fs)
glLinkProgram(shader_program)                          # 頂点＆フラグメントをリンクして実行可能に
glDeleteShader(vs)                                     # 単体のシェーダーオブジェクトは削除
glDeleteShader(fs)

# バウンディングボックス用シェーダープログラム
bbox_shader_program = glCreateProgram()
bbox_vs = compile_shader(BBOX_VERTEX_SHADER, GL_VERTEX_SHADER)
bbox_fs = compile_shader(BBOX_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
glAttachShader(bbox_shader_program, bbox_vs)
glAttachShader(bbox_shader_program, bbox_fs)
glLinkProgram(bbox_shader_program)
glDeleteShader(bbox_vs)
glDeleteShader(bbox_fs)

vector_shader_program = glCreateProgram()
vector_vs = compile_shader(VECTOR_VERTEX_SHADER, GL_VERTEX_SHADER)
vector_fs = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)  # 同じフラグメントシェーダーを使用
glAttachShader(vector_shader_program, vector_vs)
glAttachShader(vector_shader_program, vector_fs)
glLinkProgram(vector_shader_program)
glDeleteShader(vector_vs)
glDeleteShader(vector_fs)

equipotential_shader_program = glCreateProgram()
equi_vs = compile_shader(EQUIPOTENTIAL_VERTEX_SHADER, GL_VERTEX_SHADER)
equi_fs = compile_shader(EQUIPOTENTIAL_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
glAttachShader(equipotential_shader_program, equi_vs)
glAttachShader(equipotential_shader_program, equi_fs)
glLinkProgram(equipotential_shader_program)
glDeleteShader(equi_vs)
glDeleteShader(equi_fs)

# Uniform変数のロケーション取得
glUseProgram(shader_program)
view_loc = glGetUniformLocation(shader_program, "view")
proj_loc = glGetUniformLocation(shader_program, "projection")
light_pos_loc = glGetUniformLocation(shader_program, "lightPos")
view_pos_loc = glGetUniformLocation(shader_program, "viewPos")

glUseProgram(bbox_shader_program)
bbox_view_loc = glGetUniformLocation(bbox_shader_program, "view")
bbox_proj_loc = glGetUniformLocation(bbox_shader_program, "projection")
bbox_color_loc = glGetUniformLocation(bbox_shader_program, "lineColor")

glUseProgram(vector_shader_program)
vector_view_loc = glGetUniformLocation(vector_shader_program, "view")
vector_proj_loc = glGetUniformLocation(vector_shader_program, "projection")
vector_light_pos_loc = glGetUniformLocation(vector_shader_program, "lightPos")
vector_view_pos_loc = glGetUniformLocation(vector_shader_program, "viewPos")

glUseProgram(equipotential_shader_program)
equi_view_loc = glGetUniformLocation(equipotential_shader_program, "view")
equi_proj_loc = glGetUniformLocation(equipotential_shader_program, "projection")
equi_color_loc = glGetUniformLocation(equipotential_shader_program, "lineColor")

# 粒子の色設定
color_positive = np.array([0.4, 0.8, 1.0], dtype=np.float32)
color_negative = np.array([1.0, 0.5, 0.0], dtype=np.float32)

max_electric_field = 50.0
max_magnetic_field = 50.0

# 初期パラメータ設定
gui_params = {
    'num_particles': 10000,
    'max_particles': 30000,
    'external_E_field_x': 0.0,
    'external_E_field_y': 0.0, 
    'external_E_field_z': 0.0,
    'positive_charge': 1.0,
    'negative_charge': -1.0,
    'particle_mass': 1.0,
    'dt': 0.005,
    'scale_factor': 0.03,
    'boundary_mode_name': "ワープ",
    'show_bbox': True,
    'simulation_paused': False,
    'positive_color': [0.4, 0.8, 1.0],
    'negative_color': [1.0, 0.5, 0.0],
    'warp_x_negative': False,
    'warp_x_positive': False,
    'warp_y_negative': False,
    'warp_y_positive': False,
    'warp_z_negative': False,
    'warp_z_positive': False,
    'use_poisson': True,
    'electrode_mode': False,
    'electrode_voltage': 10.0,
    'electrode_positions': {'x_neg': False, 'x_pos': False, 'y_neg': True, 'y_pos': False, 'z_neg': False, 'z_pos': False},
    'geometry_mode': 0,
    'cylinder_radius': 1.0,
    'cylinder_height': 4.0,
    'geometry_mode_cylinder': 0,
    'cylinder_top_warp': False,
    'cylinder_bottom_warp': False,
    'cylinder_wall_vanish': False,
    'enable_emitter': False,
    'emitter_position': [0.0, -2.0, 0.0],
    'emitter_direction': [0.0, 1.0, 0.0],
    'emitter_speed': 5.0,
    'emitter_spread': 0.0,
    'emitter_charge_type': 0,
    'emitter_rate': 1,
    'max_emitted_particles': 5000,
    'magnetic_field_x': 0.0,
    'magnetic_field_y': 0.0,
    'magnetic_field_z': 0.0,  # デフォルトでZ軸方向に磁場
    'enable_magnetic_field': True,
    'cyclotron_frequency': 1.0,  # サイクロトロン周波数の表示用
    'coulomb_force_control': 1.0,
    'show_field_vectors': False,
    'field_vector_scale': 0.1,
    'field_sample_spacing': 2,  # グリッドのサンプリング間隔
    'show_electric_field': False,
    'show_magnetic_field': False,
    'electric_field_color': [1.0, 1.0, 0],  
    'magnetic_field_color': [0.3, 1.0, 0.3],
    'dt_safety_factor': 0.3,
    'rf_frequency': 1.0,
    'rf_amplitude': 50.0,
    'rf_phase': 0.0,
    'use_rf_voltage': False,
    'show_equipotential': False,
    'equipotential_plane': 0,
    'equipotential_slice_position': 0.0,
    'equipotential_num_lines': 10,
    'equipotential_color': [0.0, 0.5, 1.0],
    'internal_electrodes': [],
    'electrode_shapes': ['sphere', 'box', 'cylinder'],
    'current_electrode_shape': 0,
    'new_electrode_position': [0.0, 0.0, 0.0],
    'new_electrode_size': 0.3,
    'new_electrode_voltage': 10.0,
    'show_electrodes': True,
    'cube_vanish_x_neg': False,
    'cube_vanish_x_pos': False,
    'cube_vanish_y_neg': False,
    'cube_vanish_y_pos': False,
    'cube_vanish_z_neg': False,
    'cube_vanish_z_pos': False,
    'length_scale_mode': 'microscopic',
    'time_scale_mode': 'nanosecond',
    'show_physical_units': True,
    'auto_normalize_constants': True,
    'display_si_values': False,
    
    # プリセット設定
    'preset_scenarios': [
        'Custom',
        'Electron Beam (電子ビーム)',
        'Ion Trap (イオントラップ)',
        'Plasma Simulation (プラズマ)',
        'Dust Plasma (ダストプラズマ)',
        'Nanoscale Device (ナノデバイス)'
    ],
    'current_preset': 0,
}

# === 円柱内部の点かどうかを判定 ===
def is_inside_cylinder(i, j, k, nx, ny, nz, cylinder_radius, cylinder_height):
    # グリッド座標を物理座標に変換(?)
    x = (i - nx/2) * (BOX_SIZE / nx)
    y = (j - ny/2) * (BOX_SIZE / ny) 
    z = (k - nz/2) * (BOX_SIZE / nz)
    
    # 円柱の判定
    r_squared = x*x + z*z
    return (r_squared <= cylinder_radius*cylinder_radius and 
            abs(y) <= cylinder_height/2)

# === バウンディングボックスの描画処理 ===
def create_bounding_box():
    if gui_params['geometry_mode'] == 0:
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
        
        # 外枠の線を追加
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
    else:
        radius = gui_params['cylinder_radius']
        height = gui_params['cylinder_height']
        half_height = height/2
        segments = 24

        vertices = []

        # 上下の円
        for i in range(segments):
            angle1 = 2 * np.pi * i / segments
            angle2 = 2 * np.pi * ((i + 1) % segments) / segments
            
            x1, z1 = radius * np.cos(angle1), radius * np.sin(angle1)
            x2, z2 = radius * np.cos(angle2), radius * np.sin(angle2)
            
            # 上の円
            vertices.extend([[x1, half_height, z1], [x2, half_height, z2]])
            # 下の円
            vertices.extend([[x1, -half_height, z1], [x2, -half_height, z2]])
            # 縦線
            vertices.extend([[x1, -half_height, z1], [x1, half_height, z1]])
    
    return np.array(vertices, dtype=np.float32)

bbox_vertices = create_bounding_box()
bbox_VAO = glGenVertexArrays(1)  # 頂点配列オブジェクト
bbox_VBO = glGenBuffers(1)       # 頂点バッファオブジェクト

glBindVertexArray(bbox_VAO)             # 頂点属性設定の保存の有効化
glBindBuffer(GL_ARRAY_BUFFER, bbox_VBO) # 頂点データを入れるGPUメモリの有効化
glBufferData(GL_ARRAY_BUFFER, bbox_vertices.nbytes, bbox_vertices, GL_STATIC_DRAW) #頂点データをGPUに送信
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
glBindVertexArray(0)

# 等電位線用VAO/VBO
equipotential_VAO = glGenVertexArrays(1)
equipotential_VBO = glGenBuffers(1)

glBindVertexArray(equipotential_VAO)
glBindBuffer(GL_ARRAY_BUFFER, equipotential_VBO)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
glBindVertexArray(0)

# 電極ワイヤーフレーム用VAO/VBO
electrode_wireframe_VAO = glGenVertexArrays(1)
electrode_wireframe_VBO = glGenBuffers(1)

glBindVertexArray(electrode_wireframe_VAO)
glBindBuffer(GL_ARRAY_BUFFER, electrode_wireframe_VBO)
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
glBindVertexArray(0)

#GPUメモリを完全にクリアする関数
def clear_gpu_arrays():
    if hasattr(update_particles_optimized, 'gpu_arrays'):
        for arr in update_particles_optimized.gpu_arrays.values():
            try:
                arr.free()
            except:
                pass
        delattr(update_particles_optimized, 'gpu_arrays')
        if hasattr(update_particles_optimized, 'last_N'):
            delattr(update_particles_optimized, 'last_N')
    
    # update_electric_field_poisson_optimized のGPUメモリをクリア
    if hasattr(update_electric_field_poisson_optimized, 'gpu_arrays'):
        for arr in update_electric_field_poisson_optimized.gpu_arrays.values():
            try:
                arr.free()
            except:
                pass
        delattr(update_electric_field_poisson_optimized, 'gpu_arrays')
        if hasattr(update_electric_field_poisson_optimized, 'last_N'):
            delattr(update_electric_field_poisson_optimized, 'last_N')
    
    # field_gpu_arraysもクリア
    if hasattr(update_electric_field_poisson_optimized, 'field_gpu_arrays'):
        for arr in update_electric_field_poisson_optimized.field_gpu_arrays.values():
            try:
                arr.free()
            except:
                pass
        delattr(update_electric_field_poisson_optimized, 'field_gpu_arrays')
    
    # CUDAコンテキストの同期
    cuda.Context.synchronize()
    print("All GPU resources cleared")


# === 電磁気シミュレーション処理 ===
BOX_SIZE = 4.2     # シミュレーション空間のサイズ
grid_divisions = 8 # 各軸を8分割
cell_size = BOX_SIZE / grid_divisions # 各格子のサイズ
grid_size = (grid_divisions, grid_divisions, grid_divisions)

# 高解像度グリッド用の変数
high_res_grid_divisions = 16
high_res_cell_size = BOX_SIZE / high_res_grid_divisions
high_res_potential_grid = None
high_res_charge_density_grid = None
high_res_poisson_solver = None

potential_grid = np.zeros((grid_divisions, grid_divisions, grid_divisions), dtype=np.float32)
charge_density_grid = np.zeros((grid_divisions, grid_divisions, grid_divisions), dtype=np.float32)
E_field = np.zeros((grid_divisions, grid_divisions, grid_divisions, 3), dtype=np.float32)

class PhysicsScale:
    """シミュレーションの物理スケール定義"""
    
    def __init__(self):
        # === 基本物理定数 ===
        self.epsilon_0 = 8.854e-12  # 真空の誘電率 [F/m]
        self.mu_0 = 4 * np.pi * 1e-7  # 真空の透磁率 [H/m]
        self.c = 2.998e8  # 光速 [m/s]
        self.e = 1.602e-19  # 電気素量 [C]
        self.m_e = 9.109e-31  # 電子質量 [kg]
        self.m_p = 1.673e-27  # 陽子質量 [kg]
        
        # === スケール設定 ===
        self.length_scale_modes = {
            'atomic': {
                'name': 'atomic scale',
                'unit': 'nm',
                'conversion': 1e-9,  # 1単位 = 1nm
                'typical_box_size': 10.0,  # 10nm
                'description': 'nano particles, quantum dots'
            },
            'microscopic': {
                'name': 'microscopic scale',
                'unit': 'μm',
                'conversion': 1e-6,  # 1単位 = 1μm
                'typical_box_size': 100.0,  # 100μm
                'description': 'cells, microparticles'
            },
            'mesoscopic': {
                'name': 'mesoscopic scale',
                'unit': 'mm',
                'conversion': 1e-3,  # 1単位 = 1mm
                'typical_box_size': 10.0,  # 10mm = 1cm
                'description': 'sall experimental equipment'
            },
            'macroscopic': {
                'name': 'macroscopic scale',
                'unit': 'm',
                'conversion': 1.0,  # 1単位 = 1m
                'typical_box_size': 1.0,  # 1m
                'description': 'laboratory scale'
            }
        }
        
        self.time_scale_modes = {
            'femtosecond': {
                'name': 'Femtosecond',
                'unit': 'fs',
                'conversion': 1e-15,  # 1 unit = 1 fs
                'typical_dt': 1.0,    # 1 fs
                'description': 'Electron dynamics, ultrafast phenomena'
            },
            'picosecond': {
                'name': 'Picosecond',
                'unit': 'ps',
                'conversion': 1e-12,  # 1 unit = 1 ps
                'typical_dt': 1.0,    # 1 ps
                'description': 'Molecular dynamics'
            },
            'nanosecond': {
                'name': 'Nanosecond',
                'unit': 'ns',
                'conversion': 1e-9,   # 1 unit = 1 ns
                'typical_dt': 0.1,    # 0.1 ns
                'description': 'Ion motion'
            },
            'microsecond': {
                'name': 'Microsecond',
                'unit': 'μs',
                'conversion': 1e-6,   # 1 unit = 1 μs
                'typical_dt': 0.01,   # 0.01 μs
                'description': 'Plasma physics'
            },
            'millisecond': {
                'name': 'Millisecond',
                'unit': 'ms',
                'conversion': 1e-3,   # 1 unit = 1 ms
                'typical_dt': 0.1,    # 0.1 ms
                'description': 'Laboratory experiments'
            }
        }
        
        # 現在のスケール設定
        self.current_length_mode = 'microscopic'
        self.current_time_mode = 'nanosecond'
        
    def get_length_conversion(self):
        """長さスケールの変換係数を取得 [m/シミュレーション単位]"""
        return self.length_scale_modes[self.current_length_mode]['conversion']
    
    def get_time_conversion(self):
        """時間スケールの変換係数を取得 [s/シミュレーション単位]"""
        return self.time_scale_modes[self.current_time_mode]['conversion']
    
    def simulation_to_si(self, value, quantity_type):
        """シミュレーション単位からSI単位への変換"""
        if quantity_type == 'length':
            return value * self.get_length_conversion()
        elif quantity_type == 'time':
            return value * self.get_time_conversion()
        elif quantity_type == 'velocity':
            return value * self.get_length_conversion() / self.get_time_conversion()
        elif quantity_type == 'acceleration':
            return value * self.get_length_conversion() / (self.get_time_conversion() ** 2)
        elif quantity_type == 'electric_field':
            # E [V/m_sim] -> [V/m_SI]
            return value / self.get_length_conversion()
        elif quantity_type == 'magnetic_field':
            # B [T] (磁場は直接SI単位)
            return value
        elif quantity_type == 'charge':
            # 電荷は電気素量の倍数として扱う
            return value * self.e
        elif quantity_type == 'mass':
            # 質量は電子質量の倍数として扱う
            return value * self.m_e
        else:
            return value
    
    def si_to_simulation(self, value, quantity_type):
        """SI単位からシミュレーション単位への変換"""
        if quantity_type == 'length':
            return value / self.get_length_conversion()
        elif quantity_type == 'time':
            return value / self.get_time_conversion()
        elif quantity_type == 'velocity':
            return value * self.get_time_conversion() / self.get_length_conversion()
        elif quantity_type == 'acceleration':
            return value * (self.get_time_conversion() ** 2) / self.get_length_conversion()
        elif quantity_type == 'electric_field':
            return value * self.get_length_conversion()
        elif quantity_type == 'magnetic_field':
            return value
        elif quantity_type == 'charge':
            return value / self.e
        elif quantity_type == 'mass':
            return value / self.m_e
        else:
            return value
    
    def get_coulomb_constant_normalized(self):
        """正規化されたクーロン定数を計算"""
        # k = 1/(4πε₀) を正規化
        L = self.get_length_conversion()
        T = self.get_time_conversion()
        
        # SI単位でのクーロン定数
        k_si = 8.988e9  # [N⋅m²/C²]
        
        # シミュレーション単位に変換
        # F = k * q1 * q2 / r²
        # [N] = k_si [N⋅m²/C²] * [C] * [C] / [m²]
        # シミュレーション単位: [F_sim] = k_sim * [q_sim] * [q_sim] / [L_sim²]
        # [m_e * L/T²] = k_sim * [e²] / [L²]
        
        k_normalized = k_si * (self.e ** 2) / (self.m_e * L ** 3 / T ** 2)
        
        return k_normalized
    
    def calculate_plasma_frequency(self, n_e):
        """プラズマ周波数を計算 [Hz]
        n_e: 電子密度 [m⁻³]
        """
        omega_pe = np.sqrt(n_e * self.e ** 2 / (self.epsilon_0 * self.m_e))
        return omega_pe / (2 * np.pi)
    
    def calculate_debye_length(self, n_e, T_e):
        """デバイ長を計算 [m]
        n_e: 電子密度 [m⁻³]
        T_e: 電子温度 [eV]
        """
        k_B = 1.381e-23  # ボルツマン定数 [J/K]
        T_e_joule = T_e * self.e  # eVをJに変換
        lambda_D = np.sqrt(self.epsilon_0 * T_e_joule / (n_e * self.e ** 2))
        return lambda_D
    
    def calculate_cyclotron_frequency(self, B, charge_state=1, particle='electron'):
        """サイクロトロン周波数を計算 [Hz]
        B: 磁場 [T]
        charge_state: 電荷状態 (1 = 単電荷, 2 = 二価イオンなど)
        particle: 'electron' or 'proton' or 'ion'
        """
        if particle == 'electron':
            m = self.m_e
        elif particle == 'proton':
            m = self.m_p
        else:
            m = self.m_p  # デフォルトは陽子質量
        
        q = charge_state * self.e
        omega_c = q * B / m
        return omega_c / (2 * np.pi)
    
    def get_info_string(self):
        """現在のスケール設定の情報文字列を取得"""
        length_info = self.length_scale_modes[self.current_length_mode]
        time_info = self.time_scale_modes[self.current_time_mode]
        
        info = f"""
        === 物理スケール設定 ===
        長さスケール: {length_info['name']} ({length_info['unit']})
        - 説明: {length_info['description']}
        - 変換: 1単位 = {self.get_length_conversion():.2e} m
        - 推奨ボックスサイズ: {length_info['typical_box_size']:.1f} {length_info['unit']}

        時間スケール: {time_info['name']} ({time_info['unit']})
        - 説明: {time_info['description']}
        - 変換: 1単位 = {self.get_time_conversion():.2e} s
        - 推奨dt: {time_info['typical_dt']:.3f} {time_info['unit']}

        速度スケール: {self.get_length_conversion()/self.get_time_conversion():.2e} m/s per sim_unit
        正規化クーロン定数: {self.get_coulomb_constant_normalized():.2e}
        ========================
        """
        return info
    
physics_scale = PhysicsScale()
particle_arrays_cache = {
    'positions': None,
    'velocities': None,
    'charges': None,
    'masses': None,
    'last_N': 0
}

soa = {
    "positions": None,
    "velocities": None,
    "charges": None,
    "masses": None,
    "N": 0
}

print(f"ポアソン方程式電場グリッド設定:")
print(f"  グリッドサイズ: {grid_size}")
print(f"  セルサイズ: {cell_size}")
print(f"  ボックスサイズ: {BOX_SIZE}")

def initialize_high_res_grid():
    """高解像度グリッドの初期化"""
    global high_res_potential_grid, high_res_charge_density_grid, high_res_poisson_solver
    
    print(f"高解像度グリッド初期化: {high_res_grid_divisions}×{high_res_grid_divisions}×{high_res_grid_divisions}")
    
    high_res_potential_grid = np.zeros(
        (high_res_grid_divisions, high_res_grid_divisions, high_res_grid_divisions), 
        dtype=np.float32
    )
    high_res_charge_density_grid = np.zeros(
        (high_res_grid_divisions, high_res_grid_divisions, high_res_grid_divisions), 
        dtype=np.float32
    )
    
    # 高解像度用のPoissonSolverを作成
    high_res_poisson_solver = PoissonSolver(
        high_res_grid_divisions, 
        high_res_grid_divisions, 
        high_res_grid_divisions, 
        high_res_cell_size
    )
    
    return True

def generate_field_vectors():
    """電場・磁場の矢印データを生成"""
    vectors = []
    
    if not gui_params['show_field_vectors']:
        return np.array(vectors, dtype=np.float32)
    
    spacing = gui_params['field_sample_spacing']
    scale = gui_params['field_vector_scale']
    
    # グリッド上でサンプリング
    for i in range(0, grid_divisions, spacing):
        for j in range(0, grid_divisions, spacing):
            for k in range(0, grid_divisions, spacing):
                # ワールド座標に変換
                x = (i - grid_divisions/2) * cell_size
                y = (j - grid_divisions/2) * cell_size
                z = (k - grid_divisions/2) * cell_size
                
                # 電場ベクトル
                if gui_params['show_electric_field']:
                    ex = E_field[k, j, i, 0]
                    ey = E_field[k, j, i, 1]
                    ez = E_field[k, j, i, 2]
                    
                    e_magnitude = np.sqrt(ex*ex + ey*ey + ez*ez)
                    if e_magnitude > 0.01:  # 閾値以下は描画しない
                        # 方向ベクトルを正規化
                        ex_norm = ex / e_magnitude
                        ey_norm = ey / e_magnitude
                        ez_norm = ez / e_magnitude
                        
                        # 強度に応じた色の濃さ（0.2〜1.0の範囲）
                        intensity = min(e_magnitude / 5.0, 1.0)  # 5.0で正規化
                        color_intensity = 0.2 + 0.8 * intensity
                        vector_length_E = scale * e_magnitude / 60.0
                        
                        # [位置x,y,z, 方向x,y,z, スケール, 色r,g,b, フィールドタイプ]
                        vectors.append([
                            x, y, z,
                            ex_norm, ey_norm, ez_norm,
                            vector_length_E,
                            gui_params['electric_field_color'][0] * color_intensity,
                            gui_params['electric_field_color'][1] * color_intensity,
                            gui_params['electric_field_color'][2] * color_intensity,
                            0  # 電場=0
                        ])
                
                # 磁場ベクトル
                if gui_params['show_magnetic_field'] and gui_params['enable_magnetic_field']:
                    bx = gui_params['magnetic_field_x']
                    by = gui_params['magnetic_field_y']
                    bz = gui_params['magnetic_field_z']
                    
                    b_magnitude = np.sqrt(bx*bx + by*by + bz*bz)
                    if b_magnitude > 0.01:
                        bx_norm = bx / b_magnitude
                        by_norm = by / b_magnitude
                        bz_norm = bz / b_magnitude
                        
                        max_B_magnitude = np.sqrt(max_magnetic_field * max_magnetic_field + max_magnetic_field * max_magnetic_field + max_magnetic_field * max_magnetic_field)
                        vector_length_B = scale * b_magnitude / max_B_magnitude

                        vectors.append([
                            x, y + 0.2, z,  # 少しオフセット
                            bx_norm, by_norm, bz_norm,
                            vector_length_B,
                            gui_params['magnetic_field_color'][0] * b_magnitude / max_B_magnitude,
                            gui_params['magnetic_field_color'][1] * b_magnitude / max_B_magnitude,
                            gui_params['magnetic_field_color'][2] * b_magnitude / max_B_magnitude,
                            1  # 磁場=1
                        ])
    
    return np.array(vectors, dtype=np.float32)

def generate_electrode_wireframes():
    """内部電極のワイヤーフレームを生成"""
    if not gui_params['show_electrodes'] or not gui_params['internal_electrodes']:
        return np.array([], dtype=np.float32)
    
    vertices = []
    
    for elec in gui_params['internal_electrodes']:
        pos = elec['position']
        size = elec['size']
        shape = elec['shape']
        
        # 電圧に応じた色（赤=正、青=負）
        voltage = elec['voltage']
        if voltage > 0:
            color = [1.0, 0.2, 0.2]  # 赤
        elif voltage < 0:
            color = [0.2, 0.2, 1.0]  # 青
        else:
            color = [0.5, 0.5, 0.5]  # 灰色
        
        if shape == 'sphere':
            # 球のワイヤーフレーム（緯線・経線）
            segments = 16
            for i in range(segments):
                # 経線
                theta1 = 2 * np.pi * i / segments
                theta2 = 2 * np.pi * (i + 1) / segments
                for j in range(segments):
                    phi1 = np.pi * j / segments
                    phi2 = np.pi * (j + 1) / segments
                    
                    x1 = pos[0] + size * np.sin(phi1) * np.cos(theta1)
                    y1 = pos[1] + size * np.cos(phi1)
                    z1 = pos[2] + size * np.sin(phi1) * np.sin(theta1)
                    
                    x2 = pos[0] + size * np.sin(phi2) * np.cos(theta1)
                    y2 = pos[1] + size * np.cos(phi2)
                    z2 = pos[2] + size * np.sin(phi2) * np.sin(theta1)
                    
                    vertices.extend([x1, y1, z1] + color + [x2, y2, z2] + color)
                
                # 緯線
                for j in range(segments):
                    phi = np.pi * j / segments
                    
                    x1 = pos[0] + size * np.sin(phi) * np.cos(theta1)
                    y1 = pos[1] + size * np.cos(phi)
                    z1 = pos[2] + size * np.sin(phi) * np.sin(theta1)
                    
                    x2 = pos[0] + size * np.sin(phi) * np.cos(theta2)
                    y2 = pos[1] + size * np.cos(phi)
                    z2 = pos[2] + size * np.sin(phi) * np.sin(theta2)
                    
                    vertices.extend([x1, y1, z1] + color + [x2, y2, z2] + color)
        
        elif shape == 'box':
            # ボックスのエッジ12本
            half = size / 2
            corners = [
                [pos[0]-half, pos[1]-half, pos[2]-half],
                [pos[0]+half, pos[1]-half, pos[2]-half],
                [pos[0]+half, pos[1]+half, pos[2]-half],
                [pos[0]-half, pos[1]+half, pos[2]-half],
                [pos[0]-half, pos[1]-half, pos[2]+half],
                [pos[0]+half, pos[1]-half, pos[2]+half],
                [pos[0]+half, pos[1]+half, pos[2]+half],
                [pos[0]-half, pos[1]+half, pos[2]+half],
            ]
            
            edges = [
                (0,1), (1,2), (2,3), (3,0),  # 底面
                (4,5), (5,6), (6,7), (7,4),  # 上面
                (0,4), (1,5), (2,6), (3,7),  # 縦
            ]
            
            for start, end in edges:
                vertices.extend(corners[start] + color + corners[end] + color)
        
        elif shape == 'cylinder':
            # シリンダーのワイヤーフレーム
            segments = 16
            height = size
            radius = size
            
            for i in range(segments):
                angle1 = 2 * np.pi * i / segments
                angle2 = 2 * np.pi * ((i + 1) % segments) / segments
                
                x1 = pos[0] + radius * np.cos(angle1)
                z1 = pos[2] + radius * np.sin(angle1)
                x2 = pos[0] + radius * np.cos(angle2)
                z2 = pos[2] + radius * np.sin(angle2)
                
                # 上の円
                vertices.extend([x1, pos[1]+height/2, z1] + color + [x2, pos[1]+height/2, z2] + color)
                # 下の円
                vertices.extend([x1, pos[1]-height/2, z1] + color + [x2, pos[1]-height/2, z2] + color)
                # 縦線
                vertices.extend([x1, pos[1]-height/2, z1] + color + [x1, pos[1]+height/2, z1] + color)
    
    return np.array(vertices, dtype=np.float32)

def debug_potential_grid():
    """電位グリッドの統計情報を表示"""
    if high_res_potential_grid is not None:
        print("=== 電位グリッド統計 ===")
        print(f"最小電位: {np.min(high_res_potential_grid):.3f} V")
        print(f"最大電位: {np.max(high_res_potential_grid):.3f} V")
        print(f"平均電位: {np.mean(high_res_potential_grid):.3f} V")
        print(f"中央値: {np.median(high_res_potential_grid):.3f} V")
        
        # 中心付近の電位を確認
        center_idx = high_res_grid_divisions // 2
        center_potential = high_res_potential_grid[center_idx, center_idx, center_idx]
        print(f"中心点の電位: {center_potential:.3f} V")
        print("=======================")
    else:
        print("高解像度グリッドが初期化されていません")

def generate_equipotential_lines_high_res(gui_params):
    """
    高解像度グリッドを使用して等電位線を生成
    """
    if not gui_params['show_equipotential'] or not gui_params['simulation_paused']:
        return np.array([], dtype=np.float32)
    
    # 高解像度グリッドが初期化されていない場合は初期化
    if high_res_potential_grid is None:
        print("高解像度グリッドが初期化されていません")
        return np.array([], dtype=np.float32)
    debug_potential_grid()
    plane = gui_params['equipotential_plane']
    slice_pos = gui_params['equipotential_slice_position']
    num_lines = gui_params['equipotential_num_lines']
    
    pot_min = np.min(high_res_potential_grid)
    pot_max = np.max(high_res_potential_grid)
    
    if pot_max - pot_min < 0.01:
        return np.array([], dtype=np.float32)
    
    levels = np.linspace(pot_min, pot_max, num_lines + 2)[1:-1]
    
    vertices = []
    
    def world_to_grid(pos, axis_size):
        normalized = (pos + BOX_SIZE/2) / BOX_SIZE
        grid_pos = int(normalized * axis_size)
        return np.clip(grid_pos, 0, axis_size - 1)
    
    def get_potential_color(level, pot_min, pot_max):
        """電位に応じた色を計算(正:青、負:オレンジ)"""
        # 基準となる最大電圧（例: 50V。または max(abs(pot_min), abs(pot_max))）
        v_range = max(abs(pot_min), abs(pot_max), 1.0)
        
        # -1.0(負) ～ 0.0(0V) ～ 1.0(正) に正規化
        norm_val = level / v_range
        
        if norm_val > 0:
            # 正の電位：白(1,1,1)から青(0.4, 0.8, 1.0)へ
            t = min(norm_val, 1.0)
            r = 1.0 - t * (1.0 - 0.4)
            g = 1.0 - t * (1.0 - 0.8)
            b = 1.0
            return [r, g, b]
        else:
            # 負の電位：白(1,1,1)からオレンジ(1.0, 0.5, 0.0)へ
            t = min(abs(norm_val), 1.0)
            r = 1.0
            g = 1.0 - t * (1.0 - 0.5)
            b = 1.0 - t * (1.0 - 0.0)
            return [r, g, b]
    
    if plane == 0:  # XZ平面
        j = world_to_grid(slice_pos, high_res_grid_divisions)
        grid_2d = high_res_potential_grid[:, j, :]
        
        for level in levels:
            color = get_potential_color(level, pot_min, pot_max)
            contour_points = extract_contour_2d_optimized(
                grid_2d, level, plane='xz', slice_pos=slice_pos, 
                color=color, grid_divisions=high_res_grid_divisions
            )
            vertices.extend(contour_points)
            
    elif plane == 1:  # YX平面
        k = world_to_grid(slice_pos, high_res_grid_divisions)
        grid_2d = high_res_potential_grid[k, :, :]
        
        for level in levels:
            color = get_potential_color(level, pot_min, pot_max)
            contour_points = extract_contour_2d_optimized(
                grid_2d, level, plane='yx', slice_pos=slice_pos, 
                color=color, grid_divisions=high_res_grid_divisions
            )
            vertices.extend(contour_points)
            
    elif plane == 2:  # YZ平面
        i = world_to_grid(slice_pos, high_res_grid_divisions)
        grid_2d = high_res_potential_grid[:, :, i]
        
        for level in levels:
            color = get_potential_color(level, pot_min, pot_max)
            contour_points = extract_contour_2d_optimized(
                grid_2d, level, plane='yz', slice_pos=slice_pos, 
                color=color, grid_divisions=high_res_grid_divisions
            )
            vertices.extend(contour_points)
    
    return np.array(vertices, dtype=np.float32)

def extract_contour_2d_optimized(grid_slice, level, plane='xz', slice_pos=0.0, 
                                 color=[1.0, 1.0, 1.0], grid_divisions=16):
    """
    マーチングスクエア法による等電位線抽出(高解像度グリッド対応版)
    """
    h, w = grid_slice.shape
    
    # 各セルのエッジ情報を保存
    edge_points = {}
    
    # 全セルをスキャンしてエッジ交点を計算
    for i in range(h - 1):
        for j in range(w - 1):
            v00 = grid_slice[i, j]
            v10 = grid_slice[i, j + 1]
            v01 = grid_slice[i + 1, j]
            v11 = grid_slice[i + 1, j + 1]
            
            # マーチングスクエアのケース判定
            case = 0
            if v00 >= level: case |= 1
            if v10 >= level: case |= 2
            if v11 >= level: case |= 4
            if v01 >= level: case |= 8
            
            if case == 0 or case == 15:
                continue
            
            # エッジ上の交点を計算して保存
            # 下エッジ (bottom)
            if (case & 1) != (case & 2):
                t = (level - v00) / (v10 - v00 + 1e-10)
                t = np.clip(t, 0, 1)
                edge_points[(i, j, 'bottom')] = (j + t, i)
            
            # 右エッジ (right)
            if (case & 2) != (case & 4):
                t = (level - v10) / (v11 - v10 + 1e-10)
                t = np.clip(t, 0, 1)
                edge_points[(i, j, 'right')] = (j + 1, i + t)
            
            # 上エッジ (top)
            if (case & 8) != (case & 4):
                t = (level - v01) / (v11 - v01 + 1e-10)
                t = np.clip(t, 0, 1)
                edge_points[(i, j, 'top')] = (j + t, i + 1)
            
            # 左エッジ (left)
            if (case & 1) != (case & 8):
                t = (level - v00) / (v01 - v00 + 1e-10)
                t = np.clip(t, 0, 1)
                edge_points[(i, j, 'left')] = (j, i + t)
    
    # エッジ接続テーブル
    def get_connected_edges(i, j, case):
        """各ケースでどのエッジが接続されるかを返す"""
        connections = []
        
        # 標準的なマーチングスクエアのルックアップテーブル
        lookup = {
            1: [('left', 'bottom')],
            2: [('bottom', 'right')],
            3: [('left', 'right')],
            4: [('right', 'top')],
            5: [('left', 'top'), ('bottom', 'right')],  # 曖昧なケース
            6: [('bottom', 'top')],
            7: [('left', 'top')],
            8: [('left', 'top')],
            9: [('bottom', 'top')],
            10: [('left', 'bottom'), ('right', 'top')],  # 曖昧なケース
            11: [('right', 'top')],
            12: [('left', 'right')],
            13: [('bottom', 'right')],
            14: [('left', 'bottom')]
        }
        
        if case in lookup:
            return lookup[case]
        
        # 曖昧なケース(5と10)の処理
        v00 = grid_slice[i, j]
        v10 = grid_slice[i, j + 1]
        v01 = grid_slice[i + 1, j]
        v11 = grid_slice[i + 1, j + 1]
        center_val = (v00 + v10 + v01 + v11) / 4.0
        
        if case == 5:
            if center_val >= level:
                return [('left', 'bottom'), ('right', 'top')]
            else:
                return [('left', 'top'), ('bottom', 'right')]
        elif case == 10:
            if center_val >= level:
                return [('bottom', 'top'), ('left', 'right')]
            else:
                return [('left', 'bottom'), ('right', 'top')]
        
        return []
    
    # 線分を生成
    vertices = []
    
    for i in range(h - 1):
        for j in range(w - 1):
            v00 = grid_slice[i, j]
            v10 = grid_slice[i, j + 1]
            v01 = grid_slice[i + 1, j]
            v11 = grid_slice[i + 1, j + 1]
            
            case = 0
            if v00 >= level: case |= 1
            if v10 >= level: case |= 2
            if v11 >= level: case |= 4
            if v01 >= level: case |= 8
            
            if case == 0 or case == 15:
                continue
            
            connections = get_connected_edges(i, j, case)
            
            for edge1_name, edge2_name in connections:
                key1 = (i, j, edge1_name)
                key2 = (i, j, edge2_name)
                
                if key1 in edge_points and key2 in edge_points:
                    x1, y1 = edge_points[key1]
                    x2, y2 = edge_points[key2]
                    
                    # グリッド座標をワールド座標に変換
                    if plane == 'xz':
                        wx1 = (x1 / w - 0.5) * BOX_SIZE
                        wz1 = (y1 / h - 0.5) * BOX_SIZE
                        wx2 = (x2 / w - 0.5) * BOX_SIZE
                        wz2 = (y2 / h - 0.5) * BOX_SIZE
                        wy = slice_pos
                        vertices.extend([
                            wx1, wy, wz1, color[0], color[1], color[2],
                            wx2, wy, wz2, color[0], color[1], color[2]
                        ])
                        
                    elif plane == 'yx':
                        wx1 = (x1 / w - 0.5) * BOX_SIZE
                        wy1 = (y1 / h - 0.5) * BOX_SIZE
                        wx2 = (x2 / w - 0.5) * BOX_SIZE
                        wy2 = (y2 / h - 0.5) * BOX_SIZE
                        wz = slice_pos
                        vertices.extend([
                            wx1, wy1, wz, color[0], color[1], color[2],
                            wx2, wy2, wz, color[0], color[1], color[2]
                        ])
                        
                    elif plane == 'yz':
                        wz1 = (y1 / h - 0.5) * BOX_SIZE
                        wy1 = (x1 / w - 0.5) * BOX_SIZE
                        wz2 = (y2 / h - 0.5) * BOX_SIZE
                        wy2 = (x2 / w - 0.5) * BOX_SIZE
                        wx = slice_pos
                        vertices.extend([
                            wx, wy1, wz1, color[0], color[1], color[2],
                            wx, wy2, wz2, color[0], color[1], color[2]
                        ])
    
    return vertices

def create_poisson_matrix_optimized(nx, ny, nz, dx, electrode_params, geometry_mode=0, cylinder_params=None, internal_electrodes=None):
    n = nx * ny * nz
    rows = []
    cols = []
    data = []
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = k * ny * nx + j * nx + i
                
                # グリッド座標から物理座標へ
                x = (i - nx/2) * dx
                y = (j - ny/2) * dx
                z = (k - nz/2) * dx
                
                # 内部電極チェック
                is_internal_electrode = False
                if internal_electrodes:
                    for elec in internal_electrodes:
                        if is_point_in_electrode(x, y, z, elec):
                            is_internal_electrode = True
                            break
                
                if geometry_mode == 1:
                    if not is_inside_cylinder(i, j, k, nx, ny, nz, 
                                            cylinder_params['radius'], 
                                            cylinder_params['height']):
                        rows.append(idx)
                        cols.append(idx)
                        data.append(1.0)
                        continue
                
                is_boundary = (i == 0 or i == nx-1 or j == 0 or j == ny-1 or k == 0 or k == nz-1)
                
                if is_internal_electrode or is_boundary:
                    # ディリクレ境界条件
                    rows.append(idx)
                    cols.append(idx)
                    data.append(1.0)
                else:
                    # 内部点での7点差分
                    rows.append(idx)
                    cols.append(idx)
                    data.append(-6.0)
                    
                    if i > 0:
                        rows.append(idx)
                        cols.append(idx - 1)
                        data.append(1.0)
                    if i < nx - 1:
                        rows.append(idx)
                        cols.append(idx + 1)
                        data.append(1.0)
                    
                    if j > 0:
                        rows.append(idx)
                        cols.append(idx - nx)
                        data.append(1.0)
                    if j < ny - 1:
                        rows.append(idx)
                        cols.append(idx + nx)
                        data.append(1.0)
                    
                    if k > 0:
                        rows.append(idx)
                        cols.append(idx - nx * ny)
                        data.append(1.0)
                    if k < nz - 1:
                        rows.append(idx)
                        cols.append(idx + nx * ny)
                        data.append(1.0)
    
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    A = A.tocsr()
    
    return A

# 境界条件を適用
def apply_boundary_conditions(A, b, nx, ny, nz, electrode_params):
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = k * ny * nx + j * nx + i
                
                # 境界での処理
                is_boundary = (i == 0 or i == nx-1 or j == 0 or j == ny-1 or k == 0 or k == nz-1)
                
                if is_boundary:
                    # 電極設定を確認
                    voltage = 0.0
                    
                    if electrode_params['electrode_mode']:
                        if i == 0 and electrode_params['electrode_positions']['x_neg']:
                            voltage = electrode_params['electrode_voltage']
                        elif i == nx-1 and electrode_params['electrode_positions']['x_pos']:
                            voltage = electrode_params['electrode_voltage']
                        elif j == 0 and electrode_params['electrode_positions']['y_neg']:
                            voltage = electrode_params['electrode_voltage']
                        elif j == ny-1 and electrode_params['electrode_positions']['y_pos']:
                            voltage = electrode_params['electrode_voltage']
                        elif k == 0 and electrode_params['electrode_positions']['z_neg']:
                            voltage = electrode_params['electrode_voltage']
                        elif k == nz-1 and electrode_params['electrode_positions']['z_pos']:
                            voltage = electrode_params['electrode_voltage']
                        
                        # ディリクレ境界条件: φ = voltage
                        A[idx, :] = 0
                        A[idx, idx] = 1
                        b[idx] = voltage
                    else:
                        # 電極OFFの場合:ノイマン境界条件(∂φ/∂n = 0)
                        # 境界点の電位 = 隣接する内部点の電位
                        A[idx, :] = 0
                        A[idx, idx] = 1
                        
                        # 隣接内部点のインデックスを取得
                        if i == 0:
                            A[idx, idx+1] = -1
                        elif i == nx-1:
                            A[idx, idx-1] = -1
                        elif j == 0:
                            A[idx, idx+nx] = -1
                        elif j == ny-1:
                            A[idx, idx-nx] = -1
                        elif k == 0:
                            A[idx, idx+nx*ny] = -1
                        elif k == nz-1:
                            A[idx, idx-nx*ny] = -1
                        
                        b[idx] = 0  # 右辺はゼロ
                    
                    

def is_point_in_electrode(x, y, z, electrode):
    """点が電極内部にあるかを判定"""
    pos = electrode['position']
    size = electrode['size']
    shape = electrode['shape']
    
    dx = x - pos[0]
    dy = y - pos[1]
    dz = z - pos[2]
    
    if shape == 'sphere':
        return (dx*dx + dy*dy + dz*dz) <= (size*size)
    elif shape == 'box':
        return (abs(dx) <= size/2 and abs(dy) <= size/2 and abs(dz) <= size/2)
    elif shape == 'cylinder':
        # Y軸方向の円柱
        r_squared = dx*dx + dz*dz
        return (r_squared <= (size*size) and abs(dy) <= size)
    
    return False

class PoissonSolver:
    # 初期化
    def __init__(self, nx, ny, nz, dx):
        # 格子点数
        self.nx = nx
        self.ny = ny
        self.nz = nz
        # 格子幅
        self.dx = dx 
        self.cached_matrix = None           # 作成済みラプラシアン行列
        self.cached_electrode_params = None # 行列を作ったときの電極設定
    # キャッシュされた行列を使用してポアソン方程式を解く
    def solve(self, charge_density, electrode_params, geometry_mode=0, cylinder_params=None, internal_electrodes=None):
        """
        キャッシュされた行列を使用してポアソン方程式を解く
        """
        # パラメータが変更された場合のみ行列を再構築
        current_params = (electrode_params, geometry_mode, cylinder_params, tuple(map(lambda e: (tuple(e['position']), e['size'], e['shape']), internal_electrodes or [])))
        if (self.cached_matrix is None or self.cached_params != current_params):
            print("Rebuilding Poisson matrix with internal electrodes...")
            self.cached_matrix = create_poisson_matrix_optimized(
                self.nx, self.ny, self.nz, self.dx, electrode_params,
                geometry_mode, cylinder_params, internal_electrodes
            )
            self.cached_params = current_params

        # 右辺ベクトルの構築
        epsilon_0 = 1.0
        b = np.zeros(self.nx * self.ny * self.nz)
        
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    idx = k * self.ny * self.nx + j * self.nx + i
                    
                    # グリッド座標から物理座標へ変換
                    x = (i - self.nx/2) * self.dx
                    y = (j - self.ny/2) * self.dx
                    z = (k - self.nz/2) * self.dx
                    
                    is_boundary = (i == 0 or i == self.nx-1 or 
                                j == 0 or j == self.ny-1 or 
                                k == 0 or k == self.nz-1)
                    
                    # 内部電極上の点かチェック
                    is_internal_electrode = False
                    electrode_voltage_value = 0.0
                    
                    if internal_electrodes:
                        for elec in internal_electrodes:
                            if is_point_in_electrode(x, y, z, elec):
                                is_internal_electrode = True
                                electrode_voltage_value = elec['voltage']
                                break
                    
                    if is_internal_electrode:
                        # ★★★ 修正: 符号を統一 ★★★
                        b[idx] = electrode_voltage_value  # マイナスを削除
                    elif is_boundary:
                        voltage = 0.0
                        if electrode_params['electrode_mode']:
                            # 面電極の電圧設定(既に正しい)
                            if i == 0 and electrode_params['electrode_positions']['x_neg']:
                                voltage = electrode_params['electrode_voltage']
                            elif i == self.nx-1 and electrode_params['electrode_positions']['x_pos']:
                                voltage = electrode_params['electrode_voltage']
                            elif j == 0 and electrode_params['electrode_positions']['y_neg']:
                                voltage = electrode_params['electrode_voltage']
                            elif j == self.ny-1 and electrode_params['electrode_positions']['y_pos']:
                                voltage = electrode_params['electrode_voltage']
                            elif k == 0 and electrode_params['electrode_positions']['z_neg']:
                                voltage = electrode_params['electrode_voltage']
                            elif k == self.nz-1 and electrode_params['electrode_positions']['z_pos']:
                                voltage = electrode_params['electrode_voltage']
                        b[idx] = voltage  # ★修正: マイナスを削除
                    else:
                        b[idx] = -(charge_density[k, j, i] * self.dx * self.dx / epsilon_0)

        # 方程式を解く
        try:
            phi_flat = spsolve(self.cached_matrix, b)
            phi = phi_flat.reshape((self.nz, self.ny, self.nx))
            return phi.astype(np.float32)
        except Exception as e:
            print(f"ポアソン方程式の解法に失敗しました: {e}")
            return np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)

# 粒子生成クラス
class ChargedParticle:
    def __init__(self, position, velocity, charge, mass):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.charge = charge
        self.mass = mass

def create_particles(num_particles):
    particles = []
    
    spawn_range = 1.5
    
    # 正電荷を生成
    for _ in range(num_particles // 2):
        if gui_params['geometry_mode'] == 0:
            position = np.random.uniform(-spawn_range, spawn_range, 3)
        else:
            radius = gui_params['cylinder_radius'] * 0.7
            height = gui_params['cylinder_height'] * 0.7
            r = radius * np.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2 * np.pi)
            y = np.random.uniform(-height/2, height/2)
            position = np.array([r * np.cos(theta), y, r * np.sin(theta)])
        velocity = np.zeros(3)  
        charge = gui_params['positive_charge']
        mass = gui_params['particle_mass']
        particles.append(ChargedParticle(position, velocity, charge, mass))

    # 負電荷を生成
    for _ in range(num_particles - num_particles // 2):
        if gui_params['geometry_mode'] == 0:
            position = np.random.uniform(-spawn_range, spawn_range, 3)
        else:
            radius = gui_params['cylinder_radius'] * 0.7
            height = gui_params['cylinder_height'] * 0.7
            r = radius * np.sqrt(np.random.uniform(0, 1))
            theta = np.random.uniform(0, 2 * np.pi)
            y = np.random.uniform(-height/2, height/2)
            position = np.array([r * np.cos(theta), y, r * np.sin(theta)])
        velocity = np.zeros(3)  
        charge = gui_params['negative_charge']
        mass = gui_params['particle_mass']
        particles.append(ChargedParticle(position, velocity, charge, mass))
    
    print(f"粒子生成完了: {len(particles)}個")
    return particles

audio_analyzer = None
particles = create_particles(gui_params['num_particles'])
dt = gui_params['dt']

def calculate_safe_dt(particles, E_field, gui_params):
    """
    数値積分の安定性を保つための安全なdtを計算（最適化版）
    """
    if len(particles) == 0:
        return 0.005
    
    # サンプリング：全粒子ではなく一部のみチェック
    sample_size = min(100, len(particles))  # 最大100個まで
    sample_indices = np.random.choice(len(particles), sample_size, replace=False)
    
    max_velocity = 0.0
    max_E_field = 0.0
    
    for idx in sample_indices:
        p = particles[idx]
        v_mag = np.linalg.norm(p.velocity)
        max_velocity = max(max_velocity, v_mag)
        
        # 電場の最大値を取得（グリッド全体から）
        if idx == sample_indices[0]:  # 1回だけ実行
            max_E_field = np.max(np.linalg.norm(E_field, axis=3))
    
    # 最大電荷と最小質量を使用
    max_charge = max(abs(gui_params['positive_charge']), abs(gui_params['negative_charge']))
    min_mass = gui_params['particle_mass']
    
    # 最大加速度の推定
    max_acceleration = (max_charge * max_E_field) / min_mass if min_mass > 0 else 0
    
    # 磁場によるサイクロトロン周波数
    dt_cyclotron = 1.0  # デフォルト値
    if gui_params['enable_magnetic_field']:
        B_mag = np.sqrt(
            gui_params['magnetic_field_x']**2 + 
            gui_params['magnetic_field_y']**2 + 
            gui_params['magnetic_field_z']**2
        )
        if B_mag > 1e-6 and min_mass > 0:
            cyclotron_freq = max_charge * B_mag / min_mass
            dt_cyclotron = 0.1 * (2 * np.pi / cyclotron_freq)  # 周期の1/10
    
    # CFL条件
    dt_cfl = 0.5 * cell_size / max_velocity if max_velocity > 1e-6 else 0.05
    
    # 加速度条件
    dt_accel = 0.5 * np.sqrt(2 * cell_size / max_acceleration) if max_acceleration > 1e-6 else 0.05
    
    # 最も厳しい条件を採用
    dt_safe = min(dt_cfl, dt_accel, dt_cyclotron)
    
    # 安全係数を適用
    dt_safe *= gui_params['dt_safety_factor']
    
    # 実用的な範囲に制限
    dt_safe = np.clip(dt_safe, 0.0001, 0.02)
    
    return dt_safe

poisson_solver = PoissonSolver(grid_divisions, grid_divisions, grid_divisions, cell_size)

# ポアソン方程式を用いて電場を更新
def update_electric_field_poisson_optimized(particles):
    if not gui_params['use_poisson']:
        # 外部電場のみを使用
        E_field[:, :, :, 0] = gui_params['external_E_field_x']
        E_field[:, :, :, 1] = gui_params['external_E_field_y']
        E_field[:, :, :, 2] = gui_params['external_E_field_z']
        return
    
    charge_density_grid[:] = 0.0

    # 電荷密度配列の初期化
    charge_density_grid.fill(0.0)
    
    N = len(particles)
    if N == 0:
        E_field[:,:,:,0] = gui_params['external_E_field_x']
        E_field[:,:,:,1] = gui_params['external_E_field_y']
        E_field[:,:,:,2] = gui_params['external_E_field_z']
        return
    
    # 粒子データの準備
    vanish_flags = np.zeros(N, dtype=np.int32)
    positions_np = np.array([p.position for p in particles], dtype=np.float32)
    charges_np = np.array([p.charge for p in particles], dtype=np.float32)
    charge_density_flat = charge_density_grid.reshape(-1).astype(np.float32)
    
    # GPUメモリの確保と再利用
    if (not hasattr(update_electric_field_poisson_optimized, 'gpu_arrays') or 
        update_electric_field_poisson_optimized.last_N != N):
        if hasattr(update_electric_field_poisson_optimized, 'gpu_arrays'):
            for arr in update_electric_field_poisson_optimized.gpu_arrays.values():
                arr.free()
        
        update_electric_field_poisson_optimized.gpu_arrays = {
            'pos': cuda.mem_alloc(positions_np.nbytes),
            'chg': cuda.mem_alloc(charges_np.nbytes),
            'rho': cuda.mem_alloc(charge_density_flat.nbytes),
            'vanish': cuda.mem_alloc(vanish_flags.nbytes)
        }
        update_electric_field_poisson_optimized.last_N = N
    
    gpu_arrays = update_electric_field_poisson_optimized.gpu_arrays #GPUメモリへの参照を取得
    
    # デバイスへのデータ転送
    cuda.memcpy_htod(gpu_arrays['pos'], positions_np)
    cuda.memcpy_htod(gpu_arrays['chg'], charges_np)
    cuda.memcpy_htod(gpu_arrays['rho'], charge_density_flat)
    cuda.memcpy_htod(gpu_arrays['vanish'], vanish_flags)
    
    block_size = 256   # CUDAブロック数のスレッド数
    grid_size_cuda = (N + block_size - 1) // block_size  # 必要なブロック数
    
    charge_density_flat = np.zeros_like(charge_density_grid.reshape(-1))
    cuda.memcpy_htod(gpu_arrays['rho'], charge_density_flat)

    # 電荷密度作成カーネルの起動
    charge_density_func(
        gpu_arrays['pos'], gpu_arrays['chg'], gpu_arrays['rho'],
        np.int32(grid_divisions), np.int32(grid_divisions), np.int32(grid_divisions),
        np.float32(cell_size), np.int32(N), np.float32(BOX_SIZE), gpu_arrays['vanish'],
        block=(block_size, 1, 1), grid=(grid_size_cuda, 1)
    )
    
    # 電荷密度を取り出す
    cuda.memcpy_dtoh(charge_density_flat, gpu_arrays['rho'])
    charge_density_grid[:] = charge_density_flat.reshape(grid_size)
    
    # 電極パラメータの設定
    electrode_params = {
        'electrode_mode': gui_params['electrode_mode'],
        'electrode_voltage': gui_params['electrode_voltage'],
        'electrode_positions': gui_params['electrode_positions']
    }
    # 円柱形状パラメータ
    cylinder_params = {
        'radius': gui_params['cylinder_radius'],
        'height': gui_params['cylinder_height']
    } if gui_params['geometry_mode'] == 1 else None
    # キャッシュされたソルバーを使用してポアソン方程式を解く
    potential_grid[:] = poisson_solver.solve(
        charge_density_grid, electrode_params, 
        gui_params['geometry_mode'], cylinder_params,
        gui_params['internal_electrodes']
    )
    
    # 電位から電場を計算
    potential_flat = potential_grid.reshape(-1).astype(np.float32)
    E_field_flat = E_field.reshape(-1, 3).astype(np.float32)
    
    # 電場計算用GPUメモリ
    if not hasattr(update_electric_field_poisson_optimized, 'field_gpu_arrays'):
        update_electric_field_poisson_optimized.field_gpu_arrays = {
            'pot': cuda.mem_alloc(potential_flat.nbytes),
            'E': cuda.mem_alloc(E_field_flat.nbytes)
        }
    
    field_gpu_arrays = update_electric_field_poisson_optimized.field_gpu_arrays
    
    # 電位データをGPUに転送
    cuda.memcpy_htod(field_gpu_arrays['pot'], potential_flat)
    cuda.memcpy_htod(field_gpu_arrays['E'], E_field_flat)
    
    # 電場計算のためのCUDA設定
    total_cells = grid_divisions ** 3
    block_size_field = 256
    grid_size_field = (total_cells + block_size_field - 1) // block_size_field
    
    # 電位から電場を求めるCUDAカーネル
    electric_field_func(
        field_gpu_arrays['pot'], field_gpu_arrays['E'],
        np.int32(grid_divisions), np.int32(grid_divisions), np.int32(grid_divisions),
        np.float32(cell_size),
        np.float32(gui_params['external_E_field_x']),
        np.float32(gui_params['external_E_field_y']),
        np.float32(gui_params['external_E_field_z']),
        block=(block_size_field, 1, 1), grid=(grid_size_field, 1)
    )
    
    # 電場を取り出す
    cuda.memcpy_dtoh(E_field_flat, field_gpu_arrays['E'])
    E_field[:] = E_field_flat.reshape(grid_size + (3,))

def update_high_res_electric_field(particles, gui_params):
    """
    ポーズモード時に高解像度グリッドで電場を再計算
    通常の8分割グリッドから16分割グリッドへの補間も含む
    """
    global high_res_potential_grid, high_res_charge_density_grid, high_res_poisson_solver
    
    # 高解像度グリッドが初期化されていない場合は初期化
    if high_res_potential_grid is None:
        initialize_high_res_grid()
    
    # 高解像度電荷密度グリッドの初期化
    high_res_charge_density_grid.fill(0.0)
    
    N = len(particles)
    if N == 0:
        return high_res_potential_grid
    
    # 粒子データの準備
    positions_np = np.array([p.position for p in particles], dtype=np.float32)
    charges_np = np.array([p.charge for p in particles], dtype=np.float32)
    
    # 高解像度グリッドに電荷密度を配置
    for i in range(N):
        pos = positions_np[i]
        charge = charges_np[i]
        
        # グリッド座標に変換
        grid_x = (pos[0] + BOX_SIZE/2) / high_res_cell_size
        grid_y = (pos[1] + BOX_SIZE/2) / high_res_cell_size
        grid_z = (pos[2] + BOX_SIZE/2) / high_res_cell_size
        
        # 最近傍グリッド点を取得
        ix = int(np.clip(grid_x, 0, high_res_grid_divisions - 1))
        iy = int(np.clip(grid_y, 0, high_res_grid_divisions - 1))
        iz = int(np.clip(grid_z, 0, high_res_grid_divisions - 1))
        
        # 電荷密度を加算
        cell_volume = high_res_cell_size ** 3
        high_res_charge_density_grid[iz, iy, ix] += charge / cell_volume
    
    # 電極パラメータの設定
    electrode_params = {
        'electrode_mode': gui_params['electrode_mode'],
        'electrode_voltage': gui_params['electrode_voltage'],
        'electrode_positions': gui_params['electrode_positions']
    }
    
    # 円柱形状パラメータ
    cylinder_params = {
        'radius': gui_params['cylinder_radius'],
        'height': gui_params['cylinder_height']
    } if gui_params['geometry_mode'] == 1 else None
    
    # 高解像度でポアソン方程式を解く
    high_res_potential_grid[:] = high_res_poisson_solver.solve(
        high_res_charge_density_grid, 
        electrode_params,
        gui_params['geometry_mode'], 
        cylinder_params,
        gui_params['internal_electrodes']
    )
    
    print(f"高解像度電位計算完了: min={np.min(high_res_potential_grid):.3f}, max={np.max(high_res_potential_grid):.3f}")
    
    return high_res_potential_grid

class ParticleSystem:
    def _init_(self, max_particles):
        self.N = 0

        self.positions = np.zeros((max_particles, 3), dtype=np.float32)
        self.velocities = np.zeros((max_particles, 3), dtype=np.float32)
        self.charges = np.zeros(max_particles, dtype=np.float32)
        self.masses = np.zeros(max_particles, dtype=np.float32)

    def spawn_particles(self, pos, vel, charge, mass):
        i = self.N
        self.positions[i] = pos
        self.velocities[i] = vel
        self.charges[i] = charge
        self.masses[i] = mass
        self.N += 1
    
    def remove_particles(self, alive_indices):
        self.positions = self.positions[alive_indices]
        self.velocities = self.velocities[alive_indices]
        self.charges = self.charges[alive_indices]
        self.masses = self.masses[alive_indices]
        self.N = len(alive_indices)

def convert_particles_to_soa(particles):
    N = len(particles)

    pos = np.zeros((N, 3), dtype=np.float32)
    vel = np.zeros((N, 3), dtype=np.float32)
    chg = np.zeros(N, dtype=np.float32)
    mas = np.zeros(N, dtype=np.float32)

    for i, p in enumerate(particles):
        pos[i] = p.position
        vel[i] = p.velocity
        chg[i] = p.charge
        mas[i] = p.mass
    
    soa['positions'] = pos
    soa['velocities'] = vel
    soa['charges'] = chg
    soa['masses'] = mas
    soa['N'] = N

# 粒子位置・速度更新関数    
def update_particles_optimized(soa_data, E_field_array):
    global vbo_resource, cuda_instance_data

    timings = {}
    
    if gui_params['simulation_paused']:
        return {
            'prepare_numpy': 0,
            'gpu_alloc': 0,
            'htod': 0,
            'cuda_func': 0,
            'dtoh': 0,
            'particle_update': 0,
            'total': 0
        }
    
    N = soa_data['N']
    
    # N=0の場合は早期リターン（GPUメモリ確保前に判定）
    if N == 0:
        return {
            'prepare_numpy': 0,
            'gpu_alloc': 0,
            'htod': 0,
            'cuda_func': 0,
            'dtoh': 0,
            'particle_update': 0,
            'total': 0
        }
    t0 = time.time()
    # 粒子データの準備
    #if particle_arrays_cache['last_N'] != N or particle_arrays_cache['positions'] is None:
    #    particle_arrays_cache['positions'] = np.empty((N, 3), dtype=np.float32)
    #    particle_arrays_cache['velocities'] = np.empty((N, 3), dtype=np.float32)
    #    particle_arrays_cache['charges'] = np.empty(N, dtype=np.float32)
    #    particle_arrays_cache['masses'] = np.empty(N, dtype=np.float32)
    #    particle_arrays_cache['last_N'] = N
    
    #positions_np = particle_arrays_cache['positions']
    #velocities_np = particle_arrays_cache['velocities']
    #charges_np = particle_arrays_cache['charges']
    #masses_np = particle_arrays_cache['masses']

    #for i, p in enumerate(particles):
    #    positions_np[i] = p.position
    #    velocities_np[i] = p.velocity
    #    charges_np[i] = p.charge
    #    masses_np[i] = p.mass

    N = soa_data['N']

    positions_np = soa["positions"][:soa["N"]]
    velocities_np = soa["velocities"][:soa["N"]]
    charges_np = soa["charges"][:soa["N"]]
    masses_np = soa["masses"][:soa["N"]]

    
    if not hasattr(update_particles_optimized, 'vanish_flags') or len(update_particles_optimized.vanish_flags) != N:
        update_particles_optimized.vanish_flags = np.zeros(N, dtype=np.int32)
    else:
        update_particles_optimized.vanish_flags.fill(0)
    vanish_flags = update_particles_optimized.vanish_flags

    E_field_flat = E_field_array.reshape(-1, 3).astype(np.float32)
    
    # 境界条件フラグの設定
    warp_flags = np.array([
        gui_params['warp_x_negative'], gui_params['warp_x_positive'],
        gui_params['warp_y_negative'], gui_params['warp_y_positive'],
        gui_params['warp_z_negative'], gui_params['warp_z_positive']
    ], dtype=np.int32)

    cube_vanish_flags = np.array([
        gui_params['cube_vanish_x_neg'], gui_params['cube_vanish_x_pos'],
        gui_params['cube_vanish_y_neg'], gui_params['cube_vanish_y_pos'],
        gui_params['cube_vanish_z_neg'], gui_params['cube_vanish_z_pos']
    ], dtype=np.int32)

    # 円柱の境界条件の設定
    cylinder_flags = np.array([
        gui_params['cylinder_top_warp'],
        gui_params['cylinder_bottom_warp'],
        gui_params['cylinder_wall_vanish']
    ], dtype=np.int32)

    # 磁場の設定
    magnetic_field = np.array([
        gui_params['magnetic_field_x'],
        gui_params['magnetic_field_y'], 
        gui_params['magnetic_field_z']
    ], dtype=np.float32)

    t1 = time.time()
    timings["prepare_numpy"] = (t1 - t0) * 1000

    t2 = time.time()
    # GPUメモリの確保
    need_realloc = False
    if not hasattr(update_particles_optimized, 'gpu_arrays'):
        need_realloc = True
    elif not hasattr(update_particles_optimized, 'last_N'):
        need_realloc = True
    elif update_particles_optimized.last_N != N:
        need_realloc = True
        for arr in update_particles_optimized.gpu_arrays.values():
            arr.free()
    
    # GPUメモリの再確保処理
    if need_realloc:
        print(f"GPUメモリを再確保: N={N}")
        instance_data_size = N * 7 * 4
        update_particles_optimized.gpu_arrays = {
            'pos': cuda.mem_alloc(positions_np.nbytes),
            'vel': cuda.mem_alloc(velocities_np.nbytes),
            'chg': cuda.mem_alloc(charges_np.nbytes),
            'mas': cuda.mem_alloc(masses_np.nbytes),
            'ef': cuda.mem_alloc(E_field_flat.nbytes),
            'warp': cuda.mem_alloc(warp_flags.nbytes),
            'vanish': cuda.mem_alloc(vanish_flags.nbytes),
            'cyl_flags': cuda.mem_alloc(cylinder_flags.nbytes),
            'mag_field': cuda.mem_alloc(magnetic_field.nbytes),
            'instance': cuda.mem_alloc(instance_data_size),
            'cube_vanish': cuda.mem_alloc(cube_vanish_flags.nbytes)
        }
        update_particles_optimized.last_N = N

    t3 = time.time()
    timings['gpu_alloc'] = (t3 - t2) * 1000

    gpu_arrays = update_particles_optimized.gpu_arrays
    
    t4 = time.time()
    # GPUへ転送
    cuda.memcpy_htod(gpu_arrays['pos'], positions_np)
    cuda.memcpy_htod(gpu_arrays['vel'], velocities_np)
    cuda.memcpy_htod(gpu_arrays['chg'], charges_np)
    cuda.memcpy_htod(gpu_arrays['mas'], masses_np)
    cuda.memcpy_htod(gpu_arrays['ef'], E_field_flat)
    cuda.memcpy_htod(gpu_arrays['warp'], warp_flags)
    cuda.memcpy_htod(gpu_arrays['vanish'], vanish_flags)
    cuda.memcpy_htod(gpu_arrays['cyl_flags'], cylinder_flags)
    cuda.memcpy_htod(gpu_arrays['mag_field'], magnetic_field)
    cuda.memcpy_htod(gpu_arrays['cube_vanish'], cube_vanish_flags)
    t5 = time.time()
    timings['htod'] = (t5 - t4) * 1000

    t6 = time.time()
    # VBOをCUDAにマップ
    mapping = vbo_resource.map()
    cuda_instance_data = np.intp(mapping.device_ptr())  # mappingから直接デバイスポインタを取得

    block_size = 512
    grid_size_cuda = (N + block_size - 1) // block_size

    # ニュートンの運動方程式を解く
    cuda_func(
        gpu_arrays['pos'], gpu_arrays['vel'], gpu_arrays['chg'], gpu_arrays['mas'],
        gpu_arrays['ef'], 
        np.int32(grid_size[0]), np.int32(grid_size[1]), np.int32(grid_size[2]),
        np.float32(cell_size), np.int32(N), 
        np.float32(gui_params['coulomb_force_control']),  
        np.float32(0.1),  
        np.float32(gui_params['dt']), 
        np.float32(BOX_SIZE),
        np.int32(boundary_mode), 
        gpu_arrays['warp'],
        np.float32(gui_params['cylinder_radius']),
        np.float32(gui_params['cylinder_height']),
        np.int32(gui_params['geometry_mode']),
        gpu_arrays['vanish'],
        gpu_arrays['cyl_flags'],
        gpu_arrays['mag_field'],
        np.int32(gui_params['enable_magnetic_field']),
        cuda_instance_data,  # VBOへの直接ポインタ
        np.float32(gui_params['scale_factor']),
        np.array(gui_params['positive_color'], dtype=np.float32),
        np.array(gui_params['negative_color'], dtype=np.float32),
        gpu_arrays['cube_vanish'],
        block=(block_size, 1, 1), grid=(grid_size_cuda, 1),
    )
    cuda.Context.synchronize()
    t7 = time.time()
    timings["cuda_func"] = (t7 - t6) * 1000

    mapping.unmap()  # マッピングを適切に解除

    # 計算結果を受け取る
    t8 = time.time()
    cuda.memcpy_dtoh(positions_np, gpu_arrays['pos'])
    cuda.memcpy_dtoh(velocities_np, gpu_arrays['vel'])
    cuda.memcpy_dtoh(vanish_flags, gpu_arrays['vanish'])
    t9 = time.time()
    timings["dtoh"] = (t9 - t8) * 1000

    # 消失していない粒子のインデックスを取得
    t10 = time.time()
    alive_indices = [i for i, flag in enumerate(vanish_flags) if flag == 0]
    particles_changed = len(alive_indices) != len(particles)
    if particles_changed:
        # 生存している粒子のみを保持
        alive_indices_np = np.where(vanish_flags == 0)[0]
        #particlesリスト更新
        new_particles = []
        for i in alive_indices:
            if i < len(particles):
                particles[i].position = positions_np[i]
                particles[i].velocity = velocities_np[i]
                new_particles.append(particles[i])
        particles[:] = new_particles
        
        # soaを更新
        soa_data["positions"] = soa_data["positions"][alive_indices_np]
        soa_data["velocities"] = soa_data["velocities"][alive_indices_np]
        soa_data["charges"] = soa_data["charges"][alive_indices_np]
        soa_data["masses"] = soa_data["masses"][alive_indices_np]
        soa_data["N"] = len(alive_indices_np)
        
        # 粒子数変化によりGPUメモリサイズが変わるため、次回再確保が必要
        clear_gpu_arrays()
    else:
        # 粒子数に変化がない場合は位置と速度のみ更新
        for i, p in enumerate(particles):
            p.position = positions_np[i]
            p.velocity = velocities_np[i]
        
        soa['positions'][:N] = positions_np
        soa['velocities'][:N] = velocities_np
    t11 = time.time()
    timings['particle_update'] = (t11 - t10) * 1000
    end_total = time.time()
    timings['total'] = (end_total - t0) * 1000

    return timings
    

emitter_frame_counter = 0

# 新しい荷電粒子を1個生成
def emit_particle():
    if not gui_params['enable_emitter']:
        return None
    
    if len(particles) >= gui_params['max_emitted_particles']:
        return None
    
    # エミッター位置・方向を設定
    position = np.array(gui_params['emitter_position'], dtype=np.float32).copy()  # copyを追加
    direction = np.array(gui_params['emitter_direction'], dtype=np.float32).copy()  # copyを追加

    direction_norm = np.linalg.norm(direction)
    if direction_norm > 0:
        direction = direction / direction_norm
    else:
        direction = np.array([0.0,1.0,0.0], dtype=np.float32)
    
    # 発射方向にばらつきを追加
    if gui_params['emitter_spread'] > 0:
        theta = np.random.uniform(-gui_params['emitter_spread'], gui_params['emitter_spread'])
        phi = np.random.uniform(0, 2 * np.pi)
        
        if abs(direction[1]) < 0.9:
            perpendicular1 = np.cross(direction, [0, 1, 0])
        else:
            perpendicular1 = np.cross(direction, [1, 0, 0])
        
        perpendicular1 = perpendicular1 / np.linalg.norm(perpendicular1)
        perpendicular2 = np.cross(direction, perpendicular1)
        perpendicular2 = perpendicular2 / np.linalg.norm(perpendicular2)

        spread_offset = (perpendicular1 * np.cos(phi) + perpendicular2 * np.sin(phi)) * np.sin(theta)
        direction = direction * np.cos(theta) + spread_offset
        direction = direction / np.linalg.norm(direction)
    
    velocity = direction * gui_params['emitter_speed']

    # 電荷タイプの決定 - 修正
    if gui_params['emitter_charge_type'] == 0:  # Positive
        charge = gui_params['positive_charge']
    elif gui_params['emitter_charge_type'] == 1:  # Negative
        charge = gui_params['negative_charge']
    else:  # Random
        charge = gui_params['positive_charge'] if np.random.random() > 0.5 else gui_params['negative_charge']
    
    new_particle = ChargedParticle(position, velocity, charge, gui_params['particle_mass'])
    return new_particle

# 粒子エミッターの更新処理
def update_particle_emitter():
    global emitter_frame_counter

    if not gui_params['enable_emitter'] or gui_params['simulation_paused']:
        return
    
    emitter_frame_counter += 1

    # フレームカウンターが設定された発射レート以上になった場合に粒子を射出
    if emitter_frame_counter >= gui_params['emitter_rate']:
        new_particle = emit_particle()
        if new_particle is not None:
            particles.append(new_particle)
            
            # SOA構造に新しい粒子を追加
            if soa['N'] >= len(soa['positions']):
                new_size = max(gui_params['max_emitted_particles'], len(particles))
                soa['positions'] = np.resize(soa['positions'], (new_size, 3))
                soa['velocities'] = np.resize(soa['velocities'], (new_size, 3))
                soa['charges'] = np.resize(soa['charges'], new_size)
                soa['masses'] = np.resize(soa['masses'], new_size)
            
            # 新しい粒子のデータをSOAに追加
            idx = soa['N']
            soa['positions'][idx] = new_particle.position
            soa['velocities'][idx] = new_particle.velocity
            soa['charges'][idx] = new_particle.charge
            soa['masses'][idx] = new_particle.mass
            soa['N'] += 1
            
            # GPU配列を再確保
            clear_gpu_arrays()
        
        emitter_frame_counter = 0

def cleanup_resources():
    global vbo_resource, audio_analyzer
    if vbo_resource is not None:
        vbo_resource.unregister()
        vbo_resource = None
    if audio_analyzer is not None:
        audio_analyzer.close()
        audio_analyzer = None

# === 3Dモデルの読み込み ===
sphere_mesh = trimesh.load("./3D_models/sphere.obj", force='mesh') # 3Dメッシュデータの読み込み
sphere_vertices = np.array(sphere_mesh.vertices, dtype=np.float32) # メッシュから頂点座標を抽出(x, y, z)
sphere_normals = np.array(sphere_mesh.vertex_normals, dtype=np.float32) # メッシュから法線ベクトルを抽出
sphere_indices = np.array(sphere_mesh.faces, dtype=np.uint32).flatten() # メッシュから面情報（三角形）を抽出し、1次元配列に変換

sphere_VAO = glGenVertexArrays(1) # 頂点属性の設定をまとめて保存
sphere_VBO = glGenBuffers(1) # 頂点データ（座標+法線）を格納
sphere_EBO = glGenBuffers(1) # インデックスデータを格納（頂点の結合順序）
instance_VBO = glGenBuffers(1) # インスタンシング用データ（各粒子の位置、色など）を格納

glBindVertexArray(sphere_VAO) # 以降の頂点属性設定をこのVAOに記録

sphere_vertex_data = np.hstack([sphere_vertices, sphere_normals]) # 頂点座標(3要素)と法線ベクトル(3要素)を連結して1つの配列にする

# 頂点データを転送
glBindBuffer(GL_ARRAY_BUFFER, sphere_VBO)
glBufferData(GL_ARRAY_BUFFER, sphere_vertex_data.nbytes, sphere_vertex_data, GL_STATIC_DRAW)

# インデックスデータを転送
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_indices.nbytes, sphere_indices, GL_STATIC_DRAW)

# 球体の頂点属性設定
glEnableVertexAttribArray(0) # 属性スロット0を有効化
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
glEnableVertexAttribArray(1) # 属性スロット0を有効化
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

# インスタンシング用データの準備
max_instance_data = np.zeros((gui_params['max_particles'], 7), dtype=np.float32) # 各粒子につき7つのデータ: [位置x,y,z, スケール, 色r,g,b] = 7要素

# インスタンスVBOにデータを転送
glBindBuffer(GL_ARRAY_BUFFER, instance_VBO)
glBufferData(GL_ARRAY_BUFFER, max_instance_data.nbytes, max_instance_data, GL_DYNAMIC_DRAW)

vbo_resource = cuda_gl.BufferObject(int(instance_VBO))

# インスタンシング用頂点属性の設定
#各インスタンスの位置
glEnableVertexAttribArray(2) # 属性スロット2を有効化
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(0))
glVertexAttribDivisor(2, 1) # インスタンス毎に1回更新

#各インスタンスのスケール
glEnableVertexAttribArray(3) # 属性スロット3を有効化
glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(12))
glVertexAttribDivisor(3, 1) # インスタンス毎に1回更新

#各インスタンスの色
glEnableVertexAttribArray(4) # 属性スロット4を有効化
glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 28, ctypes.c_void_p(16))
glVertexAttribDivisor(4, 1) # インスタンス毎に1回更新

glBindVertexArray(0) # 設定完了

arrow_mesh = trimesh.load("./3D_models/vector.obj", force='mesh') 
arrow_vertices = np.array(arrow_mesh.vertices, dtype=np.float32)
arrow_normals = np.array(arrow_mesh.vertex_normals, dtype=np.float32)
arrow_indices = np.array(arrow_mesh.faces, dtype=np.uint32).flatten()

arrow_VAO = glGenVertexArrays(1)
arrow_VBO = glGenBuffers(1)
arrow_EBO = glGenBuffers(1)

glBindVertexArray(arrow_VAO)

arrow_vertex_data = np.hstack([arrow_vertices, arrow_normals])

glBindBuffer(GL_ARRAY_BUFFER, arrow_VBO)
glBufferData(GL_ARRAY_BUFFER, arrow_vertex_data.nbytes, arrow_vertex_data, GL_STATIC_DRAW)

glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, arrow_EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, arrow_indices.nbytes, arrow_indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

max_vector_instances = 1000  # 最大矢印数
vector_instance_VBO = glGenBuffers(1)

# 矢印用VAOにインスタンシング設定を追加
glBindVertexArray(arrow_VAO)
glBindBuffer(GL_ARRAY_BUFFER, vector_instance_VBO)

# 矢印インスタンス用頂点属性
glEnableVertexAttribArray(2)  # 位置
glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(0))
glVertexAttribDivisor(2, 1)

glEnableVertexAttribArray(3)  # 方向
glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(12))
glVertexAttribDivisor(3, 1)

glEnableVertexAttribArray(4)  # スケール
glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(24))
glVertexAttribDivisor(4, 1)

glEnableVertexAttribArray(5)  # 色
glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 44, ctypes.c_void_p(28))
glVertexAttribDivisor(5, 1)

glBindVertexArray(0)

projection = Matrix44.perspective_projection(45.0, mode.size.width/mode.size.height, 0.1, 100.0) # 透視投影行列を作成

glEnable(GL_DEPTH_TEST) # 深度テストを有効化
glEnable(GL_CULL_FACE) # カリング（背面非表示）を有効化

# === メインループ ===
#初期設定
angle_speed = 0.02
camera_control_mode = 'keyboard'
target_camera_theta = 3.14 / 4.0
easing_speed = 0.1
is_easing = False
mode_switch_cooldown = 0.0
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
print("新機能: ポアソン方程式による電場計算")
print("===============")

frame_times = []
frame_count = 0
last_particle_count = gui_params['num_particles']
last_geometoty_mode = gui_params['geometry_mode']
last_electric_field_settings = None
field_update_counter = 0
audio_particle_scale = 0

particle_update_skip_counter = 0

poisson_times = []
particle_times = []
render_times = []

particle_detail_times = {
    "prepare_numpy": [],
    "gpu_alloc": [],
    "htod": [],
    "cuda_func": [],
    "dtoh": [],
    "particle_update": [],
    "total": []
}

simulation_time = 0.0
real_time_start = time.time()
convert_particles_to_soa(particles)
#メイン
while not glfw.window_should_close(window):
    total_start_time = time.time()


    #if frame_count % 60 == 0:  # 60フレームに1回だけ計算
    #    gui_params['dt'] = calculate_safe_dt(particles, E_field, gui_params)

    #入力処理
    glfw.poll_events()
    impl.process_inputs()
    
    #粒子数の変更検出と再生成
    if gui_params['num_particles'] != last_particle_count:
        particles = create_particles(gui_params['num_particles'])
        convert_particles_to_soa(particles)
        last_particle_count = gui_params['num_particles']
        clear_gpu_arrays()
    
    if gui_params['geometry_mode'] != last_geometoty_mode:
        bbox_vertices = create_bounding_box()
        glBindBuffer(GL_ARRAY_BUFFER, bbox_VBO)
        glBufferData(GL_ARRAY_BUFFER, bbox_vertices.nbytes, bbox_vertices, GL_STATIC_DRAW)
        particles = create_particles(gui_params['num_particles'])
        last_geometoty_mode = gui_params['geometry_mode']
        clear_gpu_arrays()
    
    # 電場設定の変更を検出
    current_electric_field_settings = {
        'use_poisson': gui_params['use_poisson'],
        'external_E_field_x': gui_params['external_E_field_x'],
        'external_E_field_y': gui_params['external_E_field_y'],
        'external_E_field_z': gui_params['external_E_field_z'],
        'electrode_mode': gui_params['electrode_mode'],
        'electrode_voltage': gui_params['electrode_voltage'],
        'electrode_positions': tuple(gui_params['electrode_positions'].values()),
        'num_particles': len(particles)
    }
    
    field_needs_update = False
    if last_electric_field_settings != current_electric_field_settings:
        field_needs_update = True
        last_electric_field_settings = current_electric_field_settings.copy()
    
    # ポアソン方程式使用時は定期的に電場を更新（粒子位置変化のため）
    if gui_params['use_poisson']:
        field_update_counter += 1
        if field_update_counter >= 5:
            field_needs_update = True
            field_update_counter = 0

    poisson_start_time = time.time()

    #電場更新
    if field_needs_update:
        update_electric_field_poisson_optimized(particles)
    
    poisson_end_time = time.time()
    poisson_time = (poisson_end_time - poisson_start_time) * 1000
    # 粒子の電荷と質量を更新
    for p in particles:
        if p.charge > 0:
            p.charge = gui_params['positive_charge']
        else:
            p.charge = gui_params['negative_charge']
        p.mass = gui_params['particle_mass']
    update_particle_emitter() #粒子エミッターの更新
    particle_start_time = time.time()
    #粒子位置・速度の更新
    detail_timings = update_particles_optimized(soa, E_field)

    particle_end_time = time.time()
    particle_time = (particle_end_time - particle_start_time) * 1000
    if not gui_params['simulation_paused']:
        simulation_time += gui_params['dt']
    if i_switch >= 0:
        i_switch -= 0.1
    
    #キー入力によるモード切替
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
    
    render_start_time = time.time()
    glClearColor(*current_bg_color)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    #カメラ制御
    if camera_control_mode == 'keyboard':
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
    elif camera_control_mode == 'easing':
        theta_diff = target_camera_theta - camera_theta
        while theta_diff > np.pi:
            theta_diff -= 2 * np.pi
        while theta_diff < -np.pi:
            theta_diff += 2 * np.pi
        
        if abs(theta_diff) > 0.001:
            camera_theta += theta_diff * easing_speed
            is_easing = True
        else:
            camera_theta = target_camera_theta
        
        if abs(theta_diff) <= 0.001:
            is_easing = False

    camX = radius * np.cos(camera_phi) * np.sin(camera_theta)
    camY = radius * np.sin(camera_phi)
    camZ = radius * np.cos(camera_phi) * np.cos(camera_theta)
    eye = np.array([camX, camY, camZ])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    view = Matrix44.look_at(eye, target, up)

    light_position = eye.copy()

    #バウンディングボックス描画
    if gui_params['show_bbox'] and bg_color_flag == False:
        glUseProgram(bbox_shader_program)
        glUniformMatrix4fv(bbox_view_loc, 1, GL_FALSE, view.astype(np.float32))
        glUniformMatrix4fv(bbox_proj_loc, 1, GL_FALSE, projection.astype(np.float32))
        glUniform3fv(bbox_color_loc, 1, np.array([0, 0, 0], dtype=np.float32))
        
        glBindVertexArray(bbox_VAO)
        glDrawArrays(GL_LINES, 0, len(bbox_vertices))
    
    #粒子描画
    if len(particles) > 0 and not gui_params['show_equipotential']:
        glUseProgram(shader_program)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view.astype(np.float32))
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection.astype(np.float32))
        glUniform3fv(light_pos_loc, 1, light_position.astype(np.float32))
        glUniform3fv(view_pos_loc, 1, eye.astype(np.float32))
        glBindVertexArray(sphere_VAO)
        glDrawElementsInstanced(GL_TRIANGLES, len(sphere_indices), GL_UNSIGNED_INT, None, len(particles))
    
    # 粒子描画の後に追加
    # 電場・磁場ベクトル描画
    if gui_params['show_field_vectors']:
        field_vectors = generate_field_vectors()
        
        if len(field_vectors) > 0:
            # VBOにデータを転送
            glBindBuffer(GL_ARRAY_BUFFER, vector_instance_VBO)
            glBufferData(GL_ARRAY_BUFFER, field_vectors.nbytes, field_vectors, GL_DYNAMIC_DRAW)
            
            glUseProgram(vector_shader_program)
            glUniformMatrix4fv(vector_view_loc, 1, GL_FALSE, view.astype(np.float32))
            glUniformMatrix4fv(vector_proj_loc, 1, GL_FALSE, projection.astype(np.float32))
            glUniform3fv(vector_light_pos_loc, 1, light_position.astype(np.float32))
            glUniform3fv(vector_view_pos_loc, 1, eye.astype(np.float32))
            
            glBindVertexArray(arrow_VAO)
            glDrawElementsInstanced(GL_TRIANGLES, len(arrow_indices), GL_UNSIGNED_INT, None, len(field_vectors))

    if gui_params['show_equipotential'] and gui_params['simulation_paused']:
        # 高解像度グリッドで電位を再計算
        update_high_res_electric_field(particles, gui_params)
        
        # 高解像度グリッドから等電位線を生成
        equipotential_lines = generate_equipotential_lines_high_res(gui_params)
        
        if len(equipotential_lines) > 0:
            glBindBuffer(GL_ARRAY_BUFFER, equipotential_VBO)
            glBufferData(GL_ARRAY_BUFFER, equipotential_lines.nbytes, equipotential_lines, GL_DYNAMIC_DRAW)
            
            glUseProgram(equipotential_shader_program)
            glUniformMatrix4fv(equi_view_loc, 1, GL_FALSE, view.astype(np.float32))
            glUniformMatrix4fv(equi_proj_loc, 1, GL_FALSE, projection.astype(np.float32))
            
            glBindVertexArray(equipotential_VAO)
            glLineWidth(2.0)
            glDrawArrays(GL_LINES, 0, len(equipotential_lines) // 6)
            glLineWidth(1.0)
    else:
        equipotential_lines = np.array([], dtype=np.float32)

    if gui_params['show_electrodes'] and gui_params['internal_electrodes']:
        electrode_wireframes = generate_electrode_wireframes()

        if len(electrode_wireframes) > 0:
            glBindBuffer(GL_ARRAY_BUFFER, electrode_wireframe_VBO)
            glBufferData(GL_ARRAY_BUFFER, electrode_wireframes.nbytes, electrode_wireframes, GL_DYNAMIC_DRAW)
            glUseProgram(equipotential_shader_program)
            glUniformMatrix4fv(equi_view_loc, 1, GL_FALSE, view.astype(np.float32))
            glUniformMatrix4fv(equi_proj_loc, 1, GL_FALSE, projection.astype(np.float32))
            
            glBindVertexArray(electrode_wireframe_VAO)
            glLineWidth(2.0)
            glDrawArrays(GL_LINES, 0, len(electrode_wireframes) // 6)
            glLineWidth(1.0)

    render_end_time = time.time()
    render_time = (render_end_time - render_start_time) * 1000
    imgui.new_frame()
    
    #ImGUIの設定
    imgui.begin("Simulation Control Panel", True)
    
    if imgui.collapsing_header("Camera Control")[0]:
        mode_changed, new_mode_index = imgui.combo(
            "Control mode", 0 if camera_control_mode == 'keyboard' else 1, ["keyboard", 'Easing']
        )

        if mode_changed:
            camera_control_mode = 'keyboard' if new_mode_index == 0 else 'easing'
            if camera_control_mode == 'easing':
                target_camera_theta = camera_theta
        
        imgui.separator()

        if camera_control_mode == 'easing':
           # 90度回転ボタン
            if imgui.button("Rotate Left 90°"):
                target_camera_theta += np.pi / 2
                is_easing = True
            
            imgui.same_line()
            if imgui.button("Rotate Right 90°"):
                target_camera_theta -= np.pi / 2
                is_easing = True 

    if imgui.collapsing_header("Physics Method")[0]:
        changed, gui_params['use_poisson'] = imgui.checkbox(
            "Use Poisson Equation", gui_params['use_poisson']
        )
        if gui_params['use_poisson']:
            imgui.text("Physical electric field from charge distribution")
        else:
            imgui.text("Simple external field + Coulomb forces")
        
        imgui.separator()
        imgui.text("External Electric Field (added to Poisson field):")
        
        changed, gui_params['external_E_field_x'] = imgui.slider_float(
            "External E_x", gui_params['external_E_field_x'], -1 * max_electric_field, max_electric_field
        )
        
        changed, gui_params['external_E_field_y'] = imgui.slider_float(
            "External E_y", gui_params['external_E_field_y'], -1 * max_electric_field, max_electric_field
        )
        
        changed, gui_params['external_E_field_z'] = imgui.slider_float(
            "External E_z", gui_params['external_E_field_z'], -1 * max_electric_field, max_electric_field
        )

        changed, gui_params['magnetic_field_x'] = imgui.slider_float(
            "B_x", gui_params['magnetic_field_x'], -1 * max_magnetic_field, max_magnetic_field
        )
        
        changed, gui_params['magnetic_field_y'] = imgui.slider_float(
            "B_y", gui_params['magnetic_field_y'], -1 * max_magnetic_field, max_magnetic_field
        )
            
        changed, gui_params['magnetic_field_z'] = imgui.slider_float(
            "B_z", gui_params['magnetic_field_z'], -1 * max_magnetic_field, max_magnetic_field
        )
        
        if imgui.button("Reset External Field"):
            gui_params['external_E_field_x'] = 0.0
            gui_params['external_E_field_y'] = 0.0
            gui_params['external_E_field_z'] = 0.0
            gui_params['magnetic_field_x'] = 0.0
            gui_params['magnetic_field_y'] = 0.0
            gui_params['magnetic_field_z'] = 0.0

    if imgui.collapsing_header("Geometry Settings")[0]:
        geometry_modes = ["Cube", "Cylinder"]
        changed, gui_params['geometry_mode'] = imgui.combo(
            "Simulation geometry", gui_params['geometry_mode'], geometry_modes
        )
        
        if gui_params['geometry_mode'] == 1:  # 円柱モード
            changed, gui_params['cylinder_radius'] = imgui.slider_float(
                "Cylinder radius", gui_params['cylinder_radius'], 1.0, 4.0
            )
            if changed:
                # バウンディングボックスを再生成
                bbox_vertices = create_bounding_box()
                print('boundingbox was changed')
                glBindBuffer(GL_ARRAY_BUFFER, bbox_VBO)
                glBufferData(GL_ARRAY_BUFFER, bbox_vertices.nbytes, bbox_vertices, GL_STATIC_DRAW)
            changed, gui_params['cylinder_height'] = imgui.slider_float(
                "Cylinder height", gui_params['cylinder_height'], 2.0, 8.0
            )
            if changed:
                # バウンディングボックスを再生成
                bbox_vertices = create_bounding_box()
                print('boundingbox was changed')
                glBindBuffer(GL_ARRAY_BUFFER, bbox_VBO)
                glBufferData(GL_ARRAY_BUFFER, bbox_vertices.nbytes, bbox_vertices, GL_STATIC_DRAW)
            
            imgui.separator()
            imgui.text("Cylinder boundary behavior:")
            changed, gui_params['cylinder_top_warp'] = imgui.checkbox(
                "Warp at top surface", gui_params['cylinder_top_warp']
            )
            changed, gui_params['cylinder_bottom_warp'] = imgui.checkbox(
                "Warp at bottom surface", gui_params['cylinder_bottom_warp']
            )
            changed, gui_params['cylinder_wall_vanish'] = imgui.checkbox(
                "Vanish at cylinder wall", gui_params['cylinder_wall_vanish']
            )
        else:
            imgui.text("X-axis boundary:")
            changed, gui_params['warp_x_negative'] = imgui.checkbox(
                "Warp X-", gui_params['warp_x_negative']
            )
            imgui.same_line()
            changed, gui_params['cube_vanish_x_neg'] = imgui.checkbox(
                "Vanish X-", gui_params['cube_vanish_x_neg']
            )
            
            changed, gui_params['warp_x_positive'] = imgui.checkbox(
                "Warp X+", gui_params['warp_x_positive']
            )
            imgui.same_line()
            changed, gui_params['cube_vanish_x_pos'] = imgui.checkbox(
                "Vanish X+", gui_params['cube_vanish_x_pos']
            )
            
            imgui.text("Y-axis boundary:")
            changed, gui_params['warp_y_negative'] = imgui.checkbox(
                "Warp Y-", gui_params['warp_y_negative']
            )
            imgui.same_line()
            changed, gui_params['cube_vanish_y_neg'] = imgui.checkbox(
                "Vanish Y-", gui_params['cube_vanish_y_neg']
            )
            
            changed, gui_params['warp_y_positive'] = imgui.checkbox(
                "Warp Y+", gui_params['warp_y_positive']
            )
            imgui.same_line()
            changed, gui_params['cube_vanish_y_pos'] = imgui.checkbox(
                "Vanish Y+", gui_params['cube_vanish_y_pos']
            )
            
            imgui.text("Z-axis boundary:")
            changed, gui_params['warp_z_negative'] = imgui.checkbox(
                "Warp Z-", gui_params['warp_z_negative']
            )
            imgui.same_line()
            changed, gui_params['cube_vanish_z_neg'] = imgui.checkbox(
                "Vanish Z-", gui_params['cube_vanish_z_neg']
            )
            
            changed, gui_params['warp_z_positive'] = imgui.checkbox(
                "Warp Z+", gui_params['warp_z_positive']
            )
            imgui.same_line()
            changed, gui_params['cube_vanish_z_pos'] = imgui.checkbox(
                "Vanish Z+", gui_params['cube_vanish_z_pos']
            )

    if imgui.collapsing_header("Electrode Settings")[0]:
        changed, gui_params['electrode_mode'] = imgui.checkbox(
            "Enable Electrodes", gui_params['electrode_mode']
        )
        
        if gui_params['electrode_mode']:
            if not gui_params['use_rf_voltage']:
                changed, gui_params['electrode_voltage'] = imgui.slider_float(
                    "Electrode Voltage", gui_params['electrode_voltage'], -50.0, 50.0
                )
                simulation_time = 0.0
            else:
                simulation_time += gui_params['dt']
                gui_params['electrode_voltage'] = gui_params['rf_amplitude'] * np.sin(
                    2 * np.pi * gui_params['rf_frequency'] * simulation_time + gui_params['rf_phase']
                )
                #if gui_params['dt'] > 1 / (10 * gui_params['rf_frequency']):
                #    gui_params['dt'] = 0.5 / (10 * gui_params['rf_frequency'])
            
            imgui.text("Select electrode surfaces:")
            
            changed, gui_params['electrode_positions']['x_neg'] = imgui.checkbox(
                "X- electrode", gui_params['electrode_positions']['x_neg']
            )
            changed, gui_params['electrode_positions']['x_pos'] = imgui.checkbox(
                "X+ electrode", gui_params['electrode_positions']['x_pos']
            )
            
            changed, gui_params['electrode_positions']['y_neg'] = imgui.checkbox(
                "Y- electrode", gui_params['electrode_positions']['y_neg']
            )
            changed, gui_params['electrode_positions']['y_pos'] = imgui.checkbox(
                "Y+ electrode", gui_params['electrode_positions']['y_pos']
            )
            
            changed, gui_params['electrode_positions']['z_neg'] = imgui.checkbox(
                "Z- electrode", gui_params['electrode_positions']['z_neg']
            )
            changed, gui_params['electrode_positions']['z_pos'] = imgui.checkbox(
                "Z+ electrode", gui_params['electrode_positions']['z_pos']
            )

            changed, gui_params['use_rf_voltage'] = imgui.checkbox(
                "use rf voltage", gui_params['use_rf_voltage']
            )

            if gui_params['use_rf_voltage']:
                changed, gui_params['rf_amplitude'] = imgui.slider_float(
                    'Amplitude', gui_params['rf_amplitude'], 0, 100
                )
                changed, gui_params['rf_frequency'] = imgui.slider_float(
                    'Frequency', gui_params['rf_frequency'], 1, 1000
                )
                changed, gui_params['rf_phase'] = imgui.slider_float(
                    'Phase', gui_params['rf_phase'], 0, 2 * np.pi
                )
            
            imgui.separator()
        else:
            imgui.text("Electrodes disabled - only particle charges create fields")
    
    if imgui.collapsing_header("Internal Electrodes")[0]:
        imgui.text(f"Number of electrodes: {len(gui_params['internal_electrodes'])}")
        
        imgui.separator()
        imgui.text("Add new electrode:")
        
        shapes = ["Sphere", "Box", "Cylinder"]
        changed, gui_params['current_electrode_shape'] = imgui.combo(
            "Shape", gui_params['current_electrode_shape'], shapes
        )
        
        changed, gui_params['new_electrode_position'][0] = imgui.slider_float(
            "Position X", gui_params['new_electrode_position'][0], -2.0, 2.0
        )
        changed, gui_params['new_electrode_position'][1] = imgui.slider_float(
            "Position Y", gui_params['new_electrode_position'][1], -2.0, 2.0
        )
        changed, gui_params['new_electrode_position'][2] = imgui.slider_float(
            "Position Z", gui_params['new_electrode_position'][2], -2.0, 2.0
        )
        
        changed, gui_params['new_electrode_size'] = imgui.slider_float(
            "Size", gui_params['new_electrode_size'], 0.1, 1.0
        )
        
        changed, gui_params['new_electrode_voltage'] = imgui.slider_float(
            "Voltage", gui_params['new_electrode_voltage'], -50.0, 50.0
        )
        
        if imgui.button("Add Electrode"):
            new_electrode = {
                'position': gui_params['new_electrode_position'].copy(),
                'size': gui_params['new_electrode_size'],
                'voltage': gui_params['new_electrode_voltage'],
                'shape': ['sphere', 'box', 'cylinder'][gui_params['current_electrode_shape']]
            }
            gui_params['internal_electrodes'].append(new_electrode)
            clear_gpu_arrays()
            print(f"Added electrode at {new_electrode['position']}")
        
        imgui.separator()
        if len(gui_params['internal_electrodes']) > 0:
            imgui.text("Existing electrodes:")
            electrodes_to_remove = []
            
            for idx, elec in enumerate(gui_params['internal_electrodes']):
                imgui.push_id(f"elec_{idx}")
                imgui.text(f"#{idx}: {elec['shape']} at {elec['position']}, V={elec['voltage']:.1f}V")
                
                if imgui.button(f"Remove##{idx}"):
                    electrodes_to_remove.append(idx)
                
                imgui.pop_id()
            
            # 削除処理
            for idx in reversed(electrodes_to_remove):
                gui_params['internal_electrodes'].pop(idx)
                clear_gpu_arrays()
            
            if imgui.button("Clear All Electrodes"):
                gui_params['internal_electrodes'].clear()
                clear_gpu_arrays()
        
        changed, gui_params['show_electrodes'] = imgui.checkbox(
            "Show electrodes", gui_params['show_electrodes']
        )

    if imgui.collapsing_header("Particle Emitter")[0]:
        changed, gui_params['enable_emitter'] = imgui.checkbox(
            "Enable particle emitter", gui_params['enable_emitter']
        )

        if gui_params['enable_emitter']:
            imgui.text("Emitter position:")
            changed, gui_params['emitter_position'][0] = imgui.slider_float(
                "Emitter X", gui_params['emitter_position'][0], -4.0, 4.0
            )
            changed, gui_params['emitter_position'][1] = imgui.slider_float(
                "Emitter Y", gui_params['emitter_position'][1], -4.0, 4.0
            )
            changed, gui_params['emitter_position'][2] = imgui.slider_float(
                "Emitter Z", gui_params['emitter_position'][2], -4.0, 4.0
            )

            imgui.separator()
            imgui.text("Emission direction:")
            changed, gui_params['emitter_direction'][0] = imgui.slider_float(
                "Direction X", gui_params['emitter_direction'][0], -1.0, 1.0
            )
            changed, gui_params['emitter_direction'][1] = imgui.slider_float(
                "Direction Y", gui_params['emitter_direction'][1], -1.0, 1.0
            )
            changed, gui_params['emitter_direction'][2] = imgui.slider_float(
                "Direction Z", gui_params['emitter_direction'][2], -1.0, 1.0
            )

            changed, gui_params['emitter_speed'] = imgui.slider_float(
                "Initial speed", gui_params['emitter_speed'], 0.1, 20.0
            )
            changed, gui_params['emitter_spread'] = imgui.slider_float(
                "Direction spread (rad)", gui_params['emitter_spread'], 0.0, 1.57
            )

            imgui.separator()
            charge_types = ["Positive", "Negative", "Random"]
            changed, gui_params['emitter_charge_type'] = imgui.combo(
                "Charge type", gui_params['emitter_charge_type'], charge_types
            )
            
            changed, gui_params['emitter_rate'] = imgui.slider_int(
                "Emission rate (frames)", gui_params['emitter_rate'], 1, 60
            )
            imgui.text(f"Particles/sec: {60.0/gui_params['emitter_rate']:.1f}")
            
            changed, gui_params['max_emitted_particles'] = imgui.slider_int(
                "Max particles", gui_params['max_emitted_particles'], 1000, 10000
            )
            
            imgui.separator()
            if imgui.button("Reset emitter position"):
                gui_params['emitter_position'] = [0.0, -2.0, 0.0]
                gui_params['emitter_direction'] = [0.0, 1.0, 0.0]
            
            imgui.same_line()
            if imgui.button("Stop emitter"):
                gui_params['enable_emitter'] = False
            
            imgui.text(f"Current particles: {len(particles)}")
        else:
            imgui.text("Emitter disabled")

    if imgui.collapsing_header("Particle Settings")[0]:
        changed, gui_params['num_particles'] = imgui.slider_int(
            "Number of particles", gui_params['num_particles'], 0, gui_params['max_particles']
        )
        
        changed, gui_params['positive_charge'] = imgui.slider_float(
            "Positive charge", gui_params['positive_charge'], 0.1, 5.0
        )
        
        changed, gui_params['negative_charge'] = imgui.slider_float(
            "Negative charge", gui_params['negative_charge'], -5.0, -0.1
        )
        
        changed, gui_params['particle_mass'] = imgui.slider_float(
            "Particle mass", gui_params['particle_mass'], 0.1, 5.0
        )
        
        changed, gui_params['scale_factor'] = imgui.slider_float(
            "Particle size", gui_params['scale_factor'], 0.01, 0.1
        )

    if imgui.collapsing_header("Simulation Settings")[0]:
        imgui.text("=== Automatic dt Calculation ===")
        
        # 現在のdt
        imgui.text(f"Current dt: {gui_params['dt']:.6f} [sim units]")
        dt_si = physics_scale.simulation_to_si(gui_params['dt'], 'time')
        if dt_si < 1e-12:
            imgui.text(f"dt (SI): {dt_si*1e15:.3f} fs")
        elif dt_si < 1e-9:
            imgui.text(f"dt (SI): {dt_si*1e12:.3f} ps")
        elif dt_si < 1e-6:
            imgui.text(f"dt (SI): {dt_si*1e9:.3f} ns")
        else:
            imgui.text(f"dt (SI): {dt_si*1e6:.3f} µs")
        
        imgui.separator()
        imgui.text("=== Stability Criteria ===")
        
        # 1. CFL条件(速度に基づく)
        if len(particles) > 0:
            max_velocity = max([np.linalg.norm(p.velocity) for p in particles])
            if max_velocity > 1e-6:
                dt_cfl = 0.5 * cell_size / max_velocity
                imgui.text(f"CFL limit: {dt_cfl:.6f}")
                imgui.text(f"  (cell_size / max_velocity)")
                if gui_params['dt'] > dt_cfl:
                    imgui.text_colored("  WARNING: dt exceeds CFL!", 1.0, 0.3, 0.3)
            else:
                dt_cfl = 0.1
                imgui.text("CFL limit: N/A (velocity too small)")
        else:
            dt_cfl = 0.1
            imgui.text("CFL limit: N/A (no particles)")
        
        imgui.separator()
        
        # 2. 加速度条件
        max_E = np.max(np.linalg.norm(E_field, axis=3))
        if max_E > 1e-6 and len(particles) > 0:
            max_charge = max(abs(gui_params['positive_charge']), 
                            abs(gui_params['negative_charge']))
            min_mass = gui_params['particle_mass']
            max_accel = (max_charge * max_E) / min_mass
            
            dt_accel = 0.5 * np.sqrt(2 * cell_size / max_accel)
        else:
            dt_accel = 0.1
        
        # 3. サイクロトロン周波数
        if gui_params['enable_magnetic_field']:
            B_mag = np.sqrt(
                gui_params['magnetic_field_x']**2 + 
                gui_params['magnetic_field_y']**2 + 
                gui_params['magnetic_field_z']**2
            )
            if B_mag > 1e-6 and len(particles) > 0:
                max_charge = max(abs(gui_params['positive_charge']), 
                                abs(gui_params['negative_charge']))
                min_mass = gui_params['particle_mass']
                
                cyclotron_freq = max_charge * B_mag / min_mass
                cyclotron_period = 2 * np.pi / cyclotron_freq
                dt_cyclotron = 0.1 * cyclotron_period
            else:
                dt_cyclotron = 0.1
        else:
            dt_cyclotron = 0.1
        
        imgui.separator()
        
        # 4. RF電圧の周波数
        if gui_params['use_rf_voltage']:
            rf_period = 1.0 / gui_params['rf_frequency']
            dt_rf = rf_period / 20.0  # 1周期を20分割
        else:
            dt_rf = 0.1
        
        imgui.separator()
        imgui.text("=== Recommended dt ===")
        
        # 最も厳しい条件を採用
        dt_recommended = min(dt_cfl, dt_accel, dt_cyclotron, dt_rf)
        dt_recommended *= gui_params['dt_safety_factor']
        dt_recommended = np.clip(dt_recommended, 0.0001, 0.05)
        
        imgui.text(f"Recommended: {dt_recommended:.6f}")
        imgui.text(f"(with safety factor {gui_params['dt_safety_factor']:.2f})")
        
        # 自動設定ボタン
        if imgui.button("Apply Recommended dt"):
            gui_params['dt'] = dt_recommended
            print(f"dt set to {dt_recommended:.6f}")
        
        imgui.same_line()
        if imgui.button("Set Conservative dt"):
            gui_params['dt'] = dt_recommended * 0.5
            print(f"dt set to conservative {dt_recommended * 0.5:.6f}")
        
        imgui.separator()
        
        # 安全係数の調整
        changed, gui_params['dt_safety_factor'] = imgui.slider_float(
            "Safety Factor", gui_params['dt_safety_factor'], 0.1, 1.0
        )
        imgui.text("Lower = more stable but slower")
        
        # 手動調整
        imgui.separator()
        imgui.text("Manual adjustment:")
        changed, gui_params['dt'] = imgui.slider_float(
            "Time step (dt)", gui_params['dt'], 0.0001, 0.05, "%.6f"
        )
        
        # プリセット値
        if imgui.button("dt = 0.001"):
            gui_params['dt'] = 0.001
        imgui.same_line()
        if imgui.button("dt = 0.005"):
            gui_params['dt'] = 0.005
        imgui.same_line()
        if imgui.button("dt = 0.01"):
            gui_params['dt'] = 0.01
        imgui.separator()
        changed, gui_params['coulomb_force_control'] = imgui.slider_float(
            "Coulomb force strength", gui_params['coulomb_force_control'], 0.0, 1.0
        )
        
        changed, gui_params['simulation_paused'] = imgui.checkbox(
            "Simulation paused", gui_params['simulation_paused']
        )
        
        if imgui.button("Reset particles"):
            particles = create_particles(gui_params['num_particles'])
            if hasattr(update_particles_optimized, 'gpu_arrays'):
                delattr(update_particles_optimized, 'gpu_arrays')
            if hasattr(update_electric_field_poisson_optimized, 'gpu_arrays'):
                delattr(update_electric_field_poisson_optimized, 'gpu_arrays')
            print("粒子をリセットしました")
        
        imgui.same_line()
        if imgui.button("Reset velocity"):
            for p in particles:
                p.velocity = np.zeros(3)
            print("全粒子の速度をリセットしました")
    
    if imgui.collapsing_header("Display Settings")[0]:
        changed, gui_params['show_bbox'] = imgui.checkbox(
            "Show bounding box", gui_params['show_bbox']
        )
        
        changed, gui_params['positive_color'] = imgui.color_edit3(
            "Positive color", *gui_params['positive_color']
        )
        
        changed, gui_params['negative_color'] = imgui.color_edit3(
            "Negative color", *gui_params['negative_color']
        )
    
    if imgui.collapsing_header("Statistics & Information")[0]:
        imgui.text(f"Current particles: {len(particles)}")
        positive_count = sum(1 for p in particles if p.charge > 0)
        negative_count = len(particles) - positive_count
        imgui.text(f"Positive: {positive_count}, Negative: {negative_count}")
        
        imgui.separator()
        imgui.text("Simulation Time")
        imgui.text(f"Simulation time: {simulation_time:.6f} [sim units]")
    
        # SI単位(秒)での時間
        time_si = physics_scale.simulation_to_si(simulation_time, 'time')
        if time_si < 1e-9:
            imgui.text(f"SI time: {time_si*1e12:.3f} ps")
        elif time_si < 1e-6:
            imgui.text(f"SI time: {time_si*1e9:.3f} ns")
        elif time_si < 1e-3:
            imgui.text(f"SI time: {time_si*1e6:.3f} µs")
        elif time_si < 1:
            imgui.text(f"SI time: {time_si*1e3:.3f} ms")
        else:
            imgui.text(f"SI time: {time_si:.3f} s")

        real_time_elapsed = time.time() - real_time_start
        imgui.text(f"Real time elapsed: {real_time_elapsed:.2f} s")
        
        # 時間スケール比
        if real_time_elapsed > 0:
            time_ratio = time_si / real_time_elapsed
            imgui.text(f"Simulation speed: {time_ratio:.2e}x real-time")
        
        # リセットボタン
        if imgui.button("Reset simulation time"):
            simulation_time = 0.0
            real_time_start = time.time()
        
        imgui.separator()

        if len(particles) > 0:
            avg_speed = np.mean([np.linalg.norm(p.velocity) for p in particles])
            imgui.text(f"Average velocity: {avg_speed:.2f}")
        
        if len(frame_times) > 0:
            avg_frame_time = np.mean(frame_times[-60:]) if len(frame_times) >= 60 else np.mean(frame_times)
            fps = 1000.0 / avg_frame_time if avg_frame_time > 0 else 0
            imgui.text(f"FPS: {fps:.1f}")
            imgui.text(f"Frame time: {avg_frame_time:.2f}ms")
        
        imgui.separator()
        imgui.text("Grid information:")
        imgui.text(f"Grid size: {grid_divisions}×{grid_divisions}×{grid_divisions}")
        imgui.text(f"Cell size: {cell_size:.3f}")
        
        if gui_params['use_poisson']:
            imgui.text("Using Poisson equation for electric field")
        else:
            imgui.text("Using simple Coulomb + external field")

    # メインループ内のGUI部分に以下を追加
    if imgui.collapsing_header("Field Visualization")[0]:
        _, gui_params['show_field_vectors'] = imgui.checkbox("Show Field Vectors", gui_params['show_field_vectors'])
        
        if gui_params['show_field_vectors']:
            _, gui_params['field_vector_scale'] = imgui.slider_float("Vector Scale", gui_params['field_vector_scale'], 0.1, 5.0)
            _, gui_params['field_sample_spacing'] = imgui.slider_int("Sample Spacing", gui_params['field_sample_spacing'], 1, 4)
            
            _, gui_params['show_electric_field'] = imgui.checkbox("Show Electric Field", gui_params['show_electric_field'])
            if gui_params['show_electric_field']:
                _, gui_params['electric_field_color'] = imgui.color_edit3("E-Field Color", *gui_params['electric_field_color'])
            
            _, gui_params['show_magnetic_field'] = imgui.checkbox("Show Magnetic Field", gui_params['show_magnetic_field'])
            if gui_params['show_magnetic_field']:
                _, gui_params['magnetic_field_color'] = imgui.color_edit3("B-Field Color", *gui_params['magnetic_field_color'])
    
    if imgui.collapsing_header("Equipotential Lines")[0]:
        if not gui_params['simulation_paused']:
            imgui.text_colored("Pause simulation to use this feature", 1.0, 0.5, 0.0)
        
        _, gui_params['show_equipotential'] = imgui.checkbox("Show Equipotential Lines", gui_params['show_equipotential'])
        
        if gui_params['show_equipotential']:
            plane_names = ["XZ Plane", "YX Plane", "YZ Plane"]
            _, gui_params['equipotential_plane'] = imgui.combo("Slice Plane", gui_params['equipotential_plane'], plane_names)
            
            axis_name = ["Y", "Z", "X"][gui_params['equipotential_plane']]
            _, gui_params['equipotential_slice_position'] = imgui.slider_float(
                f"Slice Position ({axis_name})", gui_params['equipotential_slice_position'], -2.0, 2.0
            )
            
            _, gui_params['equipotential_num_lines'] = imgui.slider_int("Number of Lines", gui_params['equipotential_num_lines'], 5, 30)
            
            _, gui_params['equipotential_color'] = imgui.color_edit3("Line Color", *gui_params['equipotential_color'])
            
            if imgui.button("Reset to center"):
                gui_params['equipotential_slice_position'] = 0.0
    
    if imgui.collapsing_header("Physics Scale Settings")[0]:
        imgui.text("=== length scale ===")
        length_modes = list(physics_scale.length_scale_modes.keys())
        length_names = [physics_scale.length_scale_modes[m]['name'] for m in length_modes]
        
        current_length_idx = length_modes.index(physics_scale.current_length_mode)
        changed, new_length_idx = imgui.combo(
            "Length Scale", current_length_idx, length_names
        )
        if changed:
            physics_scale.current_length_mode = length_modes[new_length_idx]
            print(f"change length scale: {physics_scale.length_scale_modes[physics_scale.current_length_mode]['name']}")
        
        current_length = physics_scale.length_scale_modes[physics_scale.current_length_mode]
        imgui.text(f"unit: {current_length['unit']}")
        imgui.text(f"explanation: {current_length['description']}")
        imgui.text(f"transform: 1 unit = {physics_scale.get_length_conversion():.2e} m")
        
        imgui.separator()
        imgui.text("=== time scale ===")
        time_modes = list(physics_scale.time_scale_modes.keys())
        time_names = [physics_scale.time_scale_modes[m]['name'] for m in time_modes]
        
        current_time_idx = time_modes.index(physics_scale.current_time_mode)
        changed, new_time_idx = imgui.combo(
            "Time Scale", current_time_idx, time_names
        )
        if changed:
            physics_scale.current_time_mode = time_modes[new_time_idx]
            print(f"change time scale: {physics_scale.time_scale_modes[physics_scale.current_time_mode]['name']}")
        
        current_time = physics_scale.time_scale_modes[physics_scale.current_time_mode]
        imgui.text(f"unit: {current_time['unit']}")
        imgui.text(f"explanation: {current_time['description']}")
        imgui.text(f"transform: 1 unit = {physics_scale.get_time_conversion():.2e} s")
        
        imgui.separator()
        imgui.text("=== Derived amount ===")
        velocity_scale = physics_scale.get_length_conversion() / physics_scale.get_time_conversion()
        imgui.text(f"velocity scale: {velocity_scale:.2e} m/s")
        
        if len(particles) > 0:
            avg_v_sim = np.mean([np.linalg.norm(p.velocity) for p in particles])
            avg_v_si = avg_v_sim * velocity_scale
            imgui.text(f"average velocity(sim): {avg_v_sim:.2f}")
            imgui.text(f"average velocity(SI): {avg_v_si:.2e} m/s")
        
        imgui.separator()
        if imgui.button("output scale information"):
            print(physics_scale.get_info_string())
    imgui.end()
    
    imgui.render()
    impl.render(imgui.get_draw_data())
    
    glfw.swap_buffers(window)
    
    # 各処理時間を記録
    poisson_times.append(poisson_time)
    particle_times.append(particle_time)
    render_times.append(render_time)

    for k in particle_detail_times:
        if k in detail_timings:
            particle_detail_times[k].append(detail_timings[k])
    
    # 古いデータを削除（最新60フレーム分のみ保持）
    if len(poisson_times) > 60:
        poisson_times.pop(0)
        particle_times.pop(0)
        render_times.pop(0)
    
    #フレームレートの計算
    total_end_time = time.time()
    total_frame_time = (total_end_time - total_start_time) * 1000
    frame_times.append(total_frame_time)
    frame_count += 1
    
    if frame_count % 60 == 0:
        avg_total_time = np.mean(frame_times[-60:])
        avg_poisson_time = np.mean(poisson_times) if poisson_times else 0
        avg_particle_time = np.mean(particle_times) if particle_times else 0
        avg_render_time = np.mean(render_times) if render_times else 0
        fps = 1000.0 / avg_total_time if avg_total_time > 0 else 0
        
        print(f"=== Performance Analysis ===")
        print(f"Total frame time: {avg_total_time:.2f}ms, FPS: {fps:.1f}")
        print(f"Poisson solver: {avg_poisson_time:.2f}ms ({avg_poisson_time/avg_total_time*100:.1f}%)")
        print(f"Particle update: {avg_particle_time:.2f}ms ({avg_particle_time/avg_total_time*100:.1f}%)")
        print(f"Rendering: {avg_render_time:.2f}ms ({avg_render_time/avg_total_time*100:.1f}%)")
        print(f"Other: {avg_total_time-avg_poisson_time-avg_particle_time-avg_render_time:.2f}ms")
        print("============================")

        print("--- Particle update breakdown ---")
        for k, v in particle_detail_times.items():
            if v:  # データがある場合のみ
                avg = np.mean(v)
                ratio = avg / avg_particle_time * 100 if avg_particle_time > 0 else 0
                print(f"{k:15s}: {avg:.3f} ms ({ratio:.1f}%)")
        print("============================")
        print(f"Potential stats: min={np.min(potential_grid):.3f}, max={np.max(potential_grid):.3f}, mean={np.mean(potential_grid):.3f}")
        center_idx = grid_divisions // 2
        print(f"Center potential: {potential_grid[center_idx, center_idx, center_idx]:.3f}")

        for k in particle_detail_times:
            particle_detail_times[k].clear()
    
    target_frame_time = 1000.0 / 60.0
    if total_frame_time < target_frame_time:
        time.sleep((target_frame_time - total_frame_time) / 1000.0)

cleanup_resources()
impl.shutdown()
glfw.terminate()