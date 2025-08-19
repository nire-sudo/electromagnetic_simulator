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

boundary_mode = 0
i_switch = 0.0

# === CUDAをロード ===
def load_cuda_kernel(filepath, funcname):
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    return SourceModule(code).get_function(funcname)

cuda_func = load_cuda_kernel("./cuda_program/cal_physics2.cu", "update_particles_cuda")
charge_density_func = load_cuda_kernel("./cuda_program/cal_physics2.cu", "compute_charge_density")
electric_field_func = load_cuda_kernel("./cuda_program/cal_physics2.cu", "compute_electric_field_from_potential")

# === GLFW 初期化 ===
if not glfw.init():
    raise Exception("GLFWの初期化に失敗しました")

camera_theta = 3.14 / 4.0
camera_phi = np.radians(30.0)

monitor = glfw.get_monitors()[1]
mode = glfw.get_video_mode(monitor)

xpos, ypos = glfw.get_monitor_pos(monitor)
glfw.window_hint(glfw.DECORATED, glfw.FALSE)

window = glfw.create_window(mode.size.width, mode.size.height, "Poisson Equation Particle Simulation", None, None)
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

# === シェーダープログラムのセットアップ ===
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)              # 空のシェーダーオブジェクト作成
    glShaderSource(shader, source)                    # GLSLソースコードをシェーダーにセット
    glCompileShader(shader)                           # コンパイル
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):  # コンパイル結果の確認
        raise RuntimeError(glGetShaderInfoLog(shader).decode())  # エラー内容を表示
    return shader

#通常描画用シェーダープログラム
shader_program = glCreateProgram()
vs = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)   # 頂点シェーダーをコンパイル
fs = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER) # フラグメントシェーダーをコンパイル
glAttachShader(shader_program, vs)                     # プログラムにアタッチ
glAttachShader(shader_program, fs)
glLinkProgram(shader_program)                          # 頂点＆フラグメントをリンクして実行可能に
glDeleteShader(vs)                                     # 単体のシェーダーオブジェクトは削除
glDeleteShader(fs)

#バウンディングボックス用シェーダープログラム
bbox_shader_program = glCreateProgram()
bbox_vs = compile_shader(BBOX_VERTEX_SHADER, GL_VERTEX_SHADER)
bbox_fs = compile_shader(BBOX_FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
glAttachShader(bbox_shader_program, bbox_vs)
glAttachShader(bbox_shader_program, bbox_fs)
glLinkProgram(bbox_shader_program)
glDeleteShader(bbox_vs)
glDeleteShader(bbox_fs)

#Uniform変数のロケーション取得
glUseProgram(shader_program)
view_loc = glGetUniformLocation(shader_program, "view")
proj_loc = glGetUniformLocation(shader_program, "projection")
light_pos_loc = glGetUniformLocation(shader_program, "lightPos")
view_pos_loc = glGetUniformLocation(shader_program, "viewPos")

glUseProgram(bbox_shader_program)
bbox_view_loc = glGetUniformLocation(bbox_shader_program, "view")
bbox_proj_loc = glGetUniformLocation(bbox_shader_program, "projection")
bbox_color_loc = glGetUniformLocation(bbox_shader_program, "lineColor")

#粒子の色設定
color_positive = np.array([0.4, 0.8, 1.0], dtype=np.float32)
color_negative = np.array([1.0, 0.5, 0.0], dtype=np.float32)

#初期パラメータ設定
gui_params = {
    'num_particles': 3000,
    'max_particles': 10000,
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
    'cylinder_radius': 2.0,
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
        
        #外枠の線を追加
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
bbox_VAO = glGenVertexArrays(1)  #頂点配列オブジェクト
bbox_VBO = glGenBuffers(1)       #頂点バッファオブジェクト

glBindVertexArray(bbox_VAO)             #頂点属性設定の保存の有効化
glBindBuffer(GL_ARRAY_BUFFER, bbox_VBO) #頂点データを入れるGPUメモリの有効化
glBufferData(GL_ARRAY_BUFFER, bbox_vertices.nbytes, bbox_vertices, GL_STATIC_DRAW) #頂点データをGPUに送信
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
glBindVertexArray(0)

#GPUメモリを完全にクリアする関数
def clear_gpu_arrays():
    # update_particles_optimized のGPUメモリをクリア
    if hasattr(update_particles_optimized, 'gpu_arrays'):  #GPU配列を持っているか確認
        for arr in update_particles_optimized.gpu_arrays.values():
            arr.free()     #すべての配列に対してGPUメモリを解放
        delattr(update_particles_optimized, 'gpu_arrays')  #gpu_arraysを削除
        delattr(update_particles_optimized, 'last_N')      #last_Nを削除
    
    # update_electric_field_poisson_optimized のGPUメモリをクリア
    if hasattr(update_electric_field_poisson_optimized, 'gpu_arrays'):
        for arr in update_electric_field_poisson_optimized.gpu_arrays.values():
            arr.free()
        delattr(update_electric_field_poisson_optimized, 'gpu_arrays')
        delattr(update_electric_field_poisson_optimized, 'last_N')
    
    #field_gpu_arraysも削除
    if hasattr(update_electric_field_poisson_optimized, 'field_gpu_arrays'):
        for arr in update_electric_field_poisson_optimized.field_gpu_arrays.values():
            arr.free()
        delattr(update_electric_field_poisson_optimized, 'field_gpu_arrays')


# === 電磁気シミュレーション処理 ===
BOX_SIZE = 4.2     #シミュレーション空間のサイズ
grid_divisions = 8 #各軸を8分割
cell_size = BOX_SIZE / grid_divisions #各格子のサイズ
grid_size = (grid_divisions, grid_divisions, grid_divisions)

potential_grid = np.zeros(grid_size, dtype=np.float32)     #電位
charge_density_grid = np.zeros(grid_size, dtype=np.float32)#電荷密度
E_field = np.zeros(grid_size + (3,), dtype=np.float32)     #電場(3成分)

print(f"ポアソン方程式電場グリッド設定:")
print(f"  グリッドサイズ: {grid_size}")
print(f"  セルサイズ: {cell_size}")
print(f"  ボックスサイズ: {BOX_SIZE}")

def create_poisson_matrix_optimized(nx, ny, nz, dx, electrode_params, geometry_mode=0, cylinder_params=None):
    #初期化
    n = nx * ny * nz
    rows = []
    cols = []
    data = []
    
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = k * ny * nx + j * nx + i #一次元インデックス変換
                if geometry_mode == 1:
                    #円柱領域外の処理
                    if not is_inside_cylinder(i, j, k, nx, ny, nz, cylinder_params['radius'], cylinder_params['height']):
                        rows.append(idx)
                        cols.append(idx)
                        data.append(1.0)
                        continue
                # 境界での処理
                is_boundary = (i == 0 or i == nx-1 or j == 0 or j == ny-1 or k == 0 or k == nz-1)
                
                if is_boundary:
                    # ディリクレ境界条件: φ = voltage
                    rows.append(idx)
                    cols.append(idx)
                    data.append(1.0)
                else:
                    # 内部点での5点または7点差分
                    # 中央点の係数
                    rows.append(idx)
                    cols.append(idx)
                    data.append(-6.0)
                    
                    # X方向隣接
                    if i > 0:
                        rows.append(idx)
                        cols.append(idx - 1)
                        data.append(1.0)
                    if i < nx - 1:
                        rows.append(idx)
                        cols.append(idx + 1)
                        data.append(1.0)
                    
                    # Y方向隣接
                    if j > 0:
                        rows.append(idx)
                        cols.append(idx - nx)
                        data.append(1.0)
                    if j < ny - 1:
                        rows.append(idx)
                        cols.append(idx + nx)
                        data.append(1.0)
                    
                    # Z方向隣接
                    if k > 0:
                        rows.append(idx)
                        cols.append(idx - nx * ny)
                        data.append(1.0)
                    if k < nz - 1:
                        rows.append(idx)
                        cols.append(idx + nx * ny)
                        data.append(1.0)
    
    # COO形式で構築してからCSRに変換
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    A = A.tocsr()
    
    return A

#境界条件を適用
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

class PoissonSolver:
    #初期化
    def __init__(self, nx, ny, nz, dx):
        #格子点数
        self.nx = nx
        self.ny = ny
        self.nz = nz
        #格子幅
        self.dx = dx 
        self.cached_matrix = None          #作成済みラプラシアン行列
        self.cached_electrode_params = None#行列を作ったときの電極設定
    #キャッシュされた行列を使用してポアソン方程式を解く
    def solve(self, charge_density, electrode_params, geometry_mode=0, cylinder_params=None):
        """
        キャッシュされた行列を使用してポアソン方程式を解く
        """
        # パラメータが変更された場合のみ行列を再構築
        current_params = (electrode_params, geometry_mode, cylinder_params)
        if (self.cached_matrix is None or 
            self.cached_params != current_params):
            
            print("Rebuilding Poisson matrix...")
            self.cached_matrix = create_poisson_matrix_optimized(
                self.nx, self.ny, self.nz, self.dx, electrode_params,
                geometry_mode, cylinder_params
            )
            self.cached_params = current_params
        
        # 右辺ベクトルの構築
        epsilon_0 = 1.0
        b = np.zeros(self.nx * self.ny * self.nz)
        
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    idx = k * self.ny * self.nx + j * self.nx + i
                    is_boundary = (i == 0 or i == self.nx-1 or 
                                 j == 0 or j == self.ny-1 or 
                                 k == 0 or k == self.nz-1)
                    
                    if is_boundary:
                        voltage = 0.0
                        if electrode_params['electrode_mode']:
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
                        b[idx] = voltage
                    else:
                        #ポアソン方程式の離散化で、右辺に電荷密度を変換してセット
                        b[idx] = -(charge_density[k, j, i] * self.dx * self.dx / epsilon_0)
        
        # 方程式を解く
        try:
            phi_flat = spsolve(self.cached_matrix, b)
            phi = phi_flat.reshape((self.nz, self.ny, self.nx))
            return phi.astype(np.float32)
        except Exception as e:
            print(f"ポアソン方程式の解法に失敗しました: {e}")
            return np.zeros((self.nz, self.ny, self.nx), dtype=np.float32)

#粒子生成クラス
class ChargedParticle:
    def __init__(self, position, velocity, charge, mass):
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.array(velocity, dtype=np.float32)
        self.charge = charge
        self.mass = mass

def create_particles(num_particles):
    particles = []
    
    spawn_range = 1.5
    
    #正電荷を生成
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

    #負電荷を生成
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

particles = create_particles(gui_params['num_particles'])
dt = gui_params['dt']

poisson_solver = PoissonSolver(grid_divisions, grid_divisions, grid_divisions, cell_size)

def update_electric_field_poisson_optimized(particles):
    """
    最適化されたポアソン方程式を用いて電場を更新
    """
    if not gui_params['use_poisson']:
        # 外部電場のみを使用
        E_field[:, :, :, 0] = gui_params['external_E_field_x']
        E_field[:, :, :, 1] = gui_params['external_E_field_y']
        E_field[:, :, :, 2] = gui_params['external_E_field_z']
        return

    # 電荷密度配列の初期化
    charge_density_grid.fill(0.0)
    
    N = len(particles)
    if N == 0:
        E_field[:,:,:,0] = gui_params['external_E_field_x']
        E_field[:,:,:,1] = gui_params['external_E_field_y']
        E_field[:,:,:,2] = gui_params['external_E_field_z']
        return
    
    vanish_flags = np.zeros(N, dtype=np.int32)
    positions_np = np.array([p.position for p in particles], dtype=np.float32)
    charges_np = np.array([p.charge for p in particles], dtype=np.float32)
    charge_density_flat = charge_density_grid.reshape(-1).astype(np.float32)
    
    # GPUメモリの確保と再利用（改良版電荷密度計算）
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
    
    gpu_arrays = update_electric_field_poisson_optimized.gpu_arrays
    
    # デバイスへのデータ転送
    cuda.memcpy_htod(gpu_arrays['pos'], positions_np)
    cuda.memcpy_htod(gpu_arrays['chg'], charges_np)
    cuda.memcpy_htod(gpu_arrays['rho'], charge_density_flat)
    cuda.memcpy_htod(gpu_arrays['vanish'], vanish_flags)
    
    block_size = 256
    grid_size_cuda = (N + block_size - 1) // block_size
    
    # 改良された電荷密度作成カーネルの起動
    charge_density_func(
        gpu_arrays['pos'], gpu_arrays['chg'], gpu_arrays['rho'],
        np.int32(grid_divisions), np.int32(grid_divisions), np.int32(grid_divisions),
        np.float32(cell_size), np.int32(N), np.float32(BOX_SIZE), gpu_arrays['vanish'],
        block=(block_size, 1, 1), grid=(grid_size_cuda, 1)
    )
    
    # 電荷密度を取り出す
    cuda.memcpy_dtoh(charge_density_flat, gpu_arrays['rho'])
    charge_density_grid[:] = charge_density_flat.reshape(grid_size)
    
    # ポアソン方程式を解いて電位を求める
    electrode_params = {
        'electrode_mode': gui_params['electrode_mode'],
        'electrode_voltage': gui_params['electrode_voltage'],
        'electrode_positions': gui_params['electrode_positions']
    }
    
    cylinder_params = {
        'radius': gui_params['cylinder_radius'],
        'height': gui_params['cylinder_height']
    } if gui_params['geometry_mode'] == 1 else None
    # キャッシュされたソルバーを使用
    potential_grid[:] = poisson_solver.solve(
        charge_density_grid, electrode_params, 
        gui_params['geometry_mode'], cylinder_params
    )
    
    # 電位から電場を計算
    potential_flat = potential_grid.reshape(-1).astype(np.float32)
    E_field_flat = E_field.reshape(-1, 3).astype(np.float32)
    
    if not hasattr(update_electric_field_poisson_optimized, 'field_gpu_arrays'):
        update_electric_field_poisson_optimized.field_gpu_arrays = {
            'pot': cuda.mem_alloc(potential_flat.nbytes),
            'E': cuda.mem_alloc(E_field_flat.nbytes)
        }
    
    field_gpu_arrays = update_electric_field_poisson_optimized.field_gpu_arrays
    
    cuda.memcpy_htod(field_gpu_arrays['pot'], potential_flat)
    cuda.memcpy_htod(field_gpu_arrays['E'], E_field_flat)
    
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
    
def update_particles_optimized(particles, E_field_array):
    if gui_params['simulation_paused']:
        return
        
    N = len(particles)
    if N == 0:
        return
    
    vanish_flags = np.zeros(N, dtype=np.int32)
    positions_np = np.array([tuple(p.position) for p in particles], dtype=np.float32)
    velocities_np = np.array([tuple(p.velocity) for p in particles], dtype=np.float32)
    charges_np = np.array([p.charge for p in particles], dtype=np.float32)
    masses_np = np.array([p.mass for p in particles], dtype=np.float32)

    E_field_flat = E_field_array.reshape(-1, 3).astype(np.float32)
    
    #境界条件フラグの設定
    warp_flags = np.array([
        gui_params['warp_x_negative'], gui_params['warp_x_positive'],
        gui_params['warp_y_negative'], gui_params['warp_y_positive'],
        gui_params['warp_z_negative'], gui_params['warp_z_positive']
    ], dtype=np.int32)

    cylinder_flags = np.array([
        gui_params['cylinder_top_warp'],
        gui_params['cylinder_bottom_warp'],
        gui_params['cylinder_wall_vanish']
    ], dtype=np.int32)

    magnetic_field = np.array([
        gui_params['magnetic_field_x'],
        gui_params['magnetic_field_y'], 
        gui_params['magnetic_field_z']
    ], dtype=np.float32)

    #GPUメモリの確保
    need_realloc = False
    if not hasattr(update_particles_optimized, 'gpu_arrays'):
        need_realloc = True
    elif not hasattr(update_particles_optimized, 'last_N'):
        need_realloc = True
    elif update_particles_optimized.last_N != N:
        need_realloc = True
        # 古いメモリを解放
        for arr in update_particles_optimized.gpu_arrays.values():
            arr.free()
    
    if need_realloc:
        print(f"GPUメモリを再確保: N={N}")
        update_particles_optimized.gpu_arrays = {
            'pos': cuda.mem_alloc(positions_np.nbytes),
            'vel': cuda.mem_alloc(velocities_np.nbytes),
            'chg': cuda.mem_alloc(charges_np.nbytes),
            'mas': cuda.mem_alloc(masses_np.nbytes),
            'ef': cuda.mem_alloc(E_field_flat.nbytes),
            'warp': cuda.mem_alloc(warp_flags.nbytes),
            'vanish': cuda.mem_alloc(vanish_flags.nbytes),
            'cyl_flags': cuda.mem_alloc(cylinder_flags.nbytes),
            'mag_field': cuda.mem_alloc(magnetic_field.nbytes)
        }
        update_particles_optimized.last_N = N
    
    gpu_arrays = update_particles_optimized.gpu_arrays
    
    #GPUへ転送
    cuda.memcpy_htod(gpu_arrays['pos'], positions_np)
    cuda.memcpy_htod(gpu_arrays['vel'], velocities_np)
    cuda.memcpy_htod(gpu_arrays['chg'], charges_np)
    cuda.memcpy_htod(gpu_arrays['mas'], masses_np)
    cuda.memcpy_htod(gpu_arrays['ef'], E_field_flat)
    cuda.memcpy_htod(gpu_arrays['warp'], warp_flags)
    cuda.memcpy_htod(gpu_arrays['vanish'], vanish_flags)
    cuda.memcpy_htod(gpu_arrays['cyl_flags'], cylinder_flags)
    cuda.memcpy_htod(gpu_arrays['mag_field'], magnetic_field)

    block_size = 256
    grid_size_cuda = (N + block_size - 1) // block_size

    # ポアソン方程式用のパラメータを使用
    
    #CUDAカーネルの呼び出し
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
        block=(block_size, 1, 1), grid=(grid_size_cuda, 1),
    )

    #計算結果を受け取る
    cuda.memcpy_dtoh(positions_np, gpu_arrays['pos'])
    cuda.memcpy_dtoh(velocities_np, gpu_arrays['vel'])
    cuda.memcpy_dtoh(vanish_flags, gpu_arrays['vanish'])

    alive_indices = [i for i, flag in enumerate(vanish_flags) if flag == 0]
    particles_changed = len(alive_indices) != len(particles)
    if particles_changed:
        #print(f"粒子消滅: {len(particles) - len(alive_indices)}個消滅, 残り{len(alive_indices)}個")
        # 生存している粒子のみを保持
        new_particles = []
        for i in alive_indices:
            particles[i].position = positions_np[i]
            particles[i].velocity = velocities_np[i]
            new_particles.append(particles[i])
        particles[:] = new_particles
        
        clear_gpu_arrays()
    else:
        # 粒子数に変化がない場合は位置と速度のみ更新
        for i, p in enumerate(particles):
            p.position = positions_np[i]
            p.velocity = velocities_np[i]

emitter_frame_counter = 0

def emit_particle():
    if not gui_params['enable_emitter']:
        return None
    
    if len(particles) >= gui_params['max_emitted_particles']:
        return None
    
    position = np.array(gui_params['emitter_position'], dtype=np.float32)
    direction = np.array(gui_params['emitter_direction'], dtype=np.float32)
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 0:
        direction = direction / direction_norm
    else:
        direction = np.array([0.0,1.0,0.0])
    
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

    if gui_params['emitter_charge_type'] == 0:
        charge = gui_params['positive_charge']
    elif gui_params['emitter_charge_type'] == 1:
        charge = gui_params['negative_charge']
    else:
        charge = gui_params['positive_charge'] if np.random.random() > 0.5 else gui_params['negative_charge']
    
    new_particle = ChargedParticle(position, velocity, charge, gui_params['particle_mass'])
    return new_particle

def update_particle_emitter():
    global emitter_frame_counter

    if not gui_params['enable_emitter']:
        return
    
    emitter_frame_counter += 1

    if emitter_frame_counter >= gui_params['emitter_rate']:
        new_particle = emit_particle()
        if new_particle is not None:
            particles.append(new_particle)
            clear_gpu_arrays()
        emitter_frame_counter = 0

# === 3Dモデルの読み込み ===
sphere_mesh = trimesh.load("./3D_models/sphere.obj", force='mesh')
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

# === メインループ ===
#初期設定
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
print("新機能: ポアソン方程式による電場計算")
print("===============")

frame_times = []
frame_count = 0
last_particle_count = gui_params['num_particles']
last_geometoty_mode = gui_params['geometry_mode']
last_electric_field_settings = None
field_update_counter = 0

#メイン
while not glfw.window_should_close(window):
    start_time = time.time()
    
    #入力処理
    glfw.poll_events()
    impl.process_inputs()
    
    #粒子数の変更検出と再生成
    if gui_params['num_particles'] != last_particle_count:
        particles = create_particles(gui_params['num_particles'])
        last_particle_count = gui_params['num_particles']
        if hasattr(update_particles_optimized, 'gpu_arrays'):
            delattr(update_particles_optimized, 'gpu_arrays')
        if hasattr(update_electric_field_poisson_optimized, 'gpu_arrays'):
            delattr(update_electric_field_poisson_optimized, 'gpu_arrays')
        if hasattr(update_electric_field_poisson_optimized, 'field_gpu_arrays'):
            delattr(update_electric_field_poisson_optimized, 'field_gpu_arrays')
    
    if gui_params['geometry_mode'] != last_geometoty_mode:
        bbox_vertices = create_bounding_box()
        glBindBuffer(GL_ARRAY_BUFFER, bbox_VBO)
        glBufferData(GL_ARRAY_BUFFER, bbox_vertices.nbytes, bbox_vertices, GL_STATIC_DRAW)
        particles = create_particles(gui_params['num_particles'])
        last_geometoty_mode = gui_params['geometry_mode']
        if hasattr(update_particles_optimized, 'gpu_arrays'):
            delattr(update_particles_optimized, 'gpu_arrays')
        if hasattr(update_electric_field_poisson_optimized, 'gpu_arrays'):
            delattr(update_electric_field_poisson_optimized, 'gpu_arrays')
        if hasattr(update_electric_field_poisson_optimized, 'field_gpu_arrays'):
            delattr(update_electric_field_poisson_optimized, 'field_gpu_arrays')
    
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

    update_particle_emitter() #粒子エミッターの更新

    #電場更新
    if field_needs_update:
        update_electric_field_poisson_optimized(particles)
    
    # 粒子の電荷と質量を更新
    for p in particles:
        if p.charge > 0:
            p.charge = gui_params['positive_charge']
        else:
            p.charge = gui_params['negative_charge']
        p.mass = gui_params['particle_mass']
    
    #粒子位置・速度の更新
    update_particles_optimized(particles, E_field)
    
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
    
    glClearColor(*current_bg_color)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    #カメラ制御
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

    #バウンディングボックス描画
    if gui_params['show_bbox'] and bg_color_flag == False:
        glUseProgram(bbox_shader_program)
        glUniformMatrix4fv(bbox_view_loc, 1, GL_FALSE, view.astype(np.float32))
        glUniformMatrix4fv(bbox_proj_loc, 1, GL_FALSE, projection.astype(np.float32))
        glUniform3fv(bbox_color_loc, 1, np.array([0, 0, 0], dtype=np.float32))
        
        glBindVertexArray(bbox_VAO)
        glDrawArrays(GL_LINES, 0, len(bbox_vertices))
    
    #粒子描画
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
    
    #ImGUIの設定
    imgui.begin("Poisson Equation Simulation Control", True)
    
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
            "External E_x", gui_params['external_E_field_x'], -50.0, 50.0
        )
        
        changed, gui_params['external_E_field_y'] = imgui.slider_float(
            "External E_y", gui_params['external_E_field_y'], -50.0, 50.0
        )
        
        changed, gui_params['external_E_field_z'] = imgui.slider_float(
            "External E_z", gui_params['external_E_field_z'], -50.0, 50.0
        )
        
        if imgui.button("Reset External Field"):
            gui_params['external_E_field_x'] = 0.0
            gui_params['external_E_field_y'] = 0.0
            gui_params['external_E_field_z'] = 0.0

    if imgui.collapsing_header("Magnetic Field Settings")[0]:
        changed, gui_params['enable_magnetic_field'] = imgui.checkbox(
            "Enable Magnetic Field", gui_params['enable_magnetic_field']
        )
        
        if gui_params['enable_magnetic_field']:
            imgui.text("Magnetic field vector (Tesla):")
            
            changed, gui_params['magnetic_field_x'] = imgui.slider_float(
                "B_x", gui_params['magnetic_field_x'], -10.0, 10.0
            )
            
            changed, gui_params['magnetic_field_y'] = imgui.slider_float(
                "B_y", gui_params['magnetic_field_y'], -10.0, 10.0
            )
            
            changed, gui_params['magnetic_field_z'] = imgui.slider_float(
                "B_z", gui_params['magnetic_field_z'], -10.0, 10.0
            )
            
            # サイクロトロン周波数の計算と表示
            B_magnitude = np.sqrt(gui_params['magnetic_field_x']**2 + 
                                gui_params['magnetic_field_y']**2 + 
                                gui_params['magnetic_field_z']**2)
            
            if B_magnitude > 0 and gui_params['particle_mass'] > 0:
                # サイクロトロン周波数 = qB/m (単位系は簡略化)
                cyclotron_freq = abs(gui_params['positive_charge']) * B_magnitude / gui_params['particle_mass']
                imgui.text(f"Cyclotron frequency: {cyclotron_freq:.3f}")
                
                if cyclotron_freq > 0:
                    period = 2 * np.pi / cyclotron_freq
                    imgui.text(f"Cyclotron period: {period:.3f}")
        
        imgui.separator()
        if imgui.button("Uniform Z-field"):
            gui_params['magnetic_field_x'] = 0.0
            gui_params['magnetic_field_y'] = 0.0
            gui_params['magnetic_field_z'] = 5.0
        
        imgui.same_line()
        if imgui.button("Uniform X-field"):
            gui_params['magnetic_field_x'] = 5.0
            gui_params['magnetic_field_y'] = 0.0
            gui_params['magnetic_field_z'] = 0.0
        
        if imgui.button("No magnetic field"):
            gui_params['magnetic_field_x'] = 0.0
            gui_params['magnetic_field_y'] = 0.0
            gui_params['magnetic_field_z'] = 0.0
    
    else:
        imgui.text("Magnetic field disabled")

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

    if imgui.collapsing_header("Electrode Settings")[0]:
        changed, gui_params['electrode_mode'] = imgui.checkbox(
            "Enable Electrodes", gui_params['electrode_mode']
        )
        
        if gui_params['electrode_mode']:
            changed, gui_params['electrode_voltage'] = imgui.slider_float(
                "Electrode Voltage", gui_params['electrode_voltage'], -50.0, 50.0
            )
            
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
            
            imgui.separator()
            if imgui.button("Parallel plate (Y-axis)"):
                gui_params['electrode_positions'] = {'x_neg': False, 'x_pos': False, 'y_neg': True, 'y_pos': True, 'z_neg': False, 'z_pos': False}
                gui_params['electrode_voltage'] = 20.0
            
            imgui.same_line()
            if imgui.button("Parallel plate (X-axis)"):
                gui_params['electrode_positions'] = {'x_neg': True, 'x_pos': True, 'y_neg': False, 'y_pos': False, 'z_neg': False, 'z_pos': False}
                gui_params['electrode_voltage'] = 20.0
            
            if imgui.button("Single electrode (Y-)"):
                gui_params['electrode_positions'] = {'x_neg': False, 'x_pos': False, 'y_neg': True, 'y_pos': False, 'z_neg': False, 'z_pos': False}
                gui_params['electrode_voltage'] = 30.0
        else:
            imgui.text("Electrodes disabled - only particle charges create fields")
    
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
    
    if imgui.collapsing_header("Boundary Conditions")[0]:
        imgui.text(f"Current boundary mode: {gui_params['boundary_mode_name']}")
        if imgui.button("Switch boundary mode"):
            boundary_mode = 1 - boundary_mode
            gui_params['boundary_mode_name'] = "Reflection" if boundary_mode == 1 else "Wrap"
        
        imgui.separator()
        imgui.text("Warp settings for each surface:")
        
        imgui.text("X-axis boundary:")
        changed, gui_params['warp_x_negative'] = imgui.checkbox(
            "Warp X-", gui_params['warp_x_negative']
        )
        changed, gui_params['warp_x_positive'] = imgui.checkbox(
            "Warp X+", gui_params['warp_x_positive']
        )
        
        imgui.text("Y-axis boundary:")
        changed, gui_params['warp_y_negative'] = imgui.checkbox(
            "Warp Y-", gui_params['warp_y_negative']
        )
        changed, gui_params['warp_y_positive'] = imgui.checkbox(
            "Warp Y+", gui_params['warp_y_positive']
        )
        
        imgui.text("Z-axis boundary:")
        changed, gui_params['warp_z_negative'] = imgui.checkbox(
            "Warp Z-", gui_params['warp_z_negative']
        )
        changed, gui_params['warp_z_positive'] = imgui.checkbox(
            "Warp Z+", gui_params['warp_z_positive']
        )

    if imgui.collapsing_header("Simulation Settings")[0]:
        changed, gui_params['dt'] = imgui.slider_float(
            "Time step", gui_params['dt'], 0.001, 0.05
        )

        changed, gui_params['coulomb_force_control'] = imgui.slider_float(
            "Coulomb force strength", gui_params['coulomb_force_control'], 0.1, 1
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
    
    if imgui.collapsing_header("Presets")[0]:
        if imgui.button("Basic Poisson test"):
            gui_params.update({
                'num_particles': 2000,
                'use_poisson': True,
                'external_E_field_x': 0.0,
                'external_E_field_y': 0.0,
                'external_E_field_z': 0.0,
                'positive_charge': 1.0,
                'negative_charge': -1.0,
                'particle_mass': 1.0,
                'dt': 0.005,
                'electrode_mode': False
            })
            particles = create_particles(gui_params['num_particles'])
            print("Basic Poisson test applied")
        
        if imgui.button("Parallel plate capacitor"):
            gui_params.update({
                'num_particles': 1500,
                'use_poisson': True,
                'electrode_mode': True,
                'electrode_voltage': 25.0,
                'electrode_positions': {'x_neg': False, 'x_pos': False, 'y_neg': True, 'y_pos': True, 'z_neg': False, 'z_pos': False},
                'external_E_field_x': 0.0,
                'external_E_field_y': 0.0,
                'external_E_field_z': 0.0,
                'positive_charge': 0.8,
                'negative_charge': -0.8,
                'dt': 0.008
            })
            particles = create_particles(gui_params['num_particles'])
            print("Parallel plate capacitor preset applied")
        
        if imgui.button("Single electrode test"):
            gui_params.update({
                'num_particles': 1000,
                'use_poisson': True,
                'electrode_mode': True,
                'electrode_voltage': 30.0,
                'electrode_positions': {'x_neg': False, 'x_pos': False, 'y_neg': True, 'y_pos': False, 'z_neg': False, 'z_pos': False},
                'external_E_field_x': 0.0,
                'external_E_field_y': 0.0,
                'external_E_field_z': 0.0,
                'positive_charge': 1.2,
                'negative_charge': -1.2,
                'dt': 0.006
            })
            particles = create_particles(gui_params['num_particles'])
            print("Single electrode test applied")
        
        if imgui.button("Classic Coulomb mode"):
            gui_params.update({
                'num_particles': 3000,
                'use_poisson': False,
                'external_E_field_x': 0.0,
                'external_E_field_y': 15.0,
                'external_E_field_z': 0.0,
                'electrode_mode': False,
                'positive_charge': 2.0,
                'negative_charge': -2.0,
                'dt': 0.01
            })
            particles = create_particles(gui_params['num_particles'])
            print("Classic Coulomb mode applied")
        
        if imgui.button("Cyclotron Motion Demo"):
            gui_params.update({
                'num_particles': 0,  # エミッターのみ使用
                'enable_emitter': True,
                'emitter_position': [0.0, 0.0, 0.0],
                'emitter_direction': [1.0, 0.0, 0.0],
                'emitter_speed': 8.0,
                'emitter_rate': 10,
                'emitter_charge_type': 0,  # 正電荷のみ
                'enable_magnetic_field': True,
                'magnetic_field_x': 0.0,
                'magnetic_field_y': 0.0,
                'magnetic_field_z': 5.0,
                'use_poisson': False,
                'external_E_field_x': 0.0,
                'external_E_field_y': 0.0,
                'external_E_field_z': 0.0,
                'positive_charge': 1.0,
                'particle_mass': 1.0,
                'dt': 0.005,
                'max_emitted_particles': 1000,
                'geometry_mode': 1,
                'cylinder_top_warp': False,
                'cylinder_bottom_warp': False,
                'cylinder_wall_vanish': True,
                'emitter_charge_type': 1,
                'coulomb_force_control': 0
            })
            particles.clear()
            clear_gpu_arrays()
            print("Cyclotron motion demo preset applied")
        
    imgui.end()
    
    imgui.begin("Operation Help", True)
    imgui.text("Camera controls:")
    imgui.text("  Arrow keys: Rotate camera")
    imgui.text("  W/S: Zoom in/out")
    imgui.text("  C: Toggle background color")
    imgui.text("  I: Toggle boundary conditions")
    imgui.separator()
    imgui.text("Physics modes:")
    imgui.text("1. Poisson Equation: Physically accurate")
    imgui.text("   - Solves ∇²φ = -ρ/ε₀ for electric potential")
    imgui.text("   - Calculates E = -∇φ from potential")
    imgui.text("   - Supports electrode boundary conditions")
    imgui.text("2. Classic: Simple Coulomb + external field")
    imgui.separator()
    imgui.text("Electrode examples:")
    imgui.text("- Parallel plate: Creates uniform field")
    imgui.text("- Single electrode: Creates radial field")
    imgui.text("- Try different voltage values!")
    imgui.end()
    
    imgui.render()
    impl.render(imgui.get_draw_data())
    
    glfw.swap_buffers(window)
    
    #フレームレートの計算
    end_time = time.time()
    frame_time = (end_time - start_time) * 1000
    frame_times.append(frame_time)
    frame_count += 1
    
    if frame_count % 60 == 0:
        avg_frame_time = np.mean(frame_times[-60:])
        fps = 1000.0 / avg_frame_time
        print(f"Average frame time: {avg_frame_time:.2f}ms, FPS: {fps:.1f}")
    
    target_frame_time = 1000.0 / 60.0
    if frame_time < target_frame_time:
        time.sleep((target_frame_time - frame_time) / 1000.0)

impl.shutdown()
glfw.terminate()