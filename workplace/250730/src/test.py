import glfw
from OpenGL.GL import *
import numpy as np
from pyrr import Matrix44
import time
import math

# === GLFW 初期化 ===
if not glfw.init():
    raise Exception("GLFWの初期化に失敗しました")

# カメラの角度
camera_theta = 3.14 / 4.0  # 水平方向の角度
camera_phi = np.radians(30.0)  # 垂直方向の角度（初期30度）

# モニター設定
monitors = glfw.get_monitors()
print(f"検出されたモニター数: {len(monitors)}")

# サブモニターを使用（存在する場合）
if len(monitors) > 1:
    monitor = monitors[2]  # サブモニター
    print("サブモニターを使用します")
else:
    monitor = monitors[0]  # プライマリモニター
    print("プライマリモニターを使用します（サブモニターが見つかりません）")

mode = glfw.get_video_mode(monitor)
print(f"モニター解像度: {mode.size.width}x{mode.size.height}")

# モニターの位置を取得
xpos, ypos = glfw.get_monitor_pos(monitor)
print(f"モニター位置: ({xpos}, {ypos})")

# フルスクリーン設定
glfw.window_hint(glfw.DECORATED, glfw.FALSE)  # ウィンドウ装飾を無効化
glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)  # リサイズ無効化

# 全画面ウィンドウを作成
window = glfw.create_window(mode.size.width, mode.size.height, "プロジェクションマッピング テストパターン", None, None)
if not window:
    glfw.terminate()
    raise Exception("ウィンドウの作成に失敗しました")

# ウィンドウをサブモニターに配置
glfw.set_window_pos(window, xpos, ypos)

glfw.make_context_current(window)

# === テストパターン用の変数 ===
pattern_mode = 0  # 0:グリッド, 1:チェッカーボード, 2:色分け, 3:アニメーション
animation_time = 0.0
grid_divisions = 9
show_edges = True
show_face_labels = True

# === OpenGL情報出力 ===
print("Renderer:", glGetString(GL_RENDERER).decode())
print("OpenGL Version:", glGetString(GL_VERSION).decode())

# === シェーダーソース ===
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

out vec3 vertexColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    vertexColor = color;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

void main()
{
    FragColor = vec4(vertexColor, 1.0);
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader).decode())
    return shader

# === シェーダー作成 ===
shader_program = glCreateProgram()
vs = compile_shader(VERTEX_SHADER, GL_VERTEX_SHADER)
fs = compile_shader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
glAttachShader(shader_program, vs)
glAttachShader(shader_program, fs)
glLinkProgram(shader_program)

glDeleteShader(vs)
glDeleteShader(fs)

glUseProgram(shader_program)
model_loc = glGetUniformLocation(shader_program, "model")
view_loc = glGetUniformLocation(shader_program, "view")
proj_loc = glGetUniformLocation(shader_program, "projection")

# === 立方体の基本構造 ===
cube_size = 4.3
half_size = cube_size / 2.0

# 立方体の6面の頂点とインデックス
faces = [
    # 前面 (Z+)
    [[-half_size, -half_size, half_size], [half_size, -half_size, half_size], 
     [half_size, half_size, half_size], [-half_size, half_size, half_size]],
    # 後面 (Z-)
    [[half_size, -half_size, -half_size], [-half_size, -half_size, -half_size], 
     [-half_size, half_size, -half_size], [half_size, half_size, -half_size]],
    # 上面 (Y+)
    [[-half_size, half_size, half_size], [half_size, half_size, half_size], 
     [half_size, half_size, -half_size], [-half_size, half_size, -half_size]],
    # 下面 (Y-)
    [[-half_size, -half_size, -half_size], [half_size, -half_size, -half_size], 
     [half_size, -half_size, half_size], [-half_size, -half_size, half_size]],
    # 右面 (X+)
    [[half_size, -half_size, half_size], [half_size, -half_size, -half_size], 
     [half_size, half_size, -half_size], [half_size, half_size, half_size]],
    # 左面 (X-)
    [[-half_size, -half_size, -half_size], [-half_size, -half_size, half_size], 
     [-half_size, half_size, half_size], [-half_size, half_size, -half_size]]
]

# 面の色（識別用）
face_colors = [
    [1.0, 0.0, 0.0],  # 前面: 赤
    [0.0, 1.0, 0.0],  # 後面: 緑
    [0.0, 0.0, 1.0],  # 上面: 青
    [1.0, 1.0, 0.0],  # 下面: 黄
    [1.0, 0.0, 1.0],  # 右面: マゼンタ
    [0.0, 1.0, 1.0]   # 左面: シアン
]

def create_grid_pattern(face_vertices, divisions, base_color):
    """グリッドパターンを生成"""
    vertices = []
    colors = []
    
    # 面の4つの頂点から補間してグリッドを生成
    for i in range(divisions + 1):
        for j in range(divisions + 1):
            u = i / divisions
            v = j / divisions
            
            # 双線形補間で頂点を計算
            p1 = np.array(face_vertices[0]) * (1-u) + np.array(face_vertices[1]) * u
            p2 = np.array(face_vertices[3]) * (1-u) + np.array(face_vertices[2]) * u
            point = p1 * (1-v) + p2 * v
            
            vertices.append(point)
            
            # チェッカーボードパターンの色
            if pattern_mode == 1:  # チェッカーボード
                if (i + j) % 2 == 0:
                    colors.append([1.0, 1.0, 1.0])  # 白
                else:
                    colors.append([0.0, 0.0, 0.0])  # 黒
            else:  # 通常のグリッド
                colors.append(base_color)
    
    # グリッドの線を生成
    grid_vertices = []
    grid_colors = []
    
    # 水平線
    for i in range(divisions + 1):
        for j in range(divisions):
            idx1 = i * (divisions + 1) + j
            idx2 = i * (divisions + 1) + j + 1
            grid_vertices.extend([vertices[idx1], vertices[idx2]])
            grid_colors.extend([base_color, base_color])
    
    # 垂直線
    for i in range(divisions):
        for j in range(divisions + 1):
            idx1 = i * (divisions + 1) + j
            idx2 = (i + 1) * (divisions + 1) + j
            grid_vertices.extend([vertices[idx1], vertices[idx2]])
            grid_colors.extend([base_color, base_color])
    
    if pattern_mode == 1:  # チェッカーボード - 面を描画
        face_vertices = []
        face_colors = []
        
        for i in range(divisions):
            for j in range(divisions):
                # 各セルの4つの頂点
                idx1 = i * (divisions + 1) + j
                idx2 = i * (divisions + 1) + j + 1
                idx3 = (i + 1) * (divisions + 1) + j + 1
                idx4 = (i + 1) * (divisions + 1) + j
                
                # 2つの三角形で四角形を作成
                face_vertices.extend([vertices[idx1], vertices[idx2], vertices[idx3]])
                face_vertices.extend([vertices[idx1], vertices[idx3], vertices[idx4]])
                
                # 色を決定
                if (i + j) % 2 == 0:
                    cell_color = [1.0, 1.0, 1.0]  # 白
                else:
                    cell_color = [0.0, 0.0, 0.0]  # 黒
                
                face_colors.extend([cell_color] * 6)
        
        return face_vertices, face_colors, grid_vertices, grid_colors
    
    return [], [], grid_vertices, grid_colors

def create_test_pattern():
    """テストパターンを生成"""
    all_vertices = []
    all_colors = []
    all_grid_vertices = []
    all_grid_colors = []
    
    for i, face_verts in enumerate(faces):
        base_color = face_colors[i] if pattern_mode == 2 else [1.0, 1.0, 1.0]
        
        # アニメーション効果
        if pattern_mode == 3:
            intensity = 0.5 + 0.5 * math.sin(animation_time + i * 0.5)
            base_color = [intensity, intensity, intensity]
        
        face_verts_list, face_colors_list, grid_verts, grid_colors_list = create_grid_pattern(
            face_verts, grid_divisions, base_color
        )
        
        all_vertices.extend(face_verts_list)
        all_colors.extend(face_colors_list)
        all_grid_vertices.extend(grid_verts)
        all_grid_colors.extend(grid_colors_list)
    
    return all_vertices, all_colors, all_grid_vertices, all_grid_colors

# === VAO/VBO 作成 ===
face_VAO = glGenVertexArrays(1)
face_VBO = glGenBuffers(1)
face_color_VBO = glGenBuffers(1)

grid_VAO = glGenVertexArrays(1)
grid_VBO = glGenBuffers(1)
grid_color_VBO = glGenBuffers(1)

def update_buffers():
    """バッファを更新"""
    face_vertices, face_colors, grid_vertices, grid_colors = create_test_pattern()
    
    # 面のバッファ更新
    if face_vertices:
        glBindVertexArray(face_VAO)
        
        face_vertices_np = np.array(face_vertices, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, face_VBO)
        glBufferData(GL_ARRAY_BUFFER, face_vertices_np.nbytes, face_vertices_np, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        face_colors_np = np.array(face_colors, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, face_color_VBO)
        glBufferData(GL_ARRAY_BUFFER, face_colors_np.nbytes, face_colors_np, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
    
    # グリッドのバッファ更新
    if grid_vertices:
        glBindVertexArray(grid_VAO)
        
        grid_vertices_np = np.array(grid_vertices, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, grid_VBO)
        glBufferData(GL_ARRAY_BUFFER, grid_vertices_np.nbytes, grid_vertices_np, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)
        
        grid_colors_np = np.array(grid_colors, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, grid_color_VBO)
        glBufferData(GL_ARRAY_BUFFER, grid_colors_np.nbytes, grid_colors_np, GL_DYNAMIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)
    
    return len(face_vertices), len(grid_vertices)

# === プロジェクション設定 ===
width, height = mode.size.width, mode.size.height  # モニターの実際の解像度を使用
projection = Matrix44.perspective_projection(45.0, width/height, 0.1, 100.0)
glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection.astype(np.float32))

glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# === カメラ設定 ===
radius = 10.0
angle_speed = 0.02
zoom_speed = 0.1
min_radius = 1.0
max_radius = 20.0

# === メイン描画ループ ===
print("=== プロジェクションマッピング テストパターン ===")
print(f"表示モニター: {mode.size.width}x{mode.size.height}")
print("操作方法:")
print("  矢印キー: カメラ回転")
print("  W/S: ズーム")
print("  1: グリッドパターン")
print("  2: チェッカーボードパターン")
print("  3: 色分けパターン")
print("  4: アニメーションパターン")
print("  G: グリッド表示ON/OFF")
print("  +/-: グリッド分割数変更")
print("  F: フルスクリーン切り替え")
print("  ESC: 終了")

# フルスクリーン状態の管理
is_fullscreen = True
windowed_width, windowed_height = 1200, 800

last_time = time.time()
face_count = 0
grid_count = 0

while not glfw.window_should_close(window):
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time
    animation_time += dt
    
    glfw.poll_events()
    
    # === キー入力処理 ===
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        break
    
    # フルスクリーン切り替え
    if glfw.get_key(window, glfw.KEY_F) == glfw.PRESS:
        if is_fullscreen:
            # ウィンドウモードに切り替え
            glfw.set_window_monitor(window, None, 100, 100, windowed_width, windowed_height, 0)
            is_fullscreen = False
        else:
            # フルスクリーンモードに切り替え
            glfw.set_window_monitor(window, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
            is_fullscreen = True
        
        # プロジェクション行列を更新
        current_width, current_height = glfw.get_window_size(window)
        projection = Matrix44.perspective_projection(45.0, current_width/current_height, 0.1, 100.0)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection.astype(np.float32))
        
        time.sleep(0.2)  # キーリピート防止
    
    # パターン切り替え
    if glfw.get_key(window, glfw.KEY_1) == glfw.PRESS:
        pattern_mode = 0
    elif glfw.get_key(window, glfw.KEY_2) == glfw.PRESS:
        pattern_mode = 1
    elif glfw.get_key(window, glfw.KEY_3) == glfw.PRESS:
        pattern_mode = 2
    elif glfw.get_key(window, glfw.KEY_4) == glfw.PRESS:
        pattern_mode = 3
    
    # グリッド表示切り替え
    if glfw.get_key(window, glfw.KEY_G) == glfw.PRESS:
        show_edges = not show_edges
        time.sleep(0.1)  # キーリピート防止
    
    # グリッド分割数変更
    if glfw.get_key(window, glfw.KEY_EQUAL) == glfw.PRESS:  # + キー
        grid_divisions = min(grid_divisions + 1, 20)
    elif glfw.get_key(window, glfw.KEY_MINUS) == glfw.PRESS:  # - キー
        grid_divisions = max(grid_divisions - 1, 2)
    
    # カメラ操作
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        camera_theta -= angle_speed
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        camera_theta += angle_speed
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        camera_phi += angle_speed
        camera_phi = min(camera_phi, np.radians(89.0))
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        camera_phi -= angle_speed
        camera_phi = max(camera_phi, np.radians(-89.0))
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        radius -= zoom_speed
        radius = max(radius, min_radius)
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        radius += zoom_speed
        radius = min(radius, max_radius)
    
    # === カメラ行列更新 ===
    camX = radius * np.cos(camera_phi) * np.sin(camera_theta)
    camY = radius * np.sin(camera_phi)
    camZ = radius * np.cos(camera_phi) * np.cos(camera_theta)
    eye = np.array([camX, camY, camZ])
    target = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    view = Matrix44.look_at(eye, target, up)
    
    # === 描画 ===
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0.1, 0.1, 0.1, 1.0)
    
    # バッファ更新
    face_count, grid_count = update_buffers()
    
    # マトリックス設定
    model = Matrix44.identity()
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model.astype(np.float32))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view.astype(np.float32))
    
    # チェッカーボードパターンの面を描画
    if pattern_mode == 1 and face_count > 0:
        glBindVertexArray(face_VAO)
        glDrawArrays(GL_TRIANGLES, 0, face_count)
    
    # グリッド線を描画
    if show_edges and grid_count > 0:
        glBindVertexArray(grid_VAO)
        glDrawArrays(GL_LINES, 0, grid_count)
    
    # 情報表示（コンソール）
    pattern_names = ["グリッド", "チェッカーボード", "色分け", "アニメーション"]
    if int(animation_time) % 2 == 0:  # 2秒おきに表示
        print(f"\rパターン: {pattern_names[pattern_mode]}, 分割数: {grid_divisions}, グリッド: {'ON' if show_edges else 'OFF'}", end="")
    
    glfw.swap_buffers(window)

glfw.terminate()
print("\nプログラムを終了しました。")