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
    monitor = monitors[1]  # サブモニター
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
window = glfw.create_window(mode.size.width, mode.size.height, "プロジェクションマッピング フォトニックフラクタルキューブ", None, None)
if not window:
    glfw.terminate()
    raise Exception("ウィンドウの作成に失敗しました")

# ウィンドウをサブモニターに配置
glfw.set_window_pos(window, xpos, ypos)

glfw.make_context_current(window)

# === テストパターン用の変数 ===
pattern_mode = 0  # 0:グリッド, 1:チェッカーボード, 2:色分け, 3:アニメーション, 4:フォトニックフラクタル
animation_time = 0.0
grid_divisions = 9
show_edges = True
show_face_labels = True

# フォトニックフラクタル用の変数
fractal_levels = 3  # フラクタルの階層数
fractal_intensity = 1.0  # 光の強度
fractal_phase = 0.0  # 位相
photonic_frequency = 2.0  # フォトニック周波数

# === OpenGL情報出力 ===
print("Renderer:", glGetString(GL_RENDERER).decode())
print("OpenGL Version:", glGetString(GL_VERSION).decode())

# === シェーダーソース ===
VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

out vec3 vertexColor;
out vec3 worldPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    worldPos = (model * vec4(position, 1.0)).xyz;
    gl_Position = projection * view * model * vec4(position, 1.0);
    vertexColor = color;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 vertexColor;
in vec3 worldPos;
out vec4 FragColor;

uniform float time;
uniform float fractalIntensity;
uniform float photonicFreq;
uniform int patternMode;

// フォトニック結晶の干渉パターンを計算
vec3 calculatePhotonicPattern(vec3 pos, float t) {
    // 3D格子による干渉パターン
    vec3 latticePos = pos * photonicFreq;
    
    // 各軸方向の波
    float waveX = sin(latticePos.x + t);
    float waveY = sin(latticePos.y + t * 1.1);
    float waveZ = sin(latticePos.z + t * 0.9);
    
    // 対角方向の波（より複雑な干渉）
    float waveDiag1 = sin((latticePos.x + latticePos.y) * 0.707 + t * 0.8);
    float waveDiag2 = sin((latticePos.y + latticePos.z) * 0.707 + t * 1.2);
    float waveDiag3 = sin((latticePos.z + latticePos.x) * 0.707 + t * 0.7);
    
    // 干渉パターンの合成
    float interference = (waveX + waveY + waveZ + waveDiag1 + waveDiag2 + waveDiag3) / 6.0;
    
    // フラクタルノイズの追加
    vec3 fractalPos = pos * 4.0;
    float fractal = 0.0;
    float amplitude = 1.0;
    for(int i = 0; i < 4; i++) {
        fractal += sin(fractalPos.x * amplitude + t) * 
                   sin(fractalPos.y * amplitude + t * 1.1) * 
                   sin(fractalPos.z * amplitude + t * 0.9) / amplitude;
        fractalPos *= 2.0;
        amplitude *= 0.5;
    }
    
    // 最終的な強度計算
    float intensity = (interference + fractal * 0.3) * fractalIntensity;
    intensity = abs(intensity);
    
    // スペクトル色の計算（干渉による色分散をシミュレート）
    float hue = fract(intensity * 2.0 + t * 0.1);
    
    // HSVからRGBへの変換（簡易版）
    vec3 color;
    if(hue < 0.166) {
        color = mix(vec3(1,0,1), vec3(0,0,1), hue * 6.0);  // マゼンタ→青
    } else if(hue < 0.333) {
        color = mix(vec3(0,0,1), vec3(0,1,1), (hue - 0.166) * 6.0);  // 青→シアン
    } else if(hue < 0.5) {
        color = mix(vec3(0,1,1), vec3(0,1,0), (hue - 0.333) * 6.0);  // シアン→緑
    } else if(hue < 0.666) {
        color = mix(vec3(0,1,0), vec3(1,1,0), (hue - 0.5) * 6.0);  // 緑→黄
    } else if(hue < 0.833) {
        color = mix(vec3(1,1,0), vec3(1,0,0), (hue - 0.666) * 6.0);  // 黄→赤
    } else {
        color = mix(vec3(1,0,0), vec3(1,0,1), (hue - 0.833) * 6.0);  // 赤→マゼンタ
    }
    
    return color * intensity;
}

void main()
{
    vec3 finalColor = vertexColor;
    
    if(patternMode == 4) {  // フォトニックフラクタルパターン
        vec3 photonicColor = calculatePhotonicPattern(worldPos, time);
        // 元の色と合成
        finalColor = mix(vertexColor, photonicColor, 0.8);
        
        // 明度の調整
        float brightness = dot(finalColor, vec3(0.299, 0.587, 0.114));
        if(brightness > 1.0) {
            finalColor = finalColor / brightness;
        }
    }
    
    FragColor = vec4(finalColor, 1.0);
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

# フラグメントシェーダー用のユニフォーム
time_loc = glGetUniformLocation(shader_program, "time")
fractal_intensity_loc = glGetUniformLocation(shader_program, "fractalIntensity")
photonic_freq_loc = glGetUniformLocation(shader_program, "photonicFreq")
pattern_mode_loc = glGetUniformLocation(shader_program, "patternMode")

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

def create_fractal_structure(face_vertices, level, max_level):
    """フラクタル構造を生成（メンガーのスポンジ風）"""
    if level >= max_level:
        return face_vertices, [1.0, 1.0, 1.0]
    
    vertices = []
    
    # 面を9分割（3x3）
    subdivisions = 3
    for i in range(subdivisions):
        for j in range(subdivisions):
            # 中央のセルは空洞にする（フラクタルホール）
            if i == 1 and j == 1:
                continue
                
            u_start = i / subdivisions
            u_end = (i + 1) / subdivisions
            v_start = j / subdivisions  
            v_end = (j + 1) / subdivisions
            
            # サブ面の4つの頂点を計算
            p1 = np.array(face_vertices[0]) * (1-u_start) * (1-v_start) + \
                 np.array(face_vertices[1]) * u_start * (1-v_start) + \
                 np.array(face_vertices[2]) * u_start * v_start + \
                 np.array(face_vertices[3]) * (1-u_start) * v_start
                 
            p2 = np.array(face_vertices[0]) * (1-u_end) * (1-v_start) + \
                 np.array(face_vertices[1]) * u_end * (1-v_start) + \
                 np.array(face_vertices[2]) * u_end * v_start + \
                 np.array(face_vertices[3]) * (1-u_end) * v_start
                 
            p3 = np.array(face_vertices[0]) * (1-u_end) * (1-v_end) + \
                 np.array(face_vertices[1]) * u_end * (1-v_end) + \
                 np.array(face_vertices[2]) * u_end * v_end + \
                 np.array(face_vertices[3]) * (1-u_end) * v_end
                 
            p4 = np.array(face_vertices[0]) * (1-u_start) * (1-v_end) + \
                 np.array(face_vertices[1]) * u_start * (1-v_end) + \
                 np.array(face_vertices[2]) * u_start * v_end + \
                 np.array(face_vertices[3]) * (1-u_start) * v_end
            
            # 2つの三角形で四角形を構成
            vertices.extend([p1, p2, p3])
            vertices.extend([p1, p3, p4])
    
    return vertices

def create_photonic_fractal_pattern(face_vertices, divisions, base_color):
    """フォトニックフラクタルパターンを生成"""
    vertices = []
    colors = []
    grid_vertices = []
    grid_colors = []
    
    # フラクタル構造を生成
    fractal_verts = create_fractal_structure(face_vertices, 0, fractal_levels)
    
    # 高密度のメッシュを生成（フォトニック結晶の詳細表現）
    high_res_divisions = divisions * 2  # より細かい分割
    
    # 面の4つの頂点から補間して詳細メッシュを生成
    for i in range(high_res_divisions + 1):
        for j in range(high_res_divisions + 1):
            u = i / high_res_divisions
            v = j / high_res_divisions
            
            # 双線形補間で頂点を計算
            p1 = np.array(face_vertices[0]) * (1-u) + np.array(face_vertices[1]) * u
            p2 = np.array(face_vertices[3]) * (1-u) + np.array(face_vertices[2]) * u
            point = p1 * (1-v) + p2 * v
            
            vertices.append(point)
            
            # フォトニック結晶の基本パターン色
            phase = animation_time * photonic_frequency
            distance_from_center = np.linalg.norm(point)
            
            # 干渉パターンによる色の計算
            interference = (
                np.sin(point[0] * photonic_frequency + phase) +
                np.sin(point[1] * photonic_frequency + phase * 1.1) +
                np.sin(point[2] * photonic_frequency + phase * 0.9)
            ) / 3.0
            
            intensity = abs(interference) * fractal_intensity
            
            # スペクトル色の生成
            hue = (intensity + phase * 0.1) % 1.0
            if hue < 0.33:
                color = [1.0, hue * 3, 0.0]  # 赤→黄
            elif hue < 0.66:
                color = [1.0 - (hue - 0.33) * 3, 1.0, (hue - 0.33) * 3]  # 黄→緑→シアン
            else:
                color = [0.0, 1.0 - (hue - 0.66) * 3, 1.0]  # シアン→青
            
            # 強度による明度調整
            color = [c * intensity for c in color]
            colors.append(color)
    
    # フォトニック結晶面を生成
    face_vertices_list = []
    face_colors_list = []
    
    for i in range(high_res_divisions):
        for j in range(high_res_divisions):
            idx1 = i * (high_res_divisions + 1) + j
            idx2 = i * (high_res_divisions + 1) + j + 1
            idx3 = (i + 1) * (high_res_divisions + 1) + j + 1
            idx4 = (i + 1) * (high_res_divisions + 1) + j
            
            # 2つの三角形で四角形を作成
            face_vertices_list.extend([vertices[idx1], vertices[idx2], vertices[idx3]])
            face_vertices_list.extend([vertices[idx1], vertices[idx3], vertices[idx4]])
            
            # 各頂点の色
            face_colors_list.extend([colors[idx1], colors[idx2], colors[idx3]])
            face_colors_list.extend([colors[idx1], colors[idx3], colors[idx4]])
    
    # 格子線の生成（オプション）
    if show_edges:
        # 水平線
        for i in range(high_res_divisions + 1):
            for j in range(0, high_res_divisions, 2):  # 間引いて表示
                idx1 = i * (high_res_divisions + 1) + j
                idx2 = i * (high_res_divisions + 1) + j + 1
                if j + 1 <= high_res_divisions:
                    grid_vertices.extend([vertices[idx1], vertices[idx2]])
                    grid_colors.extend([base_color, base_color])
        
        # 垂直線
        for i in range(0, high_res_divisions, 2):  # 間引いて表示
            for j in range(high_res_divisions + 1):
                idx1 = i * (high_res_divisions + 1) + j
                idx2 = (i + 1) * (high_res_divisions + 1) + j
                if i + 1 <= high_res_divisions:
                    grid_vertices.extend([vertices[idx1], vertices[idx2]])
                    grid_colors.extend([base_color, base_color])
    
    return face_vertices_list, face_colors_list, grid_vertices, grid_colors

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
        face_vertices_list = []
        face_colors_list = []
        
        for i in range(divisions):
            for j in range(divisions):
                # 各セルの4つの頂点
                idx1 = i * (divisions + 1) + j
                idx2 = i * (divisions + 1) + j + 1
                idx3 = (i + 1) * (divisions + 1) + j + 1
                idx4 = (i + 1) * (divisions + 1) + j
                
                # 2つの三角形で四角形を作成
                face_vertices_list.extend([vertices[idx1], vertices[idx2], vertices[idx3]])
                face_vertices_list.extend([vertices[idx1], vertices[idx3], vertices[idx4]])
                
                # 色を決定
                if (i + j) % 2 == 0:
                    cell_color = [1.0, 1.0, 1.0]  # 白
                else:
                    cell_color = [0.0, 0.0, 0.0]  # 黒
                
                face_colors_list.extend([cell_color] * 6)
        
        return face_vertices_list, face_colors_list, grid_vertices, grid_colors
    
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
        
        # フォトニックフラクタルパターン
        if pattern_mode == 4:
            face_verts_list, face_colors_list, grid_verts, grid_colors_list = create_photonic_fractal_pattern(
                face_verts, grid_divisions, base_color
            )
        else:
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
print("=== プロジェクションマッピング フォトニックフラクタルキューブ ===")
print(f"表示モニター: {mode.size.width}x{mode.size.height}")
print("操作方法:")
print("  矢印キー: カメラ回転")
print("  W/S: ズーム")
print("  1: グリッドパターン")
print("  2: チェッカーボードパターン")
print("  3: 色分けパターン")
print("  4: アニメーションパターン")
print("  5: フォトニックフラクタルパターン ← NEW!")
print("  G: グリッド表示ON/OFF")
print("  +/-: グリッド分割数変更")
print("  Q/E: フラクタル階層数変更")
print("  A/D: フォトニック周波数変更")
print("  Z/X: フラクタル強度変更")
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
    elif glfw.get_key(window, glfw.KEY_5) == glfw.PRESS:
        pattern_mode = 4  # フォトニックフラクタルパターン
    
    # グリッド表示切り替え
    if glfw.get_key(window, glfw.KEY_G) == glfw.PRESS:
        show_edges = not show_edges
        time.sleep(0.1)  # キーリピート防止
    
    # グリッド分割数変更
    if glfw.get_key(window, glfw.KEY_EQUAL) == glfw.PRESS:  # + キー
        grid_divisions = min(grid_divisions + 1, 20)
    elif glfw.get_key(window, glfw.KEY_MINUS) == glfw.PRESS:  # - キー
        grid_divisions = max(grid_divisions - 1, 2)
    
    # フォトニックフラクタル専用コントロール
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        fractal_levels = max(fractal_levels - 1, 1)
        time.sleep(0.1)
    elif glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        fractal_levels = min(fractal_levels + 1, 6)
        time.sleep(0.1)
    
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        photonic_frequency = max(photonic_frequency - 0.1, 0.5)
    elif glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        photonic_frequency = min(photonic_frequency + 0.1, 10.0)
    
    if glfw.get_key(window, glfw.KEY_Z) == glfw.PRESS:
        fractal_intensity = max(fractal_intensity - 0.1, 0.1)
    elif glfw.get_key(window, glfw.KEY_X) == glfw.PRESS:
        fractal_intensity = min(fractal_intensity + 0.1, 3.0)
    
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
    
    # === シェーダーユニフォーム更新 ===
    glUniform1f(time_loc, animation_time)
    glUniform1f(fractal_intensity_loc, fractal_intensity)
    glUniform1f(photonic_freq_loc, photonic_frequency)
    glUniform1i(pattern_mode_loc, pattern_mode)
    
    # === 描画 ===
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glClearColor(0, 0, 0, 1.0)  # 暗い青背景（フォトニック効果を強調）
    
    # バッファ更新
    face_count, grid_count = update_buffers()
    
    # マトリックス設定
    model = Matrix44.identity()
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, model.astype(np.float32))
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view.astype(np.float32))
    
    # 面を描画（チェッカーボードまたはフォトニックフラクタル）
    if (pattern_mode == 1 or pattern_mode == 4) and face_count > 0:
        glBindVertexArray(face_VAO)
        glDrawArrays(GL_TRIANGLES, 0, face_count)
    
    # グリッド線を描画
    if show_edges and grid_count > 0:
        glBindVertexArray(grid_VAO)
        glDrawArrays(GL_LINES, 0, grid_count)
    
    # 情報表示（コンソール）
    pattern_names = ["グリッド", "チェッカーボード", "色分け", "アニメーション", "フォトニックフラクタル"]
    if int(animation_time) % 2 == 0:  # 2秒おきに表示
        status = f"\rパターン: {pattern_names[pattern_mode]}, 分割: {grid_divisions}"
        if pattern_mode == 4:  # フォトニックフラクタル情報
            status += f", フラクタル階層: {fractal_levels}, 周波数: {photonic_frequency:.1f}, 強度: {fractal_intensity:.1f}"
        status += f", グリッド: {'ON' if show_edges else 'OFF'}"
        print(status, end="")
    
    glfw.swap_buffers(window)

glfw.terminate()
print("\nプログラムを終了しました。")