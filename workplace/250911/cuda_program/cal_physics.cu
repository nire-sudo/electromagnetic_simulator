// 電荷密度を計算するカーネル
extern "C" __global__ void compute_charge_density(
    float3 *positions, float *charges, float *charge_density,
    int grid_x, int grid_y, int grid_z, float cell_size, int N, float box_size,
    int *vanish_flags
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    if (vanish_flags[i] == 1) return;

    float3 pos = positions[i];
    float charge = charges[i];
    float half_box = box_size / 2.0f;

    // 正規化座標 [0,1]
    float norm_x = (pos.x + half_box) / box_size;
    float norm_y = (pos.y + half_box) / box_size;
    float norm_z = (pos.z + half_box) / box_size;

    // セルインデックス（floatで計算）
    float gx_f = norm_x * grid_x - 0.5f;
    float gy_f = norm_y * grid_y - 0.5f;
    float gz_f = norm_z * grid_z - 0.5f;

    int gx0 = floorf(gx_f);
    int gy0 = floorf(gy_f);
    int gz0 = floorf(gz_f);

    int gx1 = (gx0 + 1) % grid_x;
    int gy1 = (gy0 + 1) % grid_y;
    int gz1 = (gz0 + 1) % grid_z;

    float wx = gx_f - gx0;
    float wy = gy_f - gy0;
    float wz = gz_f - gz0;

    float cell_volume = cell_size * cell_size * cell_size;

    // 8セルに分配
    int idx;
    float w;

    // (gx0, gy0, gz0)
    idx = gz0 * grid_y * grid_x + gy0 * grid_x + gx0;
    w = (1 - wx) * (1 - wy) * (1 - wz);
    atomicAdd(&charge_density[idx], charge * w / cell_volume);

    // (gx1, gy0, gz0)
    idx = gz0 * grid_y * grid_x + gy0 * grid_x + gx1;
    w = (wx) * (1 - wy) * (1 - wz);
    atomicAdd(&charge_density[idx], charge * w / cell_volume);

    // (gx0, gy1, gz0)
    idx = gz0 * grid_y * grid_x + gy1 * grid_x + gx0;
    w = (1 - wx) * (wy) * (1 - wz);
    atomicAdd(&charge_density[idx], charge * w / cell_volume);

    // (gx1, gy1, gz0)
    idx = gz0 * grid_y * grid_x + gy1 * grid_x + gx1;
    w = (wx) * (wy) * (1 - wz);
    atomicAdd(&charge_density[idx], charge * w / cell_volume);

    // (gx0, gy0, gz1)
    idx = gz1 * grid_y * grid_x + gy0 * grid_x + gx0;
    w = (1 - wx) * (1 - wy) * (wz);
    atomicAdd(&charge_density[idx], charge * w / cell_volume);

    // (gx1, gy0, gz1)
    idx = gz1 * grid_y * grid_x + gy0 * grid_x + gx1;
    w = (wx) * (1 - wy) * (wz);
    atomicAdd(&charge_density[idx], charge * w / cell_volume);

    // (gx0, gy1, gz1)
    idx = gz1 * grid_y * grid_x + gy1 * grid_x + gx0;
    w = (1 - wx) * (wy) * (wz);
    atomicAdd(&charge_density[idx], charge * w / cell_volume);

    // (gx1, gy1, gz1)
    idx = gz1 * grid_y * grid_x + gy1 * grid_x + gx1;
    w = (wx) * (wy) * (wz);
    atomicAdd(&charge_density[idx], charge * w / cell_volume);
}

// 電位から電場を計算するカーネル
extern "C" __global__ void compute_electric_field_from_potential(
    float *potential, float3 *E_field, 
    int grid_x, int grid_y, int grid_z, float cell_size,
    float external_Ex, float external_Ey, float external_Ez
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_cells = grid_x * grid_y * grid_z;
    
    if (idx >= total_cells) return;

    // 1次元インデックスから3次元座標を復元
    int gz = idx / (grid_y * grid_x);
    int gy = (idx % (grid_y * grid_x)) / grid_x;
    int gx = idx % grid_x;

    float3 E = make_float3(0.0f, 0.0f, 0.0f);

    // 中央差分で電場を計算: E = -∇φ
    // X方向
    if (gx > 0 && gx < grid_x - 1) {
        int idx_xp = gz * grid_y * grid_x + gy * grid_x + (gx + 1);
        int idx_xm = gz * grid_y * grid_x + gy * grid_x + (gx - 1);
        E.x = -(potential[idx_xp] - potential[idx_xm]) / (2.0f * cell_size);
    } else if (gx == 0) {
        int idx_xp = gz * grid_y * grid_x + gy * grid_x + 1;
        E.x = -(potential[idx_xp] - potential[idx]) / cell_size;
    } else if (gx == grid_x - 1) {
        int idx_xm = gz * grid_y * grid_x + gy * grid_x + (gx - 1);
        E.x = -(potential[idx] - potential[idx_xm]) / cell_size;
    }

    // Y方向
    if (gy > 0 && gy < grid_y - 1) {
        int idx_yp = gz * grid_y * grid_x + (gy + 1) * grid_x + gx;
        int idx_ym = gz * grid_y * grid_x + (gy - 1) * grid_x + gx;
        E.y = -(potential[idx_yp] - potential[idx_ym]) / (2.0f * cell_size);
    } else if (gy == 0) {
        int idx_yp = gz * grid_y * grid_x + 1 * grid_x + gx;
        E.y = -(potential[idx_yp] - potential[idx]) / cell_size;
    } else if (gy == grid_y - 1) {
        int idx_ym = gz * grid_y * grid_x + (gy - 1) * grid_x + gx;
        E.y = -(potential[idx] - potential[idx_ym]) / cell_size;
    }

    // Z方向
    if (gz > 0 && gz < grid_z - 1) {
        int idx_zp = (gz + 1) * grid_y * grid_x + gy * grid_x + gx;
        int idx_zm = (gz - 1) * grid_y * grid_x + gy * grid_x + gx;
        E.z = -(potential[idx_zp] - potential[idx_zm]) / (2.0f * cell_size);
    } else if (gz == 0) {
        int idx_zp = 1 * grid_y * grid_x + gy * grid_x + gx;
        E.z = -(potential[idx_zp] - potential[idx]) / cell_size;
    } else if (gz == grid_z - 1) {
        int idx_zm = (gz - 1) * grid_y * grid_x + gy * grid_x + gx;
        E.z = -(potential[idx] - potential[idx_zm]) / cell_size;
    }

    // 外部電場を追加
    E.x += external_Ex;
    E.y += external_Ey;
    E.z += external_Ez;

    E_field[idx] = E;
}

// 元の粒子更新カーネル（修正版）
extern "C" __global__ void update_particles_cuda(
    float3 *positions, float3 *velocities, float *charges, float *masses,
    float3 *E_field, int E_field_x_size, int E_field_y_size, int E_field_z_size,
    float cell_size, int N, float k, float min_dist, float dt, float box_size, 
    int boundary_mode, int *warp_flags, float cylinder_radius, float cylinder_height,
    int geometry_mode, int *vanish_flags, int *cylinder_flags,
    float3 *magnetic_field, int enable_magnetic_field,
    float *instance_vbo,  // OpenGL VBOへの直接ポインタ
    float scale_factor,
    float3 positive_color,
    float3 negative_color
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    if (vanish_flags[i] == 1) return;

    float3 pi = positions[i];
    float3 vi = velocities[i];
    float qi = charges[i];
    float mi = masses[i];

    // クーロン力計算（最適化版）
    float3 fi = make_float3(0.0f, 0.0f, 0.0f);
    
    // ポアソン方程式使用時はクーロン力を抑制（重複を避ける）
    if (k > 0.0f) {
        for (int j = 0; j < N; ++j) {
            if (i == j || vanish_flags[j] == 1) continue;
            float3 pj = positions[j];
            float qj = charges[j];

            float3 r = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
            float dist = sqrtf(r.x * r.x + r.y * r.y + r.z * r.z);
            if (dist < min_dist) dist = min_dist;

            float factor = k * qi * qj / (dist * dist * dist);
            fi.x += factor * r.x;
            fi.y += factor * r.y;
            fi.z += factor * r.z;
        }
    }

    // 電場インデックス計算を修正
    // ボックス座標系(-box_size/2 ~ +box_size/2)を電場グリッド座標系(0 ~ grid_size-1)にマッピング
    float half_box = box_size / 2.0f;
    
    // 粒子位置を正規化 (0.0 ~ 1.0)
    float norm_x = (pi.x + half_box) / box_size;
    float norm_y = (pi.y + half_box) / box_size;
    float norm_z = (pi.z + half_box) / box_size;
    
    // グリッドインデックスに変換（範囲チェック付き）
    int3 idx;
    idx.x = min(max((int)(norm_x * E_field_x_size), 0), E_field_x_size - 1);
    idx.y = min(max((int)(norm_y * E_field_y_size), 0), E_field_y_size - 1);
    idx.z = min(max((int)(norm_z * E_field_z_size), 0), E_field_z_size - 1);
    
    // 1次元インデックスに変換
    int field_index = idx.z * E_field_y_size * E_field_x_size + idx.y * E_field_x_size + idx.x;
    
    // 安全性チェック
    int max_index = E_field_x_size * E_field_y_size * E_field_z_size - 1;
    if (field_index < 0 || field_index > max_index) {
        field_index = 0;  // デフォルト値
    }
    
    float3 E = E_field[field_index];

    // 電場による力
    fi.x += qi * E.x;
    fi.y += qi * E.y;
    fi.z += qi * E.z;

    // 磁場による力（ローレンツ力)
    if (enable_magnetic_field) {
        float3 B = magnetic_field[0];  // 一様磁場として扱う
        float B_magnitude = sqrtf(B.x * B.x + B.y * B.y + B.z * B.z);

        if(B_magnitude > 1e-10){
            float3 a_E = make_float3(fi.x / mi, fi.y / mi, fi.z / mi);
            float3 v_minus = make_float3(
                vi.x + 0.5f * a_E.x * dt,
                vi.y + 0.5f * a_E.y * dt,
                vi.z + 0.5f * a_E.z * dt
            );

            float cyclotron_freq = qi * B_magnitude / mi;
            float tan_half_angle = 0.5f * cyclotron_freq * dt;

            float3 t = make_float3(
                B.x * tan_half_angle / B_magnitude,
                B.y * tan_half_angle / B_magnitude,
                B.z * tan_half_angle / B_magnitude
            );

            float3 v_cross_t = make_float3(
                v_minus.y * t.z - v_minus.z * t.y,
                v_minus.z * t.x - v_minus.x * t.z,
                v_minus.x * t.y - v_minus.y * t.x
            );

            float3 v_prime = make_float3(
                v_minus.x + v_cross_t.x,
                v_minus.y + v_cross_t.y,
                v_minus.z + v_cross_t.z
            );

            float t_squared = t.x * t.x + t.y * t.y + t.z * t.z;
            float s_factor = 2.0f / (1.0f + t_squared);
            float3 s = make_float3(
                t.x * s_factor,
                t.y * s_factor,
                t.z * s_factor
            );

            float3 v_prime_cross_s = make_float3(
                v_prime.y * s.z - v_prime.z * s.y,
                v_prime.z * s.x - v_prime.x * s.z,
                v_prime.x * s.y - v_prime.y * s.x
            );
            
            float3 v_plus = make_float3(
                v_minus.x + v_prime_cross_s.x,
                v_minus.y + v_prime_cross_s.y,
                v_minus.z + v_prime_cross_s.z
            );

            vi = make_float3(
                v_plus.x + 0.5f * a_E.x * dt,
                v_plus.y + 0.5f * a_E.y * dt,
                v_plus.z + 0.5f * a_E.z * dt
            );
        } else {
            float3 a = make_float3(fi.x / mi, fi.y / mi, fi.z / mi);
            vi.x += a.x * dt;
            vi.y += a.y * dt;
            vi.z += a.z * dt;
        }
    } else {
        float3 a = make_float3(fi.x / mi, fi.y / mi, fi.z / mi);
        vi.x += a.x * dt;
        vi.y += a.y * dt;
        vi.z += a.z * dt;
    }

    // 加速度 a = F / m
    float3 a = make_float3(fi.x / mi, fi.y / mi, fi.z / mi);

    // 速度更新（制限付き）
    float max_speed =50.0f;  // 最大速度を上げる（ポアソン方程式で高精度なため）
    float speed = sqrtf(vi.x * vi.x + vi.y * vi.y + vi.z * vi.z);
    if (speed < max_speed) {
        vi.x += a.x * dt;
        vi.y += a.y * dt;
        vi.z += a.z * dt;
        
        // 更新後の速度制限
        float new_speed = sqrtf(vi.x * vi.x + vi.y * vi.y + vi.z * vi.z);
        if (new_speed > max_speed) {
            float scale = max_speed / new_speed;
            vi.x *= scale;
            vi.y *= scale;
            vi.z *= scale;
        }
    }

    // 位置更新
    pi.x += vi.x * dt;
    pi.y += vi.y * dt;
    pi.z += vi.z * dt;

    // 改良された境界条件処理
    // warp_flags配列の構造:
    // [0]: X負側, [1]: X正側, [2]: Y負側, [3]: Y正側, [4]: Z負側, [5]: Z正側
    if(geometry_mode == 0) {
        // X軸境界処理
        if (pi.x < -half_box) {
            if (warp_flags[0]) {
                // X負側ワープ
                pi.x = half_box - 0.01f;  // 小さなオフセットで重複回避
            } else {
                // X負側跳ね返り
                pi.x = -half_box;
                vi.x *= -0.8f;  // 減衰係数を調整
            }
        } else if (pi.x > half_box) {
            if (warp_flags[1]) {
                // X正側ワープ
                pi.x = -half_box + 0.01f;
            } else {
                // X正側跳ね返り
                pi.x = half_box;
                vi.x *= -0.8f;
            }
        }
        
        // Y軸境界処理
        if (pi.y < -half_box) {
            if (warp_flags[2]) {
                // Y負側ワープ
                pi.y = half_box - 0.01f;
            } else {
                // Y負側跳ね返り
                pi.y = -half_box;
                vi.y *= -0.8f;
            }
        } else if (pi.y > half_box) {
            if (warp_flags[3]) {
                // Y正側ワープ
                pi.y = -half_box + 0.01f;
            } else {
                // Y正側跳ね返り
                pi.y = half_box;
                vi.y *= -0.8f;
            }
        }
        
        // Z軸境界処理
        if (pi.z < -half_box) {
            if (warp_flags[4]) {
                // Z負側ワープ
                pi.z = half_box - 0.01f;
            } else {
                // Z負側跳ね返り
                pi.z = -half_box;
                vi.z *= -0.8f;
            }
        } else if (pi.z > half_box) {
            if (warp_flags[5]) {
                // Z正側ワープ
                pi.z = -half_box + 0.01f;
            } else {
                // Z正側跳ね返り
                pi.z = half_box;
                vi.z *= -0.8f;
            }
        }
    } else {
        // 円柱境界処理
        float half_height = cylinder_height / 2.0f;

        // Y軸（高さ）方向の境界
        if (pi.y < -half_height) {
            if (cylinder_flags[1]) {
                pi.y = half_height - 0.01f;
            } else if (cylinder_flags[2]) {
                vanish_flags[i] = 1;
                return;
            } else {
                pi.y = -half_height;
                vi.y *= -0.8f;
            }
        } else if (pi.y > half_height) {
            if (cylinder_flags[0]) {
                pi.y = -half_height + 0.01f;
            } else if (cylinder_flags[2]) {
                vanish_flags[i] = 1;
                return;
            } else {
                pi.y = half_height;
                vi.y *= -0.8f;
            }
        }

        //XZ平面での円柱境界
        float r = sqrtf(pi.x * pi.x + pi.z * pi.z);
        if (r > cylinder_radius * 0.95) {
            if (cylinder_flags[2]) {
                vanish_flags[i] = 1;
                return;
            } else {
                // 円柱表面に押し戻し
                float scale = cylinder_radius * 0.95 / r;
                pi.x *= scale;
                pi.z *= scale;
                
                // 法線方向の速度を反転（減衰付き）
                float3 normal = make_float3(pi.x / cylinder_radius * 0.95, 0.0f, pi.z / cylinder_radius);
                float vn_dot = vi.x * normal.x + vi.z * normal.z;
                vi.x -= 1.8f * vn_dot * normal.x;
                vi.z -= 1.8f * vn_dot * normal.z;
            }
        }
    }

    // 書き戻し
    positions[i] = pi;
    velocities[i] = vi;

    if (vanish_flags[i] == 0) {  // 生存している粒子のみ
        int vbo_offset = i * 7;  // 各粒子7要素（位置3 + スケール1 + 色3）
        
        // 位置データを書き込み
        instance_vbo[vbo_offset + 0] = pi.x;
        instance_vbo[vbo_offset + 1] = pi.y;
        instance_vbo[vbo_offset + 2] = pi.z;
        
        // スケールと色を電荷に応じて設定
        if (qi > 0) {
            instance_vbo[vbo_offset + 3] = scale_factor;
            instance_vbo[vbo_offset + 4] = positive_color.x;
            instance_vbo[vbo_offset + 5] = positive_color.y;
            instance_vbo[vbo_offset + 6] = positive_color.z;
        } else {
            instance_vbo[vbo_offset + 3] = scale_factor * 0.8f;
            instance_vbo[vbo_offset + 4] = negative_color.x;
            instance_vbo[vbo_offset + 5] = negative_color.y;
            instance_vbo[vbo_offset + 6] = negative_color.z;
        }
    }
}
