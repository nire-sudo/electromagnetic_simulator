extern "C" __global__ void update_particles_cuda(
    float3 *positions, float3 *velocities, float *charges, float *masses,
    float3 *E_field, int E_field_x_size, int E_field_y_size, int E_field_z_size,
    float cell_size, int N, float k, float min_dist, float dt, float box_size, 
    int boundary_mode, int *warp_flags
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float3 pi = positions[i];
    float3 vi = velocities[i];
    float qi = charges[i];
    float mi = masses[i];

    // クーロン力計算（最適化版）
    float3 fi = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < N; ++j) {
        if (i == j) continue;
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

    // 外部電場による力（正確な計算）
    fi.x += qi * E.x;
    fi.y += qi * E.y;
    fi.z += qi * E.z;

    // 加速度 a = F / m
    float3 a = make_float3(fi.x / mi, fi.y / mi, fi.z / mi);

    // 速度更新（制限付き）
    float max_speed = 8.0f;  // 最大速度を上げる
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
    
    // X軸境界処理
    if (pi.x < -half_box) {
        if (warp_flags[0]) {
            // X負側ワープ
            pi.x = half_box - 0.01f;  // 小さなオフセットで重複回避
        } else {
            // X負側跳ね返り
            pi.x = -half_box;
            vi.x *= -0.7f;  // 減衰係数
        }
    } else if (pi.x > half_box) {
        if (warp_flags[1]) {
            // X正側ワープ
            pi.x = -half_box + 0.01f;
        } else {
            // X正側跳ね返り
            pi.x = half_box;
            vi.x *= -0.7f;
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
            vi.y *= -0.7f;
        }
    } else if (pi.y > half_box) {
        if (warp_flags[3]) {
            // Y正側ワープ
            pi.y = -half_box + 0.01f;
        } else {
            // Y正側跳ね返り
            pi.y = half_box;
            vi.y *= -0.7f;
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
            vi.z *= -0.7f;
        }
    } else if (pi.z > half_box) {
        if (warp_flags[5]) {
            // Z正側ワープ
            pi.z = -half_box + 0.01f;
        } else {
            // Z正側跳ね返り
            pi.z = half_box;
            vi.z *= -0.7f;
        }
    }

    // 書き戻し
    positions[i] = pi;
    velocities[i] = vi;
}