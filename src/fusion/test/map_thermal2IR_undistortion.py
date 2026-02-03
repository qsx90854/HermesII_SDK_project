import cv2
import numpy as np
import json

def load_params(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data["camera_matrix"]), np.array(data["dist_coeff"])

def load_extrinsics(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return np.array(data["R"]), np.array(data["T"])

def create_rectified_maps(output_size, new_K_ir, K_ir, D_ir, K_th, D_th, R, T, dist_meters):
    """
    建立兩個映射表 (Map)：
    1. map_x_ir, map_y_ir: 將原始 IR 轉正 (Undistort)
    2. map_x_th, map_y_th: 將原始 Thermal 對齊到轉正後的 IR (Alignment + Undistort)
    """
    w, h = output_size
    
    # 1. 產生「理想無畸變影像」的像素座標網格
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    
    # 展平成 (N, 1, 2)
    # 這些是我們 "希望" 輸出的像素位置
    points_2d_rect = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).reshape(-1, 1, 2).astype(np.float32)

    # 2. 將理想像素反投影回 3D 空間 (Normalized Coordinate)
    # 因為是理想相機 (new_K_ir)，沒有畸變 (dist=0)，所以直接用數學回推
    # u = fx * x + cx  =>  x = (u - cx) / fx
    fx = new_K_ir[0, 0]
    fy = new_K_ir[1, 1]
    cx = new_K_ir[0, 2]
    cy = new_K_ir[1, 2]

    x_norm = (points_2d_rect[:, 0, 0] - cx) / fx
    y_norm = (points_2d_rect[:, 0, 1] - cy) / fy
    
    # 建立 3D 點 (在 IR 相機座標系下)
    # shape: (N, 1, 3)
    points_3d_ir = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=1).reshape(-1, 1, 3)
    points_3d_ir = points_3d_ir * dist_meters

    # --- 分支 A: 製作 IR 的 Map (找回原始 IR 像素) ---
    # 將 3D 點投影回 "原始有畸變" 的 IR 感光元件
    points_2d_raw_ir, _ = cv2.projectPoints(points_3d_ir, np.zeros(3), np.zeros(3), K_ir, D_ir)
    map_x_ir = points_2d_raw_ir[:, 0, 0].reshape(h, w).astype(np.float32)
    map_y_ir = points_2d_raw_ir[:, 0, 1].reshape(h, w).astype(np.float32)

    # --- 分支 B: 製作 Thermal 的 Map (找回原始 Thermal 像素) ---
    # 座標轉換：IR 座標系 -> Thermal 座標系
    # P_th = R_inv * (P_ir - T)
    points_3d_ir_centered = points_3d_ir - T.reshape(1, 1, 3)
    R_inv = R.T
    points_3d_th = np.matmul(points_3d_ir_centered, R_inv.T)

    # 將 3D 點投影回 "原始有畸變" 的 Thermal 感光元件
    points_2d_raw_th, _ = cv2.projectPoints(points_3d_th, np.zeros(3), np.zeros(3), K_th, D_th)
    map_x_th = points_2d_raw_th[:, 0, 0].reshape(h, w).astype(np.float32)
    map_y_th = points_2d_raw_th[:, 0, 1].reshape(h, w).astype(np.float32)

    return (map_x_ir, map_y_ir), (map_x_th, map_y_th)

def main():
    # --- 1. 檔案路徑設定 ---
    thermal_img_path = "Thermal_4.bmp"
    ir_img_path = "IR_4.bmp"
    
    thermal_json = "thermal_camera_params.json"
    ir_json = "ir_camera_params.json"
    stereo_json = "stereo_extrinsics.json"

    ASSUMED_DISTANCE = 1000.0 # mm

    # --- 2. 載入資料 ---
    print("載入參數與影像...")
    K_th, D_th = load_params(thermal_json)
    K_ir, D_ir = load_params(ir_json)
    R, T = load_extrinsics(stereo_json)

    img_th = cv2.imread(thermal_img_path)
    img_ir = cv2.imread(ir_img_path)

    if img_th is None or img_ir is None:
        print("錯誤：找不到圖片")
        return

    h, w = img_ir.shape[:2]

    # --- 3. 計算最佳的新相機矩陣 (New Camera Matrix) ---
    # alpha=1: 保留所有像素 (可能有黑邊)
    # alpha=0: 裁切掉無效區域 (放大)
    print("計算最佳無畸變視角...")
    new_K_ir, roi = cv2.getOptimalNewCameraMatrix(K_ir, D_ir, (w, h), 0, (w, h))

    # --- 4. 計算兩個映射表 ---
    print(f"正在計算雙重映射表 (假設距離: {ASSUMED_DISTANCE})...")
    (map_ir_x, map_ir_y), (map_th_x, map_th_y) = create_rectified_maps(
        (w, h), 
        new_K_ir,   # 用新的無畸變矩陣作為基準
        K_ir, D_ir, 
        K_th, D_th, 
        R, T, 
        ASSUMED_DISTANCE
    )

    # --- 5. 執行影像變形 (Warping) ---
    # 兩張圖都要 remap
    print("正在執行 Remap...")
    rectified_ir = cv2.remap(img_ir, map_ir_x, map_ir_y, cv2.INTER_LINEAR)
    rectified_th = cv2.remap(img_th, map_th_x, map_th_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite("ir_undistortion.bmp", rectified_ir)
    cv2.imwrite("rectified_th.bmp", rectified_th)
    # --- 6. 疊加顯示 ---
    rectified_th_gray = cv2.cvtColor(rectified_th, cv2.COLOR_BGR2GRAY)
    rectified_th_color = cv2.applyColorMap(rectified_th_gray, cv2.COLORMAP_JET)
    
    mask = (rectified_th_gray > 0).astype(np.uint8)
    
    result = rectified_ir.copy()
    alpha = 0.5
    beta = 1.0 - alpha
    
    np.copyto(result, cv2.addWeighted(rectified_ir, alpha, rectified_th_color, beta, 0), where=mask[:,:,None].astype(bool))

    # --- 7. 顯示結果 ---
    vis = cv2.resize(result, (0, 0), fx=0.8, fy=0.8)
    cv2.imshow(f"Rectified & Fused (Dist: {ASSUMED_DISTANCE})", vis)
    
    print("按 's' 儲存結果，按其他鍵離開")
    key = cv2.waitKey(0)
    if key == ord('s'):
        cv2.imwrite("rectified_result.jpg", result)
        print("已儲存 rectified_result.jpg")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()