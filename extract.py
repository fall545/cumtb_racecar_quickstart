import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from math import cos, sin, radians
import copy

img = "your_image_path_here"

def binarize_extract_black(img, thresh=60):
    # 将图像转为灰度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化并反色，使赛道黑色部分变成255白，便于处理
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    return bw

def cluster_black_pixels(rect_img, eps=5, min_samples=20):
    # 查找矩形区域中所有非零像素（即黑色赛道部分）
    ys, xs = np.where(rect_img==255)
    if len(xs) == 0:
        return []
    pts = np.column_stack((xs, ys))
    # 用DBSCAN聚类赛道像素（eps为距离阈值，min_samples为最小聚类数）
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = db.labels_
    clusters = []
    # 对每个聚类计算质心用于作为赛道点
    for lab in set(labels):
        if lab == -1:  
            continue  # -1 为噪声
        mask = labels == lab
        centroid = pts[mask].mean(axis=0)
        clusters.append((centroid[0], centroid[1]))
    return clusters

def quadratic_bezier_control_point(pts):
    # 用最小二乘法拟合二次贝塞尔曲线控制点（P0 起点, P1 控制点, P2 终点）
    if len(pts) < 3:
        return None
    pts = np.array(pts, dtype=np.float64)
    P0 = pts[0]
    P2 = pts[-1]

    # 用弦长参数化计算每一点对应的时间参数t
    chord = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
    if chord.sum() == 0:
        t = np.linspace(0, 1, len(pts))
    else:
        cum = np.concatenate(([0.0], np.cumsum(chord)))
        t = cum / cum[-1]

    A = []
    bx = []
    by = []
    for i, ti in enumerate(t):
        if ti == 0 or ti == 1:
            continue
        # 二次贝塞尔： B(t)= (1-t)^2*P0 + 2(1-t)t*P1 + t^2*P2
        # 求P1时，把其它项移到一边
        a = 2*(1-ti)*ti
        A.append([a])
        xi = pts[i,0]
        yi = pts[i,1]
        bx.append(xi - ((1-ti)**2 * P0[0] + ti**2 * P2[0]))
        by.append(yi - ((1-ti)**2 * P0[1] + ti**2 * P2[1]))

    A = np.array(A)
    if A.size == 0:
        return None
    bx = np.array(bx)
    by = np.array(by)

    # 最小二乘求解 P1
    px, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
    py, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
    P1 = np.array([px[0], py[0]])

    return P0, P1, P2

def sample_quadratic_bezier(P0, P1, P2, k):
    # 从拟合的二次贝塞尔曲线采样k个点
    ts = np.linspace(0, 1, k)
    pts = []
    for t in ts:
        p = (1-t)**2 * P0 + 2*(1-t)*t * P1 + t**2 * P2
        pts.append((float(p[0]), float(p[1])))
    return pts


def polar_bfs_scan(start_pt, bw, angle_step=10, dis=100, step=5, cluster_eps=4, cluster_min=10):
    # 使用广度优先搜索（Breadth First Search）方式沿角度与距离扫描赛道
    # start_pts: 初始路径点
    # angle_step: 扫描角度步长
    # dis: 扫描最大距离
    # step: 扫描细分步长
    h, w = bw.shape
    print(h, w)
    results = [[start_pt]]
    find=False
    for _ in range(step):  # 最大迭代次数，防止死循环
        # 对当前所有路径继续扩展
        for idx, path in enumerate(results):
            cur_x, cur_y = path[-1]
            branches = []

            # 扫描0°到180°范围
            ang=0
            # for ang in np.arange(0, 180+1e-6, angle_step):
            while ang<=180:
                rad = radians(ang)

                # 每隔 step 往前扫描
                nx = int(round(cur_x + dis * cos(rad)))
                ny = int(round(cur_y - dis * sin(rad)))

                # 越界则停止该方向
                if nx < 0 or nx >= w or ny < 0 or ny >= h:
                    print('hit bolder')
                    return results

                # 在扫描点附近开窗口检测黑色面积
                win_r = 6
                x0 = max(0, nx-win_r)
                y0 = max(0, ny-win_r)
                x1 = min(w, nx+win_r+1)
                y1 = min(h, ny+win_r+1)

                window = bw[y0:y1, x0:x1]
                if window.size == 0:
                    continue

                black_area = np.count_nonzero(window)

                # 此处约定窗口至少20%黑则认为是赛道
                if black_area >= max(5, int(0.2 * window.size)):
                    # 再次聚类得到该点所在赛道区域质心
                    clusters = cluster_black_pixels(window, eps=cluster_eps, min_samples=cluster_min)
                    if len(clusters) == 0:
                        raise Exception("no clusters")

                    cx, cy = clusters[0]
                    global_x = cx + x0
                    global_y = cy + y0
                    branches.append((global_x, global_y))
                    if len(branches)>=2:
                        break  # 最多取两个分支
                    ang+=90
                else:
                    ang+=angle_step
                        

            # 若没有分支，停止扩展
            if len(branches) == 0:
                continue
            elif len(branches) == 1:
                path.append(branches[0])
            else:
                # 对每个分支复制路径
                
                new_path = copy.deepcopy(path)
                new_path.append(branches[0])
                path.append(branches[1])
                results.append(new_path)

    return results

def compute_target_from_bezier_points(bezier_pts, k):
    # 按从下到上排序（y 越大越靠下）
    pts = sorted(bezier_pts, key=lambda p: p[1], reverse=True)
    if len(pts) < k:
        k = len(pts)

    # 取前k个点（越靠下越重要）
    selected = pts[:k]

    # 权重为：从下往上，权重按二次函数递减
    xs = np.array([p[0] for p in selected])
    indices = np.arange(len(selected))
    weights = 1 - (indices/(k-1))**2 if k>1 else np.array([1.0])

    weights = weights / weights.sum()
    target = (xs * weights).sum()
    return float(target), selected

def draw_results(orig, paths, bezier_samples_per_path, target_x, error):
    # 绘制结果显示
    out = orig.copy()
    h, w = out.shape[:2]


    colors=((255, 255, 0),(0, 255, 255),(128, 255, 128))

    # 绘制路径点
    for idx, path in enumerate(paths):
        # for i in range(len(path)-1):
        #     p0 = (int(path[i][0]), int(path[i][1]))
        #     p1 = (int(path[i+1][0]), int(path[i+1][1]))
        #     cv2.line(out, p0, p1, color[0], 30)
        for p in path:
            cp = (int(p[0]), int(p[1]))
            cv2.circle(out, cp, 20, colors[0], -1)

    # 绘制贝塞尔曲线
    for idx, bs in enumerate(bezier_samples_per_path):
        for i in range(len(bs)-1):
            p0 = (int(bs[i][0]), int(bs[i][1]))
            p1 = (int(bs[i+1][0]), int(bs[i+1][1]))
            cv2.line(out, p0, p1, colors[1], 10)

    # 绘制 target 点
    ty = int(target_x)
    cv2.rectangle(out, (ty-20, h-60), (ty+20, h-40), (128,128,255), -1)

    # 绘制误差值
    cv2.putText(out, f'error: {error:.2f}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # 绘制图像中心线
    cx = int(w/2)
    cv2.line(out, (cx,0),(cx,h),(200,200,200),5)

    return out

def process_track_pipeline(image,
                           bin_thresh=60,
                           bottom_layer_height=10,
                           n_layers=5,
                           cluster_eps=6,
                           cluster_min=30,
                           angle_step=10,
                           scan_dis=120,
                           scan_step=5,
                           cluster_eps_scan=4,
                           cluster_min_scan=12,
                           path_length_threshold=30,
                           bezier_k=20):
    # 获取图像尺寸
    h, w = image.shape[:2]

    # STEP 1：二值化提取黑色赛道
    bw = binarize_extract_black(image, thresh=bin_thresh)

    # STEP 2：在底部区域聚类得到起点
    start_y0 = h - bottom_layer_height
    start_y1 = max(0, start_y0 - n_layers)

    x0 = 0
    x1 = w
    rect = bw[start_y1:start_y0, x0:x1]
    print('preprocess done')
    clusters = cluster_black_pixels(rect, eps=cluster_eps, min_samples=cluster_min)
    print('clustering done')
    # 将矩形区域内坐标映射回全局坐标
    start_points = [ (c[0]+x0, c[1]+start_y1) for c in clusters ]

    if len(start_points) == 0:
        print("No start points found.")
        return image, [], [], None, None
    if len(start_points) != 1:
        print(f"Warning: Multiple start points found: {len(start_points)}")
        start_point= start_points[0]  # 仅取第一个起点,也可以试一试平均值
    start_point= start_points[0]
    print('start point:', start_point)
    # STEP 3：从起点使用BFS方式沿角度/距离扫描赛道
    paths = polar_bfs_scan(start_point, bw,
                           angle_step=angle_step,
                           dis=scan_dis,
                           step=scan_step,
                           cluster_eps=cluster_eps_scan,
                           cluster_min=cluster_min_scan)

    # 移除只有一个点的路径
    
    final_paths = [p for p in paths if len(p) > 1]
    print('scan done')
    print(f'Number of paths found: {len(final_paths)}')
    print(len(final_paths[0]))
    # 按路径长度过滤（保留长路径）
    pruned = [p for p in final_paths if len(p) >= path_length_threshold]
    if len(pruned) == 0:
        pruned = final_paths

    bezier_samples_all = []
    targets = []

    # STEP 4：对每条路径拟合贝塞尔曲线 & 求目标点
    for p in pruned:
        ctrl = quadratic_bezier_control_point(p)
        if ctrl is None:
            bezier_pts = p
        else:
            P0, P1, P2 = ctrl
            bezier_pts = sample_quadratic_bezier(np.array(P0), np.array(P1), np.array(P2), bezier_k)

        bezier_samples_all.append(bezier_pts)

        # STEP 5：取下方k个点二次递减加权得到target
        t, sel = compute_target_from_bezier_points(bezier_pts, k=min(len(bezier_pts), bezier_k))
        targets.append(t)

    if len(targets) == 0:
        return image, pruned, bezier_samples_all, None, None

    # 多条路径时，用均值作为最终target
    target_x = float(np.mean(targets))

    # 误差 = target - 图像中心
    error = target_x - (w/2)

    # 可视化
    vis = draw_results(image, pruned, bezier_samples_all, target_x, error)
    return vis, pruned, bezier_samples_all, target_x, error


# 用法示例
if __name__ == "__main__":
    img = cv2.imread(img)
    print('read')
    vis, paths, beziers, target, error = process_track_pipeline(
        img,
        bin_thresh=100,
        bottom_layer_height=12,
        n_layers=20,
        cluster_eps=6,
        cluster_min=25,
        angle_step=4,
        scan_dis=250,
        scan_step=9,
        cluster_eps_scan=4,
        cluster_min_scan=10,
        path_length_threshold=200,
        bezier_k=30
    )

    if vis is not None:
        cv2.imshow("result", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()