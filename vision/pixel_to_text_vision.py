# -*- coding: utf-8 -*-
"""
像素→文本→通用物体识别 (DeepSeek V4 Pro 文本API)
特征提取(通用,非针对特定物体) → 查表匹配 → API确认
"""

import os, struct, math
from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "your-key-here")
client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# =============================================================
# Step 1: 读 BMP 像素 (通用)
# =============================================================
def read_bmp(path):
    with open(path, 'rb') as f:
        f.read(18)
        w = struct.unpack('<I', f.read(4))[0]
        h = struct.unpack('<I', f.read(4))[0]
        f.read(28)
        pixels = []
        for y in range(h):
            row = []
            for x in range(w):
                b, g, r = f.read(3)
                row.append((r, g, b))
            pixels.insert(0, row)
        return w, h, pixels

# =============================================================
# Step 2: 通用特征提取
# =============================================================

def detect_edges(w, h, pixels):
    """一阶差分边缘, 返回边缘像素坐标和边缘强度"""
    edges = []
    for y in range(1, h-1):
        for x in range(1, w-1):
            r0, g0, b0 = pixels[y][x]
            r1, g1, b1 = pixels[y][x+1]
            rd, gd, bd = pixels[y+1][x]
            diff = abs(r0-r1) + abs(g0-g1) + abs(b0-b1) + abs(r0-rd) + abs(g0-gd) + abs(b0-bd)
            if diff > 150:
                edges.append((x, y, diff))
    return edges


def color_distribution(w, h, pixels):
    """颜色分布: 主色调 + 各通道统计"""
    all_r, all_g, all_b = [], [], []
    for y in range(h):
        for x in range(w):
            r, g, b = pixels[y][x]
            all_r.append(r); all_g.append(g); all_b.append(b)

    # 主色调判定
    avg_r, avg_g, avg_b = sum(all_r)/len(all_r), sum(all_g)/len(all_g), sum(all_b)/len(all_b)
    if avg_r > avg_g + 50 and avg_r > avg_b + 50:   dominant = "red"
    elif avg_g > avg_r + 30 and avg_g > avg_b + 30:  dominant = "green"
    elif avg_b > avg_r + 30 and avg_b > avg_g + 30:  dominant = "blue"
    elif max(avg_r,avg_g,avg_b) - min(avg_r,avg_g,avg_b) < 30: dominant = "gray/white"
    elif avg_r > 150 and avg_g > 100 and avg_b < 80: dominant = "yellow/orange"
    else: dominant = "mixed"

    # 各通道标准差(纹理均匀度)
    def std(vals): m=sum(vals)/len(vals); return math.sqrt(sum((x-m)**2 for x in vals)/len(vals))
    return {
        "dominant": dominant,
        "avg_rgb": (round(avg_r), round(avg_g), round(avg_b)),
        "std_rgb": (round(std(all_r)), round(std(all_g)), round(std(all_b))),
        "brightness": round((avg_r + avg_g + avg_b) / 3),
    }


def connectivity(w, h, pixels):
    """连通域计数: 有多少块独立的非背景区域"""
    # 背景 = 亮度接近白色
    bg_threshold = 400  # R+G+B > 400 视为背景(白色)
    visited = [[False]*w for _ in range(h)]
    regions = 0
    region_sizes = []

    for y in range(h):
        for x in range(w):
            r, g, b = pixels[y][x]
            if r + g + b > bg_threshold or visited[y][x]:
                continue
            # BFS 淹灌一个连通域
            regions += 1
            size = 0
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if not (0 <= cx < w and 0 <= cy < h): continue
                if visited[cy][cx]: continue
                pr, pg, pb = pixels[cy][cx]
                if pr + pg + pb > bg_threshold: continue
                visited[cy][cx] = True
                size += 1
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    stack.append((cx+dx, cy+dy))
            if size > 0:
                region_sizes.append(size)

    return {"region_count": regions, "region_sizes": region_sizes,
            "single_object": regions == 1}


def center_of_mass(w, h, pixels):
    """重心偏移: 非背景像素的几何中心位置"""
    total_w, total_n = 0, 0
    min_y = h
    for y in range(h):
        for x in range(w):
            r, g, b = pixels[y][x]
            if r + g + b < 400:  # 非背景
                total_w += y * 1.0  # 垂直方向权重
                total_n += 1
                if y < min_y: min_y = y
    if total_n == 0: return {"vertical_center": 0, "top_offset": 0}
    center = total_w / total_n  # 0 = 顶部, h-1 = 底部
    return {
        "vertical_center_ratio": round(center / h, 2),
        "top_has_element": min_y < h * 0.3,  # 顶部30%区域有物体
    }


def shape_fill_ratio(w, h, pixels):
    """填充率: 边缘包围面积 / 宽高矩形面积"""
    # 简化: 非背景像素 / 总像素
    fg = sum(1 for y in range(h) for x in range(w) if sum(pixels[y][x]) < 400)
    return {"fill_ratio": round(fg / (w * h), 2)}


# =============================================================
# Step 3: 简单查表 (物体特征库, 可扩展)
# =============================================================
OBJECT_DB = [
    {
        "name": "apple",
        "dominant_color": "red",
        "fill_ratio_range": (0.55, 0.85),
        "single_object": True,
        "top_feature": True,         # 顶部有梗
        "texture_hint": "斑驳渐变,自然纹理"
    },
    {
        "name": "sun",
        "dominant_color": "yellow/orange",
        "fill_ratio_range": (0.60, 0.95),
        "single_object": True,
        "top_feature": False,
        "texture_hint": "均匀辐射,中心更亮"
    },
    {
        "name": "leaf",
        "dominant_color": "green",
        "fill_ratio_range": (0.40, 0.70),
        "single_object": True,
        "top_feature": False,
        "texture_hint": "叶脉纹理,细长形状"
    },
]

def match_object(features):
    """查表匹配"""
    matches = []
    color = features["color"]["dominant"]
    fill = features["fill"]["fill_ratio"]
    single = features["connectivity"]["single_object"]
    top = features["center"]["top_has_element"]
    texture = "斑驳" if features["color"]["std_rgb"][0] > 60 else "均匀"

    for obj in OBJECT_DB:
        score = 0
        reasons = []
        if obj["dominant_color"] in color or color in obj["dominant_color"]:
            score += 30; reasons.append("颜色匹配")
        lo, hi = obj["fill_ratio_range"]
        if lo <= fill <= hi:
            score += 25; reasons.append("填充率匹配")
        if obj["single_object"] == single:
            score += 20; reasons.append("连通域匹配")
        if obj["top_feature"] == top:
            score += 15; reasons.append("顶部特征匹配")
        if obj["texture_hint"] in texture:
            score += 10; reasons.append("纹理匹配")
        matches.append((obj["name"], score, reasons))

    matches.sort(key=lambda x: -x[1])
    return matches


# =============================================================
# Step 4: 组装描述 + 调用 DeepSeek
# =============================================================
def identify(path):
    w, h, pixels = read_bmp(path)
    edges = detect_edges(w, h, pixels)
    color = color_distribution(w, h, pixels)
    conn = connectivity(w, h, pixels)
    center = center_of_mass(w, h, pixels)
    fill = shape_fill_ratio(w, h, pixels)

    features = {"color": color, "connectivity": conn, "center": center, "fill": fill}
    matches = match_object(features)

    description = f"""这是一张{w}x{h}像素的BMP图片, 提取到的通用视觉特征:

特征提取(非针对特定物体):
  边缘: {len(edges)}个边缘像素
  颜色: 主色调{color['dominant']}, 平均RGB{color['avg_rgb']}
        纹理均匀度(标准差): {color['std_rgb']} (值越大纹理越斑驳)
        整体亮度: {color['brightness']}
  连通域: {conn['region_count']}个独立区域, 各区域大小{conn['region_sizes']}
          是否为单物体: {conn['single_object']}
  重心: 垂直重心位于 {center['vertical_center_ratio']*100:.0f}% 处
        顶部区域是否有物体: {center['top_has_element']}
  填充率: 物体像素占总面积 {fill['fill_ratio']*100:.0f}%

特征库匹配结果(最高分):
  {matches[0][0]} (得分:{matches[0][1]}, 匹配:{', '.join(matches[0][2])})
  {matches[1][0] if len(matches)>1 else '无'} (得分:{matches[1][1] if len(matches)>1 else 0})

请根据以上通用特征数据, 判断图片中最可能是什么物体。只输出物体名称和一句推理依据。"""

    resp = client.chat.completions.create(
        model="deepseek-chat", temperature=0.3, max_tokens=400,
        messages=[
            {"role": "system", "content": "你是基于文本特征进行视觉推理的引擎。你将收到一份通用特征报告和一个查表匹配候选。请输出你自己的完整推理逻辑链：特征解读→逐个物体比对→加权评分→最终结论。输出格式:\n[逻辑链]\n...\n[最终结论]\n物体名称, 置信度(百分数)"},
            {"role": "user", "content": description}
        ]
    )
    return resp.choices[0].message.content, description, matches

# =============================================================
# 主流程
# =============================================================
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "test_apple.bmp"
    if not os.path.exists(path):
        print(f"文件不存在: {path}")
        sys.exit(1)

    print(f"分析: {path}\n")
    result, desc, matches = identify(path)
    print(f"[特征提取]\n{desc}\n")
    print(f"[DeepSeek 识别]\n{result}")
