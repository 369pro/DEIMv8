import albumentations as A
import cv2

def night_pipeline(img, light_prob=0.7):
    """
    输入 BGR 图像，返回仿真黑夜图像
    适配 albumentations 1.4.x+
    """
    transform = A.Compose([
        # 1. 整体压暗 + 伽玛
        A.AdjustBrightness(brightness_scale=0.45, p=1),  # scale∈(0,1) 变暗
        A.AdjustGamma(gamma=0.4, gain=1, p=1),           # 伽玛≈0.4

        # 2. 随机光束/眩光
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.4),
                         angle_lower=0.3, angle_upper=0.7,
                         num_flare_circles_lower=3, num_flare_circles_upper=7,
                         src_radius=100, src_color=(255, 255, 220), p=light_prob),

        # 3. 饱和度下降 + 轻微色偏
        A.ColorJitter(brightness=0, contrast=0.1, saturation=0.4, hue=0.02, p=0.8),

        # 4. 噪声
        A.GaussNoise(var_limit=(5, 25), p=0.7),

        # 5. 随机阴影
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1),
                       num_shadows_lower=1, num_shadows_upper=3,
                       shadow_dimension=5, p=0.5)
    ], p=1)
    return transform(image=img)['image']

# ---------------- demo ----------------
if __name__ == "__main__":
    day = cv2.imread("imgs/001940_14171.jpg")
    night = night_pipeline(day)
    cv2.imwrite("night1.jpg", night)