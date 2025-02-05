import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from collections import deque
import math

def convolve(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    convolved_image = np.zeros((image_height, image_width), dtype=np.float32)
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    for i in range(image_height):
        for j in range(image_width):
            region = padded_image[i : i + kernel_height, j : j + kernel_width]
            convolved_image[i, j] = np.sum(region * kernel)
    return convolved_image

def gaussian_blur(image):
    gaussian_kernel = np.array([
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ], dtype=np.float32)
    return convolve(image, gaussian_kernel)

def sobel_edge_detection(image, threshold=50, normalize=True, visualize=False):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=np.float32)

    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)

    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    if normalize:
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max()) * 255

    edge_image = (gradient_magnitude > threshold) * 255
    edge_image = edge_image.astype(np.uint8)
    
    return edge_image


def dbscan(points, eps=2, min_samples=5):
    n_points, _ = points.shape
    labels = -2 * np.ones(n_points, dtype=int)
    cluster_id = 0
    def region_query(point_idx):
        point = points[point_idx]
        return np.where(np.linalg.norm(points - point, axis=1) < eps)[0]
    def expand_cluster(point_idx, neighbors):
        labels[point_idx] = cluster_id
        queue = deque(neighbors)
        while queue:
            neighbor_idx = queue.popleft()
            if labels[neighbor_idx] == -2:
                labels[neighbor_idx] = cluster_id
                neighbor_neighbors = region_query(neighbor_idx)
                if len(neighbor_neighbors) >= min_samples:
                    queue.extend(neighbor_neighbors)
            elif labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
    for point_idx in range(n_points):
        if labels[point_idx] != -2:
            continue
        neighbors = region_query(point_idx)
        if len(neighbors) < min_samples:
            labels[point_idx] = -1  
        else:
            expand_cluster(point_idx, neighbors)
            cluster_id += 1
    return labels

def calculate_bounding_circle(cluster_points):
    rows = cluster_points[:, 0]
    cols = cluster_points[:, 1]
    min_y, max_y = np.min(rows), np.max(rows)
    min_x, max_x = np.min(cols), np.max(cols)
    centroid_y = (min_y + max_y) // 2
    centroid_x = (min_x + max_x) // 2
    dists = np.sqrt((rows - centroid_y)**2 + (cols - centroid_x)**2)
    r = np.max(dists)
    return (centroid_x, centroid_y, r)

def sort_targets(circles, shooter_point):
    sx, sy = shooter_point
    def distance(circle):
        cx, cy, _ = circle
        return math.hypot(cx - sx, cy - sy)
    return sorted(circles, key=distance)

def ball_dynamics(v, k_m=0.005, g=200.0):
    vx, vy = v
    speed = np.sqrt(vx*vx + vy*vy)
    ax = -k_m * vx * speed
    ay = g - k_m * vy * speed
    return np.array([ax, ay])

def euler_method_step(p, v, f, dt=0.01):
    p_new = p + v * dt
    v_new = v + f(v) * dt
    return p_new, v_new
  
def runge_kutta_2_step(p, v, f, dt=0.01):
    k1_v = f(v)
    k1_p = v
    k2_v = f(v + k1_v * dt)
    k2_p = v + k1_v * dt
    v_new = v + (k1_v + k2_v) * dt / 2
    p_new = p + (k1_p + k2_p) * dt / 2
    return p_new, v_new

def shooting_v(position, target, steps, h=0.01):
    position = position.astype(np.float64)
    target   = target.astype(np.float64)
    velocity = (target - position) / (steps * h)
    for _ in range(5):
        p  = position.copy()
        v  = velocity.copy()
        p1 = position.copy()
        v1 = velocity + np.array([h, 0.0])
        p2 = position.copy()
        v2 = velocity + np.array([0.0, h])
        for __ in range(steps):
            # p,  v  = euler_method_step(p,  v,  ball_dynamics)
            # p1, v1 = euler_method_step(p1, v1, ball_dynamics)
            # p2, v2 = euler_method_step(p2, v2, ball_dynamics)
            p,  v  = runge_kutta_2_step(p,  v,  ball_dynamics)
            p1, v1 = runge_kutta_2_step(p1, v1, ball_dynamics)
            p2, v2 = runge_kutta_2_step(p2, v2, ball_dynamics)
        dv1 = (p1 - p) / h
        dv2 = (p2 - p) / h
        jacobian = np.array([
            [dv1[0], dv2[0]],
            [dv1[1], dv2[1]]
        ])
        error = p - target
        velocity -= np.linalg.inv(jacobian) @ error
    return velocity

x0, y0 = 330, 200
shooter_point = np.array([x0, y0])
image_path = "./targets.png"
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
blurred = gaussian_blur(img_gray)
edges = sobel_edge_detection(blurred)
points = np.column_stack(np.where(edges > 0))
labels = dbscan(points)
unique_labels = set(labels)

circles = []

for cid in unique_labels:
    if cid == -1: 
        continue
    cluster_pts = points[labels == cid]
    cx, cy, r = calculate_bounding_circle(cluster_pts)
    circles.append((cx, cy, r))
sorted_circles = sort_targets(circles, shooter_point)

paths = []
for (cx, cy, r) in sorted_circles:
    velocity = shooting_v(np.array([x0, y0]), np.array([cx, cy]), steps=200)
    p = np.array([x0, y0], dtype=np.float64)
    v = velocity
    x_vals, y_vals = [p[0]], [p[1]]
    for _ in range(200):
        p, v = euler_method_step(p, v, ball_dynamics)
        x_vals.append(p[0])
        y_vals.append(p[1])
    paths.append((x_vals, y_vals))

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("Targets Plot")
ax.set_aspect("equal", "box")
ax.invert_yaxis()
ax.set_xlim([0, img_gray.shape[1]])
ax.set_ylim([img_gray.shape[0], 0])
ax.plot(shooter_point[0], shooter_point[1], 'bo', markersize=10, label='Shooter')

circle_patches = []
centroid_patches = []
for (cx, cy, r) in sorted_circles:
    circle = plt.Circle(
        (cx, cy), radius=r,
        color='black',
        linewidth=2
    )
    ax.add_patch(circle)
    centroid = ax.plot(cx, cy, 'ro', markersize=3)[0]
    circle_patches.append(circle)
    centroid_patches.append(centroid)
    
ax.legend(loc='upper right')
path_line, = ax.plot([], [], 'r-', linewidth=1)

def update(frame, path_line, paths, circle_patches, centroid_patches):
    current_path_index = frame // 200
    current_frame = frame % 200
    if current_path_index < len(paths):
        x_vals, y_vals = paths[current_path_index]
        path_line.set_data(x_vals[:current_frame + 1], y_vals[:current_frame + 1])
        if current_frame == 199:
            circle_patches[current_path_index].remove()
            centroid_patches[current_path_index].remove()
            path_line.set_data([], [])
    return path_line,

ani = FuncAnimation(fig, update, frames=len(paths) * 200, fargs=(path_line, paths, circle_patches, centroid_patches), interval=30, blit=False)
output_video_path = "p1_output.mp4"
ani.save(output_video_path, writer=FFMpegWriter(fps=30))
plt.show()
