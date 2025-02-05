import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

image = cv2.imread('./content/fly.webp')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def bezier_curve(p1, p2, p3, p4, t):
    return (1 - t) ** 3 * p1 + 3 * (1 - t) ** 2 * t * p2 + 3 * (1 - t) * t ** 2 * p3 + t ** 3 * p4

def generate_grid(width, height, pieces):
    cols = int(np.sqrt(pieces * (width / height)))
    rows = int(np.ceil(pieces / cols))
    cell_w, cell_h = width / cols, height / rows
    points = [(j * cell_w, i * cell_h) for i in range(rows + 1) for j in range(cols + 1)]
    return points, rows, cols, cell_w, cell_h

def create_knob_curve(start, end, neck=1, width=0.5, height=0.5, pull=0.1):
    x1, y1, x2, y2 = *start, *end
    mx, my = (x1 + x2) / 2, max(y1, y2) + abs(x2 - x1) * height
    ctrls = [
        (x1 + (mx - x1) * neck, y1 + (my - y1) * pull),
        (mx - (mx - x1) * width, my),
        (mx + (x2 - mx) * width, my),
        (x2 - (x2 - mx) * neck, y2 + (my - y2) * pull),
    ]
    t = np.linspace(0, 1, 100)
    curve1 = bezier_curve(x1, ctrls[0][0], ctrls[1][0], mx, t), bezier_curve(y1, ctrls[0][1], ctrls[1][1], my, t)
    curve2 = bezier_curve(mx, ctrls[2][0], ctrls[3][0], x2, t), bezier_curve(my, ctrls[2][1], ctrls[3][1], y2, t)
    return np.concatenate([curve1[0], curve2[0]]), np.concatenate([curve1[1], curve2[1]])

def draw_knobs(ax, points, rows, cols):
    params = dict(neck=1.5, width=0.8, height=0.3, pull=0.2)
    color = "black"

    # horizontal
    for r in range(1, rows):
        for c in range(cols):
            p1, p2 = points[r * (cols + 1) + c], points[r * (cols + 1) + c + 1]
            x, y = create_knob_curve(p1, p2, **params)
            if random.choice([0, 1]): y = [2 * p1[1] - yi for yi in y]
            ax.plot(x, y, color=color, linewidth=1.5)

    # vertical
    for c in range(1, cols):
        for r in range(rows):
            p1, p2 = points[r * (cols + 1) + c], points[(r + 1) * (cols + 1) + c]
            y, x = create_knob_curve((p1[1], p1[0]), (p2[1], p2[0]), **params)
            if random.choice([0, 1]): x = [2 * p1[0] - xi for xi in x]
            ax.plot(x, y, color=color, linewidth=1.5)

def render_puzzle(image, width, height, points, rows, cols):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.plot([0, width, width, 0, 0], [0, 0, height, height, 0], color="black")
    for x, y in points: ax.plot(x, y, 'ro', markersize=2)
    draw_knobs(ax, points, rows, cols)
    plt.axis('off')
    plt.show()

img_w, img_h = image.shape[1], image.shape[0]
points, rows, cols, cell_w, cell_h = generate_grid(img_w, img_h, 250)
render_puzzle(image, img_w, img_h, points, rows, cols)