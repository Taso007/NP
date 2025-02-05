import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def ball_motion(vx, vy, g0, k_m0, x0, y0, tx, ty, dt, ms=50):
    x, y = x0, y0
    g, k_m = g0, k_m0
    for _ in range(ms):
        ax = -k_m * vx * np.sqrt(vx*vx + vy*vy)
        ay = -g - k_m * vy * np.sqrt(vx*vx + vy*vy)
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
    return tx - x, ty - y

def shooting_method(vx0, vy0, x0, y0, tx, ty, dt, tol=0.01, it=100):
    g, k_m = 0.0, 0.0
    h = 1e-5
    for _ in range(it):
        ex, ey = ball_motion(vx0, vy0, g, k_m, x0, y0, tx, ty, dt)
        if abs(ex) < tol and abs(ey) < tol:
            return g, k_m
        exg, eyg = ball_motion(vx0, vy0, g + h, k_m, x0, y0, tx, ty, dt)
        exk, eyk = ball_motion(vx0, vy0, g, k_m + h, x0, y0, tx, ty, dt)
        dExdg, dEydg = (exg - ex)/h, (eyg - ey)/h
        dExdk, dEydk  = (exk - ex)/h, (eyk - ey)/h
        J = np.array([[dExdg, dExdk],[dEydg, dEydk]])
        F = np.array([ex, ey])
        try:
            dp = np.linalg.solve(J, -F)
        except:
            break
        g += dp[0]
        k_m += dp[1]
    return g, k_m

def ball_throw_trajectory(vx0, vy0, x0, y0, t, dt, g, k_m):
    steps = int(round(t/dt))
    vx, vy = vx0, vy0
    x, y = x0, y0
    for _ in range(steps):
        ax = -k_m * vx * np.sqrt(vx*vx + vy*vy)
        ay = -g - k_m * vy * np.sqrt(vx*vx + vy*vy)
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
    return x, y

def shoot_v_for_T(xs, ys, xe, ye, T, dt, g, k_m, vxg=0.0, vyg=-0.0, tol=1e-2, mi=50):
    vx0, vy0 = vxg, vyg
    h = 1e-5
    for _ in range(mi):
        xf, yf = ball_throw_trajectory(vx0, vy0, xs, ys, T, dt, g, k_m)
        fx, fy = xe - xf, ye - yf
        if abs(fx)<tol and abs(fy)<tol:
            return vx0, vy0
        xfg, yfg = ball_throw_trajectory(vx0+h, vy0, xs, ys, T, dt, g, k_m)
        xfh, yfh = ball_throw_trajectory(vx0, vy0+h, xs, ys, T, dt, g, k_m)
        fxg, fyg = xe - xfg, ye - yfg 
        fxh, fyh = xe - xfh, ye - yfh
        dfxdvx, dfydvx = (fxg - fx)/h, (fyg - fy)/h
        dfxdvy, dfydvy = (fxh - fx)/h, (fyh - fy)/h
        J = np.array([[dfxdvx, dfxdvy],[dfydvx, dfydvy]])
        F = np.array([fx, fy])
        try:
            dV = np.linalg.solve(J, -F)
        except:
            break
        vx0 += dV[0]
        vy0 += dV[1]
    return vx0, vy0

video_path = './content/slow_throw_and_fall.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
positions = []
vels_x = []
vels_y = []
prev_c = None
frame_count = 0

while cap.isOpened() and frame_count < 70:
    ret, frame = cap.read()
    if not ret or frame is None:
        break
    frame_count += 1
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    m = back_sub.apply(g)
    _, th = cv2.threshold(m, 50, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    c1 = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
    c2 = cv2.morphologyEx(c1, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(c2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours and frame_count > 10:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(largest_contour)
        if 5 < r < 100:
            c = (int(x), int(y))
            cv2.circle(frame, c, int(r), (0, 0, 255), 2)
            positions.append(c)
            if prev_c is not None:
                dtf = 1/fps
                dx, dy = c[0] - prev_c[0], c[1] - prev_c[1]
                vx, vy = dx/dtf, dy/dtf
                vels_x.append(vx)
                vels_y.append(vy)
            prev_c = c
    cv2.imshow('Single Moving Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

if vels_x and vels_y:
    dt = 1/fps
    x0, y0 = positions[0]
    tx, ty = positions[-1]
    vx0, vy0 = vels_x[0], vels_y[0]
    g, km = shooting_method(vx0, vy0, x0, y0, tx, ty, dt)
else:
    g, km = 0, 0

if positions and vels_x and vels_y:
    vxe, vye = vels_x[-1], vels_y[-1]
    px, py = positions[-1]
    ex_pos = []
    for _ in range(50):
        ax = -km * vxe * np.sqrt(vxe*vxe + vye*vye)
        ay = -g - km * vye * np.sqrt(vxe*vxe + vye*vye)
        vxe += ax * dt
        vye += ay * dt
        px += vxe * dt
        py += vye * dt
        ex_pos.append((px, py))
    ex_arr = np.array(ex_pos)
    pos_arr = np.array(positions)
    full_trajectory = np.concatenate((pos_arr, ex_arr), axis=0)
else:
    full_trajectory = np.array([])

T = 3.5
xt, yt = None, None
if len(full_trajectory) > 0:
    idxT = int(T/dt)
    if idxT < len(full_trajectory):
        xt, yt = full_trajectory[idxT]
    else:
        xt, yt = full_trajectory[-1]

second_arr = np.array([])
if xt is not None and yt is not None:
    shooter_x, shooter_y = 900, 400
    gvx, gvy = 0.0, 0.0
    ivx, ivy = shoot_v_for_T(shooter_x, shooter_y, xt, yt, T, dt, g, km, gvx, gvy)
    vx2, vy2 = ivx, ivy
    x2, y2 = shooter_x, shooter_y
    sp = []
    stp = int(round(T/dt))
    for _ in range(stp):
        ss2 = np.sqrt(vx2*vx2 + vy2*vy2)
        ax2 = -km * vx2 * ss2
        ay2 = -g - km * vy2 * ss2
        vx2 += ax2*dt
        vy2 += ay2*dt
        x2 += vx2*dt
        y2 += vy2*dt
        sp.append((x2, y2))
    second_arr = np.array(sp)

fig, ax = plt.subplots()
ax.invert_yaxis()
line1, = ax.plot([], [], 'b-o')
line2, = ax.plot([], [],  marker='o', color='purple')
pointT, = ax.plot([], [], marker = 'o', color='black', markersize=10)
if len(full_trajectory):
    ax.set_xlim(full_trajectory[:,0].min()-50, full_trajectory[:,0].max()+50)
    ax.set_ylim(full_trajectory[:,1].max()+50, full_trajectory[:,1].min()-50)
    
frames = max(len(full_trajectory), len(second_arr))
cx, cy = [], []
cx2, cy2 = [], []
txl, tyl = [xt], [yt]

def init():
    line1.set_data([], [])
    line2.set_data([], [])
    pointT.set_data(txl, tyl)
    return line1, line2, pointT

def update(frame):
    if frame < len(full_trajectory):
        cx.append(full_trajectory[frame,0])
        cy.append(full_trajectory[frame,1])
    if frame < len(second_arr):
        cx2.append(second_arr[frame,0])
        cy2.append(second_arr[frame,1])
    else:
        plt.close(fig)
    line1.set_data(cx, cy)
    line2.set_data(cx2, cy2)
    return line1, line2, pointT

ani = FuncAnimation(fig, update, frames=frames, init_func=init, interval=40, blit=True)
plt.show()