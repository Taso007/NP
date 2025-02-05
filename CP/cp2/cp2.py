import numpy as np
import cv2
import matplotlib.pyplot as plt


def transform_to_px(g, density, scale, frame_rate):
    scaled_g = g * scale / frame_rate ** 2
    scaled_density = density / scale ** 3
    return scaled_g, scaled_density
    

def calculate_mass(radius, density):
    volume = (4 / 3) * np.pi * (radius ** 3)
    mass = density * volume
    return mass

def detect_single_moving_object(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # I will say that hypothetically 100px is 1m
    # g = 9.81 Gravitational acceleration (m/s^2)
    # density = 0.92 Natural rubber density (kg/m^3)
    g, density = transform_to_px(9.81, 0.92, 100, fps)
    
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

    prev_centroid = None  
    prev_vx = None  
    prev_vy = None
    velocities = []  
    drag_coefficients = []  
    radiuses = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = back_sub.apply(gray)
        _, thresh = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            radiuses.append(radius)

            if radius > 5: 
                center = (int(x), int(y))
                cv2.circle(frame, center, int(radius), (0, 0, 255), 2)

                mass = calculate_mass(radius, density)

                vx, vy = 0, 0

                if prev_centroid is not None:
                    dx = center[1] - prev_centroid[1]
                    dy = center[0] - prev_centroid[0]

                    dt = 1/fps

                    vx = dx / dt  
                    vy = dy / dt  

                    v_final = np.sqrt(vx**2 + vy**2)
                    velocities.append(v_final)  

                    if vx != 0 or vy != 0:
                        if prev_vx is not None and prev_vy is not None:
                            dvx_dt = (vx - prev_vx) / dt
                            dvy_dt = (vy - prev_vy) / dt

                            if vx != 0 and vy != 0:
                                k_vx = (-mass * dvx_dt) / (vx * np.sqrt(vx**2 + vy**2))
                                k_vy = -mass * (dvy_dt + g) / (vy * np.sqrt(vx**2 + vy**2))
                            else:
                                k_vx = 0
                                k_vy = 0

                            k = (k_vx + k_vy) / 2 if (vx != 0 and vy != 0) else 0

                            if k > 0: drag_coefficients.append(k) 

                            print(f"vx: {vx} px/s, vy: {vy} px/s, v_final: {v_final} px/s")

                prev_centroid = center
                prev_vx = vx
                prev_vy = vy

        cv2.imshow('Single Moving Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    avg_radius = np.mean(radiuses)
    mass = calculate_mass(avg_radius, density)

    avg_drag = np.mean(drag_coefficients)

    print(f"\n \n Mass: {mass}kg, Drag coefficient: {avg_drag} \n \n")

    plt.figure(figsize=(10, 5))
    plt.plot(velocities)
    plt.title("Final Velocity over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Velocity (m/s)")
    plt.show()

video_path = './content/cp2_fail.mp4'
detect_single_moving_object(video_path)