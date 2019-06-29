import cv2
import environments.path_planners as path_planners

def imshow(img, name, scale):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, *scale)
    cv2.imshow(name, img)
    cv2.waitKey(0)

def run_path_planner_test():
    print('PATH PLANNING TEST')
    env_map_path = 'environments/data/env_map.png'
    env_map = cv2.cvtColor(cv2.imread(env_map_path), cv2.COLOR_BGR2RGB)
    h, w, c = env_map.shape
    path = path_planners.run_planning_task(env_map, sample=False)
    img = env_map.copy()
    for x, y in path:
        img[y, x] = (0, 255, 0)
    imshow(img, 'path', (w * 50, h * 50))
    path = path_planners.run_planning_task(env_map, sample=True)
    img = env_map.copy()
    for x, y in path:
        img[y, x] = (0, 255, 0)
    imshow(img, 'path', (w * 50, h * 50))
