import time
import requests
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from clover import srv
from std_srvs.srv import Trigger
from clover.srv import SetLEDEffect
import math

rospy.init_node('flight')

get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
navigate = rospy.ServiceProxy('navigate', srv.Navigate)
set_effect = rospy.ServiceProxy('led/set_effect', SetLEDEffect)
land = rospy.ServiceProxy('land', Trigger)

def navigate_wait(x=0, y=0, z=0, yaw=float('nan'), speed=0.5, frame_id='', auto_arm=False, tolerance=0.2):
    print(f"Лечу в точку: x={x}, y={y}, z={z}, frame_id={frame_id}")
    navigate(x=x, y=y, z=z, yaw=yaw, speed=speed, frame_id=frame_id, auto_arm=auto_arm)
    while not rospy.is_shutdown():
        telem = get_telemetry(frame_id='navigate_target')
        distance = math.sqrt(telem.x ** 2 + telem.y ** 2 + telem.z ** 2)
        if distance < tolerance:
            print("Точка достигнута!")
            break
        rospy.sleep(0.2)

def check_altitude(min_altitude=1.0):
    telem = get_telemetry()
    if telem.z < min_altitude:
        print(f"Высота слишком мала: {telem.z} м, поднимаюсь!")
        set_effect(r=255, g=165, b=0)
        navigate_wait(z=min_altitude, frame_id='body')
        return False
    return True

SERVER_IP = "192.168.11.112"
SERVER_PORT = 8000
TIMEOUT_TX = 5
TOPIC = "/main_camera/image_raw"
bridge = CvBridge()

def capture_frame() -> bytes:
    print("Захватываю кадр с камеры...")
    set_effect(effect='flash', r=255, g=255, b=255)
    rospy.sleep(0.3)
    msg = rospy.wait_for_message(TOPIC, Image, timeout=5.0)
    img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
    _, buf = cv2.imencode(".jpg", img)
    print("Кадр успешно захвачен!")
    return buf.tobytes()

def send_frame(jpeg: bytes):
    url = f"http://{SERVER_IP}:{SERVER_PORT}/upload"
    files = {"image": ("frame.jpg", jpeg, "image/jpeg")}
    try:
        print("Отправляю кадр на сервер...")
        set_effect(r=0, g=0, b=255)
        r = requests.post(url, files=files, timeout=TIMEOUT_TX)
        rospy.loginfo(f"Отправлено {len(jpeg)} байт → код ответа: {r.status_code}")
        set_effect(r=0, g=0, b=0)
        return r.text
    except requests.exceptions.RequestException as e:
        rospy.logerr(f"Ошибка отправки: {e}")
        set_effect(effect='blink', r=255, g=0, b=0)
        return None
movements = [
    (0, -1),
    (0, -1),
    (1, 0),
    (1, 0),
    (0, 1),
    (0, 1),
    (-1, 0),
    (-1, 0)
]

print("Начинаю полет!")
print("Взлет - фиолетовый LED")
set_effect(r=128, g=0, b=128)
navigate_wait(z=1.5, frame_id='body', auto_arm=True)
rospy.sleep(2)

path = []

for step, (dx, dy) in enumerate(movements, 1):
    print(f"Шаг {step}: Движение на dx={dx}, dy={dy}")
    set_effect(r=0, g=255, b=0)
    navigate_wait(x=dx, y=dy, z=0, frame_id='body')
    path.append((dx, dy))
    if not check_altitude():
        print("Корректировка высоты выполнена")
    print("Ожидание перед фото - голубой LED")
    set_effect(r=0, g=191, b=255)
    rospy.sleep(5)
    jpeg = capture_frame()
    response = send_frame(jpeg)
    print(f"Ответ сервера: {response}")
    if response == "недорога":
        print("Получено 'недорога' - возврат на старт, желтый LED")
        set_effect(r=255, g=255, b=0)
        for rev_dx, rev_dy in reversed(path):
            print(f"Возврат: dx={-rev_dx}, dy={-rev_dy}")
            navigate_wait(x=-rev_dx, y=-rev_dy, z=0, frame_id='body')
        print("Посадка после возврата - красный LED")
        set_effect(r=255, g=0, b=0)
        land()
        break
    elif response == "трещина":
        print("Обнаружена 'трещина' - красное мигание")
        set_effect(effect='blink', r=255, g=0, b=0)
        rospy.sleep(3)
        set_effect(r=0, g=0, b=0)
    elif response == "яма":
        print("Обнаружена 'яма' - фиолетовое мигание")
        set_effect(effect='blink', r=128, g=0, b=128)
        rospy.sleep(3)
        set_effect(r=0, g=0, b=0)
    elif response == "нормальная дорога":
        print("Дорога нормальная - зеленый LED")
        set_effect(r=0, g=255, b=0)
        rospy.sleep(1)
    else:
        print("Неизвестный ответ или ошибка - оранжевый LED")
        set_effect(r=255, g=165, b=0)
        rospy.sleep(2)

else:
    print("Маршрут завершен успешно - зеленое мигание!")
    set_effect(effect='blink', r=0, g=255, b=0)
    rospy.sleep(4)
    set_effect(r=0, g=0, b=0)
    print("Посадка после успеха - белый LED")
    set_effect(r=255, g=255, b=255)
    land()

print("Полет завершен!")