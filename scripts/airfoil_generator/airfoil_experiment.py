import numpy as np
from PIL import Image
IMAGE_X = 6700
IMAGE_Y = 5000
AIRFOIL = "./naca4412.dat"
AIRFOIL_LEN = 5000
ANGLE_DEG = 15.0

ACCURACY = 3


class Airfoil:
    def __init__(self, filename):
        self.upper_arr = []
        self.lower_arr = []
        with open(filename, 'r') as f:
            self.name = f.readline()

            upper = True
            for line in f:
                x, y = line.split()
                x, y = float(x), float(y)
                if x == 0.0:
                    upper = False
                    self.upper_arr.append(np.array([y, x]))
                    self.lower_arr.append(np.array([y, x]))
                elif upper:
                    self.upper_arr.append(np.array([y, x]))
                else:
                    self.lower_arr.append(np.array([y, x]))
        self.upper_arr.reverse()

    def __interpolate(self, x, arr: list[np.ndarray]):
        if x <= 0:
            return arr[0][0]
        if x >= 1.0:
            return arr[-1][0]

        lower, upper = 0, 0
        for i, x_arr in enumerate(arr):
            if x < x_arr[1]:
                lower = i - 1
                upper = i
                break
        return np.interp(x, [arr[lower][1], arr[upper][1]], [arr[lower][0], arr[upper][0]])

    def y_lower(self, x):
        return self.__interpolate(x, self.lower_arr)

    def y_upper(self, x):
        return self.__interpolate(x, self.upper_arr)


def get_rot_mat(angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R


def main():
    airfoil = Airfoil(AIRFOIL)
    frame = np.zeros((IMAGE_Y, IMAGE_X), dtype=np.uint8)

    rot_mat = get_rot_mat(ANGLE_DEG)
    offset = np.array([IMAGE_Y/2 + np.sin(np.radians(ANGLE_DEG))/2
                      * AIRFOIL_LEN, IMAGE_X/2 - AIRFOIL_LEN/2])

    for x in range(ACCURACY*AIRFOIL_LEN):
        x_frac = x / (ACCURACY*AIRFOIL_LEN)
        vec = np.array([airfoil.y_lower(x_frac), x_frac])
        vec = vec * AIRFOIL_LEN
        vec = np.matmul(rot_mat, vec) + offset
        frame[int(vec[0]):, int(vec[1])] = 255
    for x in range(ACCURACY*AIRFOIL_LEN):
        x_frac = x / (ACCURACY*AIRFOIL_LEN)
        vec = np.array([airfoil.y_upper(x_frac), x_frac])
        vec = vec * AIRFOIL_LEN
        vec = np.matmul(rot_mat, vec) + offset
        frame[int(vec[0]):, int(vec[1])] = 0

    frame = np.flip(frame, axis=0)
    Image.fromarray(frame).save("out.png")

    with open("out.txt", 'w') as f:
        for y in range(frame.shape[0]):
            for x in range(frame.shape[1]):
                if frame[y, x] == 255:
                    f.write('1 ')
                else:
                    f.write('0 ')


if __name__ == "__main__":
    main()
