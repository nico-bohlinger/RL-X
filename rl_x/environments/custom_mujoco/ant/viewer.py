import glfw
import mujoco
import time
from itertools import cycle


class MujocoViewer:
    def __init__(self, model, dt):
        self.model = model
        self.dt = dt

        self.button_left, self.button_right, self.button_middle = False, False, False
        self.last_x, self.last_y = 0, 0
        self.frames = 0
        self.loop_count = 0
        self.target_render_time = 1 / 60.
        self.time_per_render = self.target_render_time
        self.run_speed_factor = 1.0
        self.paused = False
        self.hide_menu = False
        self.overlay = {}
        self.font_scale = 100

        glfw.init()
        glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)

        primary_monitor = glfw.get_primary_monitor()
        video_mode = glfw.get_video_mode(primary_monitor)
        window_width, window_height = video_mode.size
        self.window = glfw.create_window(width=window_width, height=window_height, title="MuJoCo", monitor=None, share=None)
        glfw.make_context_current(self.window)

        glfw.set_mouse_button_callback(self.window, self.mouse_button)
        glfw.set_cursor_pos_callback(self.window, self.mouse_move)
        glfw.set_key_callback(self.window, self.keyboard)
        glfw.set_scroll_callback(self.window, self.scroll)

        self.scene = mujoco.MjvScene(model, 1000)
        self.scene_option = mujoco.MjvOption()

        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, self.camera)
        self.all_camera_modes = ("static", "follow")
        self.camera_mode_iter = cycle(self.all_camera_modes)
        self.camera_mode = next(self.camera_mode_iter)
        self.camera_mode_target = self.camera_mode
        self.set_camera()

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale(self.font_scale))

    def mouse_button(self, window, button, act, mods):
        self.button_left = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self.button_right = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self.button_middle = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS

        self.last_x, self.last_y = glfw.get_cursor_pos(self.window)

    def mouse_move(self, window, x_pos, y_pos):
        if not self.button_left and not self.button_right and not self.button_middle:
            return

        dx = x_pos - self.last_x
        dy = y_pos - self.last_y
        self.last_x, self.last_y = x_pos, y_pos

        width, height = glfw.get_window_size(self.window)

        mod_shift = glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(self.window,
                                                                                                  glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

        if self.button_right:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(self.model, action, dx / width, dy / height, self.scene, self.camera)

    def keyboard(self, window, key, scancode, act, mods):
        if act != glfw.RELEASE:
            return
        elif key == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif key == glfw.KEY_H:
            self.hide_menu = not self.hide_menu
        elif key == glfw.KEY_TAB:
            self.camera_mode_target = next(self.camera_mode_iter)
        elif key == glfw.KEY_S:
            self.run_speed_factor /= 2.0
        elif key == glfw.KEY_F:
            self.run_speed_factor *= 2.0

    def scroll(self, window, x_offset, y_offset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, 0.05 * y_offset, self.scene, self.camera)

    def render(self, data):
        def render_inner_loop(self):
            self.create_overlay()
            render_start = time.time()

            mujoco.mjv_updateScene(self.model, data, self.scene_option, None, self.camera,
                                   mujoco.mjtCatBit.mjCAT_ALL,
                                   self.scene)
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(self.window)
            mujoco.mjr_render(self.viewport, self.scene, self.context)

            if not self.hide_menu:
                for gridpos, [t1, t2] in self.overlay.items():
                    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_SHADOW, gridpos, self.viewport, t1, t2, self.context)

            glfw.swap_buffers(self.window)
            glfw.poll_events()

            self.frames += 1
            self.overlay.clear()

            if glfw.window_should_close(self.window):
                self.stop()
                exit(0)

            time.sleep(max(0, self.target_render_time - (time.time() - render_start) - 0.0002))
            self.time_per_render = time.time() - render_start

        if self.paused:
            while self.paused:
                render_inner_loop(self)

        self.loop_count += self.dt / (self.time_per_render * self.run_speed_factor)
        while self.loop_count > 0:
            render_inner_loop(self)
            self.set_camera()
            self.loop_count -= 1

    def stop(self):
        glfw.destroy_window(self.window)

    def close(self):
        glfw.set_window_should_close(self.window, True)

    def create_overlay(self):
        topleft = mujoco.mjtGridPos.mjGRID_TOPLEFT
        bottomright = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT

        self.overlay[bottomright] = ["Framerate:", str(int(1/self.time_per_render * self.run_speed_factor))]
        self.overlay[topleft] = ["", ""]
        self.overlay[topleft][0] += "Press SPACE to pause.\n"
        self.overlay[topleft][1] += "\n"
        self.overlay[topleft][0] += "Press H to hide the menu.\n"
        self.overlay[topleft][1] += "\n"
        self.overlay[topleft][0] += "Press TAB to switch cameras.\n"
        self.overlay[topleft][1] += "\n"
        self.overlay[topleft][0] += "Camera mode:\n"
        self.overlay[topleft][1] += self.camera_mode+"\n"
        self.overlay[topleft][0] += "Run speed = %.3f x real time" % self.run_speed_factor
        self.overlay[topleft][1] += "[S]lower, [F]aster"

    def set_camera(self):
        if self.camera_mode_target == "static" and self.camera_mode != "static":
                self.camera.fixedcamid = 0
                self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.camera.trackbodyid = -1
                self.camera.distance = 15.0
                self.camera.elevation = -45.0
                self.camera.azimuth = 90.0
        if self.camera_mode_target == "follow" and self.camera_mode != "follow":
                self.camera.fixedcamid = -1
                self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.camera.trackbodyid = 0
                self.camera.distance = 3.5
                self.camera.elevation = 0.0
                self.camera.azimuth = 90.0
        self.camera_mode = self.camera_mode_target
