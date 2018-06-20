#!/usr/bin/env python3
import os
import pickle
import socket
import sys
from copy import deepcopy
from threading import Thread

import mujoco_py
import numpy as np
from gym import spaces
from gym import utils
from gym.envs.mujoco import mujoco_env
from scipy import misc


import glfw
import OpenGL.GL as gl



class ParticleEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, dist_epsilon=0.1):
        mujoco_env.MujocoEnv.__init__(self, os.path.join(os.getcwd(), 'environment/assets/particle.xml'), 5)
        utils.EzPickle.__init__(self)
        self.dist_epsilon = dist_epsilon

        #self.viewer = mujoco_py.MjViewer(init_width=100, init_height=100)
        #self.viewer = mujoco_py.MjViewer(self.sim)
        #self.viewer.start()
        #self.viewer.set_model(self.model)

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done
        self.obs_dim = observation.size

        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = spaces.Box(low=low, high=high)

        high = np.inf*np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)

        self.state = None
        self.state_dim = self.state_vector().shape
        high = np.inf*np.ones(self.state_dim)
        low = -high
        self.state_space = spaces.Box(low, high)



        self.seed()


    def step(self, a):
        xpos = self.get_body_com("agent")[0]
        self.do_simulation(a, self.frame_skip)
        self.state = state = self.state_vector()

        notdone = np.isfinite(state).all() \
            and not self._at_goal() and not self._escaped()

        reward = 0

        if self._at_goal():
            reward = 1

        done = not notdone

        image_obs = np.zeros([100, 100, 3])
        if self.viewer is not None:
            image_obs = self._get_image()

        return image_obs, state, reward, done


    def _get_image(self):
        self.render()
        data = self.get_image(self._get_viewer(mode="human"))

        img_data = data[0]
        width = data[1]
        height = data[2]

        tmp = np.fromstring(img_data, dtype=np.uint8)
        image_obs = np.reshape(tmp, [height, width, 3])
        image_obs = np.flipud(image_obs)

        return image_obs


    def _get_state(self, body='agent'):
        state = self.get_body_com(body)[:2]
        return state


    def _at_goal(self):
        try:
            self.goal_state
        except:
            return False

        return np.linalg.norm(self._get_state() - self.goal_state) \
            < self.dist_epsilon


    def _escaped(self):
        state = self._get_state()
        return state.any() > 2


    def _set_goal(self, goal):
        qpos = goal['qpos']
        qvel = goal['qvel']

        self.goal_state = qpos
        self.set_state(qpos, qvel)
        self.step(qvel)

        goal_obs = self._get_image()

        return goal_obs


    def generate_goal(self, e):
        goalposx = np.random.uniform(low=-2.0, high=2.0)
        goalposy = np.random.uniform(low=-2.0, high=2.0)

        goalpos = np.array([goalposx, goalposy])
        goalvel = np.zeros(shape=self.model.nq)

        print('Episode: {}, Goal {}'.format(e, goalpos))

        goal = dict()
        goal['qpos'] = goalpos
        goal['qvel'] = goalvel

        goal_obs = self._set_goal(goal)

        return goal_obs


    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_image()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -90
        self.viewer.cam.distance = 4.85

    def get_image(self,viewer_obj):
        glfw.make_context_current(viewer_obj.window)
        width,height = glfw.get_framebuffer_size(viewer_obj.window)
        gl.glReadBuffer(gl.GL_BACK)
        data = gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        return (data, width, height)



