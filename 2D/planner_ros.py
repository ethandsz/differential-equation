import numpy as np, rospy
from geometry_msgs.msg import Wrench

def send_wrench(a_world, v_world, psi):
    ax, ay = a_world; vx, vy = v_world
    c, s = np.cos(psi), np.sin(psi)
    Rt = np.array([[ c, s],[-s, c]])     # R^T
    a_b = Rt @ np.array([ax, ay])
    v_b = Rt @ np.array([vx, vy])

    Fx = m_x*a_b[0] + d_u*v_b[0]
    Fy = m_y*a_b[1] + d_v*v_b[1]
    Fx, Fy = np.clip([Fx, Fy], -Fmax, Fmax)

    msg = Wrench()
    msg.force.x, msg.force.y, msg.force.z = float(Fx), float(Fy), 0.0
    msg.torque.x = msg.torque.y = 0.0
    msg.torque.z = 0.0  # optional small yaw damping if desired
    pub.publish(msg)

m_x, m_y = 30.0, 30.0         # incl. added mass
d_u, d_v = 20.0, 30.0         # linearized drag
Fmax = 60.0                   # per-axis force limit


pub = rospy.Publisher("/bluerov2/thruster_manager/input", Wrench, queue_size=10)

state_vec = np.load("state_vec.npy")[:,1:]
control_vec = np.load("control_vec.npy")

timesteps = state_vec.shape[1]

for t in range(timesteps):
    send_wrench(control_vec[:,t], state_vec[2:,t], 0.0)
