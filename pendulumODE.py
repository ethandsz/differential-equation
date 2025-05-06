import numpy as np 
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def sys_ode(y, t, u, l, g):
    theta, theta_d = y
    thetha_dd = -u * theta_d - (g/l) * np.sin(theta)
    return theta_d, thetha_dd
def main():
    u = 0.1
    g = 9.81
    l = 1.0 

    N0 = [1.1, 0]
    time_steps = np.linspace(0,50,5000)


    sol = odeint(sys_ode, N0, time_steps, args=(u,l,g))

    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True)

    # Pendulum rod and bob

    line, = ax.plot([], 'o-', lw=2, markersize=4)
    # Initialization function
    def init():
        line.set_data([], [])
        return line,

    # Animation update function
    def update(frame):
        x = [sol[frame,0]]
        y = [sol[frame,1]]
        line.set_data(x, y)
        return line,

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(time_steps),
                                  init_func=init, blit=True, interval=10)

    plt.title("Damped Pendulum Animation")
    plt.show()




if __name__ == "__main__":
    main()
