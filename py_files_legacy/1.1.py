import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

L=1
c=1
dt=0.001
N=200
dx=L/N
x=np.linspace(0, L, N+1)


def wav(psi, T, dt=dt, dx=dx, c=c):
    r= c* (dt/dx)
    r2=r**2
    n_steps = int(T/dt)
    
    psi_prev = np.copy(psi)
    psi_curr = np.copy(psi)
    psi_next = np.zeros_like(psi)
    
    for i in range(1,N):
        psi_next[i]= psi_curr[i]+0.5*r2*(psi_curr[i+1]+psi_curr[i-1]-2*psi_curr[i])
    psi_next[0]=0
    psi_next[N]=0
    times = [0]
    states=[np.copy(psi)]
    psi_prev=np.copy(psi_curr)
    psi_curr=np.copy(psi_next)
    times.append(dt)
    states.append(np.copy(psi_curr))
        
        
    for s in range(2, n_steps+1):
        for i in range(1,N):
            psi_next[i]=(r2*(psi_curr[i+1]+psi_curr[i-1]-2*psi_curr[i])+2*psi_curr[i]-psi_prev[i])
        psi_next[0]=0
        psi_next[N]=0
        psi_prev[:]=psi_curr
        psi_curr[:]=psi_next
        times.append(s*dt)
        states.append(np.copy(psi_curr))
                
    return times, states


#(a)
psi_a = np.sin(2*np.pi*x)
psi_a[0] = 0
psi_a[N] = 0

#(b)
psi_b = np.sin(5*np.pi*x)
psi_b[0]=0
psi_b[N]=0

#(c)
psi_c = np.zeros(N+1)
for i in range(N+1):
    if 1/5 < x[i] < 2/5:
        psi_c[i]= np.sin(5*np.pi*x[i])


T_end = 2

times_a, states_a = wav(psi_a, T_end)    
times_b, states_b = wav(psi_b, T_end)   
times_c, states_c = wav(psi_c, T_end) 

#####################################

snap_times = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] 
colors = plt.cm.viridis(np.linspace(0, 1, len(snap_times)))

output_dir = "outputs_1.1"
os.makedirs(output_dir, exist_ok=True)

cases = [
    (times_a, states_a, r"(i) $\Psi(x,0) = \sin(2\pi x)$", "case_a"),
    (times_b, states_b, r"(ii) $\Psi(x,0) = \sin(5\pi x)$", "case_b"),
    (times_c, states_c, r"(iii) $\Psi(x,0) = \sin(5\pi x)$ on $(1/5,2/5)$", "case_c"),
]

for t_list, s_list, title, name in cases:
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, t_snap in enumerate(snap_times):
        idx = min(range(len(t_list)), key=lambda k: abs(t_list[k]-t_snap))
        ax.plot(x, s_list[idx], color=colors[j], label=f"t={t_list[idx]:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Psi(x,t)$")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, L)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_snap.png"), dpi=150)
    # plt.show()

for t_list, s_list, title, name in cases:
    frame_dir = os.path.join(output_dir, f"{name}_frames")
    os.makedirs(frame_dir, exist_ok=True)
    
    ymax = max(np.max(np.abs(s)) for s in s_list) * 1.1
    
    skip = 20
    frame_num = 0
    for n in range(0, len(s_list), skip):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, s_list[n], 'b-')
        ax.set_xlim(0, L)
        ax.set_ylim(-ymax, ymax)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$\Psi(x,t)$")
        ax.set_title(f"{title},  t = {t_list[n]:.3f}")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(frame_dir, f"frame_{frame_num:04d}.png"), dpi=100)
        plt.close()
        frame_num += 1
    
    out_path = os.path.join(output_dir, f"{name}_anim.gif")
    palette = os.path.join(frame_dir, "palette.png")
    os.system(f"ffmpeg -y -framerate 30 -i {frame_dir}/frame_%04d.png -vf palettegen {palette}")
    os.system(f"ffmpeg -y -framerate 30 -i {frame_dir}/frame_%04d.png -i {palette} -lavfi paletteuse {out_path}")