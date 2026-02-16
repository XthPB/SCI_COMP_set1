import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.special import erfc

N=50
D=1
dx=1/N
dt=dx**2/(4*D)*0.9
x=np.linspace(0,1,N+1)
y=np.linspace(0,1,N+1)

c=np.zeros((N+1,N+1))
c[:,N]=1
c[N,:]=0

def bc(c):
    c[:,N]=1
    c[:,0]=0
    c[0,:]=c[N-1,:]
    c[N,:]=c[1,:]
    
def step(c,dt,dx,D):
    c_n = np.copy(c)
    r=D*dt/dx**2
    for i in range(1,N):
        for j in range(1,N):
            c_n[i,j]=c[i,j]+r*(c[i+1,j]+c[i-1,j]+c[i,j+1]+c[i,j-1]-4*c[i,j])
    bc(c_n)
    return c_n
    
    

out_dir = "outputs_1.2"
os.makedirs(out_dir, exist_ok=True)

snap_times = [0, 0.001, 0.01, 0.1, 1.0]
snaps = {}
snaps[0] = np.copy(c)

data_dir = os.path.join(out_dir, "data")
os.makedirs(data_dir, exist_ok=True)

########################################

t=0
T_end=1
snap_idx=1
step_co=0

while t<T_end:
    c=step(c,dt,dx,D)
    t+=dt
    step_co+=1
    
    if step_co%100 ==0:
        np.savetxt(os.path.join(data_dir, f"c_step_{step_co:06d}.npy"), c)

    if snap_idx<len(snap_times) and t>=snap_times[snap_idx]:
        snaps[snap_times[snap_idx]]=np.copy(c)
        snap_idx+=1


X, Y = np.meshgrid(x, y, indexing='ij')

fig, axs = plt.subplots(2, 3, figsize=(14, 8))
axes = axs.flatten()

for k, t_s in enumerate(snap_times):
    if t_s not in snaps:
        continue
    ax = axes[k]
    im = ax.pcolormesh(X, Y, snaps[t_s], vmin=0, vmax=1, cmap='hot')
    ax.set_title(f"t = {t_s}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')
    fig.colorbar(im, ax=ax)

axes[-1].axis('off')
plt.suptitle("2D Diffusion", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "part_f.png"), dpi=150)
# plt.show()

###############################################

c = np.zeros((N+1, N+1))
c[:, N]=1
c[:, 0]=0
bc(c)

frame_dir = os.path.join(out_dir, "diffusion_frames")
os.makedirs(frame_dir, exist_ok=True)

t=0
frame_num = 0
save_freq = 50
step_co = 0
tol = 1e-6

while True:
    c_old = np.copy(c)
    c=step(c, dt, dx, D)
    t+= dt
    step_co+=1

    if step_co % save_freq == 0:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(X, Y, c, vmin=0, vmax=1, cmap='hot')
        ax.set_title(f"t = {t:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(frame_dir, f"frame_{frame_num:04d}.png"), dpi=100)
        plt.close()
        frame_num+=1

    max_change = np.max(np.abs(c - c_old))
    if max_change < tol:
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.pcolormesh(X, Y, c, vmin=0, vmax=1, cmap='hot')
        ax.set_title(f"equilibrium, t = {t:.4f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(frame_dir, f"frame_{frame_num:04d}.png"), dpi=100)
        plt.close()
        frame_num+=1
        break

frames = []
for i in range(frame_num):
    img = Image.open(os.path.join(frame_dir, f"frame_{i:04d}.png"))
    frames.append(img)

out_path = os.path.join(out_dir, "diffusion_anim.gif")
frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=50, loop=0)

###################################################

# verification

def analytical(y_vals, t):
    if t<=0:
        return np.zeros_like(y_vals)
    res=np.zeros_like(y_vals)
    for i in range(50):
        res+=(erfc((1 - y_vals + 2*i) / (2*np.sqrt(D*t)))- erfc((1 + y_vals + 2*i) / (2*np.sqrt(D*t))))
    return res

fig, ax = plt.subplots(figsize=(9, 6))
col_e = plt.cm.viridis(np.linspace(0, 1, len(snap_times)))
mid = N//2

for k, t_snap in enumerate(snap_times):
    if t_snap not in snaps:
        continue
    
    num=snaps[t_snap][mid, :]
    
    if t_snap>0:
        ana=analytical(y, t_snap)
        ax.plot(y, ana, '--', color=col_e[k], linewidth=3, alpha=0.5)
    
    ax.plot(y, num, '-', color=col_e[k], linewidth=1.5, label=f"num t={t_snap}")

ax.set_xlabel("y")
ax.set_ylabel("c(y, t)")
ax.set_title("Numerical (solid) vs Analytical (dashed)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "part_e.png"), dpi=150)