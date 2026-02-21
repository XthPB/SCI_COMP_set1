import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os

out_dir = "outputs_1.3"
os.makedirs(out_dir, exist_ok=True)

N=50
eps=1e-5
y=np.linspace(0,1,N+1)

@njit
def jacobi(N, eps):
    c=np.zeros((N+1,N+1))
    c[:,N]=1
    c_n=np.zeros_like(c)
    dels=[]
    k=0
    
    while True:
        for i in range(N):
            ip=(i+1)%N
            im=(i-1)%N
            for j in range(1,N):
                c_n[i,j]=0.25*(c[ip,j]+c[im,j]+c[i,j+1]+c[i,j-1])
                
        c_n[:,0]=0
        c_n[:,N]=1
        
        delta=np.max(np.abs(c_n-c))
        dels.append(delta)
        
        c[:,:]=c_n
        k+=1
        
        if delta<eps:
            break
    return c, dels, k


@njit 
def seidel(N, eps):
    c=np.zeros((N+1,N+1))
    c[:,N]=1
    dels=[]
    k=0
    
    while True:
        delta=0
        
        for i in range(N):
            ip=(i+1)%N
            im=(i-1)%N
            for j in range(1,N):
                old= c[i,j]
                c[i,j]=0.25*(c[ip,j]+c[im,j]+c[i,j+1]+c[i,j-1])
                delta=max(delta, abs(c[i,j]-old))
                
        c[:,0]=0
        c[:,N]=1
        dels.append(delta)
        k+=1
        
        if delta<eps:
            break
        
    return c,dels,k


@njit
def sor(N, eps, omega):
    c=np.zeros((N+1,N+1))
    c[:,N]=1
    dels=[]
    k=0
    
    while True:
        delta=0
        
        for i in range(N):
            ip=(i+1)%N
            im=(i-1)%N
            for j in range(1,N):
                old=c[i,j]
                gs=0.25*(c[ip,j]+c[im,j]+c[i,j+1]+c[i,j-1])
                c[i,j]=(1-omega)*old + omega*gs
                delta=max(delta,abs(c[i,j]-old))
        
        c[:,0]=0
        c[:,N]=1
        
        dels.append(delta)
        k+=1
        
        if delta<eps:
            break
    
    return c,dels,k

########################################

def plot_sol(c, name, fname):
    mid=N//2
    plt.figure(figsize=(6,5))
    plt.plot(y, c[mid, :], 'b-', label='numerical')
    plt.plot(y, y, 'r--', label='c = y')
    plt.xlabel('y')
    plt.ylabel('c')
    plt.title(f"{name} (N={N})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()




def plot_conv(curves, fname):
    plt.figure(figsize=(8,6))
    for lbl, d in curves.items():
        plt.semilogy(range(1, len(d)+1), d, label=lbl)

    plt.xlabel('Iteration k')
    plt.ylabel('δ')
    plt.title('Convergence')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close()



def plot_opt(omegas, iters):
    b_idx=np.argmin(iters)
    b_omega=omegas[b_idx]
    b_iters=iters[b_idx]

    plt.figure(figsize=(7,5))
    plt.plot(omegas, iters, 'bo-', label='iterations')
    plt.plot(b_omega, b_iters, 'ro', markersize=8, label=f'optimal ω = {b_omega:.2f}')
    plt.axvline(b_omega, color='r', linestyle='--')

    plt.annotate(
        f'ω* = {b_omega:.2f}\n{b_iters} iterations',
        xy=(b_omega, b_iters),
        xytext=(b_omega + 0.05, b_iters * 1.1),
        arrowprops=dict(arrowstyle="->")
    )

    plt.xlabel('ω')
    plt.ylabel('Iterations')
    plt.title('SOR: Iterations vs ω')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "part_j.png"), dpi=150)
    plt.close()

    print(f"Optimal ω = {b_omega:.3f}, iterations = {b_iters}")
    
############################################################

def test(c, name):
    mid=N//2
    num = c[mid, :]
    analytic = y

    max_err = np.max(np.abs(num - analytic))
    l2_err = np.sqrt(np.mean((num - analytic)**2))

    print(f"{name}:")
    print(f"  max error = {max_err:.3e}")
    print(f"  L2 error  = {l2_err:.3e}")
    print()
    

############################################################

c_j, d_j, k_j = jacobi(N, eps)
c_g, d_g, k_g = seidel(N, eps)
c_s, d_s, k_s = sor(N, eps, 1.8)

plot_sol(c_j, "Jacobi", "part_h_jacobi.png")
plot_sol(c_g, "Gauss-Seidel", "part_h_gs.png")
plot_sol(c_s, "SOR ω=1.8", "part_h_sor.png")

test(c_j, "Jacobi")
test(c_g, "Gauss-Seidel")
test(c_s, "SOR")

plot_conv({
    f'Jacobi ({k_j})': d_j,
    f'Gauss-Seidel ({k_g})': d_g,
    f'SOR ω=1.8 ({k_s})': d_s
}, "part_i.png")

omegas = np.arange(1.0, 2.0, 0.02)
iters = []

for w in omegas:
    _, _, k = sor(N, eps, w)
    iters.append(k)

plot_opt(omegas, iters)


################################################################

N_vals = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
o_vals = np.arange(1.0, 2.0, 0.02)

b_omega = []
b_iters = []
all_iters = {}

for n in N_vals:
    iters_n = []
    for w in o_vals:
        _, _, k = sor(n, eps, w)
        iters_n.append(k)
    all_iters[n] = iters_n
    b_idx = np.argmin(iters_n)
    b_omega.append(o_vals[b_idx])
    b_iters.append(iters_n[b_idx])


plt.figure(figsize=(8, 6))
for n in N_vals:
    plt.plot(o_vals, all_iters[n], marker='o', ms=3.5, lw=1.8, label=f'N={n}')
plt.xlabel('ω')
plt.ylabel('iterations')
plt.title('SOR iteration count vs ω')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(N_vals, b_omega, marker='o', lw=2.0)
plt.xlabel('N')
plt.ylabel('optimal ω')
plt.title('Optimal ω vs grid size')
plt.grid(alpha=0.35)
plt.tight_layout()
plt.show()

for n, w, k in zip(N_vals, b_omega, b_iters):
    print(f'  N={n:3d}: ω_opt={w:.2f}, iters={k}')