import meep as mp
from meep import mpb
import numpy as np
import matplotlib.pyplot as plt
from pylab import plot,show

num_bands = 5

# Criar uma malha de pontos k no espaço recíproco
nk = 5  # número de pontos em cada direção
kx_vals = np.linspace(-0.5, 0.5, nk)
ky_vals = np.linspace(-0.5, 0.5, nk)
KX, KY = np.meshgrid(kx_vals, ky_vals)

theta = np.radians(20)
thetb = np.radians(-20)

# Define os vetores de rotação (e1, e2, e3)
e1 = mp.Vector3(np.cos(theta), np.sin(theta), 0)
e2 = mp.Vector3(-np.sin(theta), np.cos(theta), 0)
e3 = mp.Vector3(0, 0, 1)
e4 = mp.Vector3(np.cos(thetb), np.sin(thetb), 0)
e5 = mp.Vector3(-np.sin(thetb), np.cos(thetb), 0)

geometry = [
    mp.Block(
        size=mp.Vector3(0.6, 0.05, mp.inf),
        center=mp.Vector3(0, 0),
        e1=e1, e2=e2, e3=e3,
        material=mp.air
    ),
    mp.Block(
        size=mp.Vector3(0.6, 0.05, mp.inf),
        center=mp.Vector3(0, 0),
        e1=e4, e2=e5, e3=e3,
        material=mp.air
    )
]

geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
resolution = 32
background_material = mp.Medium(epsilon=13)

# Calcular frequências para cada ponto (kx, ky)
frequencies = np.zeros((nk, nk, num_bands))

for i, kx in enumerate(kx_vals):
    for j, ky in enumerate(ky_vals):
        k_point = [mp.Vector3(kx, ky)]

        ms = mpb.ModeSolver(num_bands=num_bands,
                            k_points=k_point,
                            geometry=geometry,
                            geometry_lattice=geometry_lattice,
                            default_material=background_material,
                            resolution=resolution)

        ms.run_tm()
        freqs = ms.all_freqs

        for band in range(num_bands):
            if band < len(freqs[0]):
                frequencies[j, i, band] = freqs[0][band]
            else:
                frequencies[j, i, band] = 0

# Plotar curvas de nível para cada banda
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for band in range(min(num_bands, 12)):  # Plotar até 12 bandas
    ax = axes[band]

    # Criar contorno
    contour = ax.contourf(KX, KY, frequencies[:, :, band], levels=20, cmap='viridis')
    ax.contour(KX, KY, frequencies[:, :, band], levels=10, colors='black', linewidths=0.5, alpha=0.5)

    # Adicionar barra de cores
    plt.colorbar(contour, ax=ax, shrink=0.8)

    # Marcar pontos simétricos importantes
    ax.plot(0, 0, 'ro', markersize=8, label='Γ')  # Gamma
    ax.plot(0.5, 0, 'go', markersize=6, label='X')  # X
    ax.plot(0, 0.5, 'bo', markersize=6, label='M')  # M
    ax.plot(0.5, 0.5, 'mo', markersize=6, label='R')  # R

    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_title(f'Banda {band + 1}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend()

plt.tight_layout()
plt.suptitle('Curvas de Nível das Frequências em função de kx e ky', y=1.02, fontsize=16)
plt.show()

# Plotar a estrutura geométrica para referência
sim = mp.Simulation(
    cell_size=mp.Vector3(1, 1),
    geometry=geometry,
    default_material=background_material,
    resolution=resolution
)
sim.init_sim()
eps_data = sim.get_epsilon()

plt.figure(figsize=(10, 8))
plt.imshow(eps_data.T, cmap='binary', interpolation='spline36',
           extent=[-0.5, 0.5, -0.5, 0.5], origin='lower')
plt.colorbar(label='Permissividade (ε)')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Estrutura com Rotação de 20 graus")
plt.grid(True, alpha=0.3)
plt.show()

# Plot adicional: Superfície 3D para a primeira banda
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(KX, KY, frequencies[:, :, 0], cmap='viridis',
                       alpha=0.8, linewidth=0, antialiased=True)

ax.set_xlabel('kx')
ax.set_ylabel('ky')
ax.set_zlabel('Frequência (ωa/2πc)')
ax.set_title('Superfície 3D - Banda 1')

plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
plt.show()

