import meep as mp
from meep import mpb
import numpy as np

num_bands = 12

k_points = [mp.Vector3(),          # Gamma
            mp.Vector3(0.5),       # X
            mp.Vector3(0, 0.5),  # M
            mp.Vector3()]          # Gamma

k_points = mp.interpolate(1 , k_points)

theta = np.radians(20)
thetb = np.radians(-20)

# Define os vetores de rotação (e1, e2, e3)
e1 = mp.Vector3(np.cos(theta), np.sin(theta), 0)  # Direção do eixo principal
e2 = mp.Vector3(-np.sin(theta), np.cos(theta), 0)  # Direção perpendicular
e3 = mp.Vector3(0, 0, 1)  # Eixo z (2D)
e4 = mp.Vector3(np.cos(thetb), np.sin(thetb), 0)  # Direção do eixo principal
e5 = mp.Vector3(-np.sin(thetb), np.cos(thetb), 0)  # Direção perpendicular

geometry = [
    mp.Block(
        size=mp.Vector3(0.6, 0.05, mp.inf),  # Tamanho (largura, altura, profundidade)
        center=mp.Vector3(0, 0),             # Centro do retângulo
        e1=e1, e2=e2, e3=e3,                 # Rotação
        material=mp.air                      # Cavidade de ar (ε=1)
    ),
    mp.Block(
        size=mp.Vector3(0.6, 0.05, mp.inf),  # Tamanho (largura, altura, profundidade)
        center=mp.Vector3(0, 0),             # Centro do retângulo
        e1=e4, e2=e5, e3=e3,                 # Rotação
        material=mp.air                      # Cavidade de ar (ε=1)
    )
]


geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
resolution = 32
background_material = mp.Medium(epsilon=13)

ms = mpb.ModeSolver(num_bands=num_bands,
                    k_points=k_points,
                    geometry=geometry,
                    geometry_lattice=geometry_lattice,
                    default_material=background_material,
                    resolution=resolution)


ms.run_tm()

import matplotlib.pyplot as plt

# Extrair frequências
freqs = ms.all_freqs

# Plotar
plt.figure()
for band in range(num_bands):
    plt.plot([f[band] for f in freqs], '-', color='black')
plt.xlabel('Vetores k')
plt.ylabel('Frequência (ωa/2πc)')
plt.xticks([0, len(k_points)//3, 2*len(k_points)//3, len(k_points)-1], ['Γ', 'X', 'M', 'Γ'])
plt.title('Bandas')
plt.grid(True)
plt.show()

sim = mp.Simulation(
    cell_size=mp.Vector3(1, 1),
    geometry=geometry,
    default_material=background_material,
    resolution=resolution
)
sim.init_sim()
eps_data = sim.get_epsilon()

import matplotlib.pyplot as plt
plt.imshow(eps_data.T, cmap='binary', interpolation='spline36')
plt.colorbar()
plt.title("Rotação de 20 graus")
plt.show()