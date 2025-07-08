import meep as mp
from meep import mpb

num_bands = 12

k_points = [mp.Vector3(),          # Gamma
            mp.Vector3(0.5),       # X
            mp.Vector3(0, 0.5),  # M
            mp.Vector3()]          # Gamma

k_points = mp.interpolate(300, k_points)

geometry = [mp.Cylinder(0.5, material=mp.Medium(epsilon=1))]
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