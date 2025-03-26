import meep as mp
import matplotlib.pyplot as plt
import numpy as np


def plot_epsilon(sim, filename="geometrycenter.png"):
    eps_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Dielectric)
    plt.imshow(eps_data.T, interpolation='spline36', cmap='binary', origin='lower')
    plt.colorbar(label="Epsilon")
    plt.axis('off')
    plt.title("Buraco Central")
    plt.savefig(filename)
    plt.close()


def main():
    resolution = 20  # pixels por unidade de comprimento (μm por exemplo)
    eps = 13  # dielétrico
    eps1 = 1 # dielétrico do buraco
    r = 0.1 # raio do buraco
    a = 1.0  # distância entre buracos

    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, mp.inf), material=mp.Medium(epsilon=eps)),
        mp.Cylinder(r, material=mp.Medium(epsilon=eps1))  # buraco central
    ]

    cell = mp.Vector3(a, a, 0)

    fcen = 0.25  # frequencia central do pulso
    df = 1.5  # largura do pulso

    s = mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Hz,
        center=mp.Vector3(0.2),
    )

    sym = mp.Mirror(direction=mp.Y, phase=-1)

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=[s],
        symmetries=[sym],
        k_point=mp.Vector3(0, 0, 0),
        resolution=resolution,
    )


    k_interp = 200  # Número de pontos kx para interpolar
    k_points = mp.interpolate(k_interp, [mp.Vector3(), mp.Vector3(0.5)])

    all_freqs = sim.run_k_points(300, k_points)

    kx_values = [k.x for k in k_points]

    max_bands = max(len(freqs) for freqs in all_freqs)

    band_data = []
    for freqs in all_freqs:
        real_freqs = [freq.real for freq in freqs]  # Extrair partes reais
        if len(real_freqs) < max_bands:
            real_freqs.extend([np.nan] * (max_bands - len(real_freqs)))  # Preencher com NaN
        band_data.append(real_freqs)

    band_data = np.array(band_data)

    plt.figure(figsize=(8, 6))
    for band in range(band_data.shape[1]):
        plt.plot(kx_values, band_data[:, band], 'ko', markersize=3)

    c = 1  # Velocidade da luz normalizada
    light_cone = c * np.abs(kx_values)
    plt.fill_between(kx_values, light_cone, max(np.nanmax(band_data, axis=1)), color='gray', alpha=0.3, label='Cone de Luz')

    plt.xlabel(r'$k_x$ ($2\pi/a$)', fontsize=14)
    plt.ylabel(r'$\omega$ ($2\pi c/a$)', fontsize=14)
    plt.title('Gráfico 2D da rede infinita', fontsize=16)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('bandas_pontos.png', dpi=300, bbox_inches='tight')
    plt.show()

    plot_epsilon(sim)


if __name__ == "__main__":
    main()