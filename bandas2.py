import meep as mp
import matplotlib.pyplot as plt
import numpy as np


def plot_epsilon(sim, filename="geometrycentral.png"):
    eps_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Dielectric)
    plt.imshow(eps_data.T, cmap='binary', origin='lower', interpolation='spline36')
    plt.colorbar(label="Epsilon")
    plt.title("Rede bidimensional infinita")
    plt.savefig(filename)
    plt.close()


def main():
    resolution = 20  # pixels por unidade de comprimento (μm por exemplo)
    eps = 13  # dielétrico (exemplo: GaAs ou Si)
    eps1 = 1
    r = 0.1  # raio do buraco
    a = 1.0  # constante de rede (distância entre buracos)

    # Geometria: bloco infinito de dielétrico + 1 buraco no centro
    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, mp.inf), material=mp.Medium(epsilon=eps)),
        mp.Cylinder(r, material=mp.Medium(epsilon=eps1))  # buraco central
    ]

    cell = mp.Vector3(a, a, 0)

    fcen = 0.25  # pulse center frequency
    df = 1.5  # pulse freq. width: large df = short impulse

    s = mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Hz,
        center=mp.Vector3(0.7),
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

    # Parâmetros para o cálculo das bandas
    k_interp = 200  # Número de pontos kx para interpolar
    k_points = mp.interpolate(k_interp, [mp.Vector3(), mp.Vector3(0.5)])

    # Executar a simulação e coletar as frequências
    all_freqs = sim.run_k_points(300, k_points)

    # Processar os dados de frequências
    kx_values = [k.x for k in k_points]

    # Encontrar o número máximo de bandas em qualquer ponto kx
    max_bands = max(len(freqs) for freqs in all_freqs)

    # Preencher as sublistas com NaN para que todas tenham o mesmo comprimento
    band_data = []
    for freqs in all_freqs:
        real_freqs = [freq.real for freq in freqs]  # Extrair partes reais
        if len(real_freqs) < max_bands:
            real_freqs.extend([np.nan] * (max_bands - len(real_freqs)))  # Preencher com NaN
        band_data.append(real_freqs)

    # Converter para um array NumPy
    band_data = np.array(band_data)

    # Plotar o gráfico ω vs kx com pontos
    plt.figure(figsize=(8, 6))
    for band in range(band_data.shape[1]):
        plt.plot(kx_values, band_data[:, band], 'ko', markersize=3)

    # Adicionar o cone de luz (ω = c*kx)
    c = 1  # Velocidade da luz normalizada
    light_cone = c * np.abs(kx_values)
    plt.fill_between(kx_values, light_cone, max(np.nanmax(band_data, axis=1)), color='gray', alpha=0.3, label='Cone de Luz')

    # Configurações do gráfico
    plt.xlabel(r'$k_x$ ($2\pi/a$)', fontsize=14)
    plt.ylabel(r'$\omega$ ($2\pi c/a$)', fontsize=14)
    plt.title('Gráfico 2D da rede infinita', fontsize=16)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('bandas_pontos.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()