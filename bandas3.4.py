import meep as mp
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_epsilon(sim, filename="geometrycenter.png"):
    eps_data = sim.get_array(center=mp.Vector3(), size=sim.cell_size, component=mp.Dielectric)
    plt.imshow(eps_data.T, interpolation='spline36', cmap='binary', origin='lower')
    plt.colorbar(label="Epsilon")
    plt.axis('off')
    plt.title("Buraco Central")
    plt.savefig(filename)
    plt.close()


def main():
    resolution = 20
    eps = 13
    eps1 = 1
    r = 0.5
    a = 1.0
    b = 2.0

    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, mp.inf), material=mp.Medium(epsilon=eps)),
        mp.Cylinder(r, material=mp.Medium(epsilon=eps1))
    ]

    cell = mp.Vector3(a, b, 0)

    fcen = 0.25
    df = 1.5

    s = mp.Source(
        src=mp.GaussianSource(fcen, fwidth=df),
        component=mp.Hz,
        center=mp.Vector3(0.0),
    )

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=[s],
        k_point=mp.Vector3(0, 0, 0),
        resolution=resolution,
    )

    k_interp = 300
    GX = mp.interpolate(k_interp, [mp.Vector3(), mp.Vector3(0.5, 0)])
    print(f'GX: {GX}')
    XM = mp.interpolate(k_interp, [mp.Vector3(0.5, 0), mp.Vector3(0.5, 0.5)])
    print(f'XM: {XM}')
    MG = mp.interpolate(k_interp, [mp.Vector3(0.5, 0.5), mp.Vector3()])
    print(f'MG: {MG}')


    GX_freqs = sim.run_k_points(300, GX)
    XM_freqs = sim.run_k_points(300, XM)
    MG_freqs = sim.run_k_points(300, MG)

    # Processamento dos dados
    def prepare_band_data(freqs):
        max_bands = max(len(f) for f in freqs)
        band_data = []
        for f in freqs:
            real_freqs = [fr.real for fr in f]
            if len(real_freqs) < max_bands:
                real_freqs.extend([np.nan] * (max_bands - len(real_freqs)))
            band_data.append(real_freqs)
        return np.array(band_data)

    GX_band_data = prepare_band_data(GX_freqs)
    XM_band_data = prepare_band_data(XM_freqs)
    MG_band_data = prepare_band_data(MG_freqs)

    # Plotagem combinada no eixo y positivo
    plt.figure(figsize=(10, 6))

    GX_values = [k.x for k in GX]
    print('K_x:', GX_values)
    for band in range(GX_band_data.shape[1]):
        plt.plot(GX_values, GX_band_data[:, band], 'ko', markersize=2, linewidth=0.2, label='' if band == 0 else "")

    XM_values = [k.x+0.5 for k in GX]
    print('K_y:', XM_values)
    for band in range(XM_band_data.shape[1]):
        plt.plot(XM_values, XM_band_data[:, band], 'ko', markersize=2, linewidth=0.2, label='' if band == 0 else "")

    MG_values = [k.x+1 for k in GX]
    print('K_xy:', MG_values)
    for band in range(MG_band_data.shape[1]):
        plt.plot(MG_values, MG_band_data[:, band], 'ko', markersize=2, linewidth=0.2, label='' if band == 0 else "")

    # Configurações do gráfico
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.2)
    plt.text(0.25, 0.02, '', ha='center', va='bottom')
    plt.text(-0.25, 0.02, '', ha='center', va='bottom')

    # Light cone (considerando |k|)
    k_combined = np.linspace(-0.5, 0.5, 100)
    light_cone = np.abs(k_combined)
    #plt.fill_between(k_combined, light_cone, plt.ylim()[1], color='gray', alpha=0.2, label='Cone de Luz')

    plt.xlabel('k (2π/a)', fontsize=12)
    plt.ylabel('Frequência (ωa/2πc)', fontsize=12)
    plt.title('Diagrama de Bandas', fontsize=14)
    plt.xlim(0, 0.5)
    plt.ylim(0, 0.5)
    #plt.ylim(0, max(np.nanmax(kx_band_data), np.nanmax(ky_band_data)) * 1.1)

    # Adicionando marcadores de simetria
    plt.xticks([ 0, 0.25, 0.5, 0.75, 1.0, 1.5])

    plt.grid(True, linestyle=':', alpha=0.5)
    #plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('bandas_combinadas_positivo.png', dpi=300, bbox_inches='tight')
    plt.show()

    plot_epsilon(sim)


if __name__ == "__main__":
    main()
