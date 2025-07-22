import numpy as np
import matplotlib.pyplot as plt

# Dados de exemplo
x = np.array([0.13888, 0.1875, 0.2222])  # Variável independente
y = np.array([1546146, 2082415, 2441406])  # Variável dependente

# Cálculo dos coeficientes manualmente
n = len(x)
soma_x = np.sum(x)
soma_y = np.sum(y)
soma_xy = np.sum(x * y)
soma_x2 = np.sum(x ** 2)

# Fórmulas para regressão linear simples:
# Coeficiente angular (a)
coeficiente_angular = (n * soma_xy - soma_x * soma_y) / (n * soma_x2 - soma_x ** 2)

# Coeficiente linear (b)
coeficiente_linear = (soma_y - coeficiente_angular * soma_x) / n

print(f"Equação da reta: y = {coeficiente_angular:.2f}x + {coeficiente_linear:.2f}")
print(f"Coeficiente angular (inclinação): {coeficiente_angular:.2f}")
print(f"Coeficiente linear (intercepto): {coeficiente_linear:.2f}")

# Previsões do modelo
y_pred = coeficiente_angular * x + coeficiente_linear

# Plotando os dados e a linha de regressão
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Dados observados')
plt.plot(x, y_pred, color='red', label='Linha de regressão')
plt.title('Regressão Linear Simples (implementação manual)')
plt.xlabel('Variável Independente (X)')
plt.ylabel('Variável Dependente (Y)')
plt.legend()
plt.grid(True)

# Mostrando o gráfico
plt.show()

# Cálculo do R² (coeficiente de determinação)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f"Coeficiente de determinação (R²): {r2:.4f}")