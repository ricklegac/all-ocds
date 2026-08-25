"""
ML básico en 5 pasos: datos -> split -> entrenar -> predecir -> evaluar.
Requiere: pip install scikit-learn pandas numpy
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

rng = np.random.default_rng(42)
N = 2000

# ---------- PASO 1: los datos ----------
# Dataset SINTÉTICO (inventado). Cada fila = una licitación ficticia.
monto = rng.lognormal(mean=17, sigma=1.2, size=N)          # guaraníes
n_oferentes = rng.integers(1, 9, size=N)                    # cuántos se presentaron
dias_publicacion = rng.integers(5, 45, size=N)              # plazo del llamado
modificaciones = rng.poisson(0.6, size=N)                   # cambios al pliego

# Yo mismo "planto" la relación: pocos oferentes + plazo corto + monto alto
# => más probabilidad de impugnación. Después vemos si el modelo la descubre.
riesgo = (
    1.2 * (n_oferentes <= 2)
    + 0.9 * (dias_publicacion < 15)
    + 0.7 * (monto > np.percentile(monto, 75))
    + 0.5 * (modificaciones >= 2)
    - 2.2
)
prob = 1 / (1 + np.exp(-riesgo))
y = rng.binomial(1, prob)  # 1 = hubo impugnación, 0 = no

X = pd.DataFrame({
    "monto": monto,
    "n_oferentes": n_oferentes,
    "dias_publicacion": dias_publicacion,
    "modificaciones": modificaciones,
})

print(f"Filas: {N} | Casos positivos (impugnadas): {y.mean():.1%}\n")

# ---------- PASO 2: separar entrenamiento y prueba ----------
# El modelo NUNCA debe ver los datos con los que lo vas a evaluar.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ---------- PASO 3: entrenar ----------
modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_train, y_train)

# ---------- PASO 4: predecir ----------
y_pred = modelo.predict(X_test)

# ---------- PASO 5: evaluar ----------
print("=== Resultados sobre el conjunto de prueba ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"F1       : {f1_score(y_test, y_pred):.3f}")

# Comparación honesta: ¿qué pasa si predigo SIEMPRE la clase mayoritaria?
baseline = np.zeros_like(y_test)
print(f"\nAccuracy de un modelo tonto (todo 0): {accuracy_score(y_test, baseline):.3f}")
print(f"F1 de ese modelo tonto              : {f1_score(y_test, baseline, zero_division=0):.3f}")

print("\nMatriz de confusión [filas=real, columnas=predicho]:")
print(confusion_matrix(y_test, y_pred))

print("\n=== Qué aprendió el árbol ===")
print(export_text(modelo, feature_names=list(X.columns), decimals=0))

print("=== Importancia de cada variable ===")
for nombre, imp in sorted(
    zip(X.columns, modelo.feature_importances_), key=lambda t: -t[1]
):
    print(f"  {nombre:<18} {imp:.3f}")
