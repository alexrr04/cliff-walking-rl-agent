# Agente RL para Cliff Walking

Implementación de varios algoritmos de aprendizaje por refuerzo para el entorno Cliff Walking para la segunda práctica de la asignatura de Sistemas Inteligentes Distribuidos del curso 2024-2025 Q2 en la Universidad Politécnica de Catalunya (UPC).

## Instalación

1. Desde la raíz del proyecto, ejecuta el siguiente comando para crear un entorno virtual:

```bash
python -m venv venv
```

2. Activa el entorno virtual:

```bash
# En Windows
venv\Scripts\activate

# En macOS y Linux
source venv/bin/activate
```

3. Instala las dependencias:

```bash
pip install -r requirements.txt
```

## Algoritmos

El proyecto incluye implementaciones de:

-   Iteración de Valor (Value Iteration)
-   Estimación Directa (Direct Estimation)
-   Q-Learning

## Ejecución de los Algoritmos

Puedes ejecutar los algoritmos de dos maneras:

### 1. Modo CLI Interactivo

Ejecutar sin archivo de configuración:

```bash
python scripts/run_experiment.py
```

Esto:

-   Te permitirá seleccionar un algoritmo
-   Solicitará los parámetros específicos del algoritmo
-   Ejecutará el experimento
-   Guardará los resultados en `experiments/[algorithm]/experiment_[timestamp]`

### 2. Usando Configuración YAML

Ejecutar con un archivo de configuración YAML:

```bash
python scripts/run_experiment.py --config experiments/[algoritmo]/[config-file].yaml
```

## Formato de Configuración YAML

Los archivos de configuración deben especificar el algoritmo y sus parámetros. Los parámetros pueden ser valores únicos o listas para experimentos con barrido de parámetros.

### Iteración de Valor

```yaml
algorithm: valueIteration
parameters:
    gamma: 0.95 # Factor de descuento
    epsilon: 0.00000001 # Umbral de convergencia
    num_episodes: 500 # Número de episodios para evaluación
```

### Estimación Directa

```yaml
algorithm: directEstimation
parameters:
    gamma: 0.95 # Factor de descuento
    num_trajectories: 500 # Número de trayectorias para muestreo
    max_iters: 500 # Máximo de iteraciones para entrenamiento
    patience: 100 # Paciencia para convergencia
    num_runs: 5 # Número de ejecuciones del experimento
    num_episodes: 500 # Número de episodios para evaluación
```

### Q-Learning

```yaml
algorithm: qlearning
parameters:
    gamma: [0.5, 0.7, 0.9] # Factor de descuento (puede ser lista para barrido de parámetros)
    learning_rate: 0.1 # Tasa de aprendizaje
    epsilon: 0.9 # Tasa inicial de exploración
    ep_decay: 0.95 # Tasa de decaimiento de epsilon
    lr_decay: 0.99 # Decaimiento de la tasa de aprendizaje
    num_episodes: 1000 # Episodios de entrenamiento
    eval_episodes: 500 # Episodios de evaluación
    penalty: -1.0 # Penalización por moverse a la derecha
    num_runs: 5 # Número de ejecuciones del experimento
```

### Recrear los experimentos

Para recrear los experimentos de la práctica, existen archivos de configuración YAML en la carpeta `experiments/[algoritmo]/`.

Cada archivo de configuración tiene un nombre que indica el algoritmo y el número de experimento, los cuales coinciden con el orden en qué aparecen los experimentos en el informe final. Por ejemplo, para ejecutar el experimento 1 de Q-Learning, puedes usar:

```bash
python scripts/run_experiment.py --config experiments/qlearning/qlearning-1.yaml
```

## Resultados

Después de ejecutar un experimento, los resultados se almacenan en `experiments/[algoritmo]/experiment_[timestamp]` con:

-   Resultados del experimento
-   Registros del experimento

Puedes limpiar todos los resultados de experimentos usando:

```bash
python scripts/run_experiment.py --clean
```
