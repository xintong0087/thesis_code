import numpy as np

def load_outerScenarios(path="./macroData/"):
    outerScenarios = []
    try:
        for i in range(10):
            outerScenario = np.load(f"{path}outerScenarios_{i}.npy")
        outerScenarios.append(outerScenario)
    except:
        raise FileNotFoundError(f"outerScenarios_{i}.npy not found in {path}")
    return np.concatenate(outerScenarios, axis=0)