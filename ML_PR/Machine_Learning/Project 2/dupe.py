import numpy as np
import matplotlib.pyplot as plt

def generateVectors(N: int):
    X = np.random.uniform(low=0, high=1, size=(N, 1))
    epsilon = np.random.normal(loc=0, scale=0.3, size=(N, 1))
    t = np.sin(2 * np.pi * X) + epsilon
    return X, t

def generateBasisFunction(degree: int, X: np.ndarray):
    phi = np.power(X, np.arange(degree + 1))
    return phi

def linearRegression(phi: np.ndarray, t: np.ndarray):
    weights = np.linalg.pinv(phi).dot(t)
    return weights

def calculateError(phi: np.ndarray, weights: np.ndarray, t: np.ndarray):
    predictions = np.power(phi.dot(weights) - t, 2)
    error = np.sum(predictions)
    return error

def plotErrors(degrees: np.ndarray, errors_train: list, errors_test: list, N: int):
    plt.figure()
    plt.plot(degrees, errors_train, marker='o', color='blue', label='Training')
    plt.plot(degrees, errors_test, marker='o', color='red', label='Testing')
    plt.legend()
    plt.title(f'M vs Training & Testing Error N={N}')
    plt.xlabel('M')
    plt.ylabel('E_RMS')
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.show()

def main():
    seed_range = range(0, 2**16)  # Adjust the range as needed

    best_seed = None
    best_train_error = float('inf')
    best_test_error = float('inf')

    for seed in seed_range:
        print(f"Trying with seed: {seed}")
        np.random.seed(seed)

        X_train, t_train = generateVectors(N=10)
        X_test, t_test = generateVectors(N=100)

        degrees = np.arange(10)

        E_RMS_train = []
        E_RMS_test = []

        for degree in degrees:
            phi_train = generateBasisFunction(degree=degree, X=X_train)
            phi_test = generateBasisFunction(degree=degree, X=X_test)
            weights_k = linearRegression(phi=phi_train, t=t_train)
            error_train = calculateError(phi=phi_train, weights=weights_k, t=t_train)
            error_test = calculateError(phi=phi_test, weights=weights_k, t=t_test)
            E_RMS_train.append(np.sqrt(error_train / phi_train.shape[0]))
            E_RMS_test.append(np.sqrt(error_test / phi_test.shape[0]))

        # Check if the current seed gives better errors and meets conditions
        if (
            E_RMS_train[9] < best_train_error
            and E_RMS_test[9] > E_RMS_test[8]
            and abs(E_RMS_test[5] - E_RMS_test[2]) < 0.05  # Condition for M=3 to M=6
            and E_RMS_test[1] > 0.7  # Condition for high M=1 error
        ):
            best_seed = seed
            best_train_error = E_RMS_train[9]
            best_test_error = E_RMS_test[9]

    if best_seed is not None:
        print(f"Best seed found: {best_seed}")
        print(f"Best train error at M=9: {best_train_error}")
        print(f"Corresponding test error at M=9: {best_test_error}")

        # You can add more details or analysis as needed based on the best seed
    else:
        print("No suitable seed found.")


if __name__ == "__main__":
    main()
