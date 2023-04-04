# PREDICTING EARLY HEART DISEASES FROM THE CLEVELAND DATASET

# import required libraries
import numpy as np
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit, Aer
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.components.multiclass_extensions import AllPairs
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name

# load the dataset
dataset = np.loadtxt("processed.cleveland.data", delimiter=",")
X, y = split_dataset_to_data_and_labels(dataset)

# normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# define the feature map
feature_dim = X.shape[1]
feature_map = ZZFeatureMap(feature_dim)

# define the variational form
var_form = TwoLocal(feature_dim, ['ry', 'rz'], 'cz', reps=3)

# define the backend and quantum instance
backend = Aer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024)

# define the QSVM algorithm
qsvm = QSVM(feature_map=feature_map,
            var_form=var_form,
            training_data=X,
            training_labels=y,
            multiclass_extension=AllPairs(),
            quantum_instance=quantum_instance)

# train the algorithm
qsvm.train(X, y)

# test the algorithm and print the accuracy
testing_data = X
testing_labels = y
predicted_labels = qsvm.predict(testing_data)
accuracy = qsvm.test(testing_data, testing_labels)['testing_accuracy']
class_names = [str(i) for i in range(1, 5)]
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification report:")
print(qsvm._get_multiclass_metrics(testing_data, testing_labels, predicted_labels, class_names=class_names))
