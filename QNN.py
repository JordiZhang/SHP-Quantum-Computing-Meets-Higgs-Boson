import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import time
from tensorflow.keras.models import Sequential, load_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import ZZFeatureMap
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, hamiltonian_variational_ansatz
from qiskit_machine_learning.optimizers import COBYLA, ADAM, POWELL, NELDER_MEAD, SPSA
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.utils import algorithm_globals
plt.style.use(["science", "no-latex"])
algorithm_globals.random_seed = 100

def training_vqc(features, samples):
    # takes 6 features and trains a quantum vqc model
    rng = np.random.default_rng(100)
    samples = samples/0.8

    entries = len(features)
    labels = np.concatenate((-np.ones(int(entries/2)), np.ones(int(entries/2)))).reshape((-1, 1))
    features = np.concatenate((features, labels), axis=1)

    bkg = np.copy(features[0:int(entries/2)])
    sig = np.copy(features[int(entries/2):])

    rng.shuffle(bkg)
    rng.shuffle(sig)

    training = np.copy(np.concatenate((bkg[0:int(0.8*samples/2)], sig[0:int(0.8*samples/2)])))
    testing = np.copy(np.concatenate((bkg[int(0.8*samples/2):], sig[int(0.8*samples/2):])))

    rng.shuffle(training)
    rng.shuffle(testing)
    print("There are " + str(len(training)) + " training events.")

    x_train = training[:, 0:-1]
    y_train = training[:, -1]
    x_test = testing[:500, 0:-1]
    y_test = testing[:500, -1]

    num_features = x_train.shape[1]

    #assert num_features == 2, "There should be 8 features"
    fmap_reps = 1
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=fmap_reps)
    #feature_map.draw(output="mpl", style="clifford", fold=20)
    print("step 1")
    ansatz_reps = 3
    # this can be considered the hidden layers analogue
    ansatz = EfficientSU2(num_qubits=num_features, entanglement='full', reps=ansatz_reps)
    #ansatz.decompose().draw(output="mpl", style="clifford", fold=20)
    print("step 2")
    #plt.show()

    # rhobeg i think is the COBYLA equivalent of learning rate for ADAM
    optimizer = COBYLA(maxiter=500)

    sampler = Sampler()

    objective_func_vals = []
    plt.rcParams["figure.figsize"] = (12, 6)

    def callback_graph(weights, obj_func_eval):
        clear_output(wait=True)
        objective_func_vals.append(obj_func_eval)
        print("Optimizer Iteration:", len(objective_func_vals))
        print(time.time()-start)


    # sort of assembling the model
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=optimizer,
        callback=callback_graph,
        warm_start=False
    )


    # clear objective value history
    objective_func_vals = []

    start = time.time()
    print(start)

    """
    counter = 0
    minimums = [10000]
    vqc.optimizer = COBYLA(maxiter=10)
    # sort of early stopping
    while True:
        vqc.fit(x_train, y_train)
        min_loss = np.min(objective_func_vals[counter*10:(counter+1)*10])
        minimums.append(min_loss)
        algorithm_globals.random_seed += 1
        vqc.neural_network.sampler = Sampler()
        vqc.optimizer = COBYLA(maxiter=10)
        # if no improvement break and stop training
        if minimums[-1] > minimums[-2]:
            break

        counter += 1
        print("Epochs:", 10 * counter)
    """

    vqc.fit(x_train, y_train)
    elapsed = time.time() - start

    print(f"Training time: {elapsed} seconds")
    print("There are " + str(len(training)) + " training events.")
    fname = "[ZZ-" + str(fmap_reps) + "]_COBYLA_[EfficientSU2-" + str(ansatz_reps) + "500ite]_" + str(int(samples*0.8)) + "_Top6_" + str(int(elapsed))
    #fname = "top6_test"
    vqc.save(fname+".model")

    train_score_q4 = vqc.score(x_train, y_train)
    test_score_q4 = vqc.score(x_test, y_test)

    print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
    print(f"Quantum VQC on the test dataset:     {test_score_q4:.2f}")

    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.savefig(os.path.join(directory, fname+".png"))
    plt.show()



def feature_reduction(features):
    # removes correlated features and returns top 6 features based on permutation ranking.
    correlated = [2, 10, 13, 23, 24, 28, 29, 31]
    correlated = np.array(correlated)
    names = ["eta_y1", "eta_y2", "pt_yy", "pt_jj", "pt_jj_dagger",
             "m_jj", "delta_y_jj", "delta_phi_jj", "delta_eta_jj", "pt_yyj1",  # 10
             "m_yyj1", "pt_yyjj", "pt_yyjj_dagger", "m_yyjj", "delta_y_yyjj",
             "delta_phi_yyjj", "delta_R_yyjj", "m_jjjj", "number_jets", "central_jets",  # 20
             "high_jet_p_t", "sca_sum_pt_jjjj", "eta_jf", "m_yyjf", "pt_thrust_yy",
             "delta_eta_yy", "eta_zepp", "phi_star_yy", "cos_theta_star_yy",
             "pt_y1", "phi_y1", "pt_y2", "phi_y2",
             "jet1.p_t", "jet1.eta", "jet1.phi",
             "jet2.p_t", "jet2.eta", "jet2.phi",
             "jet3.p_t", "jet3.eta", "jet3.phi",
             "jet4.p_t", "jet4.eta", "jet4.phi",
             "pt_yy_myyscaled", "eta_yy", "pt_jf", "delta_theta_yyjf"]
    names = np.array(names)
    names = np.delete(names, correlated)
    features = np.delete(features, correlated, axis=1)

    reduced = ["number_jets", "jet3.p_t", "sca_sum_pt_jjjj", "delta_eta_jj", "delta_phi_jj", "high_jet_p_t",
               "jet1.p_t", "eta_jf", "jet4.p_t", "pt_jf", "eta_y2", "jet2.phi", "eta_y1", "jet1.phi",
               "delta_eta_yy", "jet4.phi", "jet2.p_t", "jet2.eta", "jet3.eta", "jet3.phi"]  # top 20 features
    reduced = np.array(reduced)

    to_delete = []

    top_number = 6

    for i in range(len(names)):
        if not np.any(reduced[0:top_number] == names[i]):
            to_delete.append(i)

    names = np.delete(names, to_delete)
    features = np.delete(features, to_delete, axis=1)
    scaler = MinMaxScaler((-np.pi, np.pi))
    features = scaler.fit_transform(features)

    return features


def principal_component(features):
    # takes in an array containing data, standardizes each column and runs principal component analysis to get reduced dimensionality
    # data should not include the labels though
    correlated = [2, 10, 13, 23, 24, 28, 29, 31]
    correlated = np.array(correlated)
    names = ["eta_y1", "eta_y2", "pt_yy", "pt_jj", "pt_jj_dagger",
             "m_jj", "delta_y_jj", "delta_phi_jj", "delta_eta_jj", "pt_yyj1",  # 10
             "m_yyj1", "pt_yyjj", "pt_yyjj_dagger", "m_yyjj", "delta_y_yyjj",
             "delta_phi_yyjj", "delta_R_yyjj", "m_jjjj", "number_jets", "central_jets",  # 20
             "high_jet_p_t", "sca_sum_pt_jjjj", "eta_jf", "m_yyjf", "pt_thrust_yy",
             "delta_eta_yy", "eta_zepp", "phi_star_yy", "cos_theta_star_yy",
             "pt_y1", "phi_y1", "pt_y2", "phi_y2",
             "jet1.p_t", "jet1.eta", "jet1.phi",
             "jet2.p_t", "jet2.eta", "jet2.phi",
             "jet3.p_t", "jet3.eta", "jet3.phi",
             "jet4.p_t", "jet4.eta", "jet4.phi",
             "pt_yy_myyscaled", "eta_yy", "pt_jf", "delta_theta_yyjf"]
    names = np.array(names)
    names = np.delete(names, correlated)
    features = np.delete(features, correlated, axis=1)

    reduced = ["number_jets", "jet3.p_t", "sca_sum_pt_jjjj", "delta_eta_jj", "delta_phi_jj", "high_jet_p_t",
               "jet1.p_t", "eta_jf", "jet4.p_t", "pt_jf", "eta_y2", "jet2.phi", "eta_y1", "jet1.phi",
               "delta_eta_yy", "jet4.phi", "jet2.p_t", "jet2.eta", "jet3.eta", "jet3.phi"]  # top 20 features
    reduced = np.array(reduced)

    to_delete = []


    top_number = 6

    for i in range(len(names)):
        if not np.any(reduced[0:top_number] == names[i]):
            to_delete.append(i)

    names = np.delete(names, to_delete)
    features = np.delete(features, to_delete, axis=1)

    data_standardized = StandardScaler().fit_transform(features)

    principal = PCA(n_components=6)
    principal.fit(data_standardized)
    ratios = principal.explained_variance_ratio_
    print(ratios)
    print(sum(ratios))

    x_pca = principal.transform(data_standardized)
    scaler = MinMaxScaler((-np.pi, np.pi))
    x_pca = scaler.fit_transform(x_pca)
    return x_pca



def classical_model(features):
    rng = np.random.default_rng(100)

    entries = len(features)
    labels = np.concatenate((-np.ones(int(entries / 2)), np.ones(int(entries / 2)))).reshape((-1, 1))
    features = np.concatenate((features, labels), axis=1)

    bkg = np.copy(features[0:int(entries / 2)])
    sig = np.copy(features[int(entries / 2):])

    rng.shuffle(bkg)
    rng.shuffle(sig)

    training = np.copy(np.concatenate((bkg[0:int(0.8 * entries / 2)], sig[0:int(0.8 * entries / 2)])))
    testing = np.copy(np.concatenate((bkg[int(0.8 * entries / 2):], sig[int(0.8 * entries / 2):])))

    rng.shuffle(training)
    rng.shuffle(testing)
    print("There are " + str(len(training)) + " training events.")

    x_train = training[:, 0:-1]
    y_train = training[:, -1]
    x_test = testing[:, 0:-1]
    y_test = testing[:, -1]

    train_label = []
    test_label = []

    for i in range(len(y_train)):
        if y_train[i] == 1:
            train_label.append("S")
        else:
            train_label.append("B")

    for i in range(len(y_test)):
        if y_test[i] == 1:
            test_label.append("S")
        else:
            test_label.append("B")

    train_label = np.array(train_label)
    test_label = np.array(test_label)

    svc = SVC()
    _ = svc.fit(x_train, train_label)
    train_score_c4 = svc.score(x_train, train_label)
    test_score_c4 = svc.score(x_test, test_label)

    print(f"Classical SVC on the training dataset: {train_score_c4:.2f}")
    print(f"Classical SVC on the test dataset:     {test_score_c4:.2f}")


def loaded(features, samples):
    rng = np.random.default_rng(100)

    entries = len(features)
    labels = np.concatenate((-np.ones(int(entries / 2)), np.ones(int(entries / 2)))).reshape((-1, 1))
    features = np.concatenate((features, labels), axis=1)

    bkg = np.copy(features[0:int(entries / 2)])
    sig = np.copy(features[int(entries / 2):])

    rng.shuffle(bkg)
    rng.shuffle(sig)

    testing = np.copy(np.concatenate((bkg[20000:20000 + 250], sig[20000:20000 + 250])))

    rng.shuffle(testing)

    x_test = np.copy(testing[:, 0:-1])
    y_test = np.copy(testing[:, -1])

    model = VQC.load("[ZZ-1]_COBYLA_[RealAmplitudes-3]_1000_Top6_1602.model")
    test_score_q4 = model.score(x_test, y_test)
    print("real",test_score_q4)
    predictions = model.predict_proba(x_test)

    fpr, tpr, thresholds = roc_curve(y_test, predictions[:, 1])

    model_auc = auc(fpr, tpr)

    model1 = VQC.load("[ZZ-1]_COBYLA_[EfficientSU2-3500ite]_1000_Top6_3506.model")
    test_score_q4 = model.score(x_test, y_test)
    print("efficent", test_score_q4)
    predictions1 = model1.predict_proba(x_test)

    fpr1, tpr1, thresholds1 = roc_curve(y_test, predictions1[:, 1])
    model_auc1 = auc(fpr1, tpr1)
    np.savetxt("qfpr.txt", fpr1)
    np.savetxt("qtpr", tpr1)


    f1 = plt.figure(1, figsize=(3.6, 3.6), dpi=300)
    #plt.plot(fpr, tpr, label="VQC RealAmplitudes, AUC = " + str(round(model_auc,5)))
    plt.plot(fpr1, tpr1, label="VQC EfficientSU2, AUC = " + str(round(model_auc1, 3)))
    #plt.plot(fpr1, tpr1, label="Neural Network Top 6, AUC = " + str(round(model_auc1, 5)))
    plt.xlabel('Background Efficiency (FPR)')
    plt.ylabel('Signal Efficiency (TPR)')
    #plt.title('ROC Curve')
    plt.legend()
    plt.show()