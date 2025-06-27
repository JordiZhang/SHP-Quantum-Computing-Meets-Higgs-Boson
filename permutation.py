from engineer import correlation_plots
import numpy as np
import scienceplots
from tensorflow.keras.models import load_model
from tensorflow import keras
import matplotlib.pyplot as plt

plt.style.use(["science", "no-latex"])

def permutation_ranking(testing):
    # array of the names
    names = [r"$\eta_{\gamma_1}$", r"$\eta_{\gamma_2}$", r"$p_{T,\gamma \gamma}$", r"$p_{T,jj}$", r"$p_{T,jj}^{\dagger}$",
                    r"$m_{jj}$", r"$\Delta y_{jj}$", r"$\Delta \phi_{jj}$", r"$\Delta \eta_{jj}$", r"$p_{T,\gamma \gamma j_1}$", # 10
                    r"$m_{\gamma \gamma j_1}$", r"$p_{T, \gamma \gamma jj}$", r"$p_{T, \gamma \gamma jj}^{\dagger}$", r"$m_{\gamma \gamma jj}$", r"$\Delta y_{\gamma \gamma jj}$",
                    r"$\Delta \phi_{\gamma \gamma jj}$", r"$\Delta R_{\gamma \gamma jj}$", r"$m_{jjjj}$", r"$N_{Jets}$", r"$N_{Central}$", # 20
                    r"$p_{T, High}$", r"$\sum p_{T}$", r"$\eta_{j_F}$", r"$m_{\gamma \gamma j_F}$", r"$p_{Tt,\gamma \gamma}$",
                    r"$\Delta \eta_{\gamma \gamma}$", r"$\eta^{Zepp}$", r"$\phi^{*}_{\gamma \gamma}$", r"$cos\theta^{*}_{\gamma \gamma}$",
                    r"$p_{T, \gamma_1}$", r"$\phi_{\gamma_1}$", r"$p_{T, \gamma_2}$", r"$\phi_{\gamma_2}$",
                    r"$p_{T,j_1}$", r"$\eta_{j_1}$", r"$\phi_{j_1}$",
                    r"$p_{T,j_2}$", r"$\eta_{j_2}$", r"$\phi_{j_2}$",
                    r"$p_{T,j_3}$", r"$\eta_{j_3}$", r"$\phi_{j_3}$",
                    r"$p_{T,j_4}$", r"$\eta_{j_4}$", r"$\phi_{j_4}$",
                    r"$\frac{p_{T, \gamma \gamma}}{m_{\gamma \gamma}}$", r"$\eta_{\gamma \gamma}$", r"$p_{T,j_F}$", r"$\Delta \theta_{\gamma \gamma j_F}$"]
    names = np.array(names)
    indices = correlation_plots()
    names = np.delete(names, indices)

    # permutation feature importance, for now the error is simply the loss (binary cross entropy)
    rng = np.random.default_rng()
    size = len(testing)
    print(testing)
    importance_error = []
    importance_acc = []

    for _ in range(100):
        rng.shuffle(testing, axis = 0)

        inp = np.copy(testing[:, 0:-1])
        output = np.copy(testing[:, -1])

        model = load_model("[20]-0.001-1000.keras")

        print("OG")
        #
        original_error, original_acc = model.evaluate(inp, output)



        permuted_error = []
        permuted_acc = []

        for i in range(len(inp[0])):
            inp = np.copy(testing[:, 0:-1])
            output = np.copy(testing[:, -1])
            rng.shuffle(inp[:, i])

            model = load_model("[20]-0.001-1000.keras")
            l, a = model.evaluate(inp, output)
            print(l, a)
            permuted_error.append(l)
            permuted_acc.append(a)

        permuted_error = np.array(permuted_error)
        permuted_acc = np.array(permuted_acc)
        ratio_error = permuted_error/original_error
        importance_error.append(ratio_error)
        ratio_acc = permuted_acc/original_acc
        importance_acc.append(ratio_acc)


    avg_importance_error = np.mean(importance_error, axis=0)
    avg_importance_acc = np.mean(importance_acc, axis=0)

    # sort and plot
    sorted_indices1 = (-avg_importance_error).argsort()
    y_pos = np.arange(len(sorted_indices1))

    top = 20

    fig1, ax1 = plt.subplots(1, 1, figsize=(5,5), dpi=300)
    ax1.barh(y_pos[0:top], avg_importance_error[sorted_indices1[0:top]])
    ax1.set_yticks(y_pos[0:top], labels=names[sorted_indices1[0:top]])
    ax1.invert_yaxis()
    ax1.set_xlabel("Feature Importance")
    ax1.set_title("Average Feature Importance (Loss) in 100 Permutations")
    plt.tick_params(axis='y', which='both', right=False, left=False)

    top = 20
    fig3, ax3 = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    ax3.barh(y_pos[top:], avg_importance_error[sorted_indices1[top:]])
    ax3.set_yticks(y_pos[top:], labels=names[sorted_indices1[top:]])
    ax3.invert_yaxis()
    ax3.set_xlabel("Feature Importance")
    ax3.set_title("Average Feature Importance (Loss) in 100 Permutations")
    plt.tick_params(axis='y', which='both', right=False, left=False)

    sorted_indices2 = (-avg_importance_acc).argsort()
    y_pos = np.arange(len(sorted_indices2))

    top = 41

    fig2, ax2 = plt.subplots(1, 1, figsize = (10,10))
    ax2.barh(y_pos[0:top], avg_importance_acc[sorted_indices2[0:top]])
    ax2.set_yticks(y_pos[0:top], labels= names[sorted_indices2[0:top]])
    ax2.invert_yaxis()
    ax2.set_xlabel("Feature Importance")
    #ax2.set_title("Average Feature Importance (Accuracy) in 100 Permutations")
    plt.tick_params(axis='y', which='both', right=False, left=False)
    plt.show()