import math
import pylorentz
import numpy as np
import seaborn as sns


def data_processing():
    # replaces all the unphysical events as the average of that parameter
    # the averages are calculated separately for the background and signal events, thus also replaced separately

    # imports the entire dataset
    bkg = np.genfromtxt(fname="bkg.csv", delimiter=",", skip_header=1)
    sig = np.genfromtxt(fname="ggf.csv", delimiter=",", skip_header=1)

    # processing for background
    for j in range(16):
        background = []
        ind = []
        for i in range(len(bkg)):
            if math.isclose(bkg[i, j + 9], -999.9):
                ind.append(i)
            else:
                background.append(bkg[i, j + 9])
        bkg_avg = np.average(background)

        for i in range(len(ind)):
            assert math.isclose(bkg[ind[i], j + 9], -999.9)
            # changes unphysical
            # to 0 for pt and E, to average for eta and phi
            bkg[ind[i], j + 9] = bkg_avg
            """
            if j % 4 == 1 or j % 4 == 2:
                bkg[ind[i], j + 9] = bkg_avg
            else:
                bkg[ind[i], j + 9] = 0
            """

    np.savetxt("bkg_processed.csv", bkg, delimiter=",")

    # processing for signal
    for j in range(16):
        signal = []
        ind = []
        for i in range(len(sig)):
            if math.isclose(sig[i, j + 9], -999.9):
                ind.append(i)
            else:
                signal.append(sig[i, j + 9])
        sig_avg = np.average(signal)

        for i in range(len(ind)):
            assert math.isclose(sig[ind[i], j + 9], -999.9)
            # changes unphysical
            # to 0 for pt and E, to average for eta and phi
            sig[ind[i], j + 9] = sig_avg
            """
            if j % 4 == 1 or j % 4 == 2:
                sig[ind[i], j + 9] = sig_avg
            else:
                sig[ind[i], j + 9] = 0
            """

    np.savetxt("sig_processed.csv", sig, delimiter=",")


def generate_sample(sample_number, size=125000):
    # generates a min-max scaled dataset for training/validation/testing
    # unprocessed data sets
    background = np.genfromtxt(fname="bkg_processed.csv", delimiter=",")
    signal = np.genfromtxt(fname="sig_processed.csv", delimiter=",")

    # data set is in the order of millions of events, for training I will use reduced data sets of about 125k signal and 125k background
    # data can then be split later 80% training (200000), 10% validation (25000) and 10% testing(25000)
    bkg_ind = np.random.choice(background.shape[0], size=size,
                               replace=False)  # replace false to avoid sampling same events
    sig_ind = np.random.choice(signal.shape[0], size=size, replace=False)
    bkg = background[bkg_ind]
    sig = signal[sig_ind]
    dataset = np.concatenate((bkg, sig), axis=0)
    np.savetxt("sample" + str(sample_number) + ".csv", dataset, delimiter=",")


def features(data, number):
    # function takes in dataset and gives engineered features back
    # to make engineering easier lets first make some 4-momentum vectors for the photons and jets
    photon1 = pylorentz.Momentum4.e_eta_phi_pt(data[:, 3], data[:, 1], data[:, 2], data[:, 0])
    photon2 = pylorentz.Momentum4.e_eta_phi_pt(data[:, 7], data[:, 5], data[:, 6], data[:, 4])
    jet1 = pylorentz.Momentum4.e_eta_phi_pt(data[:, 12], data[:, 10], data[:, 11], data[:, 9])
    jet2 = pylorentz.Momentum4.e_eta_phi_pt(data[:, 16], data[:, 14], data[:, 15], data[:, 13])
    jet3 = pylorentz.Momentum4.e_eta_phi_pt(data[:, 20], data[:, 18], data[:, 19], data[:, 17])
    jet4 = pylorentz.Momentum4.e_eta_phi_pt(data[:, 24], data[:, 22], data[:, 23], data[:, 21])

    print(data[:, 9])
    print(data[:, 13])
    print(data[:, 17])
    print(data[:, 21])

    # Table 2
    # eta_y1, eta_y2
    eta_y1 = photon1.eta
    eta_y2 = photon2.eta

    # di-photon
    di_photon = photon1 + photon2
    pt_yy = di_photon.p_t

    print("number 1")

    # 2 highest pt jets, i.e. jets 1 and 2
    two_high_jets = jet1 + jet2
    # pt_jj and dagger
    # I should rewrite this, it is very inefficient or at least it seems to be.
    pt_jj = np.zeros(len(data))
    pt_jj_dagger = np.zeros(len(data))

    mask1 = jet2.p_t >= 30000
    mask2 = np.invert(mask1) * (jet1.p_t >= 30000)
    mask_dagger1 = jet2.p_t >= 25000
    mask_dagger2 = np.invert(mask_dagger1) * (jet1.p_t >= 25000)

    pt_jj[mask1] = two_high_jets.p_t[mask1]
    pt_jj[mask2] = jet1.p_t[mask2]
    pt_jj_dagger[mask_dagger1] = two_high_jets.p_t[mask_dagger1]
    pt_jj_dagger[mask_dagger2] = jet1.p_t[mask_dagger2]

    """
    for i in range(len(data)):
        if jet2.p_t[i] >= 30000:
            pt_jj[i] = two_high_jets.p_t[i]
        elif jet1.p_t[i] >= 30000:
            pt_jj[i] = jet1.p_t[i]
        if jet2.p_t[i] >= 25000:
            pt_jj_dagger[i] = two_high_jets.p_t[i]
        elif jet1.p_t[i] >= 25000:
            pt_jj_dagger[i] = jet1.p_t[i]
    """

    m_jj = two_high_jets.m

    delta_y_jj = 0.5 * np.log((jet1.e + jet1.p_z) / (jet1.e - jet1.p_z)) - 0.5 * np.log(
        (jet2.e + jet2.p_z) / (jet2.e - jet2.p_z))
    delta_phi_jj = jet1.phi - jet2.phi
    delta_eta_jj = jet1.eta - jet2.eta

    print("number 2")

    # di-photon with leading jet
    di_photon_jet1 = photon1 + photon2 + jet1
    pt_yyj1 = di_photon_jet1.p_t
    m_yyj1 = di_photon_jet1.m

    # di-photon with 2 leading jets
    di_photon_jet2 = photon1 + photon2 + jet1 + jet2
    pt_yyjj = di_photon.p_t
    pt_yyjj_dagger = di_photon.p_t

    pt_yyjj[mask1] = di_photon_jet2.p_t[mask1]
    pt_yyjj[mask2] = di_photon_jet1.p_t[mask2]
    pt_yyjj_dagger[mask_dagger1] = di_photon_jet2.p_t[mask_dagger1]
    pt_yyjj_dagger[mask_dagger2] = di_photon_jet1.p_t[mask_dagger2]

    """
    for i in range(len(data)):
        if jet2.p_t[i] >= 30000:
            pt_yyjj[i] = di_photon_jet2.p_t[i]
        elif jet1.p_t[i] >= 30000:
            pt_yyjj[i] = di_photon_jet1.p_t[i]
        if jet2.p_t[i] >= 25000:
            pt_yyjj_dagger[i] = di_photon_jet2.p_t[i]
        elif jet1.p_t[i] >= 25000:
            pt_yyjj_dagger[i] = di_photon_jet1.p_t[i]
    """

    m_yyjj = di_photon_jet2.m
    print("number 3")

    # 2 leading jets
    jetjet = jet1 + jet2
    delta_y_yyjj = 0.5 * np.log((di_photon.e + di_photon.p_z) / (di_photon.e - di_photon.p_z)) - 0.5 * np.log(
        (jetjet.e + jetjet.p_z) / (jetjet.e - jetjet.p_z))
    delta_phi_yyjj = di_photon.phi - jetjet.phi

    print("delta_y_jj", delta_y_jj)
    print(jet1.e)
    print(jet1.p_z)
    print(di_photon.e)
    print(di_photon.p_z)
    print("delta_eta_jj", delta_eta_jj)
    print("delta_y_yyjj", delta_y_yyjj)

    delta_R_yyjj = np.sqrt(
        (di_photon.eta - jetjet.eta) * (di_photon.eta - jetjet.eta) + delta_phi_yyjj * delta_phi_yyjj)

    # invariant mass of all jets
    jetjetjetjet = jet1 + jet2 + jet3 + jet4
    m_jjjj = jetjetjetjet.m

    # number of jets, we ignore the requirements of dagger, good enough to just use the Number of jets
    number_jets = data[:, 8]

    # for number of central jets, we take the approximation that only the 4 leading ones matter, for most events this is true since for most events there arent more than 4 jets
    # we will ignore dagger, since this is an approximation anyways it won't differ much either ways
    # masks that give events where
    cen_mask1 = np.invert(np.isclose(jet1.p_t, 0)) * (np.abs(jet1.eta) < 2.5)
    cen_mask2 = np.invert(np.isclose(jet2.p_t, 0)) * (np.abs(jet2.eta) < 2.5)
    cen_mask3 = np.invert(np.isclose(jet3.p_t, 0)) * (np.abs(jet3.eta) < 2.5)
    cen_mask4 = np.invert(np.isclose(jet4.p_t, 0)) * (np.abs(jet4.eta) < 2.5)
    central_jets = cen_mask1 + cen_mask2 + cen_mask3 + cen_mask4

    high_jet_p_t = jet1.p_t

    # approximate all jets with 4 leading ones
    sca_sum_pt_jjjj = jet1.p_t + jet2.p_t + jet3.p_t + jet4.p_t

    # forward jet
    forward_jet = np.zeros((len(data), 4))

    jf_mask1 = number_jets >= 4
    jf_mask2 = number_jets == 3
    jf_mask3 = number_jets == 2
    jf_mask4 = number_jets <= 1

    etas1 = np.array((jet1.eta, jet2.eta, jet3.eta, jet4.eta))
    etas2 = np.array((jet1.eta, jet2.eta, jet3.eta))
    etas3 = np.array((jet1.eta, jet2.eta))
    print(etas1)
    max1 = np.argmax(etas1, axis=0)
    max2 = np.argmax(etas2, axis=0)
    max3 = np.argmax(etas3, axis=0)

    for i in range(len(data)):
        print(i)
        forward_jet[jf_mask1[i]] = data[:, 9 + max1[i] * 4:13 + max1[i] * 4]
        forward_jet[jf_mask2[i]] = data[:, 9 + max2[i] * 4:13 + max2[i] * 4]
        forward_jet[jf_mask3[i]] = data[:, 9 + max3[i] * 4:13 + max3[i] * 4]
        forward_jet[jf_mask4[i]] = data[:, 9:13]

    print(max1)
    """
    forward_jet[jf_mask1] = data[:, 9 + max1[jf_mask1]* 4:13 + max1[jf_mask1]]
    forward_jet[jf_mask2] = data[:, 9 + max2[jf_mask2] * 4:13 + max2[jf_mask2]]
    forward_jet[jf_mask3] = data[:, 9 + max3[jf_mask3] * 4:13 + max3[jf_mask3]]
    forward_jet[jf_mask4] = data[:, 9:13]


    for i in range(len(data)):
        print(i)
        if number_jets[i] >= 4:
            etas = np.array((jet1.eta[i], jet2.eta[i], jet3.eta[i], jet4.eta[i]))
            index = np.argmax(etas)
            forward_jet[i] = data[i, 9 + index*4:13 + index*4]
        elif number_jets[i] == 3:
            etas = np.array((jet1.eta[i], jet2.eta[i], jet3.eta[i]))
            index = np.argmax(etas)
            forward_jet[i] = data[i, 9 + index * 4:13 + index * 4]
        elif number_jets[i] == 2:
            etas = np.array((jet1.eta[i], jet2.eta[i]))
            index = np.argmax(etas)
            forward_jet[i] = data[i, 9 + index * 4:13 + index * 4]
        else:
            forward_jet[i] = data[i, 9:13]
    """

    print("forward:", forward_jet)
    print(forward_jet.shape)
    forward_jet = pylorentz.Momentum4.e_eta_phi_pt(forward_jet[:, 3], forward_jet[:, 1], forward_jet[:, 2],
                                                   forward_jet[:, 0])
    di_photon_forward_jet = photon1 + photon2 + forward_jet

    eta_jf = forward_jet.eta
    m_yyjf = di_photon_forward_jet.m

    print("number 4")

    # Table 3
    # https://math.stackexchange.com/questions/4386389/how-to-rotate-a-vector-through-another-vector-in-the-same-direction
    difference = photon1 - photon2
    pt_thrust_yy = np.abs(photon1.p_x * photon2.p_y - photon2.p_x * photon1.p_y) / (2 * difference.p_t)

    print("number 5")

    delta_eta_yy = photon1.eta - photon2.eta
    eta_zepp = (di_photon.eta - jetjet.eta) / 2

    phi_star_yy = np.tan((np.pi - np.absolute(photon1.phi - photon2.phi)) / 2) * np.sqrt(
        1 - np.square(np.tanh(delta_eta_yy / 2)))
    cos_theta_star_yy = np.absolute(((photon1.e + photon1.p_z) * (photon2.e - photon2.p_z) - (
                photon1.e - photon1.p_z) * (photon2.e + photon2.p_z)) / (di_photon.m + np.sqrt(
        np.square(di_photon.m) + np.square(di_photon.p_t))))

    pt_y1 = photon1.p_t
    phi_y1 = photon1.phi
    pt_y2 = photon2.p_t
    phi_y2 = photon2.phi
    print("number 6")
    pt_yy_myyscaled = di_photon.p_t / di_photon.m
    eta_yy = di_photon.eta

    pt_jf = forward_jet.p_t
    eta_jf = forward_jet.eta
    delta_theta_yyjf = 2 * np.arctan(np.exp(-di_photon.eta)) - 2 * np.arctan(np.exp(-forward_jet.eta))

    feat = np.stack((eta_y1, eta_y2, pt_yy, pt_jj, pt_jj_dagger,
                     m_jj, delta_y_jj, delta_phi_jj, delta_eta_jj, pt_yyj1,  # 10
                     m_yyj1, pt_yyjj, pt_yyjj_dagger, m_yyjj, delta_y_yyjj,
                     delta_phi_yyjj, delta_R_yyjj, m_jjjj, number_jets, central_jets,  # 20
                     high_jet_p_t, sca_sum_pt_jjjj, eta_jf, m_yyjf, pt_thrust_yy,
                     delta_eta_yy, eta_zepp, phi_star_yy, cos_theta_star_yy,
                     pt_y1, phi_y1, pt_y2, phi_y2,
                     jet1.p_t, jet1.eta, jet1.phi,
                     jet2.p_t, jet2.eta, jet2.phi,
                     jet3.p_t, jet3.eta, jet3.phi,
                     jet4.p_t, jet4.eta, jet4.phi,
                     pt_yy_myyscaled, eta_yy, pt_jf, delta_theta_yyjf), axis=1)  # 49 features total if i can count

    np.savetxt("features" + str(number) + ".csv", X=feat, delimiter=",")



def correlation_plots():
    feat = np.genfromtxt(fname = "features1.csv", delimiter = ",") # engineered features
    data = np.genfromtxt(fname = "sample1.csv", delimiter = ",") # original features

    photon1 = pylorentz.Momentum4.e_eta_phi_pt(data[:, 3], data[:, 1], data[:, 2], data[:, 0])
    photon2 = pylorentz.Momentum4.e_eta_phi_pt(data[:, 7], data[:, 5], data[:, 6], data[:, 4])

    di_photon = photon1 + photon2
    feat = np.concatenate((feat, di_photon.m.reshape(-1, 1)), axis = 1)

    background = feat[0: 125000]
    signal = feat[125000:]

    bkg_cov_matrix = np.corrcoef(background, rowvar=False)
    sig_cov_matrix = np.corrcoef(signal, rowvar=False)

    names = ["eta_y1", "eta_y2", "pt_yy", "pt_jj", "pt_jj_dagger",
                    "m_jj", "delta_y_jj", "delta_phi_jj", "delta_eta_jj", "pt_yyj1", # 10
                    "m_yyj1", "pt_yyjj", "pt_yyjj_dagger", "m_yyjj", "delta_y_yyjj",
                    "delta_phi_yyjj", "delta_R_yyjj", "m_jjjj", "number_jets", "central_jets", # 20
                    "high_jet_p_t", "sca_sum_pt_jjjj", "eta_jf", "m_yyjf", "pt_thrust_yy",
                    "delta_eta_yy", "eta_zepp", "phi_star_yy", "cos_theta_star_yy",
                    "pt_y1", "phi_y1", "pt_y2", "phi_y2",
                    "jet1.p_t", "jet1.eta", "jet1.phi",
                    "jet2.p_t", "jet2.eta", "jet2.phi",
                    "jet3.p_t", "jet3.eta", "jet3.phi",
                    "jet4.p_t", "jet4.eta", "jet4.phi",
                    "pt_yy_myyscaled", "eta_yy", "pt_jf", "delta_theta_yyjf", "m_yy"]

    """
    f1 = plt.figure(1, figsize=(36,36), dpi=50)
    sns.heatmap(bkg_cov_matrix, annot = True, fmt = ".2f", square=True, xticklabels=names, yticklabels=names)
    plt.suptitle("Di-photon Events with Jets", y=0.9)
    plt.title("Background Correlation Matrix")
    #plt.savefig("background_correlation.png")
    #plt.clf()

    f2 = plt.figure(2, figsize=(36,36), dpi=50)
    sns.heatmap(sig_cov_matrix, annot = True, fmt = ".2f", square=True, xticklabels=names, yticklabels=names)
    plt.suptitle("Di-photon Events with Jets", y=0.9)
    plt.title("Signal Correlation Matrix")
    #plt.savefig("signal_correlation.png")

    #plt.show()
    """

    indices = []
    for i in range(len(bkg_cov_matrix)-1):
        if bkg_cov_matrix[-1, i] >= 0.05 or sig_cov_matrix[-1, i] >= 0.05:
            indices.append(i)

    for i in indices:
        print(names[i])
    print(np.array(indices))
    return np.array(indices)



def training_data(feat,number):
    # takes in an array of the features and returns training, validation and testing datasets with the appropriate features removed
    rng = np.random.default_rng()

    indices = correlation_plots()
    dataset = np.delete(feat, indices, axis = 1)

    feat_mins = np.min(dataset, axis = 0)
    feat_maxs = np.max(dataset, axis = 0)

    # normalisation
    for i in range(len(dataset[0])):
        dataset[:, i] = (dataset[:, i] - feat_mins[i])/(feat_maxs[i] - feat_mins[i])
    size = feat.shape[0]
    background = dataset[0: int(size/2)]
    signal = dataset[int(size/2):]

    rng.shuffle(background)
    rng.shuffle(signal)

    # splits background into training, validation and testing
    bkg_train = background[0:int(0.4*size)]
    bkg_val = background[int(0.4*size):int(0.45*size)]
    bkg_test = background[int(0.45*size):]

    # splits signal into training, validation and testing
    sig_train = signal[0:int(0.4*size)]
    sig_val = signal[int(0.4*size):int(0.45*size)]
    sig_test = signal[int(0.45*size):]

    # input datasets
    x_train = np.concatenate((bkg_train, sig_train))
    x_val = np.concatenate((bkg_val, sig_val))
    x_test = np.concatenate((bkg_test, sig_test))

    # output datasets
    y_train = np.concatenate((np.zeros(len(bkg_train)), np.ones(len(sig_train)))).reshape((-1, 1))
    y_val = np.concatenate((np.zeros(len(bkg_val)), np.ones(len(sig_val)))).reshape((-1, 1))
    y_test = np.concatenate((np.zeros(len(bkg_test)), np.ones(len(sig_test)))).reshape((-1, 1))

    # joining both sets for ease of storage
    training = np.concatenate((x_train, y_train), axis = 1)
    validation = np.concatenate((x_val, y_val), axis = 1)
    testing = np.concatenate((x_test, y_test), axis = 1)
    # shuffles training dataset since we are training in mini-batches
    rng.shuffle(training)

    # save datasets
    np.savetxt("training"+str(number)+".csv", training, delimiter = ",")
    np.savetxt("validation"+str(number)+".csv", validation, delimiter=",")
    np.savetxt("testing"+str(number)+".csv", testing, delimiter=",")
