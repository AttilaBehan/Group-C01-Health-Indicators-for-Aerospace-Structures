import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import resample
from scipy.stats import pearsonr
from sklearn.preprocessing import Normalizer
from itertools import zip_longest
import pandas as pd
from scipy.signal import resample_poly 

def Tr(X):
    """
    Calculate trendability score for a set of HIs (Health Indicators)
    """
    m = X.shape[0]
    trendability_values = []

    for i in range(m):
        for j in range(i + 1, m):  # avoid duplicates and self-correlation
            x_i = X[i]
            x_j = X[j]

            # Resample to same length if needed
            if len(x_i) != len(x_j):
                min_len = min(len(x_i), len(x_j))
                x_i = resample_poly(x_i, min_len, len(x_i))
                x_j = resample_poly(x_j, min_len, len(x_j))

            rho = abs(pearsonr(x_i, x_j)[0])
            trendability_values.append(rho)

    return min(trendability_values)  # the minimum absolute correlation


def Pr(X):
    """
    Calculate prognosability score for a set of HIs

    Parameters:
        - X (numpy.ndarray): List of all extracted HIs, shape (m rows x n columns). Each row represents one HI
    Returns:
        - prognosability (float): Prognosability score for given set of HIs
    """
    # Compute M as the number of HIs, and Nfeatures as the number of timesteps
    M = len(X)
    Nfeatures = X.shape[1]

    # Initialize top and bottom of fraction in prognosability formula to zero
    top = np.zeros((M, Nfeatures))
    bottom = np.zeros((M, Nfeatures))

    # Iterate over each HI
    for j in range(M):

        # Set row in top to the final HI value for current HI
        top[j, :] = X[j, -1]

        # Compute absolute difference between initial and final values for current HI
        bottom[j, :] = np.abs(X[j, 0] - X[j, -1])

    # Compute prognosability score with formula
    prognosability = np.exp(-np.std(top) / np.mean(bottom))

    return prognosability

def Mo(X):
    """
    Calculate monotonicity score for a set of HIs

    Parameters:
        - X (numpy.ndarray): List of all extracted HIs, shape (m rows x n columns). Each row represents one HI
    Returns:
        - monotonicity (float): Monotonicity score for given set of HIs
    """
    # Initialize sum of individual monotonicities to 0
    sum_monotonicities = 0

    # Iterate over all HIs
    for i in range(len(X)):

        # Calculate the monotonicity of each HI with the Mo_single function, add to the sum
        monotonicity_i = Mo_single(X[i, :])
        sum_monotonicities += monotonicity_i

    # Compute monotonicity score by normalizing over number of HIs
    monotonicity = sum_monotonicities / np.shape(X)[0]

    return monotonicity

def Mo_single(X_single) -> float:
    """
    Calculate monotonicity score for a single HI

    Parameters:
        - X_single (numpy.ndarray): Array representing a single HI (1 row x n columns)
    Returns:
        - monotonicity_single (float): Monotonicity score for given HI
    """
    # Initialize sum as 0
    sum_samples = 0

    # Iterate over all timesteps
    for i in range(len(X_single)):

        # Initialize sum of measurements for a timestep and sum of denominator
        sum_measurements = 0
        div_sum = 0

        # Iterate over all timesteps again
        for k in range(len(X_single)):

            # Initialize sums for current timesteps (i,k)
            sub_sum = 0
            div_sub_sum = 0

            # When k is a future timestep in comparison to i
            if k > i:

                # Sum the signed difference between HI values at time k, i scaled by the time gap (k - i)
                sub_sum += (k - i) * np.sign(X_single[k] - X_single[i])

                # Sum the time gap to the denominator values
                div_sub_sum += k - i

            # Update the outer loop sums, don't do anything if k < i
            sum_measurements += sub_sum
            div_sum += div_sub_sum

        # If dividing by zero, ignore and continue on to next i value
        if div_sum == 0:
            sum_samples += 0

        # Else update sum_samples with the sum of measurements normalized by div_sum
        else:
            sum_samples += abs(sum_measurements / div_sum)

        # Compute monotonicity score by normalizing by total number of comparisons
        monotonicity_single = sum_samples / (len(X_single)-1)

    return monotonicity_single

def fitness(X, Mo_a=1.0, Tr_b=1.0, Pr_c=1.0):
    """
    Calculate fitness score for a set of HIs

    Parameters:
        - X (numpy.ndarray): List of all extracted HIs, shape (m rows x n columns). Each row represents one HI
        - Mo_a (float): Weight of monotonicity score in the fitness function, with default value 1
        - Tr_b (float): Weight of trendability score in the fitness function, with default value 1
        - Pr_c (float): Weight of prognosability score in the fitness function, with default value 1
    Returns:
        - ftn (float): Fitness score for given set of HIs
        - monotonicity (float): Monotonicity score for given set of HIs
        - trendability (float): Trendability score for given set of HIs
        - prognosability (float): Prognosability score for given set of HIs
        - error (float): Error value for given set of HIs, defined as the sum of weights (default value 3) / fitness
    """
    # Compute the 3 prognostic criteria scores
    monotonicity = Mo(X)
    trendability = Tr(X)
    prognosability = Pr(X)

    # Compute fitness score as sum of scores multiplied by their respective weights
    ftn = Mo_a * monotonicity + Tr_b * trendability + Pr_c * prognosability

    # Compute the error value, defined as the sum of the weights (default value 3) divided by the fitness score
    error = (Mo_a + Tr_b + Pr_c) / ftn 
    #print("Error: ", error)

    return ftn, monotonicity, trendability, prognosability, error

def reshape(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if any(os.path.isfile(os.path.join(output_dir, f)) for f in os.listdir(output_dir)):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path) 
    print(f"Reshaping {os.path.basename(input_dir)}") 
    for root, dir, samples in os.walk(input_dir):
        for sample in samples: 
            df=pd.read_csv(os.path.join(root, sample))
            df=df.iloc[:, 1:]
            for column in df.columns:
                filepath=os.path.join(output_dir, (column+".csv"))
                if not os.path.exists(filepath):
                    df[column].to_csv(filepath, index=False)
                else:
                    new_df=pd.read_csv(filepath)
                    new_df=pd.concat([new_df, df[column]], axis=1)
                    new_df.to_csv(filepath, index=False)

def calculate_fitness(dirname):
    mpts=[]
    for dir, root, files in os.walk(dirname): 
        print(f"Calculating fitness for {os.path.basename(dirname)}")
        for file in files:
            filepath=os.path.join(dir, file)
            df=pd.read_csv(filepath).dropna()
            df=df.drop(df.columns[0], axis=1)
            #df=df.T
            X = df.to_numpy() 
            scaler = Normalizer()
            X = scaler.fit_transform(X)
            trendability_score = Tr(X)
            monotonicity_score = Mo(X)
            prognosability_score = Pr(X)
            # print(f"Folder: {filepath}")
            # print(f"Trendability score: {trendability_score:.4f}")
            # print(f"Monotonicity score: {monotonicity_score:.4f}")
            # print(f"Prognosability score: {prognosability_score:.4f}")
            mpts.append([trendability_score, monotonicity_score, prognosability_score, file[:-4]])
    return mpts 

def plot_bar(fitness_list, column):
    feature_list = np.arange(len(fitness_list))

    #If you want to display the feature names on the x-axis 
    #feature_list = [feat[4] for feat in fitness_list]

    trendability = [feat[0] for feat in fitness_list]
    monotonicity = [feat[1] for feat in fitness_list]
    prognosability = [feat[2] for feat in fitness_list]

    total_heights = [t + m + p for t, m, p in zip(trendability, monotonicity, prognosability)]
    mu = np.mean(total_heights)


    # X locations
    x = np.arange(len(fitness_list))

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width=0.4 

    ax.bar(x, prognosability, width=bar_width, label='Prognosability', color='lightgreen')
    ax.bar(x, monotonicity, width=bar_width, bottom=prognosability, label='Monotonicity', color='salmon')
    bottom_stack = [p + m for p, m in zip(prognosability, monotonicity)]
    ax.bar(x, trendability, width=bar_width, bottom=bottom_stack, label='Trendability', color='skyblue')

    ax.axhline(mu, color='black', linestyle='--', linewidth=1.5, label=f'μ = {mu:.2f}')
    ax.text(len(x) - 0.2, mu -0.05, 'μ', color='red', fontsize=16, va='bottom')

    # Formatting
    ax.set_xlim(-0.5, len(x) - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_list, rotation=90, ha='center') 
    ax.set_ylabel('Fitness Score')
    ax.set_xlabel('Feature Index')
    ax.set_title(f'{column} Feature Scores')
    ax.set_ylim(0, 3)
    ax.legend()

    plt.tight_layout()
    plt.show()

    return mu

#plot_bar([[0.000706570200317988, 0.6424871691882003, 0.9469698275769394], [0.000275469662994271, 0.6416514184555421, 0.7263848799259812], [0.00011985572909433209, 0.7202076731973642, 0.7189581458714516], [0.00038447102526666804, 0.6316289552372024, 0.9889603678063209], [2.5305611694002514e-05, 0.6259570669879948, 0.9980119319848746], [0.0008403474852911735, 0.6349648175421373, 0.7733159847162758], [0.0027531118795852266, 0.6802649469659782, 0.8334387940222431], [0.001712534613939748, 0.6396435637672752, 0.8168213006469061], [0.0014915933182755026, 0.6394900403147826, 0.8118306042052615], [0.006236125399815662, 0.6435356510614243, 0.8274507907902924], [0.000622430591537354, 0.6485207003763706, 0.9132107111276122], [0.006995457965244589, 0.628592255396379, 0.8235310146289373], [0.002230193590876617, 0.6425776171136998, 0.912442181939915], [0.0003410050197737505, 0.7125698814358611, 0.34870666897130614], [6.356881533136748e-06, 0.6885418247273916, 0.7253646092995322], [2.123509846213567e-05, 0.655589473527618, 0.5696227040485519], [0.007560923740651704, 0.6239472783802682, 0.8188188940316458], [0.0004239730525645996, 0.6269902262685766, 0.9840240208061948], [0.00020313887195660962, 0.6263201975573108, 0.997215159876452], [2.950414610028429e-05, 0.6582088930542536, 0.5543161042329705], [0.00021217086334669732, 0.6646663988932029, 0.519041198178673], [5.705847737098868e-05, 0.5609482155873908, 0.8094170766329126], [0.03968223476141278, 0.6265406643757158, 0.8304294439155008], [0.0005291776504849147, 0.6163093378557297, 0.23108448627266162], [4.7081601200753276e-05, 0.6530253938501361, 0.9079117495941286], [0.00017262164812403835, 0.6369693994436263, 0.9237499783578855], [0.0006622750894726658, 0.6884655092902516, 0.7104602589430741], [5.1592767552871566e-05, 0.6501221344520314, 0.4619669949420277], [0.0004883530411041725, 0.63007631543714, 0.9653895521258118], [0.002352314327988686, 0.6493396408860329, 0.7472968130671457], [0.00012073981410783835, 0.7207453028071584, 0.7759295653535796], [0.0006499358029634118, 0.6478497790868922, 0.9891282694691079], [0.00013727298383448341, 0.6329794260722095, 0.9974672435387409], [0.0029063962841948676, 0.6442889870724926, 0.7943453570549306], [0.0006988871760918725, 0.690357328810937, 0.8288803549181449], [0.0019462550243502519, 0.6430875767989169, 0.8131688624560082], [0.0042705335605627, 0.6380705434313686, 0.831639626422325], [0.0021587170221157026, 0.6463567932640097, 0.8226163374313135], [0.0005005732134748808, 0.6505583076717097, 0.9136434282848186], [0.0004630901539740273, 0.5874903675934602, 0.789285949031476], [0.001997001499978196, 0.631609318516535, 0.9300548693322723], [0.00010648986050482839, 0.7356794751640116, 0.3733897443384394], [0.0003037067283162276, 0.6396343404590826, 0.9247940647676578], [0.00010411576304890369, 0.6489344103777095, 0.5169576382317068], [4.26385383157403e-05, 0.7074595736451408, 0.5972666346666536], [0.00015849625921461608, 0.6514615968224214, 0.993136460226939], [6.123981891638548e-05, 0.6565674417220809, 0.9993569270855452], [7.072336000218948e-05, 0.6446994242870531, 0.6130211692589267], [0.0004308815501304804, 0.6825898156825991, 0.8238974016679238], [0.0006851262949596759, 0.6534457981880665, 0.803188698374959], [0.0006421616244207326, 0.6603858913137264, 0.6965360908010497], [0.0016323231084819367, 0.6622592642180273, 0.8168368648571852], [0.002742779228596788, 0.6608293539221376, 0.9129043313952913], [0.0006445869731653842, 0.6062144270391695, 0.8029819305285169], [0.0011407474728794276, 0.645206259948528, 0.8879480631679304], [3.960595310909573e-05, 0.7112253611222682, 0.12642297043226916], [6.681624650314855e-05, 0.6264231415777811, 0.9488549432066685], [7.781176712469981e-05, 0.649847220362684, 0.6328767289606549], [0.00011793277703565398, 0.7062102616741793, 0.71683650411843], [0.00014246564393177036, 0.6343841954151233, 0.9905798708993535], [0.0002565011102379378, 0.6455485636928935, 0.9980909691586906], [5.645056436659468e-05, 0.6481420984513766, 0.6912644378775775], [0.0024359383059576745, 0.6973793903690809, 0.8311737308764457], [0.005708686224487156, 0.6317936359173474, 0.8169110692703015], [0.00021614198632065318, 0.6507354844468245, 0.7532280313378341], [0.0050725091098472175, 0.634912006664584, 0.8294993531652856], [0.003021565031505316, 0.6530822213296443, 0.9129656487470647], [0.0019938816164904727, 0.6034518974725159, 0.8216887956732604], [0.004974677971287422, 0.624361732196784, 0.9089441086393627], [4.095401134246901e-05, 0.7005044554529093, 0.2703185013819701], [0.0013303166296020907, 0.7101927968938281, 0.8423652393246839], [0.001036402006823492, 0.6630425015992027, 0.7202638924113188], [0.0003157515984173742, 0.6727617857514766, 0.8070220284325211], [6.508559272904585e-06, 0.6876907514020916, 0.9897892200580621], [3.991237968147335e-05, 0.6697238958063702, 0.9974826096728907], [0.0003719649323528154, 0.6577304711325331, 0.7628718406084732], [0.002651031991122099, 0.6467654453221463, 0.8544901926191393], [3.936077931988337e-05, 0.6961419794409486, 0.7680140764454144], [0.0015627134711543034, 0.6620773270257809, 0.8041860798458331], [0.0037755987176100736, 0.6961378140759586, 0.8176494076051132], [0.0010533418990471, 0.6590648755597207, 0.9190338051488597], [8.969909759493078e-05, 0.6283650942413829, 0.6997418880175995], [0.00032874614325276337, 0.7110522009491083, 0.7936623521569744], [0.0004830356210105413, 0.6340805700599516, 0.5903121863495477]], "FFT")

def write_scores(dir, column, fitness_scores):
    df=pd.read_csv(dir)
    if df.empty:
        df = pd.DataFrame(index=range(len(fitness_scores)), columns=df.columns)
    df[column] = np.nan
    df[column] = df[column].astype(object)
    df.loc[df.index[:len(fitness_scores)], column] = pd.Series(fitness_scores, index=df.index[:len(fitness_scores)], dtype='object')
    df.dropna()
    df.to_csv(dir, index=False) 