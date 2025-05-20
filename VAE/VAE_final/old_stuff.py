import matplotlib.pyplot as plt

''' PLOTS GRAPH OF HI's vs % OF LIFETIME FOR DIFFERENT TEST SAMPLES'''
def plot_HI_graph(HI_all, dataset_name, sp_method_name, folder_output, show_plot=True, n_plot_rows=4, n_plot_col=3):
    ''' Takes data from array HI_all, which contains stacked arrays of HI vs lifetime for samples
        Plots graph for each value of test panel
        
        Inputs:
            - HI_all (arr): stacked arrays of HI vs lifetime for samples
            - dataset_name (str): name to identify data for graph title
            - sp_method_name (str): which sp method was used on data
            - n_plot_rows, n_plot_cols (int): dimensions of subplots (product=n_samples)
            
        Outputs:
            - Prints graphs and saves figure to folder_output'''
    n_samples = n_plot_rows*n_plot_col
    x = np.linspace(0,100, HI_all.shape[1])
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink', 'gray', 'lime', 'violet', 'yellow']
    # Create grid of subplots
    fig, axes = plt.subplots(n_plot_rows, n_plot_col, figsize=(15, 12))

    # Flatten the axes array for simple iteration
    axes = axes.flatten()
    # Plot each row of y against x
    for i, ax in enumerate(axes):
        y = HI_all[i*n_samples:i*n_samples+n_samples,:]
        for j in range(y.shape[0]):  # Plot all rows/samples
            ax.plot(x, y[j,:], color=colors[j], label=f'Sample{j+1}')
            ax.set_title(f"Test sample {j + 1}")
            ax.grid(True)

    # Add a single legend and big title for all subplots
    fig.legend(loc='center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    fig.suptitle(f'HIs constructed by VAE using {dataset_name} and {sp_method_name} features')
    plt.tight_layout()

    # Save the figure to file
    filename = f'HI_graphs_VAE_{dataset_name}_{sp_method_name}.png'
    file_path = os.path.join(folder_output, filename)
    plt.savefig(file_path)
    if show_plot:
        plt.show()



    # df_test_resampled = pd.DataFrame()
    # df_val_resampled = pd.DataFrame()
    # for col in df_test.columns: # interpolates test data columns so they are sampe length as target rows of train data
    #     original = df_test[col].values
    #     og = df_val[col].values
    #     x_original = np.linspace(0, 1, len(original))
    #     x_val_original = np.linspace(0,1,len(og))
    #     x_target = np.linspace(0, 1, target_rows)
    #     interpolated = np.interp(x_target, x_original, original)
    #     interp = np.interp(x_target, x_val_original, og)
    #     df_test_resampled[col] = interpolated
    #     df_val_resampled[col] = interp
