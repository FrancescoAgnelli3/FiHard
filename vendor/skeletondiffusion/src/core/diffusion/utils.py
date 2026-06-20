import torch

def dim_null_space(matrix):
    assert matrix.shape[-1] == matrix.shape[-2], "Matrix must be square"
    # rank = torch.linalg.matrix_rank(matrix) This is not set to accuracy of PYTORCH float32
    # 1.0 + eps != 1.0
    # torhc.tensor(1.0) + 0.7e-7!= torhc.tensor(1.0)
    return torch.sum(torch.linalg.eigh(matrix)[0].abs() < 0.7e-7)
        
def is_positive_def(matrix):
    #M is symmetric or Hermitian, and all its eigenvalues are real and positive.
    assert  torch.allclose(matrix.transpose(-1, -2), matrix), "Matrix must be symmetric"
    eigenvalues = torch.linalg.eigvals(matrix)
    is_pos_def = (torch.real(eigenvalues)> 0).all()
    if is_pos_def:
        assert torch.isreal(eigenvalues).all(), "Eigenvalues must be real"
    return (torch.real(eigenvalues)> 0).all()

def make_positive_definite(matrix, epsilon=1e-6, if_submin=False):

    eigenvalues = torch.linalg.eigvals(matrix)
    # assert torch.isreal(eigenvalues).all()
    if is_positive_def(matrix):
        print("Input Matrix was positive Definitive without adding spectral norm to the diagonal")
        return matrix

    eigenvalues = torch.real(eigenvalues)
    if not if_submin:
        max_eig = eigenvalues.abs().max() #
        pos_def_matrix = matrix + torch.eye(matrix.shape[0])*(max_eig + epsilon)
    else:
        min_eig = eigenvalues.min()
        pos_def_matrix = matrix + torch.eye(matrix.shape[0])*(- min_eig + epsilon)
    assert  dim_null_space(pos_def_matrix) == 0
    return pos_def_matrix

def normalize_cov(Sigma_N:torch.Tensor, Lambda_N:torch.Tensor, U:torch.Tensor, if_sigma_n_scale=True, sigma_n_scale='spectral', **kwargs):
    N, _ = Sigma_N.shape
    assert Lambda_N.shape == (N,)
    assert U.shape == (N, N)

    if if_sigma_n_scale:
        # decrease the scale of Sigma_N to make it more similar to the identity matrix
        if sigma_n_scale == 'spectral':
            relative_scale_factor = Lambda_N.max()
        else:
            if sigma_n_scale == 'frob':
                relative_scale_factor = Lambda_N.sum()/N
            else:
                assert 0, "Not implemented"
        
        Lambda_N = Lambda_N/relative_scale_factor
        
        Sigma_N = Sigma_N/relative_scale_factor
        cond = U @ torch.diag(Lambda_N) @ U.mT
        assert torch.isclose(Sigma_N, cond, atol=1e-06).all(), "Sigma_N must be equal to U @ Lambda_N @ U.t()"
        # Sigma_N[Sigma_N>0.] = (Sigma_N + Sigma_N.t())[Sigma_N>0.]/2           
    cond = Lambda_N>0.7e-7
    assert (cond).all(), f"Lambda_N must be positive definite: {Lambda_N}"
    assert is_positive_def(Sigma_N), "Sigma_N must be positive definite" 
    # print("Frobenius Norm of SigmaN: ", torch.linalg.matrix_norm(Sigma_N, ord='fro').mean().item(), "Spectral Norm of SigmaN: ", Lambda_N.max(dim=-1)[0].mean().item())
    return Sigma_N, Lambda_N


def get_cov_from_corr(correlation_matrix: torch.Tensor, if_sigma_n_scale=True, sigma_n_scale='spectral', if_run_as_isotropic=False, diffusion_covariance_type='skeleton-diffusion', **kwargs):
    N, _ = correlation_matrix.shape
    
    if if_run_as_isotropic:
        if diffusion_covariance_type == 'skeleton-diffusion':
            Lambda_N = torch.ones(N, device=correlation_matrix.device)
            Sigma_N = torch.zeros_like(correlation_matrix)
            U = torch.eye(N, device=correlation_matrix.device)
        elif diffusion_covariance_type == 'anisotropic':
            Lambda_N = torch.ones(N, device=correlation_matrix.device)
            Sigma_N = torch.eye(N, device=correlation_matrix.device)
            U = torch.eye(N, device=correlation_matrix.device)
        else: 
            Lambda_N = torch.zeros(N, device=correlation_matrix.device)
            Sigma_N = torch.zeros_like(correlation_matrix)
            U = torch.eye(N, device=correlation_matrix.device)
    else: 
        Sigma_N = make_positive_definite(correlation_matrix)
        Lambda_N, U = torch.linalg.eigh(Sigma_N, UPLO='L')

        Sigma_N, Lambda_N = normalize_cov(Sigma_N=Sigma_N, Lambda_N=Lambda_N, U=U, if_sigma_n_scale=if_sigma_n_scale, sigma_n_scale=sigma_n_scale, **kwargs)
    return Sigma_N, Lambda_N, U


def verify_noise_scale(diffusion):
    N, *_ = diffusion.Lambda_N.shape
    alphas = 1 - diffusion.betas
    noise = diffusion.get_noise((2000, diffusion.num_timesteps, N))
    zeta_noise = torch.sqrt(diffusion.Lambda_t.unsqueeze(0)) * noise
    print("current: ", (zeta_noise**2).sum(-1).mean(0))
    print("original standard gaussian diffusion: ",(1-alphas) * zeta_noise.shape[-1])


def plot_matrix(matrix):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np

    Sigma_N = matrix.cpu().clone().numpy()
    color = 'Purples'
    cmap = matplotlib.colormaps[color].set_bad("white")
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # colormap_r = ListedColormap(cmap.colors[::-1])
    # norm = mpl.colors.Normalize(vmin=0.000, vmax=1.0)

    fig, ax = plt.subplots(1,1, figsize=(6, 6),sharex=True,  subplot_kw=dict(box_aspect=1),)
    # cax = fig.add_axes([0.93, 0.15, 0.01, 0.7])  # Adjust the position and size of the colorbar
    # for i, ax in enumerate(axes):
    vmax = Sigma_N.max()
    Sigma_N[Sigma_N <=0.0000] = np.nan
    im = ax.imshow(Sigma_N, cmap=color, vmin=0., vmax=vmax)
    ax.set_xticks(np.arange(len(Sigma_N)))
    ax.set_xticklabels(labels=list(skeleton.node_dict.values()), rotation=45, ha="right",
            rotation_mode="anchor")
    ax.set_yticks(np.arange(len(Sigma_N)))
    ax.set_yticklabels(labels=list(skeleton.node_dict.values()), rotation=45, ha="right",
            rotation_mode="anchor")
    ax.set_xticks(np.arange(-0.5, len(Sigma_N), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(Sigma_N), 1), minor=True)
    ax.grid(True, which='minor', color='gray', linestyle='--', linewidth=0.5)
    ax.grid(False, which='major')
    # ax.set_title(list(method2sigman.keys())[i])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cmap=cmap, cax=cax)
    #     plt.title('Adjancecy Matrix')
    plt.show()
        # fig.savefig("../paper_plots/sigmaN.pdf", format="pdf", bbox_inches="tight")

    # ax.set_yticklabels(labels=[f'pixel {i+1}' for i in range(len(Sigma_N))], rotation=45, ha="right",


def plot_noise(matrix):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import numpy as np
    noise_temp = self.U@((0.5 * posterior_log_variance).exp() * noise)
    matrix = noise_temp[0]
    plt.close('all')
    Sigma_N = matrix.cpu().clone().numpy()
    color = 'Greys'
    cmap = matplotlib.colormaps[color].set_bad("white")
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # colormap_r = ListedColormap(cmap.colors[::-1])
    # norm = mpl.colors.Normalize(vmin=0.000, vmax=1.0)

    fig, ax = plt.subplots(1,1, figsize=(6, 6))
    # cax = fig.add_axes([0.93, 0.15, 0.01, 0.7])  # Adjust the position and size of the colorbar
    # for i, ax in enumerate(axes):
    vmax = Sigma_N.max()
    # Sigma_N[Sigma_N <=0.0000] = np.nan
    im = ax.imshow(Sigma_N, cmap=color) #, vmin=0., vmax=vmax)
    # ax.set_xticks(np.arange(Sigma_N.shape[1]))
    # ax.set_xticklabels(labels=list(skeleton.node_dict.values()), rotation=45, ha="right",
    #         rotation_mode="anchor")
    # ax.set_yticks(np.arange(Sigma_N.shape[0]))
    # ax.set_yticklabels(labels=list(skeleton.node_dict.values()), rotation=45, ha="right",
    #         rotation_mode="anchor")
    # ax.set_xticks(np.arange(-0.5, Sigma_N.shape[1], 1), minor=True)
    # ax.set_yticks(np.arange(-0.5, Sigma_N.shape[0], 1), minor=True)
    # ax.grid(True, which='minor', color='gray', linestyle='--', linewidth=0.5)
    # ax.grid(False, which='major')
    # ax.set_title(list(method2sigman.keys())[i])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cmap=cmap, cax=cax)
    #     plt.title('Adjancecy Matrix')
    plt.show()
    fig.savefig("./corr_noise_T.png", format="png", bbox_inches="tight")
        # fig.savefig("../paper_plots/sigmaN.pdf", format="pdf", bbox_inches="tight")

    # ax.set_yticklabels(labels=[f'pixel {i+1}' for i in range(len(Sigma_N))], rotation=45, ha="right",