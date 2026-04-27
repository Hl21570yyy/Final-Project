def get_default_config(n_cat, flow_dim):
    """
    n_cat:    POI 类别数（从 data_prepare 得到）
    flow_dim: Flow 特征维度 = 28 (7天×4时段)
    """
    return dict(
        Prediction=dict(
            arch1=[48, 96, 48],
            arch2=[48, 96, 48],
        ),
        Autoencoder=dict(
            arch1=[n_cat, 128, 128, 48],     # ← 原来 244，改成你的 n_cat
            arch2=[flow_dim, 128, 128, 48],  # ← 原来 270，改成 28
            activations1='relu',
            activations2='relu',
            batchnorm=True,
        ),
        training=dict(
            seed=42,
            start_dual_prediction=50,
            epoch=1000,
            lr=0.01,
            alpha=9,
            lambda1=1,
            lambda2=1,
            sigma=0.0001,
        ),
        print_num=200,
    )
