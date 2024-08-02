


def emd_1d(x_a, x_b, a=None, b=None, metric='sqeuclidean', p=1., dense=True,
           log=False, check_marginals=True):
    """Solves the Earth Movers distance problem between 1d measures and returns
    the OT matrix


    .. math::
        \gamma = arg\min_\gamma \sum_i \sum_j \gamma_{ij} d(x_a[i], x_b[j])

        s.t. \gamma 1 = a,
             \gamma^T 1= b,
             \gamma\geq 0

    where :

    - d is the metric
    - x_a and x_b are the samples
    - a and b are the sample weights

    When 'minkowski' is used as a metric, :math:`d(x, y) = |x - y|^p`.

    Uses the algorithm detailed in [1]_

    Parameters
    ----------
    x_a : (ns,) or (ns, 1) ndarray, float64
        Source dirac locations (on the real line)
    x_b : (nt,) or (ns, 1) ndarray, float64
        Target dirac locations (on the real line)
    a : (ns,) ndarray, float64, optional
        Source histogram (default is uniform weight)
    b : (nt,) ndarray, float64, optional
        Target histogram (default is uniform weight)
    metric: str, optional (default='sqeuclidean')
        Metric to be used. Only strings listed in :func:`ot.dist` are accepted.
        Due to implementation details, this function runs faster when
        `'sqeuclidean'`, `'cityblock'`,  or `'euclidean'` metrics are used.
    p: float, optional (default=1.0)
         The p-norm to apply for if metric='minkowski'
    dense: boolean, optional (default=True)
        If True, returns math:`\gamma` as a dense ndarray of shape (ns, nt).
        Otherwise returns a sparse representation using scipy's `coo_matrix`
        format. Due to implementation details, this function runs faster when
        `'sqeuclidean'`, `'minkowski'`, `'cityblock'`,  or `'euclidean'` metrics
        are used.
    log: boolean, optional (default=False)
        If True, returns a dictionary containing the cost.
        Otherwise returns only the optimal transportation matrix.
    check_marginals: bool, optional (default=True)
        If True, checks that the marginals mass are equal. If False, skips the
        check.

    Returns
    -------
    gamma: (ns, nt) ndarray
        Optimal transportation matrix for the given parameters
    log: dict
        If input log is True, a dictionary containing the cost


    Examples
    --------

    Simple example with obvious solution. The function emd_1d accepts lists and
    performs automatic conversion to numpy arrays

    >>> import ot
    >>> a=[.5, .5]
    >>> b=[.5, .5]
    >>> x_a = [2., 0.]
    >>> x_b = [0., 3.]
    >>> ot.emd_1d(x_a, x_b, a, b)
    array([[0. , 0.5],
           [0.5, 0. ]])
    >>> ot.emd_1d(x_a, x_b)
    array([[0. , 0.5],
           [0.5, 0. ]])

    References
    ----------

    .. [1]  Peyré, G., & Cuturi, M. (2017). "Computational Optimal
        Transport", 2018.

    See Also
    --------
    ot.lp.emd : EMD for multidimensional distributions
    ot.lp.emd2_1d : EMD for 1d distributions (returns cost instead of the
        transportation matrix)
    """
    x_a, x_b = list_to_array(x_a, x_b)
    nx = get_backend(x_a, x_b)
    if a is not None:
        a = list_to_array(a, nx=nx)
    if b is not None:
        b = list_to_array(b, nx=nx)

    assert (x_a.ndim == 1 or x_a.ndim == 2 and x_a.shape[1] == 1), \
        "emd_1d should only be used with monodimensional data"
    assert (x_b.ndim == 1 or x_b.ndim == 2 and x_b.shape[1] == 1), \
        "emd_1d should only be used with monodimensional data"

    # if empty array given then use uniform distributions
    if a is None or a.ndim == 0 or len(a) == 0:
        a = nx.ones((x_a.shape[0],), type_as=x_a) / x_a.shape[0]
    if b is None or b.ndim == 0 or len(b) == 0:
        b = nx.ones((x_b.shape[0],), type_as=x_b) / x_b.shape[0]

    # ensure that same mass
    if check_marginals:
        np.testing.assert_almost_equal(
            nx.to_numpy(nx.sum(a, axis=0)),
            nx.to_numpy(nx.sum(b, axis=0)),
            err_msg='a and b vector must have the same sum',
            decimal=6
        )
    b = b * nx.sum(a) / nx.sum(b)

    x_a_1d = nx.reshape(x_a, (-1,))
    x_b_1d = nx.reshape(x_b, (-1,))
    perm_a = nx.argsort(x_a_1d)
    perm_b = nx.argsort(x_b_1d)

    G_sorted, indices, cost = emd_1d_sorted(
        nx.to_numpy(a[perm_a]).astype(np.float64),
        nx.to_numpy(b[perm_b]).astype(np.float64),
        nx.to_numpy(x_a_1d[perm_a]).astype(np.float64),
        nx.to_numpy(x_b_1d[perm_b]).astype(np.float64),
        metric=metric, p=p
    )

    G = nx.coo_matrix(
        G_sorted,
        perm_a[indices[:, 0]],
        perm_b[indices[:, 1]],
        shape=(a.shape[0], b.shape[0]),
        type_as=x_a
    )
    if dense:
        G = nx.todense(G)
    elif str(nx) == "jax":
        warnings.warn("JAX does not support sparse matrices, converting to dense")
    if log:
        log = {'cost': nx.from_numpy(cost, type_as=x_a)}
        return G, log
    return G
