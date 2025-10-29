"""
Script for helper numerical tools
"""
import jax
from jax import grad, lax, config, jit, vmap
from jax.scipy.special import gamma, factorial
from functools import partial
import numpy as np
import jax.numpy as jnp
import equinox as eqx

config.update("jax_enable_x64", True)

### BEGINNING OF WIGNER ROTATION FOR LENSING ###

def wigner_d_matrix(mu, ells, m, n):
    """
    Compute Wigner d-matrix elements for rotation.

    Recursively computes reduced Wigner d-matrix elements d^ell_{mn}(beta)
    for CMB lensing calculations using three-term recurrence relation.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle beta
    ells : array
        Multipole values [m, m+1, m+2, ..., ellmax]
    m : int
        First index (must be positive and >= |n|)
    n : int
        Second index (must satisfy |n| <= m)

    Returns:
    --------
    array
        Wigner d-matrix elements, shape (len(mu), len(ells))
    """
    
    # base case: ell = m
    def base_val(mu):
        beta = jnp.arccos(mu)
        norm = jnp.sqrt((2*m+1)/2) * jnp.sqrt(factorial(2*m)/(factorial(m+n)*factorial(m-n)))
        return norm * jnp.cos(beta/2.)**(m+n)*(-jnp.sin(beta/2.))**(m-n)
        #return norm * jnp.sqrt((1+mu)/2)**(m+n) * jnp.sqrt((1-mu)/2)**(m-n)

    normA = jnp.sqrt((2*ells+3)/(2*ells+1))
    normC = jnp.sqrt((2*ells+3)/(2*ells-1))
    denom = jnp.sqrt((ells+1)**2-m**2) * jnp.sqrt((ells+1)**2-n**2)
    A = jnp.nan_to_num(normA * (ells+1)*(2*ells+1) / denom, 0)
    B = jnp.nan_to_num(-A * m * n / ells / (ells+1), 0)
    C = jnp.nan_to_num(-normC * jnp.sqrt(ells**2-m**2) * jnp.sqrt(ells**2-n**2) / denom * (ells+1)/ells, 0)

    def one_mu(mu):
        d_start = base_val(mu) # Corresponds to ellmin = m
                
        def recursive_dlp1(carry, inputs):
            # For the first iteration, will take d^m_{mn} and d^m_{mn}=0., compute d^{m+1}_{mn}.
            dl, dlm1 = carry 
            a, b, c = inputs

            # Compute dlp1
            dlp1 = a*mu*dl + b*dl + c*dlm1

            # Save dl, then make dl->dlm1, dlp1->dl
            return (dlp1, dl), dl

        # run scan for l = 2..lmax-1
        (_, _), res = lax.scan(recursive_dlp1, (d_start, 0.), (A, B, C))
        return res * jnp.sqrt(2./(2.*ells+1))

    return vmap(one_mu)(mu)

def d00(mu, ells):
    """
    Compute Wigner d-matrix elements d^ell_{00}.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle
    ells : array
        Multipole values starting from ell=2

    Returns:
    --------
    array
        d^ell_{00} elements for ells >= 2
    """
    # ells go from (2, 3, 4, ..., ellmax)
    ells_patched = jnp.concatenate((jnp.array([0, 1]), ells))
    res = wigner_d_matrix(mu, ells_patched, 0, 0)
    return res[:, 2:] # Return only the ells >= 2

def d1n(mu, ells, n):
    """
    Compute Wigner d-matrix elements d^ell_{1n}.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle
    ells : array
        Multipole values
    n : int
        Second index (|n| <= 1)

    Returns:
    --------
    array
        d^ell_{1n} elements
    """
    # Wigner matrices where m=1, and |n|<=m.
    ells_patched = jnp.concatenate((jnp.array([1]), ells))
    res = wigner_d_matrix(mu, ells_patched, 1, n)
    return res[:, 1:]

def d2n(mu, ells, n):
    """
    Compute Wigner d-matrix elements d^ell_{2n}.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle
    ells : array
        Multipole values
    n : int
        Second index (|n| <= 2)

    Returns:
    --------
    array
        d^ell_{2n} elements
    """
    # Wigner matrices where m=2, and |n|<=m.
    res = wigner_d_matrix(mu, ells, 2, n)
    return res

def d3n(mu, ells, n):
    """
    Compute Wigner d-matrix elements d^ell_{3n}.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle
    ells : array
        Multipole values
    n : int
        Second index (|n| <= 3)

    Returns:
    --------
    array
        d^ell_{3n} elements, zero-padded for ell < 3
    """
    # Wigner matrices where m=3, and |n|<=m.
    ells_sliced = ells[1:] # Compute starting at ell=3
    res = wigner_d_matrix(mu, ells_sliced, 3, n)
    res_patched = jnp.concatenate((jnp.zeros((mu.size, 1)), res), axis=1) # Pad zeros for ell<3.
    return res_patched

def d4n(mu, ells, n):
    """
    Compute Wigner d-matrix elements d^ell_{4n}.

    Parameters:
    -----------
    mu : array
        Cosine of rotation angle
    ells : array
        Multipole values
    n : int
        Second index (|n| <= 4)

    Returns:
    --------
    array
        d^ell_{4n} elements, zero-padded for ell < 4
    """
    # Wigner matrices where m=4, and |n|<=m.
    ells_sliced = ells[2:] # Compute starting at ell=4
    res = wigner_d_matrix(mu, ells_sliced, 4, n)
    res_patched = jnp.concatenate((jnp.zeros((mu.size, 2)), res), axis=1) # Pad zeros for ell<4.
    return res_patched

### END OF WIGNER ROTATION FOR LENSING ###



def fast_interp(x, xp_min, xp_max, fp):
    """
    Fast 1D linear interpolation for uniformly-spaced grids.

    Optimized interpolation that avoids searchsorted by exploiting
    uniform grid spacing. Significantly faster than jnp.interp for
    large arrays.

    Parameters:
    -----------
    x : float or array
        Query points for interpolation
    xp_min : float
        Minimum value of interpolation grid
    xp_max : float
        Maximum value of interpolation grid
    fp : array
        Function values on uniform grid

    Returns:
    --------
    float or array
        Interpolated values at query points

    Notes:
    ------
    Credit: JAX issue #16182 (https://github.com/jax-ml/jax/issues/16182)
    Assumes fp is uniformly spaced between xp_min and xp_max.
    """
    # The official jnp.interp is very slow becuase it uses searchsorted.
    # Therefore, we leverage the fact that the fp is linearly increasing, evenly spaced, and has a known range
    # to make this operation much faster.
    eps = 1.e-6
    n = fp.shape[-1]
    i = (x - xp_min) / (xp_max - xp_min) * n
    i = jnp.clip(i, eps, n - 1.0 - eps)  # Avoid index out of range
    i_lower = jnp.floor(i).astype(jnp.int32)
    i_upper = jnp.minimum(i_lower + 1, n - 1)
    w_upper = i - i_lower
    w_lower = 1.0 - w_upper
    return w_lower * fp[i_lower] + w_upper * fp[i_upper]


def bilinear_interp(x, y, z, xq, yq):
    """
    Bilinear interpolation on 2D regular grid.

    Performs bilinear interpolation to evaluate function at query point
    (xq, yq) given values on a regular 2D grid.

    Parameters:
    -----------
    x : array
        1D array of x-coordinates (must be sorted)
    y : array
        1D array of y-coordinates (must be sorted)
    z : array
        2D array of function values, shape (len(y), len(x))
    xq : float
        Query x-coordinate
    yq : float
        Query y-coordinate

    Returns:
    --------
    float
        Interpolated value at (xq, yq)

    Notes:
    ------
    Uses standard bilinear interpolation formula with four nearest
    grid points.
    """
    # find indices for x and y
    ix = jnp.clip(jnp.searchsorted(x, xq) - 1, 0, x.size - 2)
    iy = jnp.clip(jnp.searchsorted(y, yq) - 1, 0, y.size - 2)

    # grid corner points
    x0, x1 = x[ix], x[ix + 1]
    y0, y1 = y[iy], y[iy + 1]

    # fractional positions
    tx = (xq - x0) / (x1 - x0)
    ty = (yq - y0) / (y1 - y0)

    # get z values
    z00 = z[iy, ix]
    z01 = z[iy, ix + 1]
    z10 = z[iy + 1, ix]
    z11 = z[iy + 1, ix + 1]

    # bilinear interpolation
    return (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z01 + (1 - tx) * ty * z10 + tx * ty * z11


