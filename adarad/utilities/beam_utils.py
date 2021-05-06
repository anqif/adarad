import numpy as np

# Generate xy-coordinate pairs from line angles and displacements.
def line_segments(angles, d_vec, n_grid, xlim = (-1,1), ylim = (-1,1)):
    if np.any(angles < 0) or np.any(angles > np.pi):
        raise ValueError("angles must all be in [0,pi]")

    n_angles = len(angles)
    n_offsets = len(d_vec)
    n_lines = n_angles*n_offsets
    segments = np.zeros((n_lines,2,2))   # n_lines x n_points x n_dims = 2.

    xc = (xlim[1] + xlim[0])/2
    yc = (ylim[1] + ylim[0])/2
    x_scale = (xlim[1] - xlim[0])/n_grid   # (x_max - x_min)/(x_len_pixels)
    y_scale = (ylim[1] - ylim[0])/n_grid   # (y_max - y_min)/(y_len_pixels)
    dydx_scale = (ylim[1] - ylim[0])/(xlim[1] - xlim[0])

    k = 0
    for i in range(n_angles):
        # Slope of line.
        slope = dydx_scale*np.tan(angles[i])

        # Center of line.
        x0 = xc - x_scale*d_vec*np.sin(angles[i])
        y0 = yc + y_scale*d_vec*np.cos(angles[i])

        # Endpoints of line.
        for j in range(n_offsets):
            if slope == 0:
                segments[k,0,:] = [xlim[0], y0[j]]
                segments[k,1,:] = [xlim[1], y0[j]]
            elif np.isinf(slope):
                segments[k,0,:] = [x0[j], ylim[0]]
                segments[k,1,:] = [x0[j], ylim[1]]
            else:
                # y - y0 = slope*(x - x0).
                ys = y0[j] + slope*(xlim - x0[j])
                xs = x0[j] + (1/slope)*(ylim - y0[j])

                # Save points at edge of grid.
                idx_y = (ys >= ylim[0]) & (ys <= ylim[1])
                idx_x = (xs >= xlim[0]) & (xs <= xlim[1])
                e1 = np.column_stack([xlim, ys])[idx_y]
                e2 = np.column_stack([xs, ylim])[idx_x]
                edges = np.row_stack([e1, e2])

                segments[k,0,:] = edges[0]
                segments[k,1,:] = edges[1]
            k = k + 1
    return segments

# Construct line integral matrix.
def line_integral_mat(structures, angles = 10, n_bundle = 1, offset = 10):
    m_grid, n_grid = structures.shape
    K = np.unique(structures).size

    if m_grid != n_grid:
        raise NotImplementedError("Only square grids are supported")
    if np.isscalar(angles):
        angles = np.linspace(0, np.pi, angles+1)[:-1]
    if n_bundle <= 0:
        raise ValueError("n_bundle must be a positive integer")
    if offset < 0:
        raise ValueError("offset must be a nonnegative number")

    # A_{kj} = fraction of beam j that falls in region k.
    n_angle = len(angles)
    n_bundle = int(n_bundle)
    n = n_angle*n_bundle
    A = np.zeros((K, n))

    # Orthogonal offsets of line from image center given in pixel lengths.
    # Positive direction = northwest for angle <= np.pi/2, southwest for angle > np.pi/2.
    n_half = n_bundle//2
    d_vec = np.arange(-n_half, n_half+1)
    if n_bundle % 2 == 0:
        d_vec = d_vec[:-1]
    d_vec = offset*d_vec

    j = 0
    for i in range(n_angle):
        for d in d_vec:
            # Flip angle since we measure counterclockwise from x-axis.
            L = line_pixel_length(d, np.pi-angles[i], n_grid)
            for k in range(K):
                A[k,j] = np.sum(L[structures == k])
            j = j + 1
    return A, angles, d_vec

def line_pixel_length(d, theta, n):
    """
    Image reconstruction from line measurements.

    Given a grid of n by n square pixels and a line over that grid,
    compute the length of line that goes over each pixel.

    Parameters
    ----------
    d : displacement of line, i.e., distance of line from center of image,
        measured in pixel lengths (and orthogonally to line).
    theta : angle of line, measured in radians clockwise from x-axis.
            Must be between 0 and pi, inclusive.
    n : image size is n by n.

    Returns
    -------
    Matrix of size n by n (same as image) with length of the line over
    each pixel. Most entries will be zero.
    """
    # For angle in [pi/4,3*pi/4], flip along diagonal (transpose) and call recursively.
    if theta > np.pi/4 and theta < 3*np.pi/4:
        return line_pixel_length(d, np.pi/2-theta, n).T

    # For angle in [3*pi/4,pi], redefine line to go in opposite direction.
    if theta > np.pi/2:
        d = -d
        theta = theta - np.pi

    # For angle in [-pi/4,0], flip along x-axis (up/down) and call recursively.
    if theta < 0:
        return np.flipud(line_pixel_length(-d, -theta, n))

    if theta > np.pi/2 or theta < 0:
        raise ValueError("theta must be in [0,pi]")

    L = np.zeros((n,n))
    ct = np.cos(theta)
    st = np.sin(theta)

    x0 = n/2 - d*st
    y0 = n/2 + d*ct

    y = y0 - x0*st/ct
    jy = int(np.ceil(y))
    dy = (y + n) % 1

    for jx in range(n):
        dynext = dy + st/ct
        if dynext < 1:
            if jy >= 1 and jy <= n:
                L[n-jy, jx] = 1/ct
            dy = dynext
        else:
            if jy >= 1 and jy <= n:
                L[n-jy, jx] = (1-dy)/st
            if jy+1 >= 1 and jy + 1 <= n:
                L[n-(jy+1), jx] = (dynext-1)/st
            dy = dynext - 1
            jy = jy + 1
    return L
