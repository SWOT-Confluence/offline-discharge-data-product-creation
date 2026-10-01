import numpy as np
import pandas as pd

def check_na(value):
    """ check if the specified value is None, '', pd.na, np.nan, is_empty or masked """
    #By Isadora
    if value is None:
        return True

    if value == '':
        return True

    if value == '--':
        return True

    if value is np.ma.masked:
        return True

    if np.ma.is_masked(value):
        return True

    if isinstance(value, np.ma.core.MaskedConstant):
        return True

    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass

    try:
        if np.isnan(value):
            return True
    except TypeError:
        pass

    try:
        if value.mask:
            return True
    except AttributeError:
        pass

    try:
        if value.is_empty:
            return True
    except AttributeError:
        pass

    try:
        if value < -1e8:
            return True
    except TypeError:
        pass

    return False

def fnc_APR(_z, _Ws, _Zs):
    """Computes hydraulic parameters from a water elevation and a cross section
    description. Can be run for 1z scaler value, or also vactor to process several
    z in the same call.
    NOTE Elevations Zs must be increasing
    NOTE Elevations Zs must be strictly > in the interpolation zone
    NOTE Ws and Zs must be the same length
    NOTE Ws must be >= 0

    Example input:
    results = fnc_APR([1.5], [1, 1, 2, 2], [0, 1, 1, 2])
    results = fnc_APR([1.1, 2], [1, 1, 2, 2], [0, 1, 1, 2])

    Parameters
    ----------
    _z : float vector
        a given water elevation (m). Zobs.
    _Ws : float vector
        Width of the section (m)
    _Zs : float vector
        Elevation of the section (m)

    Returns
    -------
    A : float scaler/vector
        wetted area (m^2)
    P : real scaler/vector
        wetted perimeter (m)
    R : real scaler/vector
        Hydraulic radius (m) w/ R = A/P
    w : real scaler/vector
        Width corresponding to z according to the bathymetry

    """

    nz = len(_z) ## Num. points to treat 
    ns = len(_Zs)-1 ## Num. of describing pairs (Ws, Zs) for the cross section ## NOTE changed from 'len(Zs)' to 'len(Zs)-1'

    _Ws = _Ws/2 ## Divide width by 2 to simplify the calculas
    ## Initialisation
    A = np.zeros(nz)
    W = np.zeros(nz)
    P = np.zeros(nz)
    R = np.zeros(nz)
    w = np.zeros(nz)

    for i in range(nz): 
        A[i] = 0
        w[i] = _Ws[0]
        if _z[i] >= _Zs[0]: ## NOTE updated from _z[i] > _Zs[0].
            P[i] = _Ws[0]
        else:
            # Maybe better to put also Ws(1) for P since the width is w(i)=Ws(1);
            P[i]=_Ws[0]
        j = 0
        ## Add complete areas
        if i == 60:
            pass

        #print("cs:", _Ws, _Zs)

        while j < ns and _z[i] > _Zs[j+1]:
            A[i] = A[i] + (_Ws[j] + _Ws[j+1])*(_Zs[j+1] - _Zs[j]) ## Area
            P[i] = P[i] + np.sqrt((_Ws[j+1] - _Ws[j])**2 + (_Zs[j+1] - _Zs[j])**2) ## Half perimeter.  
            j += 1
        
        ## Add remaining part using interpolation
        if _z[i] > _Zs[j]:
            if j < ns:
                w[i] = _Ws[j] + (_Ws[j+1] - _Ws[j])*(_z[i] - _Zs[j])/(_Zs[j+1] - _Zs[j])
            else:
                w[i] = _Ws[ns] ## Assume vertical banks above max _Zs
            A[i] = A[i] + (_Ws[j] + w[i])*(_z[i] - _Zs[j]) ## Area
            P[i] = P[i] + np.sqrt((w[i]-_Ws[j])**2 + (_z[i] - _Zs[j])**2) ## Perimeter

        if P[i]<= 0:
            pass
            #print(i, P[i])
            #print(bug3)

        P[i] = P[i]*2 ## Full perimeter
        w[i] = w[i]*2 ## Full width
        if P[i] > 0:
            R[i] = A[i]/P[i]
        else:
            R[i] = 0

    #print(P)
    #print(tug)
    return A, P, R, w

def call_func_APR(node_w, node_z, node_xr, node_yr, params, param_dict, coeff_array = [], t=-1.0):
    ## Calculate the dry bathymetry. Pass vector of z, xr, yr to fnc_APR.
    
    node_a = node_w.copy()
    node_p = node_w.copy()
    node_r = node_w.copy()
    node_w_simp = node_w.copy()
    
    #OPTION
    #0: APR only, 1: APR + Igor computation of bb (additional)

    depth_mean=[]
    A0_mean = []
    P0_mean = []
    W0_mean = []

    if np.array(coeff_array).size < 1.:
        coeff_array = np.ones(len(node_z[0]))

    ## Iterate over nodes
    for i in range(len(node_w)):
        if t >= 0.:
            node_a[i], node_p[i], node_r[i], node_w_simp[i] = fnc_APR(node_z[i,t] * coeff_array, node_xr[i], node_yr[i])
        else:
            node_a[i], node_p[i], node_r[i], node_w_simp[i] = fnc_APR(node_z[i] * coeff_array, node_xr[i], node_yr[i])  ## return 4x 1D arrays of A, R, P, w_simplified
        depth_section = 0.

        if param_dict["bb_computation"]:
            depth_section = np.nanmax(node_yr[i])-np.nanmin(node_yr[i])
            A0=0
            P0=node_xr[i][0]
            for j in range(0,len(node_xr[i])-1):
                
                A0=A0+((node_xr[i][j]+node_xr[i][j+1])/2)*(node_yr[i][j+1]-node_yr[i][j])
                P0=P0+2*np.sqrt((node_xr[i][j+1]/2 - node_xr[i][j]/2)**2 + (node_yr[i][j+1] - node_yr[i][j])**2)

                if (node_yr[i][j]-np.nanmin(node_yr[i])) > (np.nanmax(node_yr[i])-np.nanmin(node_yr[i]))/2:
                    depth_section = (np.nanmax(node_yr[i])-np.nanmin(node_yr[i]))/2
                    break

            if False: #Plot for testing
                results2 = f_approx_sections_v6(node_w[i], node_z[i],
                                                    params.approx_section_params[0], params.approx_section_params[1],
                                                    params.approx_section_params[2])
                plt.plot(node_xr[i], node_yr[i])
                plt.plot(results2[0], results2[1])
                plt.plot(node_w[i], node_z[i], marker=".", linestyle='None')
                print("A0, P0, R, depth, depth/R:", A0, P0, A0/P0, depth_section, depth_section/(A0/P0))
                plt.show()
                plt.clf()

            A0_mean.append(A0)
            P0_mean.append(P0)
            W0_mean.append(node_xr[i][0])
                

        index_temp = np.where(node_w_simp[i] == 0.)
        node_w_simp[i][index_temp] = np.nanmin(node_xr[i][np.where(node_xr[i] > 0.)])
    
        depth_mean.append(depth_section)

    if not param_dict["bb_computation"]:
        bb = 9999.0        

    else:
        depth_mean = np.nanmean(depth_mean)
        A0_mean = np.nanmean(A0_mean)
        P0_mean = np.nanmean(P0_mean)
        W0_mean = np.nanmean(W0_mean)

        print("depth_mean, A0_mean, P0_mean, W0_mean:", depth_mean, A0_mean, P0_mean, W0_mean)
        print("mean Radius:", A0_mean/P0_mean, depth_mean/(A0_mean/P0_mean))
        #print(bug)
        #pause=input("Pause")

        #looking for zb_lb    
        
        #depth_mean=0.92
        #A0_mean=237
        #P0_mean=303
        #W0_mean=275
        
        #bb=compute_bb(depth_mean, W0_mean, A0_mean, P0_mean)
        bb = 9999.0 #not needed for L2

        print("Final computed bb:", bb)

        if bb < 1:
            bb=1

        print("Final bb:", bb)
        #pause=input("Pause")
        #print(bug)

    return node_a, node_p, node_r, node_w_simp, bb