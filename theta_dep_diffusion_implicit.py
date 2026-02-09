from petsc4py import PETSc
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import copy 
import json 

def calculate_lagrange_function(x, x_0, x_1, x_2, y_0,y_1,y_2):

    l_0 = ((x-x_1)/(x_0-x_1))*((x-x_2)/(x_0-x_2))

    l_1 = ((x-x_0)/(x_1-x_0))*((x-x_2)/(x_1-x_2))

    l_2 = ((x-x_0)/(x_2-x_0))*((x-x_1)/(x_2-x_1))

    L_n = y_0*l_0 + y_1*l_1 + y_2*l_2

    return L_n

def calculate_deriv_of_lagrange_function(x, x_0, x_1, x_2, y_0,y_1,y_2):

    d_l_0 = (1/(x_0-x_1))*((x-x_2)/(x_0-x_2)) + ((x-x_1)/(x_0-x_1))*(1/(x_0-x_2))


    d_l_1 = (1/(x_1-x_0))*((x-x_2)/(x_1-x_2)) + ((x-x_0)/(x_1-x_0))*(1/(x_1-x_2))


    d_l_2 = (1/(x_2-x_0))*((x-x_1)/(x_2-x_1)) + ((x-x_0)/(x_2-x_0))*(1/(x_2-x_1))


    L_n = y_0*d_l_0 + y_1*d_l_1 + y_2*d_l_2

    return L_n

def calculate_second_deriv_of_lagrange_function(x, x_0, x_1, x_2, y_0,y_1,y_2):

    dd_l_0 = (1/(x_0-x_1))*((1)/(x_0-x_2)) + ((1)/(x_0-x_1))*(1/(x_0-x_2)) #note that these derivatives are no longer dependent on x, we can fix this later. It also may change if more poly present


    dd_l_1 = (1/(x_1-x_0))*((1)/(x_1-x_2)) + ((1)/(x_1-x_0))*(1/(x_1-x_2))


    dd_l_2 = (1/(x_2-x_0))*((1)/(x_2-x_1)) + ((1)/(x_2-x_0))*(1/(x_2-x_1))


    L_n = y_0*dd_l_0 + y_1*dd_l_1 + y_2*dd_l_2

    return L_n


def formFunction_theta(snes, theta_guess_global, residual_vec):
    """
    Residual R = (theta_new - theta_old)/dt - Lap( mu(theta_new, u_current) )
    Uses Neumann BCs the same way you did for laplacian.
    """
    
    da = snes.getDM()

    residual_vec.zeroEntries() 
    # local copy for stencil access
    
    theta_guess_local = da.createLocalVec()
    da.globalToLocal(theta_guess_global, theta_guess_local) 
    theta_arr = da.getVecArray(theta_guess_local)   

    ctx = snes.getApplicationContext()
    dt = ctx["dt"]
    Cbar = ctx["Cbar"]
    Lmk_vals = ctx["Lmk_vals"]
    Pmk_vals = ctx["Pmk_vals"]
    conc_vals = ctx["conc_vals"]
    h = ctx["h"]
  
    u_global = ctx["u_array"]      
    u_local = da.createLocalVec()
    da.globalToLocal(u_global, u_local)
    u_arr = da.getVecArray(u_local)   

    theta_old_global = ctx["theta_old"]
    theta_old_local = da.createLocalVec()
    da.globalToLocal(theta_old_global, theta_old_local)
    theta_old_arr = da.getVecArray(theta_old_local)   


    start, end = da.getCorners()[0][0],da.getCorners()[1][0]

    N = da.getCorners()[1][0]

    for n in range(start, end):

       # compute chemical potential mu at this node using theta_local[n] and current u
        mu_center = calculate_chem_pot(theta_arr[n], u_arr[n], Lmk_vals, Pmk_vals, conc_vals)

        # compute Lap(mu) at node n using theta_local's neighborhood for mu values
        # we need mu at neighbors -> compute mu at n-1, n, n+1   
        mu_left = calculate_chem_pot(theta_arr[n-1], u_arr[n-1], Lmk_vals, Pmk_vals, conc_vals)
        mu_right = calculate_chem_pot(theta_arr[n+1], u_arr[n+1], Lmk_vals, Pmk_vals, conc_vals)
        lap_mu = (mu_right - 2*mu_center + mu_left)/h**2

        d_theta = (theta_arr[n+1] - theta_arr[n-1])/(2.0*h)
        d_mu = (mu_right - mu_left)/(2.0*h)
        
        flow_val = (1-2.0*theta_arr[n])*d_theta*d_mu + (theta_arr[n] - theta_arr[n]**2)*lap_mu
        residual_vec.setValue(n,(theta_old_arr[n] - theta_arr[n])/dt + flow_val)
 
    residual_vec.assemble()
    return

def formJacobian_theta(snes, theta_global, J_theta, P_theta):
    """
    Jacobian: dR/dtheta_new = (1/dt) * I - Lap * dmu/dtheta
    For 1D, Lap discretization and dmu/dtheta gives tridiagonal blocks.
    Use your helper dF(theta) which returns d^2 psi/d theta^2 (the local derivative of mu wrt theta).
    """
    
    da = snes.getDM()

    theta_local = da.createLocalVec()
    da.globalToLocal(theta_global, theta_local)
    theta_arr = da.getVecArray(theta_local)   
    
    J_theta.zeroEntries()

    ctx = snes.getApplicationContext()
    dt = ctx["dt"]
    Cbar = ctx["Cbar"]
    Lmk_vals = ctx["Lmk_vals"]
    Pmk_vals = ctx["Pmk_vals"]
    conc_vals = ctx["conc_vals"]
    h = ctx["h"]
    u_global = ctx["u_array"]

    u_local = da.createLocalVec()
    da.globalToLocal(u_global,u_local)
    u_arr = da.getVecArray(u_local)
   
    #start, end = da.getRanges()[0]
    N = J.getSize()[0]
    start,end = da.getCorners()[0][0],da.getCorners()[1][0]
    for n in range(start, end):

        # compute dmu/dtheta at neighbors (we need at n, n-1, n+1)
        # dmu/dtheta at a node is d^2 psi_hom / dtheta^2 for the local part
        # Here we assume mu = d psi_hom/d theta + ... and mu depends only locally on theta for that term
        dmu_theta_left = calculate_deriv_chem_pot(theta_arr[n-1], u_arr[n-1], Lmk_vals, Pmk_vals, conc_vals)
        dmu_theta_right = calculate_deriv_chem_pot(theta_arr[n+1], u_arr[n+1], Lmk_vals, Pmk_vals, conc_vals)
        dmu_theta = calculate_deriv_chem_pot(theta_arr[n],u_arr[n],Lmk_vals, Pmk_vals, conc_vals)

        mu_center = calculate_chem_pot(theta_arr[n], u_arr[n], Lmk_vals, Pmk_vals, conc_vals)
        mu_left = calculate_chem_pot(theta_arr[n-1], u_arr[n-1], Lmk_vals, Pmk_vals, conc_vals)
        mu_right = calculate_chem_pot(theta_arr[n+1], u_arr[n+1], Lmk_vals, Pmk_vals, conc_vals)
     
        d_mu_spatial = (mu_right - mu_left)/(2.0*h)

        d_theta = (theta_arr[n+1] - theta_arr[n-1])/(2.0*h)

        lap_mu = (mu_right - 2.0*mu_center + mu_left)/(h**2)

        J_theta_nmin1_val = ((-1.0/(2.0*h))*d_mu_spatial - d_theta*(dmu_theta_left)*(1/(2.0*h)))*(1-2.0*theta_arr[n]) + (dmu_theta_left)*(theta_arr[n] - theta_arr[n]**2)*(1/(h**2))

        J_theta_cent_val = (-1.0/dt) - 2.0*(d_theta)*(d_mu_spatial) + (1-2.0*theta_arr[n])*lap_mu + (-2.0/h**2)*(theta_arr[n]-theta_arr[n]**2)*dmu_theta

        J_theta_np1_val = ((1.0/(2.0*h))*d_mu_spatial + d_theta*(dmu_theta_right)*(1/(2.0*h)))*(1-2.0*theta_arr[n]) + (dmu_theta_right)*(theta_arr[n] - theta_arr[n]**2)*(1/(h**2))


        #   print(J_theta_nmin1_val,J_theta_cent_val, J_theta_np1_val)
        J_theta.setValues(n, [(n-1)%N, n, (n+1)%N],[J_theta_nmin1_val, J_theta_cent_val , J_theta_np1_val])

    J_theta.assemble()

    if J_theta != P_theta:
        P_theta.assemble()

    return

def calculate_deriv_chem_pot(theta,phi,A_cos_vals,B_sin_vals,concentration_vals):

    A_m_of_theta = 0
    B_m_of_theta = 0

    for entry in A_cos_vals:
        x_0 = concentration_vals[0]
        x_1 = concentration_vals[1]
        x_2 = concentration_vals[2]

        y_0 = A_cos_vals[entry][0]
        y_1 =  A_cos_vals[entry][1]
        y_2 =  A_cos_vals[entry][2]

        A_m_of_theta += (calculate_second_deriv_of_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))*np.cos(2*np.pi*entry*phi) #this builds the constants in front of the cosine modes as a function of composition, theta.

    for entry in B_sin_vals:

        x_0 = concentration_vals[0]
        x_1 = concentration_vals[1]
        x_2 = concentration_vals[2]

        y_0 = B_sin_vals[entry][0]
        y_1 =  B_sin_vals[entry][1]
        y_2 =  B_sin_vals[entry][2]  #this builds the constants in front of the sine modes as a function of composition, theta.

       # print(calculate_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))
        B_m_of_theta += (calculate_second_deriv_of_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))*np.sin(2*np.pi*entry*phi)

    chem_pot_deriv = (A_m_of_theta + B_m_of_theta)

    return chem_pot_deriv

def calculate_chem_pot(theta,phi,A_cos_vals,B_sin_vals,concentration_vals):

    A_m_of_theta = 0
    B_m_of_theta = 0

    for entry in A_cos_vals:
        x_0 = concentration_vals[0]
        x_1 = concentration_vals[1]
        x_2 = concentration_vals[2]

        y_0 = A_cos_vals[entry][0]
        y_1 =  A_cos_vals[entry][1]
        y_2 =  A_cos_vals[entry][2]

        A_m_of_theta += (calculate_deriv_of_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))*np.cos(2*np.pi*entry*phi) #this builds the constants in front of the cosine modes as a function of composition, theta.

    for entry in B_sin_vals:

        x_0 = concentration_vals[0]
        x_1 = concentration_vals[1]
        x_2 = concentration_vals[2]

        y_0 = B_sin_vals[entry][0]
        y_1 =  B_sin_vals[entry][1]
        y_2 =  B_sin_vals[entry][2]  #this builds the constants in front of the sine modes as a function of composition, theta.

       # print(calculate_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))
        B_m_of_theta += (calculate_deriv_of_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))*np.sin(2*np.pi*entry*phi)

    chem_pot = (A_m_of_theta + B_m_of_theta) + calculate_psi_t_at_u_and_theta(0.0,0.0,A_cos_vals,B_sin_vals,concentration_vals) - calculate_psi_t_at_u_and_theta(0.33,1.0,A_cos_vals,B_sin_vals,concentration_vals)

    return chem_pot

def calculate_lap_of_chem(chem_pot,h): #similar structure as form func. Now we just haver P to serve as an "optional" preconditioner
    lap_of_chem_pot = np.zeros(len(chem_pot))

    for n in range(0,len(chem_pot)):
        if n == 0:
            lap_of_chem_pot[n] = (chem_pot[1]-2.0*chem_pot[0]+chem_pot[-1])/h**2  #zero flux boundary condition to start
        if n == len(chem_pot)-1:
            lap_of_chem_pot[n] = (chem_pot[n-1]-2.0*chem_pot[n]+chem_pot[0])/h**2
        else:
            lap_of_chem_pot[n] = (chem_pot[n+1]-2*chem_pot[n]+chem_pot[n-1])/h**2

    return lap_of_chem_pot
def calculate_psi_t_at_u_and_theta(phi,theta, A_cos_vals, B_sin_vals,concentration_vals):

    A_m_of_theta = 0
    B_m_of_theta = 0

    for entry in A_cos_vals:
        x_0 = concentration_vals[0]
        x_1 = concentration_vals[1]
        x_2 = concentration_vals[2]

        y_0 = A_cos_vals[entry][0]
        y_1 =  A_cos_vals[entry][1]
        y_2 =  A_cos_vals[entry][2]

        A_m_of_theta += (calculate_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))*np.cos(2*np.pi*entry*phi) #this builds the constants in front of the cosine modes as a function of composition, theta.

    for entry in B_sin_vals:

        x_0 = concentration_vals[0]
        x_1 = concentration_vals[1]
        x_2 = concentration_vals[2]

        y_0 = B_sin_vals[entry][0]
        y_1 =  B_sin_vals[entry][1]
        y_2 =  B_sin_vals[entry][2]  #this builds the constants in front of the sine modes as a function of composition, theta.

       # print(calculate_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))
        B_m_of_theta += (calculate_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))*np.sin(2*np.pi*entry*phi)

    return A_m_of_theta + B_m_of_theta

def calculate_deriv_of_psi_hom(phi,theta, A_cos_vals, B_sin_vals,concentration_vals):

    A_m_of_theta = 0
    B_m_of_theta = 0

    for entry in A_cos_vals:
        x_0 = concentration_vals[0]
        x_1 = concentration_vals[1]
        x_2 = concentration_vals[2]

        y_0 = A_cos_vals[entry][0]
        y_1 =  A_cos_vals[entry][1]
        y_2 =  A_cos_vals[entry][2]

        A_m_of_theta += -1.0*(2*np.pi*entry)*(calculate_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))*np.sin(2*np.pi*entry*phi) #this builds the constants of deriv in front of the sine modes as a function of composition, theta.

    for entry in B_sin_vals:

        x_0 = concentration_vals[0]
        x_1 = concentration_vals[1]
        x_2 = concentration_vals[2]

        y_0 = B_sin_vals[entry][0]
        y_1 =  B_sin_vals[entry][1]
        y_2 =  B_sin_vals[entry][2]  #this builds the constants of deriv in front of the  cosine modes as a function of composition, theta.

        B_m_of_theta += (2*np.pi*entry)*(calculate_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))*np.cos(2*np.pi*entry*phi)

    return (A_m_of_theta + B_m_of_theta)


def calculate_second_deriv_of_psi_hom(phi,theta, A_cos_vals, B_sin_vals,concentration_vals):

    A_m_of_theta = 0
    B_m_of_theta = 0

    for entry in A_cos_vals:
        x_0 = concentration_vals[0]
        x_1 = concentration_vals[1]
        x_2 = concentration_vals[2]

        y_0 = A_cos_vals[entry][0]
        y_1 =  A_cos_vals[entry][1]
        y_2 =  A_cos_vals[entry][2]

        A_m_of_theta += -1.0*(2*np.pi*entry)*(2*np.pi*entry)*(calculate_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))*np.cos(2*np.pi*entry*phi) #this builds the constants of deriv in front of the sine modes as a function of composition, theta.

    for entry in B_sin_vals:

        x_0 = concentration_vals[0]
        x_1 = concentration_vals[1]
        x_2 = concentration_vals[2]

        y_0 = B_sin_vals[entry][0]
        y_1 =  B_sin_vals[entry][1]
        y_2 =  B_sin_vals[entry][2]  #this builds the constants of deriv in front of the  cosine modes as a function of composition, theta.

        B_m_of_theta += -1.0*(2*np.pi*entry)*(2*np.pi*entry)*(calculate_lagrange_function(theta,x_0,x_1,x_2,y_0,y_1,y_2))*np.sin(2*np.pi*entry*phi)

    return (A_m_of_theta + B_m_of_theta)


def calculate_psi_hom(phi,theta,A_cos_vals,B_sin_vals,concentration_vals):

    psi_hom = calculate_psi_t_at_u_and_theta(phi,theta,A_cos_vals,B_sin_vals,concentration_vals) - (1-theta)*calculate_psi_t_at_u_and_theta(0,0,A_cos_vals,B_sin_vals,concentration_vals) - (theta)*calculate_psi_t_at_u_and_theta(0.33,1,A_cos_vals,B_sin_vals,concentration_vals)

    return psi_hom

def initial_theta_prof(x_array,N,sigma):
    cent = N/2
    z_var = (x_array - cent)/(sigma)
    theta_init = np.exp(-z_var**2)

    return theta_init

def initial_theta_prof_sig_step2(x_array,N,width):
    cent1 = (N)/8
    cent2 = (7*N)/8 #this ensures that the average concentration profile is 0.75. Helpful for troubleshooting long time limit. 

    x_array_trans_1 = x_array - cent1
    x_array_trans_2 = x_array - cent2

    sigmoid_denom_1 = (1+np.exp(-width*(x_array_trans_1)))
    sigmoid_1 = 1/sigmoid_denom_1

    sigmoid_denom_2 = (1+np.exp(-width*(x_array_trans_2)))
    sigmoid_2 = -1/sigmoid_denom_2

    theta_init = 1-(0.3*(sigmoid_1 + sigmoid_2)+0.2)

    return theta_init


def initial_theta_prof_sig_step(x_array,N,width):
    cent1 = (N)/8
    cent2 = (7*N)/8 #this ensures that the average concentration profile is 0.75. Helpful for troubleshooting long time limit. 

    x_array_trans_1 = x_array - cent1
    x_array_trans_2 = x_array - cent2

    sigmoid_denom_1 = (1+np.exp(-width*(x_array_trans_1)))
    sigmoid_1 = 1/sigmoid_denom_1

    sigmoid_denom_2 = (1+np.exp(-width*(x_array_trans_2)))
    sigmoid_2 = -1/sigmoid_denom_2

    theta_init = (sigmoid_1 + sigmoid_2)

    return theta_init


def nonlinear_term(u_entry, theta_entry, Lmk_vals, Pmk_vals,concentration_vals):

    f_nonlin_val = calculate_deriv_of_psi_hom(u_entry,theta_entry,Lmk_vals,Pmk_vals,concentration_vals)

    return f_nonlin_val

def dF(u_entry, theta_entry, Lmk_vals, Pmk_vals,concentration_vals):

    df_nonlin_val = calculate_second_deriv_of_psi_hom(u_entry,theta_entry,Lmk_vals,Pmk_vals,concentration_vals)
    return df_nonlin_val


def formFunction(snes, u_global, residual): #the signature of formFunction must be "snes, guess, residual" for petsc to understand. Since I also need to pass "context" I can use ctx flag
    
    da = snes.getDM()
    
    #u_arr = u_global.getArray()

    residual.zeroEntries()

    u_local = da.createLocalVec()
    da.globalToLocal(u_global, u_local)
    u_arr = da.getVecArray(u_local)

    ctx = snes.getApplicationContext()

    Cbar = ctx["Cbar"]
    Lmk_vals = ctx["Lmk_vals"]
    Pmk_vals = ctx["Pmk_vals"]
    concentration_vals = ctx["conc_vals"]
    h = ctx["h"]
    theta_global = ctx["theta"]
    #N_max = da.getRanges()[0][1]

    theta_local = da.createLocalVec()
    da.globalToLocal(theta_global,theta_local)
    theta_arr = da.getVecArray(theta_local)

    start,end = da.getCorners()[0][0],da.getCorners()[1][0]
    
    for n in range(start,end):  #the laplacian is calculated here. We use neumann boundary conditions, which places a constraint on how the laplacian is calculated
        
        #left = (n-1)%N_max
        #right = (n+1)%N_max
        lap = (u_arr[n+1]-2.0*u_arr[n]+u_arr[n-1])/h**2

        Fn = (nonlinear_term(u_arr[n],theta_arr[n], Lmk_vals, Pmk_vals, concentration_vals)) - Cbar*lap  #this is an entry in the vector that includes all of the equations of motion that must be solved simultaneously.
        residual.setValue(n,Fn/Cbar)
    
    residual.assemble()

    return

def generate_x_vals(da,x_max,x_min,N):

   x_global = da.createGlobalVec()
   x_local = da.createLocalVec()

   #x_arr = x_local.getArray()
   #print(da.getCorners())
   xs,xe = da.getCorners()[0][0],da.getCorners()[1][0]

   #ghost_start, ghost_end = da.getGhostRanges()[0]

   dx = (x_max - x_min)/N
   for n in range(xs,xe):
       x_global[n] = x_min + n * dx

   da.globalToLocal(x_global, x_local)
   #x_arr = x_local.getArray(readonly=True)

   return x_global

def generate_theta_vals(da,x_global,N,width):

   theta_global = da.createGlobalVec()

   #theta_arr = theta_local.getArray()
   #x_local = da.createLocalVec()

   xs,xe = da.getCorners()[0][0],da.getCorners()[1][0]

   #ghost_start, ghost_end = da.getGhostRanges()[0]

   for n in range(xs,xe):
       theta_global[n] = initial_theta_prof_sig_step2(x_global[n],N,width)

   #da.localToGlobal(theta_local, theta_global)

   #theta_arr = theta_local.getArray(readonly=True)

   return theta_global

def generate_u_guess(da,x_global,N,width):

   u_guess_global = da.createGlobalVec()
   #theta_local = da.createLocalVec()

   #theta_arr = theta_local.getArray()
   #x_local = da.createLocalVec()

   xs,xe = da.getCorners()[0][0],da.getCorners()[1][0]

   #ghost_start, ghost_end = da.getGhostRanges()[0]

   for n in range(xs,xe):
       u_guess_global[n] = 0.3*initial_theta_prof_sig_step(x_global[n],N,width) + 0.3

   #da.localToGlobal(theta_local, theta)

   #theta_arr = theta_local.getArray(readonly=True)

   return u_guess_global



def formJacobian(snes, u_global, J, P): #similar structure as form func. Now we just haver P to serve as an "optional" preconditioner
    
    da = snes.getDM()
    
    u_local = da.createLocalVec()
    da.globalToLocal(u_global,u_local)
    u_arr = da.getVecArray(u_local)   

    J.zeroEntries()
    ctx = snes.getApplicationContext()
    
    Cbar = ctx["Cbar"]
    Lmk_vals = ctx["Lmk_vals"]
    Pmk_vals = ctx["Pmk_vals"]
    concentration_vals = ctx["conc_vals"]
    h = ctx["h"]
    theta_global = ctx["theta"]
    
    theta_local = da.createLocalVec()
    da.globalToLocal(theta_global, theta_local)
    theta_arr = da.getVecArray(theta_local)   

    N = J.getSize()[0]
    #N_max = da.getRanges()[0][1]
    start,end = da.getCorners()[0][0],da.getCorners()[1][0]
    #print(start,end)
    for n in range(start,end):

        cols = [(n-1)%N, n, (n+1)%N]
        
        left = -1.0/h**2
        right = -1.0/h**2
        cent = (dF(u_arr[n],theta_arr[n],Lmk_vals,Pmk_vals,concentration_vals)+2*Cbar/h**2)/Cbar
        J.setValues(n,cols,[left, cent, right])
        #print('completed the following jacobian entry: ', 'row: ', n, 'with columns: ', n-1,n,n+1)
    J.assemble()

    if J != P:
      P.assemble()

    return

############################################### main starts here ###################################################################################################################
fig_filename = 'long_time_lim_backward_difference_1emin4_tstep_vary_L.png'
data_filename = 'long_time_lim_backward_difference_1emin4_tstep_vary_L.json'

plt.rcParams['font.size'] = 15
plt.rcParams['axes.linewidth'] = 1
fig,ax=plt.subplots(nrows=2,ncols=2, figsize = (20,20))

N = 400

ax[0][0].set_xlim(0,N)
ax[0][1].set_xlim(0, N)
ax[1][0].set_xlim((N)/8 - 20, (N)/8 + 20)
ax[1][1].set_xlim((N)/8 - 20, (N)/8 + 20)

ax[0][0].set_xlabel("Length $L/a_{0}$")
ax[0][0].set_ylabel("Displacement $u/a_{0}$")

ax[0][1].set_xlabel("Length $L/a_{0}$")
ax[0][1].set_ylabel("Composition")

ax[1][0].set_xlabel("Length $L/a_{0}$")
ax[1][0].set_ylabel("Displacement $u/a_{0}$")

ax[1][1].set_xlabel("Length $L/a_{0}$")
ax[1][1].set_ylabel("Composition")

plt.tight_layout()

############################################ end of figure set up -- on to data init #############################################################################
Lmk_vals = {0:[-39.91161,-55.7967,-55.7967], 1:[-3.582353,5.68681,5.68681], 2:[-1.7983,2.794065,2.794065] ,3:[0.0,0.2969012,0.2969012]}

Pmk_vals = {1:[-4.692103,-0.429589,-0.429589], 2:[1.963535,0.5694743,-0.5694743], 3:[0.1799266,0.0375,-0.0375], 4:[ 0.1799266,0.0,0.0]}

#Lmk_vals = {0:[0.0,0.0,0.0]}
#Pmk_vals = {0:[0.0,0.0,0.0]}

concentration_vals = [0.08,0.7,0.4]

#N = 1200
width = 10.0

x_max = 400.0 

x_min = 0.0

Cbar = 5e3

delta_t = 1e-3

iterations = 500001

##################################### end of parameter initialization -- start of grid initialization ############################################################################


da = PETSc.DMDA().create(dim=1,dof=1,sizes=(N+1,),boundary_type=(PETSc.DM.BoundaryType.PERIODIC,),stencil_width=1)  #creates a 1D "distributed mesh" (DMDA = distributed mesh for distributed arrays)

print('initializing x array...')

x_global = generate_x_vals(da,x_max,x_min,N) #builds an array of x_values called "x" with an x_local and x_global array. This array can be called to build the concentration profile.

#x_local = da.createLocalVec()
#da.globalToLocal(x_global,x_local)

x_arr = x_global.getArray() #required for plotting below

#print(len(x_arr),x_arr)

print('*****************************************************************************************')
print('initializing theta array...')
print('*****************************************************************************************')

theta_global = generate_theta_vals(da,x_global,N,width) #builds an array of theta_values based on initial theta prof function and x array.

print(np.average(theta_global.getArray()))

#theta_local = da.createLocalVec()
#da.globalToLocal(theta_global,theta_local)

#theta_arr = theta_global.getArray()

#print(len(theta_arr),theta_arr)

params = {}

params["Cbar"] = Cbar
params["Lmk_vals"] = Lmk_vals
params["Pmk_vals"] = Pmk_vals
params["conc_vals"] = concentration_vals
params["h"] = (x_max - x_min)/N
params["theta"] = da.createGlobalVector()
params["theta"].setArray(theta_global.getArray())

print(np.average(params["theta"].getArray()))
print('displacement solver context dictionary has been built: ', params)
print('*****************************************************************************************')

#u = da.createGlobalVector()
#ulocal = da.createLocalVector()
u_global = generate_u_guess(da,x_global,N,width)

#ulocal.setArray(python_u)

##################################### end of field initialization -- start of solver init ###################################################################################################################
snes = PETSc.SNES().create(comm = PETSc.COMM_WORLD)
snes.setDM(da)
snes.setType('newtonls') 
snes.setApplicationContext(params)
snes.setTolerances(max_it=200)

residual = da.createGlobalVector()
snes.setFunction(formFunction,residual)
J=da.createMatrix()

#row,cols = J.getSize()
#print(row,cols)

snes.setJacobian(formJacobian,J)

theta_ctx = {
    "dt": delta_t,
    "Cbar": Cbar,
    "Lmk_vals": Lmk_vals,
    "Pmk_vals": Pmk_vals,
    "conc_vals": concentration_vals,
    "h": params["h"],
    "theta_old": da.createGlobalVector(),
    "u_array": da.createGlobalVector()
}

print('theta solver context dictionary has been built: ', theta_ctx)
print('*****************************************************************************************')


snes_theta = PETSc.SNES().create(comm = PETSc.COMM_WORLD)
snes_theta.setDM(da)
snes_theta.setType('vinewtonrsls') #these three lines initialize the scalable nonlinear equation solver(s) "SNES". Newtonls is a newton algorithm with line search.
snes_theta.setApplicationContext(theta_ctx)

snes_theta.setTolerances(max_it=200)

theta_residual =  da.createGlobalVector()
snes_theta.setFunction(formFunction_theta,theta_residual)

J_theta = da.createMatrix()
snes_theta.setJacobian(formJacobian_theta,J_theta)
theta_guess = generate_theta_vals(da,x_global,N,width) #initial theta_guess is just the initial concentration profile  

print('Displacement and Theta solvers have been initialized, starting solve step..............')
print('*****************************************************************************************')

###################################################### solvers initialized -- iterative solve begins below ###############################################################################################

for i in range(iterations):
    ####################   solve mechanics for u given current theta (your existing snes.solve call)
   
    #params["theta"].copy(theta_global)

    snes.setApplicationContext(params)

    before_u_solve = (u_global.getArray()).copy()
    snes.solve(None, u_global)
    after_u_solve = (u_global.getArray()).copy()

    if (i % 500) == 0: 
        print('disp change norm:', np.linalg.norm(after_u_solve-before_u_solve))
        print('snes_disp converged reason:', snes.getConvergedReason(), 'iterations:', snes.getIterationNumber())
    
    ###################### update context for theta solve
 
    theta_ctx["theta_old"].setArray(params["theta"].getArray())
    #print(np.average(theta_ctx["theta_old"].getArray()))

    theta_ctx["u_array"].setArray(u_global.getArray())
    
    snes_theta.setApplicationContext(theta_ctx)
    
    before = (theta_global.getArray()).copy()
    #print('initial theta avg: ', np.average(before))

    ###################### before theta solve, plot theta field and corresponding mech equillibrium.
    if (i % 500) == 0:
        store_data_dict = {'time_' + str(i) : i, 'x_arr': list(x_arr), 'disp' : list(after_u_solve), 'conc': list(before) }
        
        with open(data_filename, "a") as f: 
            f.write(json.dumps(store_data_dict) + '\n')

        ax[0][0].plot(x_arr, after_u_solve,label = r'$\tau$ = ' + str(round(i*delta_t*Cbar,2)), linewidth = 4)
        ax[1][0].plot(x_arr, after_u_solve,label = r'$\tau$ = ' + str(round(i*delta_t*Cbar,2)), linewidth = 4)

        ax[0][1].plot(x_arr, before, label = r'$\tau$ = ' + str(round(i*delta_t*Cbar,2)), linewidth = 4)
        ax[1][1].plot(x_arr, before, label = r'$\tau$ = ' + str(round(i*delta_t*Cbar,2)), linewidth = 4)
        
        ax[0][0].legend(loc = 'upper right')
        plt.savefig(fig_filename)

    #################################### solve implicit backward Euler for theta^{n+1}
    snes_theta.solve(None, theta_global)  
    after = (theta_global.getArray()).copy()
    params["theta"].setArray(theta_global.getArray())
   
    if (i % 500) == 0: 
        print('theta change norm:', np.linalg.norm(after-before), ' theta average: ', np.average(after))
        print('snes_theta converged reason:', snes_theta.getConvergedReason(), 'iterations:', snes_theta.getIterationNumber())

        print('Step ', i, 'complete!')
        print('**************************************************************************************************************') 

############################################## solve step complete .... saving data and figure(s) below ####################################################################

print('solving complete!')
#plt.legend()
plt.savefig(fig_filename)

################################# end code #############################################################################################
