import math
import cupy as cp
import pykokkos as pk
from . import stack_profiler as sp

@pk.workunit(
    d_data_ar = pk.ViewTypeInfo(layout=pk.LayoutRight, space=pk.CudaSpace),
    d_tosum_ar = pk.ViewTypeInfo(layout=pk.LayoutRight, space=pk.CudaSpace),
    d_R_vec = pk.ViewTypeInfo(layout=pk.LayoutRight, space=pk.CudaSpace),
    d_E_ar = pk.ViewTypeInfo(space=pk.CudaSpace),
    d_ns_ar = pk.ViewTypeInfo(space=pk.CudaSpace),
    d_currxbins = pk.ViewTypeInfo(pk.CudaSpace),
    d_forE_xbins = pk.ViewTypeInfo(pk.CudaSpace),
    d_curr_count = pk.ViewTypeInfo(pk.CudaSpace),
)

# Kernel for electron movement and collisions
def electron_kernel_1D(
    tid: int,
    stride: int,
    Nc: int,
    nn: float,
    dt: float,
    L: float,
    q_e: float,
    m_e: float,
    m_n: float,
    j_ev: float,
    e_ion_full: float,
    e_exc: float,
    e_ion_step: float,
    d_data_ar, pk.View2D[float],
    d_tosum_ar, pk.View2D[float],
    d_R_vec, pk.View2D[float],
    d_E_ar: pk.View1D[float],
    d_ns_ar: pk.View1D[float],
    d_currxbins: pk.View1D[int64],
    d_forE_xbins: pk.View1D[int64],
    d_curr_count: pk.View1D[pk.int32],
):   

    num_steps: int = 1

    xx: int = 0
    xy: int = 1
    xz: int = 2
    vx: int = 3
    vy: int = 4
    vz: int = 5
    
    wt: int = 0
    ai: int = 1
    ae: int = 2
    en: int = 3

    epsilon: float = 0.01

    # Loop over # steps (operator splitting)
    for mystep in range(num_steps):
        # Looping and doing all the particles 
        for i in range(start, Nc, stride):
            d_w: float = d_tosum_ar[i,wt]
            nz_ind: int = d_w > 0

            d_xx: float = d_data_ar[i,xx]
            d_xy: float = d_data_ar[i,xy]
            d_xz: float = d_data_ar[i,xz]
            
            d_vx: float = d_data_ar[i,vx] * nz_ind + epsilon * (1 - nz_ind) 
            d_vy: float = d_data_ar[i,vy] * nz_ind + epsilon * (1 - nz_ind)
            d_vz: float = d_data_ar[i,vz] * nz_ind + epsilon * (1 - nz_ind)

            # Energy / velocity of particle
            e_el: float = d_vx**2 + d_vy**2 + d_vz**2
            v_mag: float = math.sqrt(e_el)
            v_inc_x: float = d_vx / v_mag
            v_inc_y: float = d_vy / v_mag
            v_inc_z: float = d_vz / v_mag
            e_el = 0.5*m_e*j_ev*e_el
            log_e_el: float = math.log10(e_el)

            ## G0: Elastic 
            temp_ev: float = e_el + 1e-13
            a0: float = 0.008787
            b0: float = 0.07243
            c: float = 0.007048
            d: float = 0.9737
            a1: float = 3.27
            b1: float = 3.679
            x0: float = 0.2347
            x1: float = 11.71
            sig_g0: float = 9.9e-20*(a1+b1*(math.log(temp_ev/x1))**2)/(1+b1*(math.log(temp_ev/x1))**2)*(a0+b0*(math.log(temp_ev/x0))**2)/(1+b0*(math.log(temp_ev/x0))**2)/(1+c*temp_ev**d)
 
            ## G2: Ionization
            sig_g2: float = 0.
            if (e_el > 15.76):
                sig_g2: float = (2.86e-20 / math.log(90-15.76)) * math.log((e_el-15.76 + 1)) * math.exp(-1e-2*((e_el-90)/90)**2)

            ## G1 and G3 collisions disabled for now
            sig_g1: float = 0.
            sig_g3: float = 0.

            # Scale by heavy density
            sig_g0 *= nn 
            sig_g1 *= nn
            sig_g2 *= nn
            sig_g3 *= 1e6*d_ns_ar[d_currxbins[i]]

            sig_g1 = 0.
            sig_g3 = 0.
            
            # Total coll probability
            sig_tot: float = sig_g0 + sig_g1 + sig_g2 + sig_g3
            P_coll: float = 1 - math.exp(-dt*v_mag*sig_tot)
            if (P_coll > 0.1 and e_el < 0.001): # for 0-D testing w/ weird x-sections
                P_coll = 0
            
            # Determine which collision occurs 
            pcst: float = P_coll *(1./sig_tot)
            sig_range_g0: float = sig_g0 * pcst
            sig_range_g1: float = sig_range_g0 + sig_g1 * pcst
            sig_range_g2: float = sig_range_g1 + sig_g2 * pcst
            sig_range_g3: float = sig_range_g2 + sig_g3 * pcst

            # For grouping g2 and g3
            coll_indicator_g2: bool = (sig_range_g1 < d_R_vec[i,0] < sig_range_g2)
            coll_indicator_g3: bool = (sig_range_g2 < d_R_vec[i,0] < sig_range_g3)
            
            ## G0: ELASTIC
            if (d_R_vec[i,0] < sig_range_g0):
 
                ## Original electron deflection direction (vscat)      
                cos_chi: float = 1 - (2*d_R_vec[i,1]) / (1 + 8*e_el*(1 - d_R_vec[i,1]))
                chi: float = math.acos(cos_chi)
                phi: float = 2*math.pi*d_R_vec[i,2]
                theta: float = math.acos(v_inc_x)
                sign_sintheta_g0: float = (d_R_vec[i,4] > 0.5)*1 + (d_R_vec[i,4] < 0.5)*(-1)
                # TERM 1
                v_scat_x: float = cos_chi * v_inc_x
                v_scat_y: float = cos_chi * v_inc_y
                v_scat_z: float = cos_chi * v_inc_z
                # TERM 2
                fac: float = math.sin(chi)*math.sin(phi)/(math.sin(theta)*sign_sintheta_g0)
                v_scat_x += 0 
                v_scat_y += v_inc_z * fac 
                v_scat_z -= v_inc_y * fac
                # TERM 3
                fac = math.sin(chi)*math.cos(phi)/(math.sin(theta)*sign_sintheta_g0)
                v_scat_x += fac*(v_inc_y**2 + v_inc_z**2)
                v_scat_y -= fac*(v_inc_x * v_inc_y)
                v_scat_z -= fac*(v_inc_x * v_inc_z)

                v_mag_new: float = v_mag * math.sqrt(1 - (2*m_e/m_n)*(1 - cos_chi))
                d_vx = v_scat_x * v_mag_new
                d_vy = v_scat_y * v_mag_new
                d_vz = v_scat_z * v_mag_new

            ## G1: EXCITATION
            elif (sig_range_g0 < d_R_vec[i,0] < sig_range_g1):
                
                ## Original electron deflection direction (vscat)      
                cos_chi: float = 1 - (2*d_R_vec[i,1]) / (1 + 8*e_el*(1 - d_R_vec[i,1]))
                chi: float = math.acos(cos_chi)
                phi: float = 2*math.pi*d_R_vec[i,2]
                theta: float = math.acos(v_inc_x)
                sign_sintheta_g0: float = (d_R_vec[i,4] > 0.5)*1 + (d_R_vec[i,4] < 0.5)*(-1)
                # TERM 1
                v_scat_x: float = cos_chi * v_inc_x
                v_scat_y: float = cos_chi * v_inc_y
                v_scat_z: float = cos_chi * v_inc_z
                # TERM 2
                fac: float = math.sin(chi)*math.sin(phi)/(math.sin(theta)*sign_sintheta_g0)
                v_scat_x += 0 
                v_scat_y += v_inc_z * fac 
                v_scat_z -= v_inc_y * fac
                # TERM 3
                fac = math.sin(chi)*math.cos(phi)/(math.sin(theta)*sign_sintheta_g0)
                v_scat_x += fac*(v_inc_y**2 + v_inc_z**2)
                v_scat_y -= fac*(v_inc_x * v_inc_y)
                v_scat_z -= fac*(v_inc_x * v_inc_z)
            
                v_mag_new: float = math.sqrt(2*(e_el-e_exc)/(m_e*j_ev)) 
                d_vx = v_scat_x * v_mag_new
                d_vy = v_scat_y * v_mag_new
                d_vz = v_scat_z * v_mag_new
                d_tosum_ar[i,ae] = 1.

            ## G2 & G3: IONIZATION & STEPWISE
            elif (coll_indicator_g2 or coll_indicator_g3):
                ## Original electron deflection direction (vscat)      
                cos_chi: float = 1 - (2*d_R_vec[i,1]) / (1 + 8*e_el*(1 - d_R_vec[i,1]))
                chi: float = math.acos(cos_chi)
                phi: float = 2*math.pi*d_R_vec[i,2]
                theta: float = math.acos(v_inc_x)
                sign_sintheta_g0: float = (d_R_vec[i,4] > 0.5)*1 + (d_R_vec[i,4] < 0.5)*(-1)
                # TERM 1
                v_scat_x: float = cos_chi * v_inc_x
                v_scat_y: float = cos_chi * v_inc_y
                v_scat_z: float = cos_chi * v_inc_z
                # TERM 2
                fac: float = math.sin(chi)*math.sin(phi)/(math.sin(theta)*sign_sintheta_g0)
                v_scat_x += 0 
                v_scat_y += v_inc_z * fac 
                v_scat_z -= v_inc_y * fac
                # TERM 3
                fac = math.sin(chi)*math.cos(phi)/(math.sin(theta)*sign_sintheta_g0)
                v_scat_x += fac*(v_inc_y**2 + v_inc_z**2)
                v_scat_y -= fac*(v_inc_x * v_inc_y)
                v_scat_z -= fac*(v_inc_x * v_inc_z)
            
                write_ind: int = cuda.atomic.add(d_curr_count, 0, 1)
                write_ind += Nc 
                
                # Energy splitting
                e_ej: float = coll_indicator_g2 * abs(0.5*(e_el - e_ion_full)) + coll_indicator_g3 * abs(0.5*(e_el - e_ion_step))
                e_scat: float = e_ej 

                ## Original electron speed and true velocity vector (uses direction from original calc)
                v_mag_new: float = math.sqrt(2*e_scat / (j_ev*m_e))
                
                d_vx = v_scat_x * v_mag_new
                d_vy = v_scat_y * v_mag_new
                d_vz = v_scat_z * v_mag_new
            
                d_tosum_ar[i,ai] = 1.
                d_tosum_ar[i,ae] = coll_indicator_g3*(-1.)

                # Ejected particle exit angle
                cos_chi = 1 - (2*d_R_vec[i,3]) / (1 + 8*e_el*(1 - d_R_vec[i,3]))
                chi = math.acos(cos_chi)
                phi = 2*math.pi*d_R_vec[i,5]
                theta = math.acos(v_inc_x)
                sign_sintheta_g2: float = (d_R_vec[i,6] > 0.5)*(1.) + (d_R_vec[i,6] < 0.5)*(-1.)
                # TERM 1
                v_scat_x = cos_chi * v_inc_x
                v_scat_y = cos_chi * v_inc_y
                v_scat_z = cos_chi * v_inc_z
                # TERM 2
                fac = math.sin(chi)*math.sin(phi)/(math.sin(theta)*sign_sintheta_g2)
                v_scat_x += 0 
                v_scat_y += v_inc_z * fac 
                v_scat_z -= v_inc_y * fac
                # TERM 3
                fac = math.sin(chi)*math.cos(phi)/(math.sin(theta)*sign_sintheta_g2)
                v_scat_x += fac*(v_inc_y*v_inc_y + v_inc_z*v_inc_z)
                v_scat_y -= fac*(v_inc_x * v_inc_y)
                v_scat_z -= fac*(v_inc_x * v_inc_z)

                # Write new particle data
                d_data_ar[write_ind,vx] = v_scat_x * v_mag_new
                d_data_ar[write_ind,vy] = v_scat_y * v_mag_new
                d_data_ar[write_ind,vz] = v_scat_z * v_mag_new
                d_data_ar[write_ind,xx] = d_data_ar[i,xx]
                d_data_ar[write_ind,xy] = d_data_ar[i,xy]
                d_data_ar[write_ind,xz] = d_data_ar[i,xz]
                d_tosum_ar[write_ind,wt] = d_tosum_ar[i,wt]
                en_x: float = (v_scat_x * v_mag_new) * (v_scat_x * v_mag_new)
                en_y: float = (v_scat_y * v_mag_new) * (v_scat_y * v_mag_new)
                en_z: float = (v_scat_z * v_mag_new) * (v_scat_z * v_mag_new)
                d_tosum_ar[write_ind,en] = 0.5*m_e*j_ev*(en_x + en_y + en_z)
   
            # X_ADV
            d_xx += 100 * dt * d_vx
            d_xy += 100 * dt * d_vy
            d_xz += 100 * dt * d_vz
 
            # V_ADV
            index: pk.int64 = d_forE_xbins[i]
            d_vx -= dt * (q_e / m_e) * (100*d_E_ar[index])
       
            # BOUNDARY CONDITIONS
            is_valid: int = (0 < d_xx and d_xx < L and d_w > 0
            d_tosum_ar[i,wt] = (is_valid) * d_w

            d_data_ar[i,xx] = d_xx # don't kill for cuda reduction purposes
            d_data_ar[i,xy] = (is_valid) * d_xy
            d_data_ar[i,xz] = (is_valid) * d_xz

            d_vx = (is_valid) * d_vx
            d_vy = (is_valid) * d_vy
            d_vz = (is_valid) * d_vz

            # For temperature calculations 
            en_x: float = d_vx*d_vx
            en_y: float = d_vy*d_vy
            en_z: float = d_vz*d_vz
            d_tosum_ar[i,en] = 0.5*m_e*j_ev*(en_x + en_y + en_z)
            
            # Putting the v back in
            d_data_ar[i,vx] = d_vx
            d_data_ar[i,vy] = d_vy
            d_data_ar[i,vz] = d_vz

def electron_kernel(
    Nc: int,
    P_null: float,
    nu_max: float,
    nc_inds_ar,
    nc_flag_ar,
    d_nc_flag_ar: cp.ndarray,
    Ncoll_offset: int,
    Nnull_max: int,
    d_data_ar: cp.ndarray,
    d_tosum_ar: cp.ndarray,
    d_E_ar: cp.ndarray,
    d_ne_ar: cp.ndarray,
    d_ni_ar: cp.ndarray,
    d_ns_ar: cp.ndarray,
    d_Te_ar: cp.ndarray,
    cp_R_vec,
    d_R_vec: cp.ndarray,
    d_currxbins: cp.ndarray,
    d_forE_xbins: cp.ndarray,
    nn: float,
    dt: float,
    L: float,
    q_e: float,
    m_e: float,
    m_n: float,
    j_ev: float,
    e_ion_full: float,
    e_exc: float,
    e_ion_step: float,
    d_curr_count: cp.ndarray,
    d_collflag: cp.ndarray,
    d_forrecomb_v_ar: cp.ndarray,
    num_blocks: int,
    threads_per_block: int,
):
  
    cp_R_vec[0:Nc,:] = cp.random.rand(Nc, 7)
    num_threads: int = num_blocks * threads_per_block

    pk.parallel_for(
        num_threads,
        electron_kernel_1D,
        stride=num_threads,
        Nc=Nc,
        nn=nn,
        dt=dt,
        L=L,
        q_e=q_e,
        m_e=m_e,
        m_n=m_n,
        j_ev=j_ev,
        e_ion_full=e_ion_full,
        e_exc=e_exc,
        e_ion_step=e_ion_step,
        d_data_ar=d_data_ar,
        d_tosum_ar=d_tosum_ar,
        d_R_vec=d_R_vec,
        d_E_ar=d_E_ar,
        d_ns_ar=d_ns_ar,
        d_currxbins=d_currxbins,
        d_forE_xbins=d_forE_xbins,
        d_curr_count=d_curr_count
    )
       
    return(0)
