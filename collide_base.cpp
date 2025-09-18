/* ----------------------------------------------------------------------
   NTC algorithm for a single group using the subcell method
------------------------------------------------------------------------- */
template < int DIM > void Collide::collisions_one_subcell()
{

  int i,j,k,m,n,ip,np;
  int nattempt,reactflag;
  double attempt,volume;
  Particle::OnePart *ipart,*jpart,*kpart;
	
  int curr_subcell, isubcell, final_subcell;
  int partner_count, temp_ctr;
  int nsubcell_bydim, nscbd_sq;
  int ibox, jbox, kbox, adj_ibox, adj_jbox, adj_kbox, radius;
  int jj, jj_new;
  double c_lo_x, c_lo_y, c_lo_z, oodx, oody, oodz, dim_inv;
  int x_id, y_id, z_id;
  if (DIM == 2) dim_inv = 1./2.;
  else dim_inv = 1./3.;
 
  // loop over cells I own  

  Grid::ChildInfo *cinfo = grid->cinfo;
 
  Particle::OnePart *particles = particle->particles;
  int *next = particle->next; 

  for (int icell = 0; icell < nglocal; icell++) {
    np = cinfo[icell].count;
    if (np <= 1) continue;
 
    ip = cinfo[icell].first;
    volume = cinfo[icell].volume / cinfo[icell].weight;
    if (volume == 0.0) error->one(FLERR, "Collision cell volume is zero");
 
    if (np > npmax) { 
      while (np > npmax) npmax += DELTAPART;
 
      memory->destroy(plist);
      memory->create(plist,npmax,"collide:plist");
 
      memory->destroy(subcell_mostrecent);
      memory->create(subcell_mostrecent,npmax,"collide:subcell_mostrecent");

      memory->destroy(subcell_next);
      memory->create(subcell_next,npmax,"collide:subcell_mostrecent");

      memory->destroy(subcell_first);
      memory->create(subcell_first,npmax,"collide:subell_first");

      memory->destroy(subcell_IDlist);
      memory->create(subcell_IDlist,npmax,"collide:subcell_IDlist");
	    
      memory->destroy(subcell_ID_ilist);
      memory->create(subcell_ID_ilist,npmax,"collide:subcell_ID_ilist");
	    
      memory->destroy(subcell_ID_jlist);
      memory->create(subcell_ID_jlist,npmax,"collide:subcell_ID_jlist");
	    
      memory->destroy(subcell_ID_klist);
      memory->create(subcell_ID_klist,npmax,"collide:subcell_ID_klist");
 
      memory->destroy(subcell_count);
      memory->create(subcell_count,npmax,"collide:subcell_count");
 
      memory->destroy(neighbor_cells);
      memory->create(neighbor_cells,npmax,"collide:neighbor_cells");
    }

    // attempt = exact collision attempt count for all particles in cell
    // nattempt = rounded attempt with RN
    // if no attempts, continue to next grid cell
    attempt = attempt_collision(icell,np,volume);
    nattempt = static_cast<int> (attempt);
    if (!nattempt) continue;
    nattempt_one += nattempt;
	
    Grid::ChildCell *cells = grid->cells;
	  
    // subcell grid size
    nsubcell_bydim = (int)(pow((double)np, dim_inv));
    nscbd_sq = nsubcell_bydim * nsubcell_bydim;
  
    // grab cell boundaries for defining subgrid
    c_lo_x = cells[icell].lo[0];
    c_lo_y = cells[icell].lo[1];
    c_lo_z = cells[icell].lo[2];
    oodx = ((double)nsubcell_bydim) / (cells[icell].hi[0] - c_lo_x);
    oody = ((double)nsubcell_bydim) / (cells[icell].hi[1] - c_lo_y);
    oodz = ((double)nsubcell_bydim) / (cells[icell].hi[2] - c_lo_z);
	  
    // clear arrays for use
    if (DIM == 2) {
      memset(subcell_count, 0, nscbd_sq*sizeof(int));
      memset(subcell_first, -1, nscbd_sq*sizeof(int));
      memset(subcell_next, -1, nscbd_sq*sizeof(int));
    } else {
      memset(subcell_count, 0, nscbd_sq*nsubcell_bydim * sizeof(int));
      memset(subcell_first, -1, nscbd_sq*nsubcell_bydim * sizeof(int));
      memset(subcell_next, -1, nscbd_sq*nsubcell_bydim * sizeof(int));
    }
	    
    n = 0;
    while (ip >= 0) {
      plist[n] = ip;

      // subcell ID
      x_id = (int)(((&particles[plist[n]])->x[0]-c_lo_x)*oodx); subcell_ID_ilist[n] = x_id;
      y_id = (int)(((&particles[plist[n]])->x[1]-c_lo_y)*oody); subcell_ID_jlist[n] = y_id;
      z_id = 0;
      if (DIM == 3) {
        z_id = (int)(((&particles[plist[n]])->x[2]-c_lo_z)*oodz); subcell_ID_klist[n] = z_id;
      }
      curr_subcell = nscbd_sq*z_id + nsubcell_bydim*y_id + x_id;
      subcell_IDlist[n] = curr_subcell;
      
      // update arrays
      if (subcell_first[curr_subcell] < 0) {
        subcell_first[curr_subcell] = n;
      } else {
        subcell_next[subcell_mostrecent[curr_subcell]] = n;
      }
      subcell_mostrecent[curr_subcell] = n;
 
      subcell_count[curr_subcell]++;      

      ip = next[ip];
      n++;
    }

    // perform collisions
    // select pair of particles according to subcell structure, cannot be the same
    // test if collision actually occurs

    for (int iattempt = 0; iattempt < nattempt; iattempt++) {
      i = np * random->uniform();

      // subcell ID for particle i
      isubcell = subcell_IDlist[i];

      // radius == 0 case
      final_subcell = isubcell;
      partner_count = subcell_count[final_subcell];
      if (partner_count >= 2) {
        j = i;
        while (j == i) {
          j = subcell_first[isubcell];
          jj = int(partner_count*random->uniform());
          while (jj > 0) {
            j = subcell_next[j];
            jj--;
          } 
        }
      }
      // radius >= 1 case
      else {
        if (DIM == 2) {
          ibox = subcell_ID_ilist[i];
          jbox = subcell_ID_jlist[i];
          radius=1;
          while (radius < nsubcell_bydim) {
            partner_count = 0;
            temp_ctr = 0;

            // 2-D looping over neighbors
            for (int adj_i = -radius; adj_i <= radius; adj_i++) {
              adj_ibox = ibox + adj_i;
              // Bottom
              adj_jbox = jbox - radius;
              if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                neighbor_cells[temp_ctr++] = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
              }
              // Top
              adj_jbox = jbox + radius;
              if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                neighbor_cells[temp_ctr++] = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
              }
            }
            for (int adj_j = -radius+1; adj_j <= radius-1; adj_j++) {
              adj_jbox = jbox + adj_j;
              // Left
              adj_ibox = ibox - radius;
              if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                neighbor_cells[temp_ctr++] = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
              }
              // Right
              adj_ibox = ibox + radius;
              if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                neighbor_cells[temp_ctr++] = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
              }
            }

            // looking for partner at this radius
            // if good, leave. if not, radius++ 
            for (int tc = 0; tc < temp_ctr; tc++) {
              partner_count += subcell_count[neighbor_cells[tc]];
            }
            if (partner_count == 0) radius++;
            else break;
          }
        }
        else if (DIM == 3) {
          ibox = subcell_ID_ilist[i];
          jbox = subcell_ID_jlist[i];
          kbox = subcell_ID_klist[i];
          radius=1;
          while (radius < nsubcell_bydim) {
            partner_count = 0;
            temp_ctr = 0;

            // 3-D looping over neighbors
            for (int adj_i = -radius; adj_i <= radius; adj_i++) {
              adj_ibox = ibox + adj_i;
              for (int adj_j = -radius; adj_j <= radius; adj_j++) {
                adj_jbox = jbox + adj_j;
                // Bottom
                adj_kbox = kbox - radius;
                if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                  neighbor_cells[temp_ctr++] = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
                }
                // Top
                adj_kbox = kbox + radius;
                if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                  neighbor_cells[temp_ctr++] = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
                }
              }
              for (int adj_k = -radius+1; adj_k <= radius-1; adj_k++) {
                adj_kbox = kbox + adj_k;
                // Front
                adj_jbox = jbox - radius;
                if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                  neighbor_cells[temp_ctr++] = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
                }
                // Back
                adj_jbox = jbox + radius;
                if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                  neighbor_cells[temp_ctr++] = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
                }
              }
            }
            for (int adj_j = -radius+1; adj_j <= radius-1; adj_j++) {
              adj_jbox = jbox + adj_j;
              for (int adj_k = -radius+1; adj_k <= radius-1; adj_k++) {
                adj_kbox = kbox + adj_k;
                // Left
                adj_ibox = ibox - radius;              
                if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                  neighbor_cells[temp_ctr++] = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
                }
                // Right
                adj_ibox = ibox + radius;              
                if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                  neighbor_cells[temp_ctr++] = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
                }
              }
            }

            // looking for partner at this radius
            // if good, leave. if not, radius++ 
            for (int tc = 0; tc < temp_ctr; tc++) {
              partner_count += subcell_count[neighbor_cells[tc]];
            }
            if (partner_count == 0) radius++;
            else break;
          }
        }
        
        // random partner selection from designated subcells
        jj = partner_count * random->uniform();
        for (int tc = 0; tc < temp_ctr; tc++) {
          jj_new = jj - subcell_count[neighbor_cells[tc]];
          if (jj_new < 0) {
            final_subcell = neighbor_cells[tc];
            break;
          }
          else jj = jj_new;
        }
        j = subcell_first[final_subcell];
        while (jj > 0) {
          j = subcell_next[j];
          jj--;
        }
      }

      ipart = &particles[plist[i]];
      jpart = &particles[plist[j]];

      // test if collision actually occurs
      // continue to next collision if no reaction
      if (!test_collision(icell,0,0,ipart,jpart)) {
        continue;
      }

      // if recombination reaction is possible for this IJ pair
      // pick a 3rd particle to participate and set cell number density
      // unless boost factor turns it off, or there is no 3rd particle
      if (recombflag && recomb_ijflag[ipart->ispecies][jpart->ispecies]) {
        if (random->uniform() > react->recomb_boost_inverse)
          react->recomb_species = -1;
        else if (np <= 2)
          react->recomb_species = -1;
        else {
          k = np * random->uniform();
          while (k == i || k == j) k = np * random->uniform();
          react->recomb_part3 = &particles[plist[k]];
          react->recomb_species = react->recomb_part3->ispecies;
          react->recomb_density = np * update->fnum / volume;
        }
      }

      // perform collision and possible reaction
      setup_collision(ipart,jpart);
      reactflag = perform_collision(ipart,jpart,kpart);
      ncollide_one++;
      if (reactflag) nreact_one++;
      else continue;
      
      // if jpart destroyed: delete from plist, add particle to deleteion list
      // exit attempt loop if only single particle left

      if (!jpart) {
        if (ndelete == maxdelete) {
          maxdelete += DELTADELETE;
          memory->grow(dellist,maxdelete,"collide:dellist");
        }
        dellist[ndelete++] = plist[j];
        np--;
        plist[j] = plist[np];
        if (np < 2) break;
      }

      // if kpart created, add to plist
      // kpart was just added to particle list, so index = nlocal-1
      // particle data structs may have been realloced by kpart

      if (kpart) {
        if (np == npmax) {
          npmax += DELTAPART;
          memory->grow(plist,npmax,"collide:plist");
          memory->grow(subcell_next,npmax,"collide:subcell_next");
          memory->grow(subcell_first,npmax,"collide:subcell_first");
          memory->grow(subcell_mostrecent,npmax,"collide:subcell_mostrecent");
          memory->grow(subcell_IDlist,npmax,"collide:subcell_IDlist");
	  	  memory->grow(subcell_IDlist,npmax,"collide:subcell_ID_ilist");
	  	  memory->grow(subcell_IDlist,npmax,"collide:subcell_ID_jlist");
          memory->grow(subcell_IDlist,npmax,"collide:subcell_ID_klist");
          memory->grow(subcell_count,npmax,"collide:subcell_count");
          memory->grow(neighbor_cells,npmax,"collide:neighbor_cells");  
        }
        plist[np++] = particle->nlocal-1;
        particles = particle->particles;
      }

    }
  }
}
