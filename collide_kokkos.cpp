/* ----------------------------------------------------------------------
   NTC algorithm for a single group with subcell method
------------------------------------------------------------------------- */

template < int DIM > void CollideVSSKokkos::collisions_one_subcell(COLLIDE_REDUCE &reduce)
{
  // loop over cells I own

  ParticleKokkos* particle_kk = (ParticleKokkos*) particle;
  particle_kk->sync(Device,PARTICLE_MASK|SPECIES_MASK);
  if (vibstyle == DISCRETE) particle_kk->sync(Device,CUSTOM_MASK);
  d_particles = particle_kk->k_particles.d_view;
  d_species = particle_kk->k_species.d_view;
  d_ewhich = particle_kk->k_ewhich.d_view;
  k_eiarray = particle_kk->k_eiarray;

  GridKokkos* grid_kk = (GridKokkos*) grid;
  grid_kk->sync(Device,CINFO_MASK);
  d_plist = grid_kk->d_plist;

  grid_kk_copy.copy(grid_kk);

  if (react) {
    ReactTCEKokkos* react_kk = (ReactTCEKokkos*) react;
    if (!react_kk)
      error->all(FLERR,"Must use TCE reactions with Kokkos");
    react_kk_copy.copy(react_kk);
  }

  copymode = 1;

  if (int(d_subcell_count.extent(0)) < nglocal || int(d_subcell_count.extent(1)) < d_plist.extent(1)) {
    d_subcell_list     = DAT::t_int_3d(Kokkos::view_alloc("collide:subcell_list",    Kokkos::WithoutInitializing),nglocal,d_plist.extent(1),d_plist.extent(1));
    d_subcell_IDlist   = DAT::t_int_2d(Kokkos::view_alloc("collide:subcell_IDlist",  Kokkos::WithoutInitializing),nglocal,d_plist.extent(1));
    d_subcell_ID_ilist = DAT::t_int_2d(Kokkos::view_alloc("collide:subcell_ID_ilist",Kokkos::WithoutInitializing),nglocal,d_plist.extent(1));
    d_subcell_ID_jlist = DAT::t_int_2d(Kokkos::view_alloc("collide:subcell_ID_jlist",Kokkos::WithoutInitializing),nglocal,d_plist.extent(1));
    d_subcell_ID_klist = DAT::t_int_2d(Kokkos::view_alloc("collide:subcell_ID_klist",Kokkos::WithoutInitializing),nglocal,d_plist.extent(1));
    d_subcell_count    = DAT::t_int_2d(Kokkos::view_alloc("collide:subcell_count",   Kokkos::WithoutInitializing),nglocal,d_plist.extent(1));
    d_neighbor_cells   = DAT::t_int_2d(Kokkos::view_alloc("collide:neighbor_cells",  Kokkos::WithoutInitializing),nglocal,d_plist.extent(1));
  }

  // ATOMIC_REDUCTION: 1 = use atomics
  //                   0 = don't need atomics
  //                  -1 = use parallel_reduce
  //

  // Reactions may create or delete more particles than existing views can hold.
  //  Cannot grow a Kokkos view in a parallel loop, so
  //  if the capacity of the view is exceeded, break out of parallel loop,
  //  reallocate on the host, and then repeat the parallel loop again.
  //  Unfortunately this leads to really messy code.

  h_retry() = 1;

  double extra_factor = sparta->kokkos->collide_extra;
  if (sparta->kokkos->collide_retry_flag) extra_factor = 1.0;

  if (react) {
    auto maxdelete_extra = maxdelete*extra_factor;
    if (d_dellist.extent(0) < maxdelete_extra) {
      memoryKK->destroy_kokkos(k_dellist,dellist);
      memoryKK->create_kokkos(k_dellist,dellist,maxdelete_extra,"collide:dellist");
      d_dellist = k_dellist.d_view;
    }

    maxcellcount = particle_kk->get_maxcellcount();
    auto maxcellcount_extra = maxcellcount*extra_factor;
    if (d_plist.extent(1) < maxcellcount_extra) {
      Kokkos::resize(grid_kk->d_plist,nglocal,maxcellcount_extra);
      d_plist = grid_kk->d_plist;
      d_subcell_list =     DAT::t_int_3d(Kokkos::view_alloc("collide:subcell_list",    Kokkos::WithoutInitializing),nglocal,maxcellcount_extra,maxcellcount_extra);
      d_subcell_IDlist =   DAT::t_int_2d(Kokkos::view_alloc("collide:subcell_IDlist",  Kokkos::WithoutInitializing),nglocal,maxcellcount_extra);
      d_subcell_ID_ilist = DAT::t_int_2d(Kokkos::view_alloc("collide:subcell_ID_ilist",Kokkos::WithoutInitializing),nglocal,maxcellcount_extra);
      d_subcell_ID_jlist = DAT::t_int_2d(Kokkos::view_alloc("collide:subcell_ID_jlist",Kokkos::WithoutInitializing),nglocal,maxcellcount_extra);
      d_subcell_ID_klist = DAT::t_int_2d(Kokkos::view_alloc("collide:subcell_ID_klist",Kokkos::WithoutInitializing),nglocal,maxcellcount_extra);
      d_subcell_count =    DAT::t_int_2d(Kokkos::view_alloc("collide:subcell_count",   Kokkos::WithoutInitializing),nglocal,maxcellcount_extra);
      d_neighbor_cells =   DAT::t_int_2d(Kokkos::view_alloc("collide:neighbor_cells",  Kokkos::WithoutInitializing),nglocal,maxcellcount_extra);
    }

    auto nlocal_extra = particle->nlocal*extra_factor;
    if (d_particles.extent(0) < nlocal_extra) {
      particle->grow(nlocal_extra - particle->nlocal);
      d_particles = particle_kk->k_particles.d_view;
      k_eiarray = particle_kk->k_eiarray;
    }
  }

  while (h_retry()) {

    if (react && sparta->kokkos->collide_retry_flag)
      backup();

    h_retry() = 0;
    h_maxdelete() = maxdelete;
    h_maxcellcount() = maxcellcount;
    h_part_grow() = 0;
    h_ndelete() = 0;
    h_nlocal() = particle->nlocal;

    Kokkos::deep_copy(d_scalars,h_scalars);

    if (sparta->kokkos->atomic_reduction) {
      if (sparta->kokkos->need_atomics)
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCollideCollisionsOneSubcell<DIM,1> >(0,nglocal),*this);
      else
        Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagCollideCollisionsOneSubcell<DIM,0> >(0,nglocal),*this);
    } else
      Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagCollideCollisionsOneSubcell<DIM,-1> >(0,nglocal),*this,reduce);

    Kokkos::deep_copy(h_scalars,d_scalars);

    if (h_retry()) {
  
      if (!sparta->kokkos->collide_retry_flag) {
        error->one(FLERR,"Ran out of space in Kokkos collisions, increase collide/extra"
                         " or use collide/retry");
      } else
        restore();

      reduce = COLLIDE_REDUCE();

      maxdelete = h_maxdelete();
      auto maxdelete_extra = maxdelete*extra_factor;
      if (d_dellist.extent(0) < maxdelete_extra) {
        memoryKK->destroy_kokkos(k_dellist,dellist);
        memoryKK->grow_kokkos(k_dellist,dellist,maxdelete_extra,"collide:dellist");
        d_dellist = k_dellist.d_view;
      }

      maxcellcount = h_maxcellcount();
      particle_kk->set_maxcellcount(maxcellcount);
      auto maxcellcount_extra = maxcellcount*extra_factor;
      if (d_plist.extent(1) < maxcellcount_extra) {
        Kokkos::resize(grid_kk->d_plist,nglocal,maxcellcount_extra);
        d_plist = grid_kk->d_plist;
      }

      auto nlocal_extra = h_nlocal()*extra_factor;
      if (d_particles.extent(0) < nlocal_extra) {
        particle->grow(nlocal_extra - particle->nlocal);
        d_particles = particle_kk->k_particles.d_view;
        k_eiarray = particle_kk->k_eiarray;
      }
    }
  }

  ndelete = h_ndelete();

  particle->nlocal = h_nlocal();

  DeviceType().fence();
  copymode = 0;

  if (h_error_flag())
    error->one(FLERR,"Collision cell volume is zero");

  particle_kk->modify(Device,PARTICLE_MASK);

  d_particles = t_particle_1d(); // destroy reference to reduce memory use
}

template < int DIM, int ATOMIC_REDUCTION >
KOKKOS_INLINE_FUNCTION
void CollideVSSKokkos::operator()(TagCollideCollisionsOneSubcell< DIM, ATOMIC_REDUCTION >, const int &icell) const {
  COLLIDE_REDUCE reduce;
  this->template operator()< DIM, ATOMIC_REDUCTION >(TagCollideCollisionsOneSubcell< DIM, ATOMIC_REDUCTION >(), icell, reduce); 
}

template < int DIM, int ATOMIC_REDUCTION >
KOKKOS_INLINE_FUNCTION
void CollideVSSKokkos::operator()(TagCollideCollisionsOneSubcell< DIM, ATOMIC_REDUCTION >, const int &icell, COLLIDE_REDUCE &reduce) const {
  if (d_retry()) return;

  int np = grid_kk_copy.obj.d_cellcount[icell];
  if (np <= 1) return;

  const double volume = grid_kk_copy.obj.k_cinfo.d_view[icell].volume / grid_kk_copy.obj.k_cinfo.d_view[icell].weight;
  if (volume == 0.0) d_error_flag() = 1;

  struct State precoln;       // state before collision
  struct State postcoln;      // state after collision

  rand_type rand_gen = rand_pool.get_state();

  // attempt = exact collision attempt count for a pair of groups
  // nattempt = rounded attempt with RN

  const double attempt = attempt_collision_kokkos(icell,np,volume,rand_gen);
  const int nattempt = static_cast<int> (attempt);
  if (!nattempt){
    rand_pool.free_state(rand_gen);
    return;
  }
  if (ATOMIC_REDUCTION == 1)
    Kokkos::atomic_add(&d_nattempt_one(),nattempt);
  else if (ATOMIC_REDUCTION == 0)
    d_nattempt_one() += nattempt;
  else
    reduce.nattempt_one += nattempt;

  int nsubcell_bydim, nscbd_sq;
  int curr_subcell, isubcell, final_subcell;
  int partner_count, temp_ctr;
  int ibox,jbox,kbox,adj_ibox,adj_jbox,adj_kbox,radius;
  int jj,jj_new;
  double c_lo_x, c_lo_y, c_lo_z, oodx, oody, oodz;
  double dim_inv;
  if (DIM == 2) dim_inv = 1./2.;
  else dim_inv = 1./3.;

  // grab cell boundares for defining subgrid
  nsubcell_bydim = (int)(pow((double)np, dim_inv));
  nscbd_sq = nsubcell_bydim * nsubcell_bydim;
  Grid::ChildCell *cells = grid->cells;
  c_lo_x = cells[icell].lo[0];
  c_lo_y = cells[icell].lo[1];
  c_lo_z = cells[icell].lo[2];
  oodx = ((double)nsubcell_bydim) / (cells[icell].hi[0] - c_lo_x);
  oody = ((double)nsubcell_bydim) / (cells[icell].hi[1] - c_lo_y);
  oodz = ((double)nsubcell_bydim) / (cells[icell].hi[2] - c_lo_z);

  for (int tp = 0; tp < np; tp++) {
    d_subcell_count(icell,tp) = 0;
  }

  // create subcell structure
  for (int tp = 0; tp < np; tp++) {
    int x_id, y_id, z_id;

    x_id = (int)(((&d_particles[d_plist(icell,tp)])->x[0]-c_lo_x)*oodx); d_subcell_ID_ilist(icell,tp) = x_id;
    y_id = (int)(((&d_particles[d_plist(icell,tp)])->x[1]-c_lo_y)*oody); d_subcell_ID_jlist(icell,tp) = y_id;
    z_id = 0;
    if (DIM == 3) {
      z_id = (int)(((&d_particles[d_plist(icell,tp)])->x[2]-c_lo_z)*oodz);
      d_subcell_ID_klist(icell,tp) = z_id;
    }
    curr_subcell = nscbd_sq*z_id + nsubcell_bydim*y_id + x_id;
    d_subcell_IDlist(icell,tp) = curr_subcell;
    d_subcell_list(icell,curr_subcell, d_subcell_count(icell,curr_subcell)) = tp;
    d_subcell_count(icell,curr_subcell)++;
  }

  // perform collisions
  // select random pair of particles, cannot be same
  // test if collision actually occurs

  for (int m = 0; m < nattempt; m++) {
    const int i = np * rand_gen.drand();
    int j;
    // subcell ID for particle i
    isubcell = d_subcell_IDlist(icell,i);

    // radius == 0 case
    final_subcell = isubcell;
    partner_count = d_subcell_count(icell, final_subcell);
    if (partner_count >= 2) {
      j = d_subcell_list(icell, final_subcell, int(partner_count*rand_gen.drand()));
      while (j == i) {
        j = d_subcell_list(icell, final_subcell, int(partner_count*rand_gen.drand()));
      }
    }
    // radius >= 1 case
    else {
      if (DIM == 2) {
        ibox = d_subcell_ID_ilist(icell, i);
        jbox = d_subcell_ID_jlist(icell, i);

        radius=1;
        while (radius < nsubcell_bydim) {
          partner_count = 0;
          temp_ctr = 0;

          // 2-D looping over neighbors
          for (int adj_i = -radius; adj_i <= radius; adj_i++) {
            adj_ibox = ibox + adj_i;
            // Bottom
            adj_jbox = jbox - radius;
            if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim) {
              d_neighbor_cells(icell, temp_ctr++) = nsubcell_bydim*adj_jbox + adj_ibox;
            }
            // Top
            adj_jbox = jbox + radius;
            if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim) {
              d_neighbor_cells(icell, temp_ctr++) = nsubcell_bydim*adj_jbox + adj_ibox;
            }
          }
          for (int adj_j = -radius+1; adj_j <= radius-1; adj_j++) {
            adj_jbox = jbox + adj_j;
            // Left
            adj_ibox = ibox - radius;
            if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim) {
              d_neighbor_cells(icell, temp_ctr++) = nsubcell_bydim*adj_jbox + adj_ibox;
            }
            // Right
            adj_ibox = ibox + radius;
            if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim) {
              d_neighbor_cells(icell, temp_ctr++) = nsubcell_bydim*adj_jbox + adj_ibox;
            }
          }
          // looking for partner at this radius
          // if good, leave. if not, radius++
          for (int tc = 0; tc < temp_ctr; tc++) {
            partner_count += d_subcell_count(icell, d_neighbor_cells(icell,tc));
          }
          if (partner_count == 0) radius++;
          else break;
        }
      }
      else if (DIM == 3) {
        ibox = d_subcell_ID_ilist(icell, i);
        jbox = d_subcell_ID_jlist(icell, i);
        kbox = d_subcell_ID_klist(icell, i);
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
                d_neighbor_cells(icell, temp_ctr++) = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
              }
              // Top
              adj_kbox = kbox + radius;
              if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                d_neighbor_cells(icell, temp_ctr++) = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
              }
            }
            for (int adj_k = -radius+1; adj_k <= radius-1; adj_k++) {
              adj_kbox = kbox + adj_k;
              // Front
              adj_jbox = jbox - radius;
              if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                d_neighbor_cells(icell, temp_ctr++) = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
              }
              // Back
              adj_jbox = jbox + radius;
              if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                d_neighbor_cells(icell, temp_ctr++) = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
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
                d_neighbor_cells(icell, temp_ctr++) = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
              }
              // Right
              adj_ibox = ibox + radius;
              if (0 <= adj_ibox && adj_ibox < nsubcell_bydim && 0 <= adj_jbox && adj_jbox < nsubcell_bydim && 0 <= adj_kbox && adj_kbox < nsubcell_bydim) {
                d_neighbor_cells(icell, temp_ctr++) = nsubcell_bydim*(nsubcell_bydim*adj_kbox + adj_jbox) + adj_ibox;
              }
            }
          }

          // looking for partner at this radius
          // if good, leave. if not, radius++
          for (int tc = 0; tc < temp_ctr; tc++) {
            partner_count = partner_count + d_subcell_count(icell, d_neighbor_cells(icell, tc));
          }
          if (partner_count == 0) radius++;
          else break;
        }
      }

      // We have a valid list of partners - select one randomly
      jj = partner_count * rand_gen.drand();
      for (int tc = 0; tc < temp_ctr; tc++) {
        jj_new = jj - d_subcell_count(icell, int(d_neighbor_cells(icell, tc)));
        if (jj_new < 0) {
          final_subcell = d_neighbor_cells(icell, tc);
          break;
        }
        else jj = jj_new;
      }
      j = d_subcell_list(icell, final_subcell, jj);
    }

    Particle::OnePart* ipart = &d_particles[d_plist(icell,i)];
    Particle::OnePart* jpart = &d_particles[d_plist(icell,j)];
    Particle::OnePart* kpart;

    // test if collision actually occurs, then perform it
    // ijspecies = species before collision chemistry
    // continue to next collision if no reaction

    if (!test_collision_kokkos(icell,0,0,ipart,jpart,precoln,rand_gen)) continue;

    // if recombination reaction is possible for this IJ pair
    // pick a 3rd particle to participate and set cell number density
    // unless boost factor turns it off, or there is no 3rd particle

    Particle::OnePart* recomb_part3 = NULL;
    int recomb_species = -1;
    double recomb_density = 0.0;
    if (recombflag && d_recomb_ijflag(ipart->ispecies,jpart->ispecies)) {
      if (rand_gen.drand() > recomb_boost_inverse)
        recomb_species = -1;
      else if (np <= 2)
        recomb_species = -1;
      else {
        int k = np * rand_gen.drand();
        while (k == i || k == j) k = np * rand_gen.drand();
        recomb_part3 = &d_particles[d_plist(icell,k)];
        recomb_species = recomb_part3->ispecies;
        recomb_density = np * fnum / volume;
      }
    }

    // perform collision and possible reaction

    int index_kpart;

    setup_collision_kokkos(ipart,jpart,precoln,postcoln);
    const int reactflag = perform_collision_kokkos(ipart,jpart,kpart,precoln,postcoln,rand_gen,
                                                   recomb_part3,recomb_species,recomb_density,index_kpart);

    if (ATOMIC_REDUCTION == 1)
      Kokkos::atomic_increment(&d_ncollide_one());
    else if (ATOMIC_REDUCTION == 0)
      d_ncollide_one()++;
    else
      reduce.ncollide_one++;

    if (reactflag) {
      if (ATOMIC_REDUCTION == 1)
        Kokkos::atomic_increment(&d_nreact_one());
      else if (ATOMIC_REDUCTION == 0)
        d_nreact_one()++;
      else
        reduce.nreact_one++;
    } else {
      rand_pool.free_state(rand_gen);
      continue;
    }

    // if jpart destroyed, delete from plist
    // also add particle to deletion list
    // exit attempt loop if only single particle left

    if (!jpart) {
      int ndelete = Kokkos::atomic_fetch_add(&d_ndelete(),1);
      if (ndelete < d_dellist.extent(0)) {
        d_dellist(ndelete) = d_plist(icell,j);
      } else {
        d_retry() = 1;
        d_maxdelete() += DELTADELETE;
        rand_pool.free_state(rand_gen);
        return;
      }
      np--;
      d_plist(icell,j) = d_plist(icell,np);
      if (np < 2) break;
    }

    // if kpart created, add to plist
    // kpart was just added to particle list, so index = nlocal-1
    // particle data structs may have been realloced by kpart

    if (kpart) {
      if (np < d_plist.extent(1)) {
        d_plist(icell,np++) = index_kpart;
      } else {
        d_retry() = 1;
        d_maxcellcount() += DELTACELLCOUNT;
        rand_pool.free_state(rand_gen);
        return;
      }

    }
  }
  rand_pool.free_state(rand_gen);
}  
